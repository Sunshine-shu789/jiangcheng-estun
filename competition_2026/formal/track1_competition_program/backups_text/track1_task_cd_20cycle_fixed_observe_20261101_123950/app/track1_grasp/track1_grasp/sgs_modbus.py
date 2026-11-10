"""Validated Modbus/TCP access for the ESTUN SGS controller.

The module intentionally separates three levels of access:

* :class:`ModbusTcpClient` implements exactly FC1/FC3/FC5/FC6 and validates
  every MBAP/PDU field before returning data.
* :class:`SgsModbusClient` exposes the known SGS holding-register map and
  read-only status decoding.
* :class:`SgsControlClient` adds *two independent* write interlocks.  Merely
  constructing a transport with ``allow_writes=True`` is not enough to pulse a
  robot command; the higher-level control wrapper must also receive
  ``allow_robot_commands=True``.

No CLI or task pipeline calls the write layer yet.  This module is deliberately
usable with the in-process fake controller before any cabinet is connected.
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Mapping, Sequence


# ---------------------------------------------------------------------------
# Modbus/TCP exceptions


class ModbusError(RuntimeError):
    """Base class for a safe-to-report Modbus failure."""


class ModbusTimeout(ModbusError):
    """The peer did not answer within the configured timeout."""


class ModbusProtocolError(ModbusError):
    """The peer returned an invalid, truncated or mismatched Modbus frame."""


class ModbusException(ModbusError):
    """The peer returned a Modbus exception PDU."""

    def __init__(self, function_code: int, exception_code: int) -> None:
        self.function_code = int(function_code)
        self.exception_code = int(exception_code)
        text = MODBUS_EXCEPTION_TEXT.get(
            self.exception_code, "unknown Modbus exception"
        )
        super().__init__(
            f"Modbus exception {self.exception_code:#04x} for "
            f"function {self.function_code:#04x}: {text}"
        )


class ModbusWriteDisabled(ModbusError):
    """A write was attempted without the transport-level write interlock."""


class SgsCommandSafetyError(ModbusError):
    """A robot command pulse was refused by the SGS safety checks."""


class SgsUnsafeCommandState(ModbusError):
    """A command register could not be proven to have returned to zero."""


MODBUS_EXCEPTION_TEXT: Mapping[int, str] = {
    0x01: "illegal function",
    0x02: "illegal data address",
    0x03: "illegal data value",
    0x04: "slave device failure",
    0x05: "acknowledge",
    0x06: "slave device busy",
    0x08: "memory parity error",
    0x0A: "gateway path unavailable",
    0x0B: "gateway target device failed to respond",
}


# ---------------------------------------------------------------------------
# Transport


def _check_u16(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if not 0 <= value <= 0xFFFF:
        raise ValueError(f"{label} must be in [0, 0xffff], got {value!r}")
    return value


def _check_count(value: int, maximum: int, label: str = "count") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if not 1 <= value <= maximum:
        raise ValueError(f"{label} must be in [1, {maximum}], got {value!r}")
    return value


def _check_address_range(
    address: int,
    count: int,
    *,
    maximum_count: int = 125,
) -> tuple[int, int]:
    start = _check_u16(address, "address")
    amount = _check_count(count, maximum_count)
    if start + amount - 1 > 0xFFFF:
        raise ValueError("register/coil range exceeds address 0xffff")
    return start, amount


@dataclass(frozen=True)
class ModbusTcpConfig:
    """Transport settings and the low-level write interlock."""

    host: str
    port: int = 502
    unit_id: int = 1
    timeout_s: float = 2.0
    connect_timeout_s: float = 2.0
    allow_writes: bool = False

    def __post_init__(self) -> None:
        if not self.host:
            raise ValueError("host must not be empty")
        if isinstance(self.port, bool) or not isinstance(self.port, int):
            raise TypeError("port must be an integer")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be in [1, 65535]")
        if isinstance(self.unit_id, bool) or not isinstance(self.unit_id, int):
            raise TypeError("unit_id must be an integer")
        if not 1 <= self.unit_id <= 247:
            raise ValueError("unit_id must be in [1, 247]")
        for name, value in (
            ("timeout_s", self.timeout_s),
            ("connect_timeout_s", self.connect_timeout_s),
        ):
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be positive")


class ModbusTcpClient:
    """Minimal, strict Modbus/TCP master for FC1/FC3/FC5/FC6.

    There is intentionally no automatically retried write.  A caller may retry
    a read after a timeout because reads are idempotent.  Writes must be
    reconciled by a higher-level state machine.
    """

    def __init__(self, config: ModbusTcpConfig) -> None:
        self.config = config
        self._socket: socket.socket | None = None
        self._transaction_id = 0
        self._lock = threading.Lock()

    @property
    def connected(self) -> bool:
        return self._socket is not None

    def connect(self) -> None:
        with self._lock:
            self._connect_unlocked()

    def _connect_unlocked(self) -> None:
        if self._socket is not None:
            return
        try:
            sock = socket.create_connection(
                (self.config.host, self.config.port),
                timeout=self.config.connect_timeout_s,
            )
        except TimeoutError as exc:
            raise ModbusTimeout(
                f"timeout connecting to {self.config.host}:{self.config.port}"
            ) from exc
        except OSError as exc:
            raise ModbusError(
                f"cannot connect to {self.config.host}:{self.config.port}: {exc}"
            ) from exc
        sock.settimeout(self.config.timeout_s)
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        except OSError:
            # These are performance/recovery options, not protocol gates.
            pass
        self._socket = sock

    def close(self) -> None:
        sock = self._socket
        self._socket = None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass

    def __enter__(self) -> "ModbusTcpClient":
        self.connect()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _next_transaction_id(self) -> int:
        self._transaction_id = (self._transaction_id % 0xFFFF) + 1
        return self._transaction_id

    def _recv_exact(self, count: int) -> bytes:
        if self._socket is None:
            raise ModbusProtocolError("Modbus socket is not connected")
        chunks = bytearray()
        while len(chunks) < count:
            try:
                chunk = self._socket.recv(count - len(chunks))
            except TimeoutError as exc:
                raise ModbusTimeout(
                    f"timeout waiting for {count} Modbus response bytes"
                ) from exc
            except OSError as exc:
                raise ModbusError(f"Modbus receive failed: {exc}") from exc
            if not chunk:
                raise ModbusProtocolError(
                    "Modbus peer closed the connection before the complete frame"
                )
            chunks.extend(chunk)
        return bytes(chunks)

    def _transact(self, pdu: bytes) -> bytes:
        """Send one PDU and return the validated normal-response PDU."""

        if not pdu:
            raise ValueError("PDU must not be empty")
        if len(pdu) > 253:
            raise ValueError("PDU exceeds the Modbus/TCP 253-byte limit")
        function = pdu[0]

        with self._lock:
            try:
                self._connect_unlocked()
                assert self._socket is not None
                transaction_id = self._next_transaction_id()
                header = struct.pack(
                    ">HHHB",
                    transaction_id,
                    0,
                    len(pdu) + 1,
                    self.config.unit_id,
                )
                self._socket.sendall(header + pdu)

                response_header = self._recv_exact(7)
                response_tid, protocol_id, length, unit_id = struct.unpack(
                    ">HHHB", response_header
                )
                if response_tid != transaction_id:
                    raise ModbusProtocolError(
                        "unexpected transaction id "
                        f"{response_tid}; expected {transaction_id}"
                    )
                if protocol_id != 0:
                    raise ModbusProtocolError(
                        f"unexpected Modbus protocol id {protocol_id}"
                    )
                if unit_id != self.config.unit_id:
                    raise ModbusProtocolError(
                        f"unexpected unit id {unit_id}; "
                        f"expected {self.config.unit_id}"
                    )
                if not 2 <= length <= 254:
                    raise ModbusProtocolError(
                        f"invalid MBAP length field {length}"
                    )

                body = self._recv_exact(length - 1)
                if not body:
                    raise ModbusProtocolError("empty Modbus response PDU")
                response_function = body[0]

                if response_function == (function | 0x80):
                    if len(body) != 2:
                        raise ModbusProtocolError(
                            "malformed Modbus exception response"
                        )
                    raise ModbusException(function, body[1])
                if response_function != function:
                    raise ModbusProtocolError(
                        "unexpected function code "
                        f"{response_function:#04x}; expected {function:#04x}"
                    )
                return body
            except ModbusError:
                # A timed-out response could arrive later and desynchronise the
                # stream, so never reuse a socket after a failed transaction.
                self.close()
                raise
            except (OSError, TimeoutError) as exc:
                self.close()
                raise ModbusError(f"Modbus transaction failed: {exc}") from exc

    # -- FC1 -------------------------------------------------------------
    def read_coils(self, address: int, count: int = 1) -> list[bool]:
        start, amount = _check_address_range(
            address, count, maximum_count=2000
        )
        body = self._transact(struct.pack(">BHH", 0x01, start, amount))
        expected_bytes = (amount + 7) // 8
        if len(body) != 2 + expected_bytes or body[1] != expected_bytes:
            raise ModbusProtocolError(
                f"malformed FC1 response for {amount} coils: {body.hex()}"
            )
        payload = body[2:]
        return [
            bool(payload[index // 8] & (1 << (index % 8)))
            for index in range(amount)
        ]

    # -- FC3 -------------------------------------------------------------
    def read_holding_registers(self, address: int, count: int = 1) -> list[int]:
        start, amount = _check_address_range(address, count)
        body = self._transact(struct.pack(">BHH", 0x03, start, amount))
        expected_bytes = amount * 2
        if len(body) != 2 + expected_bytes or body[1] != expected_bytes:
            raise ModbusProtocolError(
                f"malformed FC3 response for {amount} registers: {body.hex()}"
            )
        return list(struct.unpack(f">{amount}H", body[2:]))

    # -- FC5 -------------------------------------------------------------
    def write_coil(self, address: int, value: bool) -> None:
        self._require_write_enabled()
        start = _check_u16(address, "address")
        raw = 0xFF00 if bool(value) else 0x0000
        request = struct.pack(">BHH", 0x05, start, raw)
        body = self._transact(request)
        if body != request:
            raise ModbusProtocolError("FC5 response does not echo the request")

    # -- FC6 -------------------------------------------------------------
    def write_register(self, address: int, value: int) -> None:
        self._require_write_enabled()
        start = _check_u16(address, "address")
        raw = _check_u16(value, "value")
        request = struct.pack(">BHH", 0x06, start, raw)
        body = self._transact(request)
        if body != request:
            raise ModbusProtocolError("FC6 response does not echo the request")

    def _require_write_enabled(self) -> None:
        if not self.config.allow_writes:
            raise ModbusWriteDisabled(
                "Modbus transport writes are disabled; "
                "set allow_writes=True only in a supervised write test"
            )


# ---------------------------------------------------------------------------
# SGS register map


class Access(str, Enum):
    READ_ONLY = "ro"
    READ_WRITE = "rw"


@dataclass(frozen=True)
class RegisterDefinition:
    name: str
    address: int
    count: int
    access: Access
    description: str


REG_MAJOR_VERSION = 0x0000
REG_MINOR_VERSION = 0x0001
REG_MOVE_RATE = 0x0008
REG_ROBOT_MODE = 0x000E
REG_ROBOT_SAFETY_MODE = 0x000F
REG_ROBOT_STATUS = 0x0010
REG_ROBOT_COMMAND = 0x0100
REG_MOVE_RATE_COMMAND = 0x0103
REG_BOARD_DO = 0x1002
REG_TOOL_DO = 0x1003
REG_BOARD_DO_COMMAND = 0x1300
REG_BOARD_DO_MASK = 0x1301
REG_TOOL_DO_COMMAND = 0x1302
REG_TOOL_DO_MASK = 0x1303

#: Baseline derived from the SGS Modbus holding-register table and confirmed
#: by actual FC3 logs at addresses 0x0000 and 0x1002.  The PDF chapter-10 byte
#: offsets are not used as absolute holding-register addresses.
SGS_REGISTERS: tuple[RegisterDefinition, ...] = (
    RegisterDefinition("major_version", REG_MAJOR_VERSION, 1, Access.READ_ONLY, "Controller major version"),
    RegisterDefinition("minor_version", REG_MINOR_VERSION, 1, Access.READ_ONLY, "Controller minor version"),
    RegisterDefinition("move_rate", REG_MOVE_RATE, 1, Access.READ_ONLY, "Current speed override"),
    RegisterDefinition("robot_mode", REG_ROBOT_MODE, 1, Access.READ_ONLY, "Robot mode"),
    RegisterDefinition("robot_safety_mode", REG_ROBOT_SAFETY_MODE, 1, Access.READ_ONLY, "Robot safety mode"),
    RegisterDefinition("robot_status", REG_ROBOT_STATUS, 1, Access.READ_ONLY, "Robot status bit word"),
    RegisterDefinition("robot_command", REG_ROBOT_COMMAND, 1, Access.READ_WRITE, "Rising-edge command"),
    RegisterDefinition("move_rate_command", REG_MOVE_RATE_COMMAND, 1, Access.READ_WRITE, "Set speed override"),
    RegisterDefinition("board_do", REG_BOARD_DO, 1, Access.READ_ONLY, "Control-cabinet DO feedback"),
    RegisterDefinition("tool_do", REG_TOOL_DO, 1, Access.READ_ONLY, "Flange DO feedback"),
    RegisterDefinition("board_do_command", REG_BOARD_DO_COMMAND, 1, Access.READ_WRITE, "Control-cabinet DO value"),
    RegisterDefinition("board_do_mask", REG_BOARD_DO_MASK, 1, Access.READ_WRITE, "Control-cabinet DO update mask"),
    RegisterDefinition("tool_do_command", REG_TOOL_DO_COMMAND, 1, Access.READ_WRITE, "Flange DO value"),
    RegisterDefinition("tool_do_mask", REG_TOOL_DO_MASK, 1, Access.READ_WRITE, "Flange DO update mask"),
)

SGS_REGISTERS_BY_NAME: Mapping[str, RegisterDefinition] = {
    item.name: item for item in SGS_REGISTERS
}


class RobotCommand(IntEnum):
    NONE = 0
    START_PROGRAM = 1
    STOP_PROGRAM = 2
    PAUSE_PROGRAM = 3
    CONTINUE_PROGRAM = 4
    SWITCH_ON = 5
    SWITCH_OFF = 6
    RESET_ERROR = 7
    TO_AUTO = 8
    TO_MANUAL = 9
    SET_MOVE_RATE = 10


class ProgramState(str, Enum):
    STOPPED = "stopped"
    PAUSED_UNCONFIRMED = "paused_unconfirmed"
    RUNNING = "running"
    RESERVED = "reserved"


@dataclass(frozen=True)
class SgsRobotStatus:
    """Decoded ``robotStatus`` word.

    Bit combinations 9/10=01 and 11 are retained as explicit non-running
    states.  The official material only names 00=stopped and 10=running, so the
    other combinations must never be treated as proof that the robot is ready.
    """

    raw: int

    def __post_init__(self) -> None:
        _check_u16(self.raw, "robot status word")

    @property
    def emergency_stop(self) -> bool:
        return bool(self.raw & (1 << 0))

    @property
    def powered(self) -> bool:
        return bool(self.raw & (1 << 1))

    @property
    def drag_mode(self) -> bool:
        return bool(self.raw & (1 << 2))

    @property
    def moving(self) -> bool:
        return bool(self.raw & (1 << 3))

    @property
    def simulation(self) -> bool:
        return bool(self.raw & (1 << 8))

    @property
    def program_state(self) -> ProgramState:
        value = (self.raw >> 9) & 0x03
        return {
            0: ProgramState.STOPPED,
            1: ProgramState.PAUSED_UNCONFIRMED,
            2: ProgramState.RUNNING,
            3: ProgramState.RESERVED,
        }[value]

    @property
    def fault(self) -> bool:
        return bool(self.raw & (1 << 15))

    @property
    def known_mask(self) -> int:
        return sum(1 << bit for bit in (0, 1, 2, 3, 8, 9, 10, 15))

    @property
    def unknown_mask(self) -> int:
        return self.raw & ~self.known_mask

    @property
    def command_block_reason(self) -> str:
        if self.emergency_stop:
            return "emergency stop is active"
        if self.fault:
            return "controller fault bit is active"
        if self.drag_mode:
            return "controller is in drag mode"
        if self.moving:
            return "robot is already moving"
        return ""

    def as_dict(self) -> dict[str, object]:
        return {
            "raw": self.raw,
            "emergency_stop": self.emergency_stop,
            "powered": self.powered,
            "drag_mode": self.drag_mode,
            "moving": self.moving,
            "simulation": self.simulation,
            "program_state": self.program_state.value,
            "fault": self.fault,
            "unknown_mask": self.unknown_mask,
        }


@dataclass(frozen=True)
class SgsCoreState:
    move_rate: int
    robot_mode: int
    safety_mode: int
    status: SgsRobotStatus

    def as_dict(self) -> dict[str, object]:
        return {
            "move_rate": self.move_rate,
            "robot_mode": self.robot_mode,
            "safety_mode": self.safety_mode,
            "status": self.status.as_dict(),
        }


@dataclass(frozen=True)
class SgsIoState:
    board_do: int
    tool_do: int

    def as_dict(self) -> dict[str, int]:
        return {"board_do": self.board_do, "tool_do": self.tool_do}


@dataclass(frozen=True)
class SgsSnapshot:
    major_version: int
    minor_version: int
    core: SgsCoreState
    io: SgsIoState
    board_do_command: int
    board_do_mask: int
    tool_do_command: int
    tool_do_mask: int

    def as_dict(self) -> dict[str, object]:
        return {
            "firmware": f"{self.major_version}.{self.minor_version}",
            "major_version": self.major_version,
            "minor_version": self.minor_version,
            "core": self.core.as_dict(),
            "io": self.io.as_dict(),
            "do_commands": {
                "board_value": self.board_do_command,
                "board_mask": self.board_do_mask,
                "tool_value": self.tool_do_command,
                "tool_mask": self.tool_do_mask,
            },
        }


class SgsModbusClient:
    """Read-only SGS facade over :class:`ModbusTcpClient`."""

    def __init__(self, transport: ModbusTcpClient) -> None:
        self.transport = transport

    @classmethod
    def from_host(
        cls,
        host: str,
        *,
        port: int = 502,
        unit_id: int = 1,
        timeout_s: float = 2.0,
    ) -> "SgsModbusClient":
        return cls(
            ModbusTcpClient(
                ModbusTcpConfig(
                    host=host,
                    port=port,
                    unit_id=unit_id,
                    timeout_s=timeout_s,
                    connect_timeout_s=timeout_s,
                    allow_writes=False,
                )
            )
        )

    def connect(self) -> None:
        self.transport.connect()

    def close(self) -> None:
        self.transport.close()

    def __enter__(self) -> "SgsModbusClient":
        self.connect()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def read_version(self) -> tuple[int, int]:
        values = self.transport.read_holding_registers(REG_MAJOR_VERSION, 2)
        return values[0], values[1]

    def read_core_state(self) -> SgsCoreState:
        values = self.transport.read_holding_registers(REG_MOVE_RATE, 9)
        return SgsCoreState(
            move_rate=values[REG_MOVE_RATE - REG_MOVE_RATE],
            robot_mode=values[REG_ROBOT_MODE - REG_MOVE_RATE],
            safety_mode=values[REG_ROBOT_SAFETY_MODE - REG_MOVE_RATE],
            status=SgsRobotStatus(values[REG_ROBOT_STATUS - REG_MOVE_RATE]),
        )

    def read_io_state(self) -> SgsIoState:
        values = self.transport.read_holding_registers(REG_BOARD_DO, 2)
        return SgsIoState(board_do=values[0], tool_do=values[1])

    def read_snapshot(self) -> SgsSnapshot:
        major, minor = self.read_version()
        core = self.read_core_state()
        io = self.read_io_state()
        commands = self.transport.read_holding_registers(REG_BOARD_DO_COMMAND, 4)
        return SgsSnapshot(
            major_version=major,
            minor_version=minor,
            core=core,
            io=io,
            board_do_command=commands[0],
            board_do_mask=commands[1],
            tool_do_command=commands[2],
            tool_do_mask=commands[3],
        )


class SgsControlClient:
    """High-level SGS commands with a second, independent write interlock.

    This class is not connected to the picking pipeline.  It exists so offline
    tests can prove that command pulses return to zero and that fault/estop/
    motion state blocks command injection.
    """

    def __init__(
        self,
        transport: ModbusTcpClient,
        *,
        allow_robot_commands: bool = False,
    ) -> None:
        self.transport = transport
        self.reader = SgsModbusClient(transport)
        self.allow_robot_commands = bool(allow_robot_commands)

    def _require_robot_commands(self) -> None:
        if not self.transport.config.allow_writes:
            raise ModbusWriteDisabled(
                "transport writes are disabled; robot command gate is closed"
            )
        if not self.allow_robot_commands:
            raise ModbusWriteDisabled(
                "robot command gate is closed; set allow_robot_commands=True "
                "only after field write validation"
            )

    def pulse_robot_command(
        self,
        command: RobotCommand,
        *,
        hold_s: float = 0.05,
        require_idle: bool = True,
    ) -> None:
        """Write a rising-edge command and prove that the register returns to 0.

        ``STOP_PROGRAM`` and ``RESET_ERROR`` are permitted while a fault is
        present.  Every other command requires a non-estop, non-fault, idle
        status when ``require_idle`` is true.
        """

        self._require_robot_commands()
        if not isinstance(command, RobotCommand):
            command = RobotCommand(int(command))
        if command is RobotCommand.NONE:
            raise ValueError("NONE is not a command pulse")
        if hold_s < 0.0:
            raise ValueError("hold_s must be non-negative")

        if require_idle:
            status = self.reader.read_core_state().status
            if status.emergency_stop:
                raise SgsCommandSafetyError(status.command_block_reason)
            fault_tolerant = {
                RobotCommand.STOP_PROGRAM,
                RobotCommand.RESET_ERROR,
                RobotCommand.TO_MANUAL,
            }
            if status.fault and command not in fault_tolerant:
                raise SgsCommandSafetyError(status.command_block_reason)
            if status.moving:
                raise SgsCommandSafetyError(status.command_block_reason)
            if status.drag_mode and command is not RobotCommand.TO_MANUAL:
                raise SgsCommandSafetyError(status.command_block_reason)

        current = self.transport.read_holding_registers(REG_ROBOT_COMMAND, 1)[0]
        if current != int(RobotCommand.NONE):
            raise SgsCommandSafetyError(
                f"robot command register is not idle (value={current})"
            )

        write_started = False
        clear_error: Exception | None = None
        try:
            write_started = True
            self.transport.write_register(REG_ROBOT_COMMAND, int(command))
            if hold_s:
                time.sleep(hold_s)
        finally:
            if write_started:
                try:
                    self.transport.write_register(
                        REG_ROBOT_COMMAND, int(RobotCommand.NONE)
                    )
                except Exception as exc:  # preserve the primary exception
                    clear_error = exc

        if clear_error is not None:
            raise SgsUnsafeCommandState(
                "failed to clear robotCommand after pulse; "
                "immediately isolate or stop the controller"
            ) from clear_error

        readback = self.transport.read_holding_registers(REG_ROBOT_COMMAND, 1)[0]
        if readback != int(RobotCommand.NONE):
            raise SgsUnsafeCommandState(
                "robotCommand readback is not zero after pulse "
                f"(value={readback}); treat the controller as unsafe"
            )


def register_map_as_dict() -> dict[str, dict[str, object]]:
    """Stable JSON-friendly view used by tools and tests."""

    return {
        item.name: {
            "address": item.address,
            "address_hex": f"0x{item.address:04x}",
            "count": item.count,
            "access": item.access.value,
            "description": item.description,
        }
        for item in SGS_REGISTERS
    }
