"""In-process fake SGS Modbus/TCP slave for hardware-free validation.

The fake implements the same FC1/FC3/FC5/FC6 subset used by
:mod:`track1_grasp.sgs_modbus`.  It records every request and can inject a
Modbus exception for a selected function/address.  It is suitable for unit
tests and local state-machine development, never for production control.
"""

from __future__ import annotations

import socket
import struct
import threading
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class FakeModbusRequest:
    transaction_id: int
    unit_id: int
    function_code: int
    address: int | None
    count_or_value: int | None


class FakeSgsModbusServer:
    """Small single-master Modbus/TCP server backed by sparse dictionaries."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        unit_id: int = 1,
        registers: Mapping[int, int] | None = None,
        coils: Mapping[int, bool] | None = None,
    ) -> None:
        self.host = host
        self.unit_id = int(unit_id)
        self.registers: dict[int, int] = {
            int(address): int(value) & 0xFFFF
            for address, value in dict(registers or {}).items()
        }
        self.coils: dict[int, bool] = {
            int(address): bool(value)
            for address, value in dict(coils or {}).items()
        }
        self.requests: list[FakeModbusRequest] = []
        self.exception_overrides: dict[tuple[int, int], int] = {}

        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._listener: socket.socket | None = None
        self._active_connection: socket.socket | None = None
        self._port = int(port)

    @property
    def port(self) -> int:
        return self._port

    def start(self) -> "FakeSgsModbusServer":
        if self._thread is not None and self._thread.is_alive():
            return self
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.host, self._port))
        listener.listen(4)
        listener.settimeout(0.2)
        self._listener = listener
        self._port = int(listener.getsockname()[1])
        self._ready.set()
        self._thread = threading.Thread(
            target=self._serve,
            name="fake-sgs-modbus",
            daemon=True,
        )
        self._thread.start()
        return self

    def __enter__(self) -> "FakeSgsModbusServer":
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    def stop(self) -> None:
        self._stop.set()
        active = self._active_connection
        if active is not None:
            try:
                active.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                active.close()
            except OSError:
                pass
        listener = self._listener
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None
        self._listener = None
        self._active_connection = None

    def set_register(self, address: int, value: int) -> None:
        with self._lock:
            self.registers[int(address)] = int(value) & 0xFFFF

    def set_registers(self, values: Mapping[int, int]) -> None:
        with self._lock:
            for address, value in values.items():
                self.registers[int(address)] = int(value) & 0xFFFF

    def set_coil(self, address: int, value: bool) -> None:
        with self._lock:
            self.coils[int(address)] = bool(value)

    def inject_exception(
        self,
        function_code: int,
        address: int | None,
        exception_code: int,
    ) -> None:
        """Make the next matching request return one exception response.

        ``address=None`` matches any address for the function.
        """

        with self._lock:
            key = (int(function_code), -1 if address is None else int(address))
            self.exception_overrides[key] = int(exception_code) & 0xFF

    def _serve(self) -> None:
        listener = self._listener
        if listener is None:
            return
        while not self._stop.is_set():
            try:
                connection, _peer = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            self._active_connection = connection
            try:
                connection.settimeout(0.2)
                self._serve_connection(connection)
            finally:
                try:
                    connection.close()
                except OSError:
                    pass
                self._active_connection = None

    def _serve_connection(self, connection: socket.socket) -> None:
        buffer = bytearray()
        while not self._stop.is_set():
            try:
                chunk = connection.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                return
            if not chunk:
                return
            buffer.extend(chunk)
            while len(buffer) >= 7:
                tid, protocol_id, length, unit_id = struct.unpack(
                    ">HHHB", bytes(buffer[:7])
                )
                if protocol_id != 0 or not 2 <= length <= 254:
                    return
                frame_length = 6 + length
                if len(buffer) < frame_length:
                    break
                pdu = bytes(buffer[7:frame_length])
                del buffer[:frame_length]
                response = self._handle_request(tid, unit_id, pdu)
                response_header = struct.pack(
                    ">HHHB",
                    tid,
                    0,
                    len(response) + 1,
                    unit_id,
                )
                try:
                    connection.sendall(response_header + response)
                except OSError:
                    return

    def _handle_request(self, transaction_id: int, unit_id: int, pdu: bytes) -> bytes:
        if not pdu:
            return b"\x80\x03"
        function = pdu[0]
        if unit_id != self.unit_id:
            return bytes([function | 0x80, 0x0B])

        if function not in (0x01, 0x03, 0x05, 0x06) or len(pdu) != 5:
            return bytes([function | 0x80, 0x01])
        address, value = struct.unpack(">HH", pdu[1:5])
        with self._lock:
            self.requests.append(
                FakeModbusRequest(
                    transaction_id=transaction_id,
                    unit_id=unit_id,
                    function_code=function,
                    address=address,
                    count_or_value=value,
                )
            )
            exception = self.exception_overrides.pop((function, address), None)
            if exception is None:
                exception = self.exception_overrides.pop((function, -1), None)
        if exception is not None:
            return bytes([function | 0x80, exception])

        if function == 0x05:
            if value not in (0x0000, 0xFF00):
                return bytes([function | 0x80, 0x03])
            with self._lock:
                self.coils[address] = value == 0xFF00
            return pdu

        if function == 0x06:
            with self._lock:
                self.registers[address] = value
            return pdu

        if function == 0x01:
            count = value
            if not 1 <= count <= 2000 or address + count - 1 > 0xFFFF:
                return bytes([function | 0x80, 0x03])
            payload = bytearray((count + 7) // 8)
            with self._lock:
                for index in range(count):
                    if self.coils.get(address + index, False):
                        payload[index // 8] |= 1 << (index % 8)
            return bytes([function, len(payload)]) + bytes(payload)

        count = value
        if not 1 <= count <= 125 or address + count - 1 > 0xFFFF:
            return bytes([function | 0x80, 0x03])
        with self._lock:
            values = [
                self.registers.get(address + index, 0) for index in range(count)
            ]
        return bytes([function, count * 2]) + struct.pack(
            f">{count}H", *values
        )
