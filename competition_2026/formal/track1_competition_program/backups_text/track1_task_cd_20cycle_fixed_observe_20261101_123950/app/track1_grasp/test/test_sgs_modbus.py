"""Hardware-free tests for the strict ESTUN SGS Modbus layer."""

from __future__ import annotations

import pathlib
import socket
import struct
import threading
import time

import pytest
import yaml

from track1_grasp.sgs_modbus import (
    ModbusException,
    ModbusProtocolError,
    ModbusTcpClient,
    ModbusTcpConfig,
    ModbusWriteDisabled,
    ProgramState,
    REG_BOARD_DO,
    REG_BOARD_DO_COMMAND,
    REG_BOARD_DO_MASK,
    REG_MAJOR_VERSION,
    REG_MINOR_VERSION,
    REG_ROBOT_COMMAND,
    REG_ROBOT_MODE,
    REG_ROBOT_SAFETY_MODE,
    REG_ROBOT_STATUS,
    REG_TOOL_DO,
    REG_TOOL_DO_COMMAND,
    REG_TOOL_DO_MASK,
    RobotCommand,
    SgsCommandSafetyError,
    SgsControlClient,
    SgsModbusClient,
    SgsRobotStatus,
    SGS_REGISTERS,
)
from track1_grasp.sgs_modbus_fake import FakeSgsModbusServer

CONFIG_DIR = pathlib.Path(__file__).resolve().parents[1] / "config"


def _populated_server() -> FakeSgsModbusServer:
    return FakeSgsModbusServer(
        registers={
            REG_MAJOR_VERSION: 1,
            REG_MINOR_VERSION: 6,
            0x0008: 100,
            REG_ROBOT_MODE: 1,
            REG_ROBOT_SAFETY_MODE: 1,
            REG_ROBOT_STATUS: (1 << 1) | (1 << 10),
            REG_ROBOT_COMMAND: 0,
            REG_BOARD_DO: 0x0001,
            REG_TOOL_DO: 0x0000,
            REG_BOARD_DO_COMMAND: 0x0001,
            REG_BOARD_DO_MASK: 0x0001,
            REG_TOOL_DO_COMMAND: 0x0000,
            REG_TOOL_DO_MASK: 0x0000,
        }
    )


def _transport(server: FakeSgsModbusServer, *, writes: bool = False) -> ModbusTcpClient:
    return ModbusTcpClient(
        ModbusTcpConfig(
            host="127.0.0.1",
            port=server.port,
            unit_id=1,
            timeout_s=0.5,
            connect_timeout_s=0.5,
            allow_writes=writes,
        )
    )


@pytest.fixture()
def server():
    fake = _populated_server().start()
    try:
        yield fake
    finally:
        fake.stop()


# -- MBAP/PDU validation -------------------------------------------------


def test_read_holding_registers_returns_expected_values(server):
    with _transport(server) as client:
        assert client.read_holding_registers(REG_MAJOR_VERSION, 2) == [1, 6]


def test_read_coils_and_write_coil_round_trip(server):
    server.set_coil(4, True)
    with _transport(server, writes=True) as client:
        assert client.read_coils(3, 3) == [False, True, False]
        client.write_coil(3, True)
        client.write_coil(4, False)
        assert client.read_coils(3, 3) == [True, False, False]


def test_transport_write_interlock_is_closed_by_default(server):
    with _transport(server) as client:
        with pytest.raises(ModbusWriteDisabled):
            client.write_register(REG_ROBOT_COMMAND, 0)
        with pytest.raises(ModbusWriteDisabled):
            client.write_coil(0, True)
    assert server.registers[REG_ROBOT_COMMAND] == 0


def test_fc6_writes_only_when_enabled(server):
    with _transport(server, writes=True) as client:
        client.write_register(REG_BOARD_DO_COMMAND, 0x55)
        assert client.read_holding_registers(REG_BOARD_DO_COMMAND, 1) == [0x55]


def test_modbus_exception_is_reported(server):
    server.inject_exception(0x03, REG_ROBOT_STATUS, 0x02)
    with _transport(server) as client:
        with pytest.raises(ModbusException) as caught:
            client.read_holding_registers(REG_ROBOT_STATUS, 1)
    assert caught.value.exception_code == 0x02
    assert caught.value.function_code == 0x03


def test_invalid_request_ranges_fail_before_network(server):
    with _transport(server) as client:
        with pytest.raises(ValueError):
            client.read_holding_registers(0xFFFF, 2)
        with pytest.raises(ValueError):
            client.read_holding_registers(0, 0)


def test_wrong_transaction_id_is_rejected():
    class WrongTidServer(threading.Thread):
        def __init__(self):
            super().__init__(daemon=True)
            self.sock = socket.socket()
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind(("127.0.0.1", 0))
            self.sock.listen(1)
            self.port = self.sock.getsockname()[1]
            self.ready = threading.Event()

        def run(self):
            self.ready.set()
            conn, _ = self.sock.accept()
            with conn:
                self._recv(conn, 12)
                response_pdu = struct.pack(">BBHH", 0x03, 2, 0, 0)
                conn.sendall(
                    struct.pack(">HHHB", 0x7777, 0, len(response_pdu) + 1, 1)
                    + response_pdu
                )
            self.sock.close()

        @staticmethod
        def _recv(conn, count):
            data = b""
            while len(data) < count:
                chunk = conn.recv(count - len(data))
                if not chunk:
                    return data
                data += chunk
            return data

    malformed = WrongTidServer()
    malformed.start()
    malformed.ready.wait(1)
    client = ModbusTcpClient(
        ModbusTcpConfig(
            host="127.0.0.1",
            port=malformed.port,
            timeout_s=0.5,
            connect_timeout_s=0.5,
        )
    )
    try:
        with pytest.raises(ModbusProtocolError, match="transaction id"):
            client.read_holding_registers(0, 1)
    finally:
        client.close()
        malformed.join(timeout=1)


# -- SGS status and read-only snapshot ----------------------------------


@pytest.mark.parametrize(
    ("word", "state"),
    [
        (0x0000, ProgramState.STOPPED),
        (1 << 9, ProgramState.PAUSED_UNCONFIRMED),
        (1 << 10, ProgramState.RUNNING),
        ((1 << 9) | (1 << 10), ProgramState.RESERVED),
    ],
)
def test_program_state_decoding(word, state):
    assert SgsRobotStatus(word).program_state is state


def test_status_bits_and_unknown_mask():
    status = SgsRobotStatus(
        (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 8) | (1 << 15)
        | (1 << 4)
    )
    assert status.emergency_stop
    assert status.powered
    assert status.drag_mode
    assert status.moving
    assert status.simulation
    assert status.fault
    assert status.unknown_mask == 1 << 4
    assert "emergency stop" in status.command_block_reason


def test_snapshot_decodes_excel_holding_register_map(server):
    with SgsModbusClient(_transport(server)) as reader:
        snapshot = reader.read_snapshot()
    assert snapshot.major_version == 1
    assert snapshot.minor_version == 6
    assert snapshot.core.move_rate == 100
    assert snapshot.core.robot_mode == 1
    assert snapshot.core.safety_mode == 1
    assert snapshot.core.status.program_state is ProgramState.RUNNING
    assert snapshot.io.board_do == 1
    assert snapshot.io.tool_do == 0
    assert snapshot.board_do_command == 1
    assert snapshot.tool_do_mask == 0


# -- command pulse safety -----------------------------------------------


def test_control_pulse_requires_both_write_gates(server):
    transport = _transport(server)  # transport gate remains closed
    control = SgsControlClient(transport, allow_robot_commands=True)
    with pytest.raises(ModbusWriteDisabled, match="transport writes"):
        control.pulse_robot_command(RobotCommand.START_PROGRAM, hold_s=0)
    assert not any(
        request.address == REG_ROBOT_COMMAND for request in server.requests
        if request.function_code == 0x06
    )


def test_control_pulse_requires_high_level_gate(server):
    transport = _transport(server, writes=True)
    control = SgsControlClient(transport, allow_robot_commands=False)
    with pytest.raises(ModbusWriteDisabled, match="robot command gate"):
        control.pulse_robot_command(RobotCommand.START_PROGRAM, hold_s=0)
    assert server.registers[REG_ROBOT_COMMAND] == 0


def test_command_pulse_writes_value_then_zero(server):
    with _transport(server, writes=True) as transport:
        control = SgsControlClient(transport, allow_robot_commands=True)
        control.pulse_robot_command(RobotCommand.START_PROGRAM, hold_s=0)
        assert transport.read_holding_registers(REG_ROBOT_COMMAND, 1) == [0]
    command_writes = [
        request.count_or_value
        for request in server.requests
        if request.function_code == 0x06
        and request.address == REG_ROBOT_COMMAND
    ]
    assert command_writes == [int(RobotCommand.START_PROGRAM), 0]


def test_command_pulse_refuses_motion_and_does_not_write(server):
    server.set_register(REG_ROBOT_STATUS, (1 << 1) | (1 << 3))
    with _transport(server, writes=True) as transport:
        control = SgsControlClient(transport, allow_robot_commands=True)
        with pytest.raises(SgsCommandSafetyError, match="already moving"):
            control.pulse_robot_command(RobotCommand.START_PROGRAM, hold_s=0)
        assert transport.read_holding_registers(REG_ROBOT_COMMAND, 1) == [0]


def test_reset_error_command_is_allowed_while_faulted(server):
    server.set_register(REG_ROBOT_STATUS, (1 << 1) | (1 << 15))
    with _transport(server, writes=True) as transport:
        control = SgsControlClient(transport, allow_robot_commands=True)
        control.pulse_robot_command(RobotCommand.RESET_ERROR, hold_s=0)
        assert transport.read_holding_registers(REG_ROBOT_COMMAND, 1) == [0]


def test_emergency_stop_blocks_normal_command(server):
    server.set_register(REG_ROBOT_STATUS, (1 << 0) | (1 << 1))
    with _transport(server, writes=True) as transport:
        control = SgsControlClient(transport, allow_robot_commands=True)
        with pytest.raises(SgsCommandSafetyError, match="emergency stop"):
            control.pulse_robot_command(RobotCommand.START_PROGRAM, hold_s=0)


def test_nonzero_command_register_blocks_new_pulse(server):
    server.set_register(REG_ROBOT_COMMAND, int(RobotCommand.PAUSE_PROGRAM))
    with _transport(server, writes=True) as transport:
        control = SgsControlClient(transport, allow_robot_commands=True)
        with pytest.raises(SgsCommandSafetyError, match="not idle"):
            control.pulse_robot_command(RobotCommand.START_PROGRAM, hold_s=0)


# -- machine-readable register schema -----------------------------------


def test_yaml_schema_matches_python_register_map():
    payload = yaml.safe_load(
        (CONFIG_DIR / "estun_sgs_modbus.yaml").read_text(encoding="utf-8")
    )
    yaml_registers = payload["registers"]
    python_registers = {item.name: item.address for item in SGS_REGISTERS}
    assert set(yaml_registers) == set(python_registers)
    for name, address in python_registers.items():
        assert int(yaml_registers[name]["address"]) == address
    assert payload["transport"]["port"] == 502
    assert payload["transport"]["unit_id"] == 1
    assert payload["safety"]["transport_writes_enabled"] is False
    assert payload["safety"]["robot_command_writes_enabled"] is False
    assert payload["task_handshake"]["status"] == "not_assigned"


def test_controller_evidence_confirms_raw_modbus_addresses():
    payload = yaml.safe_load(
        (CONFIG_DIR / "estun_sgs_modbus.yaml").read_text(encoding="utf-8")
    )
    evidence = payload["address_evidence"]
    assert evidence["holding_register_table"]["result"] == "authoritative_baseline"
    observed = evidence["controller_readonly_logs"]["observed"]
    assert observed["major_version"] == 1
    assert observed["board_do_address"] == REG_BOARD_DO
    assert REG_ROBOT_MODE == 0x000E
    assert REG_ROBOT_SAFETY_MODE == 0x000F
    assert REG_ROBOT_STATUS == 0x0010


# -- read-only CLI -------------------------------------------------------


def test_probe_schema_has_no_write_mode(capsys):
    from track1_grasp.tools.sgs_modbus_probe import main

    assert main(["--schema", "--json"]) == 0
    import json

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "schema"
    assert payload["writes_enabled"] is False
    assert payload["registers"]["robot_status"]["address_hex"] == "0x0010"


def test_probe_offline_smoke_uses_loopback_only(capsys):
    from track1_grasp.tools.sgs_modbus_probe import main

    assert main(["--offline-smoke", "--json"]) == 0
    import json

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "offline_smoke"
    assert payload["writes_enabled"] is False
    assert payload["firmware"] == "1.6"
    assert payload["core"]["status"]["program_state"] == "running"


def test_probe_requires_explicit_live_host(capsys):
    from track1_grasp.tools.sgs_modbus_probe import main

    assert main([]) == 2
    assert "--host is required" in capsys.readouterr().err
