"""Tests for the pluggable electric-gripper command layer."""

from __future__ import annotations

import pathlib

import pytest

from track1_grasp.config import Track1Config
from track1_grasp.domain import Detection, GripperState, MotionPlan, Pose
from track1_grasp.gripper import (
    DiscreteCommandGripperBackend,
    GripperStateHoldVerifier,
    MockGripperBackend,
    build_gripper,
    build_hold_verifier,
    encode_command,
)

CONFIG_DIR = pathlib.Path(__file__).resolve().parents[1] / "config"


def _gripper_config(**overrides):
    data = {
        "task": {"mode": "C"},
        "vision": {"backend": "mock", "scenario_file": "../scenarios/mock_track1_scene.yaml"},
        "gripper": overrides,
    }
    return Track1Config.from_dict(data, base_dir=CONFIG_DIR).gripper


class FakeTransport(DiscreteCommandGripperBackend):
    """In-memory stand-in for a serial/TCP electric gripper controller."""

    def __init__(self, config, replies=None, log=None):
        super().__init__(config, log=log)
        self.sent: list[bytes] = []
        self._replies = list(replies or [])

    def _write(self, payload: bytes) -> None:
        self.sent.append(payload)

    def _read(self, length: int, timeout: float) -> bytes:
        return self._replies.pop(0) if self._replies else b""

    def _shutdown(self) -> None:
        self.closed = True


# --- command encoding ----------------------------------------------------


def test_encode_ascii_with_terminator():
    assert encode_command("OPEN", "ascii", "\\r\\n") == b"OPEN\r\n"


def test_encode_hex_accepts_spaces_and_commas():
    assert encode_command("01 06 00,00", "hex") == bytes([0x01, 0x06, 0x00, 0x00])


def test_encode_rejects_unknown_encoding():
    with pytest.raises(ValueError):
        encode_command("OPEN", "utf8")


# --- discrete command backend -------------------------------------------


def test_discrete_backend_sends_configured_bytes():
    config = _gripper_config(
        backend="serial",
        serial_port="/dev/ttyUSB0",
        open_command="A5 01 01",
        close_command="A5 01 02",
        command_encoding="hex",
        command_terminator="",
        settle_time_s=0.0,
    )
    backend = FakeTransport(config)
    backend.open()
    backend.close()
    assert backend.sent == [bytes([0xA5, 0x01, 0x01]), bytes([0xA5, 0x01, 0x02])]
    assert backend.state is GripperState.CLOSED
    assert backend.events == ["open", "close"]


def test_discrete_backend_reads_feedback_to_detect_a_missed_grasp():
    config = _gripper_config(
        backend="serial",
        serial_port="/dev/ttyUSB0",
        open_command="OPEN",
        close_command="CLOSE",
        expect_response=True,
        feedback_command="STATUS",
        feedback_open_match="OPEN",
        feedback_closed_match="CLOSED",
        settle_time_s=0.0,
    )
    # The jaws report they never reached the closed position: the object moved.
    backend = FakeTransport(config, replies=[b"", b"OPEN"])
    backend.close()
    assert backend.state is GripperState.CLOSED
    assert backend._refresh_state() is GripperState.OPEN


# --- hold verifier -------------------------------------------------------


def test_mock_backend_uses_optimistic_verifier():
    config = _gripper_config(backend="mock")
    gripper = build_gripper(config)
    assert isinstance(gripper, MockGripperBackend)
    verifier = build_hold_verifier(config, gripper)
    ok, _ = verifier.verify(_detection(), _plan())
    assert ok is True


def test_hardware_backend_uses_state_verifier_and_can_reject():
    config = _gripper_config(
        backend="serial",
        serial_port="/dev/ttyUSB0",
        open_command="OPEN",
        close_command="CLOSE",
        feedback_command="STATUS",
        feedback_open_match="OPEN",
        feedback_closed_match="CLOSED",
        settle_time_s=0.0,
    )
    # The feedback query is the only exchange this configuration reads.
    backend = FakeTransport(config, replies=[b"OPEN"])
    verifier = build_hold_verifier(config, backend)
    backend.close()
    ok, message = verifier.verify(_detection(), _plan())
    assert ok is False
    assert "open" in message.lower()


def test_verifier_accepts_when_controller_reports_closed():
    config = _gripper_config(
        backend="serial",
        serial_port="/dev/ttyUSB0",
        open_command="OPEN",
        close_command="CLOSE",
        feedback_command="STATUS",
        feedback_open_match="OPEN",
        feedback_closed_match="CLOSED",
        settle_time_s=0.0,
    )
    backend = FakeTransport(config, replies=[b"CLOSED"])
    verifier = GripperStateHoldVerifier(backend)
    backend.close()
    ok, _ = verifier.verify(_detection(), _plan())
    assert ok is True


# --- config validation ---------------------------------------------------


def test_serial_backend_requires_port_and_commands():
    with pytest.raises(ValueError):
        Track1Config.from_dict(
            {
                "task": {"mode": "C"},
                "vision": {
                    "backend": "mock",
                    "scenario_file": "../scenarios/mock_track1_scene.yaml",
                },
                "gripper": {"backend": "serial"},
            },
            base_dir=CONFIG_DIR,
        )


def test_tcp_backend_requires_host_and_port():
    with pytest.raises(ValueError):
        Track1Config.from_dict(
            {
                "task": {"mode": "C"},
                "vision": {
                    "backend": "mock",
                    "scenario_file": "../scenarios/mock_track1_scene.yaml",
                },
                "gripper": {
                    "backend": "tcp",
                    "open_command": "1",
                    "close_command": "0",
                },
            },
            base_dir=CONFIG_DIR,
        )


def test_ros_topic_backend_requires_topic():
    with pytest.raises(ValueError):
        Track1Config.from_dict(
            {
                "task": {"mode": "C"},
                "vision": {
                    "backend": "mock",
                    "scenario_file": "../scenarios/mock_track1_scene.yaml",
                },
                "gripper": {"backend": "ros_topic"},
            },
            base_dir=CONFIG_DIR,
        )


def test_shipped_configs_validate():
    for name in ("track1_mock.yaml", "track1_real_template.yaml"):
        payload = (CONFIG_DIR / name).read_text(encoding="utf-8")
        import yaml

        Track1Config.from_dict(yaml.safe_load(payload), base_dir=CONFIG_DIR)


def _detection() -> Detection:
    return Detection(label="mug", confidence=0.9, graspability=0.9, bbox=(0, 0, 10, 10))


def _plan() -> MotionPlan:
    return MotionPlan(
        target=_detection(),
        pregrasp_pose=Pose(xyz=(0.0, 0.0, 0.2)),
        grasp_pose=Pose(xyz=(0.0, 0.0, 0.1)),
        lift_pose=Pose(xyz=(0.0, 0.0, 0.2)),
        transfer_pose=Pose(xyz=(0.2, 0.2, 0.2)),
        release_pose=Pose(xyz=(0.2, 0.2, 0.1)),
        retreat_pose=Pose(xyz=(0.2, 0.2, 0.2)),
    )


# --- MS42DC (WHEELTEC FAE 2-finger electric gripper) ---------------------


def test_ms42dc_frames_match_the_manual_examples():
    """The manual's raw open/close/status frames must round-trip byte for byte."""

    from track1_grasp.gripper import encode_ms42dc_frame, ms42dc_status_frame

    common = dict(
        device_id=0x01, mode=0x02, subdivide=0x20, angle_wire=18720, speed_wire=200
    )
    assert encode_ms42dc_frame(direction=0x01, **common).hex(" ") == (
        "7b 01 02 01 20 49 20 00 c8 f8 7d"
    )
    assert encode_ms42dc_frame(direction=0x00, **common).hex(" ") == (
        "7b 01 02 00 20 49 20 00 c8 f9 7d"
    )
    assert ms42dc_status_frame(0x01).hex(" ") == "7b 01 00 00 00 00 00 00 00 7a 7d"


def test_ms42dc_bcc_is_the_xor_of_the_first_nine_bytes():
    from track1_grasp.gripper import ms42dc_bcc

    body = bytes([0x7B, 0x01, 0x02, 0x01, 0x20, 0x49, 0x20, 0x00, 0xC8])
    assert ms42dc_bcc(body) == 0xF8


def test_ms42dc_angle_and_speed_are_scaled_by_ten():
    from track1_grasp.gripper import encode_ms42dc_frame

    frame = encode_ms42dc_frame(
        device_id=1, mode=4, direction=0, subdivide=32,
        angle_wire=int(1872.0 * 10), speed_wire=int(20.0 * 10),
    )
    assert frame[0] == 0x7B and frame[-1] == 0x7D
    assert (frame[5] << 8 | frame[6]) == 18720
    assert (frame[7] << 8 | frame[8]) == 200
    assert frame[2] == 4  # absolute-angle mode survives encoding


def test_ms42dc_status_reply_is_decoded_and_checksum_verified():
    from track1_grasp.gripper import ms42dc_bcc, parse_ms42dc_status

    # id=1, arrived=1, speed=200 (20 rad/s), angle=18720 (1872.0 deg)
    head = bytes([0x01, 0x01, 0x00, 0xC8, 0x00, 0x00, 0x49, 0x20])
    reply = head + bytes([ms42dc_bcc(head)])
    status = parse_ms42dc_status(reply)
    assert status is not None
    assert status.device_id == 1
    assert status.arrived is True
    assert status.speed_raw == 200
    assert status.angle_deg == 1872.0


def test_ms42dc_status_reply_with_bad_checksum_is_rejected():
    from track1_grasp.gripper import parse_ms42dc_status

    reply = bytes([0x01, 0x01, 0x00, 0xC8, 0x00, 0x00, 0x49, 0x20, 0x00])
    assert parse_ms42dc_status(reply) is None
    assert parse_ms42dc_status(b"\x01\x02") is None


def test_ms42dc_hold_verifier_rejects_an_empty_close():
    from track1_grasp.gripper import MS42DCHoldVerifier, ms42dc_bcc

    config = _gripper_config(
        backend="ms42dc",
        serial_port="/dev/motor_serial",
        hold_min_angle_deg=1500.0,
        settle_time_s=0.0,
    )

    class StubGripper:
        def __init__(self, angle_raw):
            self._angle_raw = angle_raw

        def read_status(self):
            from track1_grasp.gripper import MS42DCStatus

            return MS42DCStatus(1, True, 0, self._angle_raw)

    lonely = MS42DCHoldVerifier(StubGripper(18720), config)
    ok, message = lonely.verify(_detection(), _plan())
    assert ok is False and "nothing between the fingers" in message

    holding = MS42DCHoldVerifier(StubGripper(9000), config)
    ok, _ = holding.verify(_detection(), _plan())
    assert ok is True


def test_ms42dc_hold_verifier_waits_for_arrival_flag():
    from track1_grasp.gripper import MS42DCHoldVerifier, MS42DCStatus

    config = _gripper_config(
        backend="ms42dc", serial_port="/dev/motor_serial", settle_time_s=0.0
    )

    class NotArrived:
        def read_status(self):
            return MS42DCStatus(1, False, 0, 100)

    ok, message = MS42DCHoldVerifier(NotArrived(), config).verify(
        _detection(), _plan()
    )
    assert ok is False and "到位" in message


def test_ms42dc_hold_verifier_reuses_arrival_confirmed_by_close():
    from track1_grasp.gripper import MS42DCHoldVerifier, MS42DCStatus

    config = _gripper_config(
        backend="ms42dc",
        serial_port="/dev/motor_serial",
        require_feedback=True,
        settle_time_s=0.0,
    )

    class ConfirmedClose:
        last_status = MS42DCStatus(1, True, 0, 195)

        def read_status(self, **_kwargs):
            return MS42DCStatus(1, False, 0, 195)

    ok, message = MS42DCHoldVerifier(ConfirmedClose(), config).verify(
        _detection(), _plan()
    )
    assert ok is True
    assert "19.5" in message


def test_ms42dc_config_rejects_bad_subdivide_and_mode():
    for overrides in (
        {"subdivide": 7},
        {"motion_mode": 3},
        {"open_angle_deg": 0.0},
    ):
        with pytest.raises(ValueError):
            Track1Config.from_dict(
                {
                    "task": {"mode": "C"},
                    "vision": {
                        "backend": "mock",
                        "scenario_file": "../scenarios/mock_track1_scene.yaml",
                    },
                    "gripper": {
                        "backend": "ms42dc",
                        "serial_port": "/dev/motor_serial",
                        **overrides,
                    },
                },
                base_dir=CONFIG_DIR,
            )


def test_ms42dc_invert_direction_swaps_the_motion_frames():
    """A reversed harness is fixed in config, not by editing the protocol."""

    from track1_grasp.gripper import MS42DCGripperBackend

    class NoSerial(MS42DCGripperBackend):
        def __init__(self, config):  # skip opening a real port
            self.config = config
            self._log = lambda message: None

    normal = _gripper_config(backend="ms42dc", serial_port="/dev/x")
    flipped = _gripper_config(
        backend="ms42dc", serial_port="/dev/x", invert_direction=True
    )
    open_normal = NoSerial(normal)._motion_frame(0)
    open_flipped = NoSerial(flipped)._motion_frame(0)
    close_normal = NoSerial(normal)._motion_frame(1)
    close_flipped = NoSerial(flipped)._motion_frame(1)
    assert open_normal[3] == 0 and close_normal[3] == 1
    assert open_flipped[3] == 1 and close_flipped[3] == 0
    assert open_flipped != open_normal and close_flipped != close_normal


def test_ms42dc_require_feedback_raises_when_arrival_is_missing():
    import pytest

    from track1_grasp.gripper import GripperState, MS42DCGripperBackend

    config = _gripper_config(
        backend="ms42dc",
        serial_port="/dev/motor_serial",
        require_feedback=True,
        settle_time_s=0.0,
    )

    class NoSerial(MS42DCGripperBackend):
        def __init__(self, cfg):
            self.config = cfg
            self._state = GripperState.UNKNOWN
            self.events = []
            self._log = lambda message: None

        def _write_frame(self, frame):
            self.last_frame = frame

        def _wait_for_arrival(self):
            return None

    backend = NoSerial(config)
    with pytest.raises(RuntimeError, match="到位标志"):
        backend.close()
    assert backend.state == GripperState.FAULT


def test_ms42dc_example_config_loads():
    import yaml

    payload = yaml.safe_load(
        (CONFIG_DIR / "ms42dc_gripper.example.yaml").read_text(encoding="utf-8")
    )
    config = Track1Config.from_dict(payload, base_dir=CONFIG_DIR)
    assert config.gripper.backend == "ms42dc"
    assert config.gripper.device_id == 1


def test_gripper_bench_refuses_motion_without_really():
    from track1_grasp.tools import gripper_bench

    config_path = str(CONFIG_DIR / "ms42dc_gripper.example.yaml")
    assert gripper_bench.main(["close", "--config", config_path]) == 2


def test_gripper_bench_status_is_read_only_and_reports_missing_feedback():
    from track1_grasp.tools import gripper_bench

    config_path = str(CONFIG_DIR / "track1_mock.yaml")
    # mock backend has no feedback channel, so status reports that and exits 1
    assert gripper_bench.main(["status", "--config", config_path]) == 1


def test_gripper_bench_measure_reports_travel_and_config_values(capsys):
    """measure drives both ends and derives travel from the two hard stops."""

    from track1_grasp.tools import gripper_bench
    from track1_grasp.gripper import MS42DCStatus

    config = _gripper_config(backend="ms42dc", serial_port="/dev/x")

    class StubGripper:
        """Reports 900 deg at the open stop and 2700 deg at the close stop."""

        def __init__(self):
            self.config = config
            self._reads = [MS42DCStatus(1, True, 0, 9000), MS42DCStatus(1, True, 0, 27000)]

        def open(self):
            pass

        def close(self):
            pass

        def read_status(self):
            return self._reads.pop(0)

        def close_backend(self):
            pass

    code = gripper_bench._measure(StubGripper(), 3600.0)
    out = capsys.readouterr().out
    assert code == 0
    assert "measured travel = 1800.0 deg = 5.00 turns" in out
    assert "open_angle_deg: 1800.0" in out


def test_gripper_bench_measure_needs_a_feedback_channel(capsys):
    from track1_grasp.tools import gripper_bench

    class NoFeedback:
        config = _gripper_config(backend="mock")

    assert gripper_bench._measure(NoFeedback(), 3600.0) == 1
    assert "no feedback channel" in capsys.readouterr().err


def test_gripper_bench_cycle_opens_closes_and_releases(monkeypatch):
    from track1_grasp.tools import gripper_bench

    class RecordingGripper:
        config = _gripper_config(backend="mock")

        def __init__(self):
            self.actions = []

        def open(self):
            self.actions.append("open")

        def close(self):
            self.actions.append("close")

        def read_status(self):
            return None

    monkeypatch.setattr(gripper_bench.time, "sleep", lambda _seconds: None)
    gripper = RecordingGripper()
    # The bench's status check is expected to fail for a mock/no-feedback
    # backend; the action sequence is what this test verifies.
    assert gripper_bench._cycle(gripper, 1) == 1
    assert gripper.actions == ["open", "close", "open"]


# --- MS42DC via the vendor ROS 2 motor_node ------------------------------


def test_ms42dc_ros_motion_fields_use_manual_wire_scaling():
    from track1_grasp.gripper import ms42dc_ros_motion_fields

    config = _gripper_config(
        backend="ms42dc_ros",
        open_angle_deg=3000.0,
        close_angle_deg=3000.0,
        speed_rad_s=20.0,
        open_dir=0,
        close_dir=1,
    )
    opened = ms42dc_ros_motion_fields(config, "open")
    closed = ms42dc_ros_motion_fields(config, "close")
    assert opened == {
        "id": 1,
        "speed": 200,
        "dir": 0,
        "mode": 2,
        "angle": 30000,
        "state": 0,
        "sub_divide": 32,
    }
    assert closed["dir"] == 1
    assert opened["angle"] == 30000


def test_ms42dc_ros_invert_direction_swaps_only_the_direction():
    from track1_grasp.gripper import ms42dc_ros_motion_fields

    normal = _gripper_config(backend="ms42dc_ros")
    flipped = _gripper_config(backend="ms42dc_ros", invert_direction=True)
    assert ms42dc_ros_motion_fields(normal, "open")["dir"] == 0
    assert ms42dc_ros_motion_fields(normal, "close")["dir"] == 1
    assert ms42dc_ros_motion_fields(flipped, "open")["dir"] == 1
    assert ms42dc_ros_motion_fields(flipped, "close")["dir"] == 0


class _FakeMotor:
    def __init__(self):
        self.id = 0
        self.speed = 0
        self.dir = 0
        self.mode = 0
        self.angle = 0
        self.state = 0
        self.sub_divide = 0


class _FakePublisher:
    def __init__(self, node):
        self.node = node
        self.messages = []

    def wait_for_matching_subscriptions(self, timeout_sec):
        return True

    def publish(self, message):
        self.messages.append(message)
        if message.state == 1 and self.node.reply_on_status is not None:
            reply = _FakeMotor()
            reply.id = message.id
            reply.state = 1
            reply.angle = self.node.reply_on_status
            self.node.on_state(reply)


class _FakeNode:
    def __init__(self, *, reply_on_status=None):
        self.reply_on_status = reply_on_status
        self.on_state = None
        self.publisher = None
        self.subscription = object()

    def create_publisher(self, message_type, topic, depth):
        self.publisher = _FakePublisher(self)
        return self.publisher

    def create_subscription(self, message_type, topic, callback, depth):
        self.on_state = callback
        return self.subscription

    def destroy_subscription(self, subscription):
        self.subscription = None

    def destroy_publisher(self, publisher):
        self.publisher = None


def test_ms42dc_ros_backend_publishes_open_and_close_messages():
    from track1_grasp.gripper import MS42DCRosGripperBackend

    config = _gripper_config(
        backend="ms42dc_ros", settle_time_s=0.0, require_feedback=False
    )
    node = _FakeNode()
    backend = MS42DCRosGripperBackend(
        config, node=node, message_type=_FakeMotor
    )
    backend.open()
    backend.close()
    assert [(m.dir, m.angle, m.speed, m.state) for m in node.publisher.messages] == [
        (0, 18720, 200, 0),
        (1, 18720, 200, 0),
    ]
    assert backend.state is GripperState.CLOSED
    assert backend.events == ["open", "close"]


def test_ms42dc_ros_backend_decodes_motor_state_reply():
    from track1_grasp.gripper import MS42DCRosGripperBackend

    config = _gripper_config(
        backend="ms42dc_ros", status_timeout_s=0.1, require_feedback=False
    )
    node = _FakeNode(reply_on_status=9000)
    backend = MS42DCRosGripperBackend(
        config, node=node, message_type=_FakeMotor
    )
    status = backend.read_status()
    assert status is not None
    assert status.arrived is True
    assert status.angle_deg == 900.0
    assert node.publisher.messages[-1].state == 1


def test_ms42dc_ros_require_feedback_fails_when_state_is_silent():
    from track1_grasp.gripper import MS42DCRosGripperBackend

    config = _gripper_config(
        backend="ms42dc_ros",
        require_feedback=True,
        settle_time_s=0.0,
        status_timeout_s=0.05,
        motion_timeout_s=0.05,
    )
    backend = MS42DCRosGripperBackend(
        config, node=_FakeNode(), message_type=_FakeMotor
    )
    with pytest.raises(RuntimeError, match="到位"):
        backend.open()
    assert backend.state is GripperState.FAULT


def test_ms42dc_ros_config_rejects_same_direction_and_uint16_overflow():
    with pytest.raises(ValueError, match="must differ"):
        Track1Config.from_dict(
            {
                "task": {"mode": "C"},
                "vision": {
                    "backend": "mock",
                    "scenario_file": "../scenarios/mock_track1_scene.yaml",
                },
                "gripper": {
                    "backend": "ms42dc_ros",
                    "open_dir": 1,
                    "close_dir": 1,
                },
            },
            base_dir=CONFIG_DIR,
        )
    with pytest.raises(ValueError, match="uint16"):
        Track1Config.from_dict(
            {
                "task": {"mode": "C"},
                "vision": {
                    "backend": "mock",
                    "scenario_file": "../scenarios/mock_track1_scene.yaml",
                },
                "gripper": {
                    "backend": "ms42dc_ros",
                    "open_angle_deg": 7000.0,
                },
            },
            base_dir=CONFIG_DIR,
        )


def test_ms42dc_ros_example_config_loads():
    import yaml

    payload = yaml.safe_load(
        (CONFIG_DIR / "ms42dc_ros_gripper.example.yaml").read_text(encoding="utf-8")
    )
    config = Track1Config.from_dict(payload, base_dir=CONFIG_DIR)
    assert config.gripper.backend == "ms42dc_ros"
    assert config.gripper.motor_control_topic == "/motor_control"
    assert config.gripper.require_feedback is True
