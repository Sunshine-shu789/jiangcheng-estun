"""Gripper backends and the post-close hold-verification boundary.

The state machine only ever calls ``open()``, ``close()`` and reads ``state``.
A two-finger electric gripper with its own electronic controller can therefore
be driven through whichever transport it exposes without touching the task
logic:

``mock``         no hardware, records every command
``taska``        reuse the pluggable gripper already present in taskA_grasp
``serial``       RS485/RS232 discrete open/close command bytes, optional feedback
``tcp``          the same command bytes over an Ethernet socket
``ros_service``  call a service exposed by the gripper vendor's ROS 2 driver
``ros_topic``    publish a boolean on a topic consumed by that driver
``ms42dc_ros``   publish step_motor/Motor messages to the vendor ROS 2 node

The discrete-command backends (serial/tcp) are configurable entirely from YAML
because most electric grippers expose exactly two triggers - open and close -
optionally with a settable force/position register.
"""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass

from .config import GripperConfig
from .domain import Detection, GripperState, MotionPlan


def _expand_escapes(value: str) -> str:
    """Turn literal backslash escapes from YAML into real characters."""

    if "\\" not in value:
        return value
    out: list[str] = []
    index = 0
    while index < len(value):
        char = value[index]
        if char == "\\" and index + 1 < len(value):
            nxt = value[index + 1]
            mapped = {"n": "\n", "r": "\r", "t": "\t", "0": "\x00"}.get(nxt)
            if mapped is not None:
                out.append(mapped)
                index += 2
                continue
            if nxt == "x" and index + 3 < len(value):
                try:
                    out.append(chr(int(value[index + 2 : index + 4], 16)))
                    index += 4
                    continue
                except ValueError:
                    pass
        out.append(char)
        index += 1
    return "".join(out)


def encode_command(command: str, encoding: str = "ascii", terminator: str = "") -> bytes:
    """Encode one YAML command string into the bytes the controller expects."""

    if encoding == "hex":
        cleaned = command.replace(",", " ").replace("0x", " ")
        payload = bytes.fromhex(cleaned)
    elif encoding == "ascii":
        payload = command.encode("ascii")
    else:
        raise ValueError(f"unsupported command_encoding: {encoding!r}")
    if terminator:
        payload += _expand_escapes(terminator).encode("ascii")
    return payload


def _decode_reply(raw: bytes) -> str:
    return raw.decode("ascii", errors="replace")


class MockGripperBackend:
    """In-memory gripper that records every open/close command."""

    def __init__(self, config: GripperConfig, log=None) -> None:
        self.config = config
        self._log = log or (lambda message: None)
        self._state = GripperState.UNKNOWN
        self.events: list[str] = []

    @property
    def state(self) -> GripperState:
        return self._state

    def open(self) -> None:
        self._state = GripperState.OPEN
        self.events.append("open")
        self._log("[gripper:mock] open")

    def close(self) -> None:
        self._state = GripperState.CLOSED
        self.events.append("close")
        self._log("[gripper:mock] close")

    def close_backend(self) -> None:
        return None


class TaskAGripperBackend:
    """Adapter over the existing pluggable gripper in taskA_grasp."""

    def __init__(self, config: GripperConfig, node=None, log=None) -> None:
        if not config.taska_config:
            raise ValueError("taska gripper needs gripper.taska_config")
        from taskA_grasp.config import load_config
        from taskA_grasp.gripper import make_gripper

        self.config = config
        self._log = log or (lambda message: None)
        self._taska_config = load_config(config.taska_config)
        self._gripper = make_gripper(
            self._taska_config.gripper, log=self._log, node=node
        )
        self._state = GripperState.UNKNOWN

    @property
    def state(self) -> GripperState:
        name = str(getattr(self._gripper, "state", "")).lower()
        if name == "open":
            return GripperState.OPEN
        if name in {"closed", "close"}:
            return GripperState.CLOSED
        if name == "moving":
            return GripperState.MOVING
        if name == "fault":
            return GripperState.FAULT
        return self._state

    def open(self) -> None:
        self._gripper.open()
        self._state = GripperState.OPEN
        time.sleep(self.config.settle_time_s)

    def close(self) -> None:
        self._gripper.close()
        self._state = GripperState.CLOSED
        time.sleep(self.config.settle_time_s)

    def close_backend(self) -> None:
        self._gripper.shutdown()


class DiscreteCommandGripperBackend:
    """Base for grippers driven by two discrete open/close command payloads.

    Subclasses only implement ``_write``/``_read``/``_shutdown``.  Optional
    reply parsing lets the same class report a real ``CLOSED``/``OPEN`` state
    (and therefore let the hold verifier see an empty grip).
    """

    def __init__(self, config: GripperConfig, log=None) -> None:
        self.config = config
        self._log = log or (lambda message: None)
        self._state = GripperState.UNKNOWN
        self.events: list[str] = []
        self.last_reply = ""

    # -- transport hooks --------------------------------------------------
    def _write(self, payload: bytes) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _read(self, length: int, timeout: float) -> bytes:  # pragma: no cover
        return b""

    def _shutdown(self) -> None:  # pragma: no cover - abstract
        return None

    # -- command layer ----------------------------------------------------
    def _send(self, command: str) -> None:
        payload = encode_command(
            command, self.config.command_encoding, self.config.command_terminator
        )
        self._write(payload)
        if self.config.expect_response:
            raw = self._read(
                self.config.response_length, self.config.response_timeout_s
            )
            self.last_reply = _decode_reply(raw)
            if self.last_reply:
                self._log(f"[gripper] reply: {self.last_reply!r}")

    def _refresh_state(self) -> GripperState:
        """Ask the controller where the jaws are and update the cached state.

        Unlike ``_send`` this always waits for a reply - a feedback query that
        is not read back cannot tell a closed grip from an empty one.
        """

        if not self.config.feedback_command:
            return self._state
        payload = encode_command(
            self.config.feedback_command,
            self.config.command_encoding,
            self.config.command_terminator,
        )
        self._write(payload)
        raw = self._read(self.config.response_length, self.config.response_timeout_s)
        self.last_reply = _decode_reply(raw)
        if self.last_reply:
            self._log(f"[gripper] feedback: {self.last_reply!r}")
        reply = self.last_reply
        if self.config.feedback_open_match and self.config.feedback_open_match in reply:
            self._state = GripperState.OPEN
        elif (
            self.config.feedback_closed_match
            and self.config.feedback_closed_match in reply
        ):
            self._state = GripperState.CLOSED
        return self._state

    @property
    def state(self) -> GripperState:
        return self._state

    def open(self) -> None:
        self._send(self.config.open_command)
        self._state = GripperState.OPEN
        self.events.append("open")
        self._log("[gripper] open command sent")
        time.sleep(self.config.settle_time_s)

    def close(self) -> None:
        self._send(self.config.close_command)
        self._state = GripperState.CLOSED
        self.events.append("close")
        self._log("[gripper] close command sent")
        time.sleep(self.config.settle_time_s)

    def close_backend(self) -> None:
        self._shutdown()


class SerialGripperBackend(DiscreteCommandGripperBackend):
    """RS485/RS232 electric gripper controller."""

    def __init__(self, config: GripperConfig, log=None) -> None:
        super().__init__(config, log=log)
        import serial  # pyserial, imported lazily

        self._serial = serial.Serial(
            port=config.serial_port,
            baudrate=config.serial_baudrate,
            bytesize=config.serial_bytesize,
            parity=config.serial_parity,
            stopbits=config.serial_stopbits,
            timeout=config.response_timeout_s,
        )
        self._log(
            f"[gripper:serial] {config.serial_port} @ {config.serial_baudrate}"
            f" {config.serial_bytesize}{config.serial_parity}{config.serial_stopbits}"
        )

    def _write(self, payload: bytes) -> None:
        self._serial.write(payload)
        self._serial.flush()

    def _read(self, length: int, timeout: float) -> bytes:
        self._serial.timeout = timeout
        if length > 0:
            return self._serial.read(length)
        waiting = getattr(self._serial, "in_waiting", 0)
        if waiting:
            return self._serial.read(waiting)
        return self._serial.read(64)

    def _shutdown(self) -> None:
        try:
            self._serial.close()
        except Exception:  # pragma: no cover - best effort on shutdown
            pass


class TcpGripperBackend(DiscreteCommandGripperBackend):
    """Electric gripper controller reachable over Ethernet/TCP."""

    def __init__(self, config: GripperConfig, log=None) -> None:
        super().__init__(config, log=log)
        self._socket = socket.create_connection(
            (config.tcp_host, config.tcp_port), timeout=config.response_timeout_s
        )
        self._log(f"[gripper:tcp] {config.tcp_host}:{config.tcp_port}")

    def _write(self, payload: bytes) -> None:
        self._socket.sendall(payload)

    def _read(self, length: int, timeout: float) -> bytes:
        self._socket.settimeout(timeout)
        try:
            if length > 0:
                return self._socket.recv(length)
            return self._socket.recv(256)
        except (TimeoutError, socket.timeout):
            return b""

    def _shutdown(self) -> None:
        try:
            self._socket.close()
        except Exception:  # pragma: no cover - best effort on shutdown
            pass


class RosServiceGripperBackend:
    """Call a service exposed by the gripper vendor's ROS 2 driver.

    The service is expected to take a boolean request (``data``) where true
    means open; ``ros_open_value``/``ros_close_value`` let a driver with the
    opposite convention be used unchanged.
    """

    def __init__(self, config: GripperConfig, node=None, log=None) -> None:
        if node is None:
            raise ValueError("ros_service gripper needs a ROS 2 node")
        from std_srvs.srv import SetBool

        self.config = config
        self._log = log or (lambda message: None)
        self._node = node
        self._client = node.create_client(SetBool, config.ros_service)
        self._state = GripperState.UNKNOWN
        self._log(f"[gripper:ros_service] {config.ros_service}")

    @property
    def state(self) -> GripperState:
        return self._state

    def _call(self, value: bool, name: str) -> None:
        from std_srvs.srv import SetBool

        if not self._client.wait_for_service(timeout_sec=1.0):
            raise RuntimeError(f"gripper service unavailable: {self.config.ros_service}")
        request = SetBool.Request()
        request.data = value
        future = self._client.call_async(request)
        deadline = time.monotonic() + max(self.config.settle_time_s, 2.0)
        while not future.done():
            time.sleep(0.01)
            if time.monotonic() > deadline:
                raise RuntimeError(f"gripper service timed out: {self.config.ros_service}")
        result = future.result()
        if result is not None and not getattr(result, "success", True):
            raise RuntimeError(
                f"gripper service rejected {name}: {getattr(result, 'message', '')}"
            )
        self._state = GripperState.OPEN if name == "open" else GripperState.CLOSED
        time.sleep(self.config.settle_time_s)

    def open(self) -> None:
        self._call(self.config.ros_open_value, "open")

    def close(self) -> None:
        self._call(self.config.ros_close_value, "close")

    def close_backend(self) -> None:
        return None


class RosTopicGripperBackend:
    """Publish a boolean on the topic consumed by a gripper driver."""

    def __init__(self, config: GripperConfig, node=None, log=None) -> None:
        if node is None:
            raise ValueError("ros_topic gripper needs a ROS 2 node")
        from std_msgs.msg import Bool

        self.config = config
        self._log = log or (lambda message: None)
        self._publisher = node.create_publisher(Bool, config.ros_topic, 10)
        self._state = GripperState.UNKNOWN
        self._log(f"[gripper:ros_topic] {config.ros_topic}")

    @property
    def state(self) -> GripperState:
        return self._state

    def _publish(self, value: bool, name: str) -> None:
        from std_msgs.msg import Bool

        message = Bool()
        message.data = value
        self._publisher.publish(message)
        self._state = GripperState.OPEN if name == "open" else GripperState.CLOSED
        time.sleep(self.config.settle_time_s)

    def open(self) -> None:
        self._publish(self.config.ros_open_value, "open")

    def close(self) -> None:
        self._publish(self.config.ros_close_value, "close")

    def close_backend(self) -> None:
        return None


# ---------------------------------------------------------------------------
# WHEELTEC FAE two-finger electric gripper (MS42DC stepper controller)
# ---------------------------------------------------------------------------
#
# 11-byte frame, TTL serial / USB-C, 115200 8N1 by default:
#
#   7B  ID  mode  dir  细分  POS_H POS_L SPD_H SPD_L  BCC  7D
#
#   dir  0 = open (counter-clockwise), 1 = close (clockwise)
#   POS  physical angle in degrees times ten
#   SPD  physical speed in rad/s times ten
#   BCC  XOR of the first nine bytes; 0x7D terminates the frame
#
# A status request is the same frame with mode/dir/细分/POS/SPD all zero and
# is answered with nine bytes:
#
#   ID  到位标志  SPD_H SPD_L  ANGLE(4 bytes, big endian, x10)  RCC
#
# where RCC is the XOR of the first eight bytes.  Rotational position is in
# the frame but the status reply is the only feedback the controller gives,
# so it cannot independently prove that the fingers reached an object.

MS42DC_HEAD = 0x7B
MS42DC_TAIL = 0x7D
MS42DC_STATUS_LENGTH = 9
MS42DC_OPEN_DIR = 0
MS42DC_CLOSE_DIR = 1


def _uint16_wire_value(value: float, name: str) -> int:
    wire = int(round(float(value) * 10.0))
    if not 0 <= wire <= 0xFFFF:
        raise ValueError(f"{name} does not fit the step_motor/Motor uint16 field")
    return wire


def ms42dc_ros_motion_fields(config: GripperConfig, action: str) -> dict[str, int]:
    """Build the step_motor/Motor field values for an open or close command."""

    if action not in {"open", "close"}:
        raise ValueError(f"unsupported MS42DC ROS action: {action!r}")
    direction = config.open_dir if action == "open" else config.close_dir
    if config.invert_direction:
        direction = config.open_dir if direction == config.close_dir else config.close_dir
    angle = config.open_angle_deg if action == "open" else config.close_angle_deg
    return {
        "id": int(config.device_id),
        "speed": _uint16_wire_value(config.speed_rad_s, "speed_rad_s"),
        "dir": int(direction),
        "mode": int(config.motion_mode),
        "angle": _uint16_wire_value(angle, f"{action}_angle_deg"),
        "state": 0,
        "sub_divide": int(config.subdivide),
    }


def ms42dc_bcc(data: bytes) -> int:
    """Exclusive-or of ``data`` - the frame's BCC/RCC byte."""

    check = 0
    for byte in data:
        check ^= byte
    return check


def encode_ms42dc_frame(
    *,
    device_id: int,
    mode: int,
    direction: int,
    subdivide: int,
    angle_wire: int,
    speed_wire: int,
) -> bytes:
    """Build one eleven-byte MS42DC frame."""

    body = bytes(
        [
            MS42DC_HEAD,
            device_id & 0xFF,
            mode & 0xFF,
            direction & 0xFF,
            subdivide & 0xFF,
            (angle_wire >> 8) & 0xFF,
            angle_wire & 0xFF,
            (speed_wire >> 8) & 0xFF,
            speed_wire & 0xFF,
        ]
    )
    return body + bytes([ms42dc_bcc(body), MS42DC_TAIL])


def ms42dc_status_frame(device_id: int) -> bytes:
    return encode_ms42dc_frame(
        device_id=device_id,
        mode=0,
        direction=0,
        subdivide=0,
        angle_wire=0,
        speed_wire=0,
    )


@dataclass(frozen=True)
class MS42DCStatus:
    """One decoded nine-byte status reply."""

    device_id: int
    arrived: bool
    speed_raw: int
    angle_raw: int

    @property
    def angle_deg(self) -> float:
        return self.angle_raw / 10.0

    @property
    def speed_rad_s(self) -> float:
        return self.speed_raw / 10.0


def parse_ms42dc_status(reply: bytes) -> MS42DCStatus | None:
    """Decode a status reply, returning ``None`` when it is malformed."""

    if len(reply) < MS42DC_STATUS_LENGTH:
        return None
    data = reply[:MS42DC_STATUS_LENGTH]
    if ms42dc_bcc(data[:8]) != data[8]:
        return None
    return MS42DCStatus(
        device_id=data[0],
        arrived=bool(data[1]),
        speed_raw=(data[2] << 8) | data[3],
        angle_raw=(data[4] << 24) | (data[5] << 16) | (data[6] << 8) | data[7],
    )


class MS42DCGripperBackend:
    """Drive the FAE two-finger electric gripper through its MS42DC controller.

    Works over the controller's TTL serial pins or its USB-C port (set the
    board's second DIP switch to ON for USB).  ``open()``/``close()`` send the
    motion frame and then wait for the controller's 到位标志, so the caller
    gets a gripper that has finished moving rather than one that merely
    accepted the command.
    """

    def __init__(self, config: GripperConfig, log=None) -> None:
        import serial  # pyserial, imported lazily

        self.config = config
        self._log = log or (lambda message: None)
        self._state = GripperState.UNKNOWN
        self.events: list[str] = []
        self.last_status: MS42DCStatus | None = None
        self._serial = serial.Serial(
            port=config.serial_port,
            baudrate=config.serial_baudrate,
            bytesize=config.serial_bytesize,
            parity=config.serial_parity,
            stopbits=config.serial_stopbits,
            timeout=config.status_timeout_s,
        )
        self._log(
            f"[gripper:ms42dc] {config.serial_port} @ {config.serial_baudrate} "
            f"id={config.device_id} subdivide={config.subdivide}"
        )

    # -- protocol ---------------------------------------------------------
    def _motion_frame(self, direction: int) -> bytes:
        angle = (
            self.config.open_angle_deg
            if direction == MS42DC_OPEN_DIR
            else self.config.close_angle_deg
        )
        if self.config.invert_direction:
            direction = MS42DC_CLOSE_DIR if direction == MS42DC_OPEN_DIR else MS42DC_OPEN_DIR
        return encode_ms42dc_frame(
            device_id=self.config.device_id,
            mode=self.config.motion_mode,
            direction=direction,
            subdivide=self.config.subdivide,
            angle_wire=int(round(angle * 10.0)),
            speed_wire=int(round(self.config.speed_rad_s * 10.0)),
        )

    def _write_frame(self, frame: bytes) -> None:
        self._serial.reset_input_buffer()
        self._serial.write(frame)
        self._serial.flush()

    def read_status(self) -> MS42DCStatus | None:
        """Send the status request and decode one reply."""

        self._write_frame(ms42dc_status_frame(self.config.device_id))
        raw = self._serial.read(MS42DC_STATUS_LENGTH)
        status = parse_ms42dc_status(raw)
        if status is None:
            if raw:
                self._log(f"[gripper:ms42dc] malformed status reply: {raw.hex(' ')}")
            else:
                self._log("[gripper:ms42dc] no status reply")
        self.last_status = status
        return status

    def _wait_for_arrival(self) -> MS42DCStatus | None:
        """Block until 到位标志 is set, or give up quietly without feedback."""

        status = self.read_status()
        if status is None and self.config.require_feedback is False:
            self._log(
                "[gripper:ms42dc] controller gave no feedback; not waiting for arrival"
            )
            return None
        deadline = time.monotonic() + self.config.motion_timeout_s
        while time.monotonic() < deadline:
            if status is not None and status.arrived:
                self._log(
                    f"[gripper:ms42dc] arrived, angle={status.angle_deg:.1f} deg"
                )
                return status
            time.sleep(0.05)
            status = self.read_status()
        self._log("[gripper:ms42dc] motion did not report arrival before timeout")
        return status

    def _move(self, direction: int, state: GripperState, name: str) -> None:
        frame = self._motion_frame(direction)
        self._write_frame(frame)
        self._state = GripperState.MOVING
        self.events.append(name)
        self._log(f"[gripper:ms42dc] {name}: {frame.hex(' ')}")
        status = self._wait_for_arrival()
        if self.config.require_feedback and (status is None or not status.arrived):
            self._state = GripperState.FAULT
            raise RuntimeError(
                "MS42DC did not report 到位标志 over the serial link; "
                "check controller power, cabling and /dev/motor_serial"
            )
        self._state = state
        time.sleep(self.config.settle_time_s)

    # -- GripperBackend ---------------------------------------------------
    @property
    def state(self) -> GripperState:
        return self._state

    def open(self) -> None:
        self._move(MS42DC_OPEN_DIR, GripperState.OPEN, "open")

    def close(self) -> None:
        self._move(MS42DC_CLOSE_DIR, GripperState.CLOSED, "close")

    def close_backend(self) -> None:
        try:
            self._serial.close()
        except Exception:  # pragma: no cover - best effort on shutdown
            pass


class MS42DCRosGripperBackend:
    """Drive the vendor ``step_motor/motor_node`` over ROS 2 topics.

    The vendor node owns the serial port.  This backend therefore never opens
    it directly and instead publishes ``step_motor/Motor`` commands on
    ``/motor_control``.  Status requests are sent through the same topic with
    ``state=1`` and the driver's decoded reply is read from ``/motor_state``.
    """

    def __init__(
        self,
        config: GripperConfig,
        *,
        node=None,
        log=None,
        message_type=None,
    ) -> None:
        if node is None:
            raise ValueError("ms42dc_ros gripper needs a ROS 2 node")
        if message_type is None:
            try:
                from step_motor.msg import Motor as message_type
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "step_motor messages are unavailable; source "
                    "/media/sunshine/ESTUN_RUNTIME/ms42dc_vendor_ws/install/setup.bash"
                ) from exc

        self.config = config
        self._log = log or (lambda message: None)
        self._node = node
        self._message_type = message_type
        self._state = GripperState.UNKNOWN
        self.events: list[str] = []
        self.last_status: MS42DCStatus | None = None
        self._last_status: MS42DCStatus | None = None
        self._status_sequence = 0
        self._condition = threading.Condition()
        self._control_ready = False
        self._publisher = node.create_publisher(
            message_type, config.motor_control_topic, 10
        )
        self._subscription = node.create_subscription(
            message_type,
            config.motor_state_topic,
            self._on_state,
            10,
        )
        self._log(
            f"[gripper:ms42dc_ros] control={config.motor_control_topic} "
            f"state={config.motor_state_topic} id={config.device_id}"
        )

    @property
    def state(self) -> GripperState:
        return self._state

    def _on_state(self, message) -> None:
        if int(message.id) != int(self.config.device_id):
            return
        status = MS42DCStatus(
            device_id=int(message.id),
            arrived=bool(message.state),
            speed_raw=int(message.speed),
            angle_raw=int(message.angle),
        )
        with self._condition:
            self._last_status = status
            self.last_status = status
            self._status_sequence += 1
            self._condition.notify_all()

    def _publish_fields(self, fields: dict[str, int]) -> None:
        message = self._message_type()
        for name, value in fields.items():
            setattr(message, name, value)
        self._publisher.publish(message)

    def _wait_for_control_subscriber(self) -> None:
        if self._control_ready:
            return
        pub = self._publisher
        waiter = getattr(pub, "wait_for_matching_subscriptions", None)
        if callable(waiter):
            matched = bool(waiter(self.config.ros_connect_timeout_s))
        else:
            getter = getattr(pub, "get_subscription_count", None)
            matched = bool(getter()) if callable(getter) else True
        if not matched:
            raise RuntimeError(
                f"no subscriber on {self.config.motor_control_topic}; "
                "confirm the vendor motor_node is running"
            )
        self._control_ready = True

    def _publish_status_request(self) -> None:
        self._publish_fields(
            {
                "id": int(self.config.device_id),
                "speed": 0,
                "dir": 0,
                "mode": 0,
                "angle": 0,
                "state": 1,
                "sub_divide": 0,
            }
        )

    def read_status(
        self,
        *,
        timeout_s: float | None = None,
        wait_for_arrival: bool = False,
    ) -> MS42DCStatus | None:
        """Request feedback and wait for a fresh ``/motor_state`` reply."""

        timeout = (
            self.config.status_timeout_s if timeout_s is None else float(timeout_s)
        )
        deadline = time.monotonic() + max(0.0, timeout)
        next_query = 0.0
        before_sequence = self._status_sequence
        self._wait_for_control_subscriber()
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_query:
                before_sequence = self._status_sequence
                self._publish_status_request()
                next_query = now + 0.20
            with self._condition:
                self._condition.wait(
                    timeout=min(0.05, max(0.001, deadline - time.monotonic()))
                )
                status = self._last_status
                sequence = self._status_sequence
            if status is not None and sequence > before_sequence:
                if not wait_for_arrival or status.arrived:
                    return status
        self._log(
            "[gripper:ms42dc_ros] no valid /motor_state reply before timeout"
        )
        return None

    def _move(self, action: str, state: GripperState) -> None:
        fields = ms42dc_ros_motion_fields(self.config, action)
        self._wait_for_control_subscriber()
        self._state = GripperState.MOVING
        self._publish_fields(fields)
        self.events.append(action)
        self._log(
            f"[gripper:ms42dc_ros] {action}: dir={fields['dir']} "
            f"angle_wire={fields['angle']} speed_wire={fields['speed']}"
        )
        if self.config.require_feedback:
            status = self.read_status(
                timeout_s=self.config.motion_timeout_s,
                wait_for_arrival=True,
            )
            if status is None:
                self._state = GripperState.FAULT
                raise RuntimeError(
                    "MS42DC did not report 到位标志 on /motor_state; "
                    "check power, /dev/motor_serial and the USB DIP switch"
                )
        self._state = state
        time.sleep(self.config.settle_time_s)

    def open(self) -> None:
        self._move("open", GripperState.OPEN)

    def close(self) -> None:
        self._move("close", GripperState.CLOSED)

    def close_backend(self) -> None:
        for method_name, resource in (
            ("destroy_subscription", self._subscription),
            ("destroy_publisher", self._publisher),
        ):
            destroy = getattr(self._node, method_name, None)
            if callable(destroy):
                try:
                    destroy(resource)
                except Exception:  # pragma: no cover - best effort on shutdown
                    pass


class MS42DCHoldVerifier:
    """Hold check driven by the MS42DC controller's own status reply.

    The controller reports 到位标志 once motion finishes.  When
    ``hold_min_angle_deg`` is set the travelled angle is also compared against
    it: a close that ran all the way to the hard stop means the fingers met
    nothing, so the grasp is rejected.  Because the check depends on a
    working reply path it stays optimistic - with a clear note - when the
    controller is silent and ``require_feedback`` is false.
    """

    def __init__(self, gripper, config: GripperConfig, log=None) -> None:
        self._gripper = gripper
        self._config = config
        self._log = log or (lambda message: None)

    def verify(self, detection: Detection, plan: MotionPlan) -> tuple[bool, str]:
        # ``close()`` already proved 到位 through the same controller reply.
        # Reuse that confirmed status: on the direct serial backend the flag can
        # be one-shot, and a second read immediately afterwards may report only
        # the settled position with arrived=False.
        status = getattr(self._gripper, "last_status", None)
        if status is None or not status.arrived:
            read_status = self._gripper.read_status
            try:
                status = read_status(
                    timeout_s=self._config.motion_timeout_s,
                    wait_for_arrival=True,
                )
            except TypeError as exc:
                if "unexpected keyword" not in str(exc):
                    raise
                status = read_status()
        if status is None:
            if self._config.require_feedback:
                return False, "no gripper status reply and require_feedback is set"
            return True, "no gripper feedback; hold unverified"
        if not status.arrived:
            return False, "controller has not reported 到位标志 after closing"
        if self._config.ignore_empty_close:
            return True, f"jaws settled at {status.angle_deg:.1f} deg"
        minimum = self._config.hold_min_angle_deg
        if minimum > 0.0 and status.angle_deg >= minimum:
            return (
                False,
                f"jaws closed {status.angle_deg:.1f} deg (>= {minimum:.1f} deg): "
                "nothing between the fingers",
            )
        return True, f"jaws settled at {status.angle_deg:.1f} deg"


class MockHoldVerifier:
    """Optimistic verifier for no-hardware runs.

    A real implementation should combine one or more of: jaw position,
    motor current, object displacement in the wrist camera, or a short
    lift-and-reobserve check.
    """

    def verify(self, detection: Detection, plan: MotionPlan) -> tuple[bool, str]:
        return True, "mock hold verifier accepted"


class GripperStateHoldVerifier:
    """Pessimistic verifier that trusts only the reported jaw state.

    Use it with an electric gripper whose controller reports whether the jaws
    actually reached the closed position (an empty close on a hard stop reads
    differently on most controllers, so this catches a slipped object).
    """

    def __init__(self, gripper, log=None) -> None:
        self._gripper = gripper
        self._log = log or (lambda message: None)

    def verify(self, detection: Detection, plan: MotionPlan) -> tuple[bool, str]:
        refresh = getattr(self._gripper, "_refresh_state", None)
        if callable(refresh):
            refresh()
        state = getattr(self._gripper, "state", GripperState.UNKNOWN)
        if state == GripperState.CLOSED:
            return True, "controller reports jaws closed"
        return False, f"controller reports jaws {state}"


def build_gripper(config: GripperConfig, node=None, log=None):
    if config.backend == "mock":
        return MockGripperBackend(config, log=log)
    if config.backend == "taska":
        return TaskAGripperBackend(config, node=node, log=log)
    if config.backend == "ms42dc":
        return MS42DCGripperBackend(config, log=log)
    if config.backend == "ms42dc_ros":
        return MS42DCRosGripperBackend(config, node=node, log=log)
    if config.backend == "serial":
        return SerialGripperBackend(config, log=log)
    if config.backend == "tcp":
        return TcpGripperBackend(config, log=log)
    if config.backend == "ros_service":
        return RosServiceGripperBackend(config, node=node, log=log)
    if config.backend == "ros_topic":
        return RosTopicGripperBackend(config, node=node, log=log)
    raise ValueError(f"unsupported gripper backend: {config.backend!r}")


def build_hold_verifier(config: GripperConfig, gripper, log=None):
    """Pick a hold verifier that matches the configured transport.

    Hardware transports get the state-based verifier so an unsuccessful close
    is not silently treated as a successful grasp; mock runs stay optimistic.
    """

    if config.backend == "mock":
        return MockHoldVerifier()
    if config.backend in {"ms42dc", "ms42dc_ros"}:
        return MS42DCHoldVerifier(gripper, config, log=log)
    return GripperStateHoldVerifier(gripper, log=log)
