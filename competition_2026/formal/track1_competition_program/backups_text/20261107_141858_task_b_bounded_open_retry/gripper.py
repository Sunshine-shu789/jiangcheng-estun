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
from math import isfinite
from pathlib import Path

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
        self.last_close_evidence: MS42DCCloseEvidence | None = None

    @property
    def state(self) -> GripperState:
        return self._state

    def open(self) -> None:
        self._state = GripperState.OPEN
        self.events.append("open")
        self._log("[gripper:mock] open")

    def close(self) -> None:
        self.last_close_evidence = None
        self._state = GripperState.CLOSED
        self.events.append("close")
        self._log("[gripper:mock] close")

    def close_for_width(self, width_m: float) -> None:
        """Record deterministic offline evidence after the same CAD gate."""

        width_m = float(width_m)
        minimum_travel_deg = _minimum_close_travel_deg(self.config, width_m)
        self._state = GripperState.CLOSED
        self.events.append("close")
        self.last_close_evidence = MS42DCCloseEvidence(
            width_m=width_m,
            minimum_travel_deg=minimum_travel_deg,
            start_angle_deg=0.0,
            end_angle_deg=minimum_travel_deg,
            travel_deg=minimum_travel_deg,
            arrived=True,
            contact_platform=True,
            stable_samples=max(3, self.config.contact_stable_samples),
            accepted=True,
            reason=(
                "mock width-aware close accepted deterministic offline "
                f"travel {minimum_travel_deg:.1f} deg"
            ),
        )
        self._log(
            "[gripper:mock] close_for_width "
            f"width={width_m:.6f} m accepted"
        )

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

# The official MS42DC tolerance below is derived from the same STEP-derived jaw
# profile used by the C/D short-edge planner.  The finger aperture is V-shaped,
# so checking one nominal depth can falsely reject objects that fit farther
# toward the fingertip.  The complete profile remains the fail-closed authority.
_JAW_PROFILE_RELATIVE_PATH = Path(
    "robot_description/estun_s3_60_g2/config/"
    "fae2mr86m_ms42dc_jaw_profile.json"
)
_JAW_DIMENSIONS_RELATIVE_PATH = Path(
    "robot_description/estun_s3_60_g2/config/"
    "fae2mr86m_ms42dc_dimensions.json"
)


def _uint16_wire_value(value: float, name: str) -> int:
    wire = int(round(float(value) * 10.0))
    if not 0 <= wire <= 0xFFFF:
        raise ValueError(f"{name} does not fit the step_motor/Motor uint16 field")
    return wire


def ms42dc_motion_speed_rad_s(config: GripperConfig, action: str) -> float:
    """Return the wire-level speed for one open or close command."""

    if action not in {"open", "close"}:
        raise ValueError(f"unsupported MS42DC action: {action!r}")
    if action == "close" and config.close_speed_rad_s > 0.0:
        return float(config.close_speed_rad_s)
    return float(config.speed_rad_s)


def ms42dc_ros_motion_fields(config: GripperConfig, action: str) -> dict[str, int]:
    """Build the step_motor/Motor field values for an open or close command."""

    if action not in {"open", "close"}:
        raise ValueError(f"unsupported MS42DC ROS action: {action!r}")
    direction = config.open_dir if action == "open" else config.close_dir
    if config.invert_direction:
        direction = config.open_dir if direction == config.close_dir else config.close_dir
    angle = config.open_angle_deg if action == "open" else config.close_angle_deg
    speed_rad_s = ms42dc_motion_speed_rad_s(config, action)
    return {
        "id": int(config.device_id),
        "speed": _uint16_wire_value(speed_rad_s, f"{action}_speed_rad_s"),
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


def _load_ms42dc_jaw_profile():
    """Load the official STEP-derived jaw profile or fail closed."""

    from .jaw_profile import JawProfile

    project_root = Path(__file__).resolve().parents[3]
    profile_path = project_root / _JAW_PROFILE_RELATIVE_PATH
    dimensions_path = project_root / _JAW_DIMENSIONS_RELATIVE_PATH
    if not profile_path.is_file() or not dimensions_path.is_file():
        raise RuntimeError(
            "MS42DC width-aware close requires the official STEP-derived jaw "
            f"profile and dimensions; missing {profile_path} or {dimensions_path}"
        )
    try:
        return JawProfile.from_files(profile_path, dimensions_path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(
            f"MS42DC could not load the CAD close-travel evidence: {exc}"
        ) from exc


def _validated_close_width_capacity(
    config: GripperConfig,
    width_m: float,
):
    """Return profile capacity for ``width + configured reserve``."""

    width = float(width_m)
    if not isfinite(width) or width <= 0.0:
        raise RuntimeError(
            "MS42DC width-aware close needs a finite positive object width"
        )
    effective_width = width + float(config.close_travel_width_margin_m)
    if not isfinite(effective_width) or effective_width <= width:
        raise RuntimeError(
            "MS42DC close-travel width margin must be finite and non-negative"
        )

    profile = _load_ms42dc_jaw_profile()
    capacity = profile.opening_capacity(effective_width)
    if not capacity.fits:
        raise RuntimeError(
            "MS42DC measured short-edge width plus safety margin "
            f"{effective_width:.6f} m exceeds the maximum CAD opening "
            f"{capacity.maximum_open_opening_m:.6f} m at "
            f"{capacity.maximum_open_depth_m * 1000.0:.1f} mm depth; "
            f"full CAD profile depth range is "
            f"{capacity.profile_depth_min_m * 1000.0:.1f}-"
            f"{capacity.profile_depth_max_m * 1000.0:.1f} mm"
        )
    return capacity


def validate_close_width(config: GripperConfig, width_m: float):
    """Fail closed when the width cannot fit anywhere in the CAD jaw envelope.

    This function is safe to call during read-only candidate precheck.  The
    execution-time :meth:`MS42DCGripper.close_for_width` path calls it again as
    defence in depth.  The returned capacity includes the configured safety
    margin and is useful for site diagnostics.
    """

    return _validated_close_width_capacity(config, width_m)


def _minimum_close_travel_deg(config: GripperConfig, width_m: float) -> float:
    """Return the minimum jaw travel that proves the close reached the object.

    The C/D planner grasps across the PCA short edge, while the actual object
    height can occupy any part of the finger.  Use the full STEP-derived CAD
    profile's maximum opening and its matching closed-state span.  A width
    margin makes the check tolerant of width underestimation, and the
    configured angle tolerance covers encoder/backlash uncertainty.  Missing
    or inconsistent CAD evidence is a hard error rather than an optimistic
    pass.
    """

    capacity = _validated_close_width_capacity(config, width_m)
    span = float(capacity.closing_span_m)
    if not isfinite(span) or span <= 1e-6:
        raise RuntimeError(
            "MS42DC maximum-open CAD profile has no usable closing span"
        )
    closing_fraction = (
        capacity.required_opening_m - capacity.closed_opening_at_maximum_m
    ) / span
    closing_fraction = min(1.0, max(0.0, closing_fraction))
    raw_minimum_travel = (1.0 - closing_fraction) * float(
        config.close_angle_deg
    )
    tolerance = float(config.close_travel_tolerance_deg)
    # A wide object can legitimately need only a few degrees of jaw travel.
    # Applying the full tolerance in that case used to turn a valid positive
    # CAD requirement into a negative number and reject the grasp before the
    # close command.  Keep the tolerance, but never let it remove the entire
    # positive travel requirement: zero measured travel must still fail.
    minimum_travel = max(0.1, raw_minimum_travel - tolerance)
    if not isfinite(raw_minimum_travel) or not isfinite(minimum_travel):
        raise RuntimeError(
            "MS42DC CAD evidence produced a non-finite minimum close travel"
        )
    return float(minimum_travel)


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


@dataclass(frozen=True)
class MS42DCCloseEvidence:
    """Independent evidence sampled during one width-aware close command."""

    width_m: float
    minimum_travel_deg: float
    start_angle_deg: float
    end_angle_deg: float
    travel_deg: float
    arrived: bool
    contact_platform: bool
    stable_samples: int
    accepted: bool
    reason: str


def _rejected_close_evidence(
    *,
    width_m: float,
    minimum_travel_deg: float,
    start_angle_deg: float,
    end_angle_deg: float,
    arrived: bool,
    contact_platform: bool,
    stable_samples: int,
    reason: str,
) -> MS42DCCloseEvidence:
    """Build a failure record without losing the measured close state."""

    return MS42DCCloseEvidence(
        width_m=float(width_m),
        minimum_travel_deg=float(minimum_travel_deg),
        start_angle_deg=float(start_angle_deg),
        end_angle_deg=float(end_angle_deg),
        travel_deg=float(end_angle_deg) - float(start_angle_deg),
        arrived=bool(arrived),
        contact_platform=bool(contact_platform),
        stable_samples=int(stable_samples),
        accepted=False,
        reason=str(reason),
    )


def _sample_ms42dc_close_evidence(
    config: GripperConfig,
    *,
    backend_name: str,
    width_m: float,
    minimum_travel_deg: float,
    start_angle_deg: float,
    read_status,
    log=None,
) -> MS42DCCloseEvidence:
    """Sample a close plateau independently of the arrival helper.

    The controller may report ``arrived=False`` while the jaws are physically
    stopped on a compliant object.  Conversely, a stale ``arrived=True`` from
    before the command must never prove contact.  This sampler therefore waits
    for post-command angle motion, then requires a consecutive near-zero-speed
    angle plateau before evaluating object-specific CAD travel.  In
    width-aware mode the CAD travel is authoritative; the legacy fixed-angle
    hard-stop ceiling is only retained when width-aware evidence is disabled.
    """

    log = log or (lambda message: None)
    deadline = time.monotonic() + float(config.motion_timeout_s)
    stable_samples = 0
    previous_stable_angle: float | None = None
    saw_motion = False
    last_status: MS42DCStatus | None = None
    required_samples = max(3, int(config.contact_stable_samples))

    while time.monotonic() < deadline:
        status = read_status()
        if status is None:
            stable_samples = 0
            previous_stable_angle = None
            time.sleep(0.05)
            continue

        last_status = status
        angle_delta_from_start = abs(status.angle_deg - start_angle_deg)
        if (
            not status.arrived
            or angle_delta_from_start > config.arrival_angle_tolerance_deg
        ):
            saw_motion = True

        if abs(status.speed_rad_s) > config.arrival_speed_tolerance_rad_s:
            stable_samples = 0
            previous_stable_angle = None
        elif (
            previous_stable_angle is None
            or abs(status.angle_deg - previous_stable_angle)
            > config.arrival_angle_tolerance_deg
        ):
            stable_samples = 1
            previous_stable_angle = status.angle_deg
        else:
            stable_samples += 1
            previous_stable_angle = status.angle_deg

        contact_platform = saw_motion and stable_samples >= required_samples
        if not contact_platform:
            time.sleep(0.05)
            continue

        end_angle_deg = status.angle_deg
        travel_deg = end_angle_deg - start_angle_deg
        width_aware = bool(config.require_close_travel_evidence)
        if travel_deg + 1e-9 < minimum_travel_deg:
            reason = (
                f"{backend_name} effective close travel {travel_deg:.1f} deg "
                "is below the CAD-derived minimum "
                f"{minimum_travel_deg:.1f} deg for object width "
                f"{float(width_m):.6f} m; refusing to treat the jaws as closed"
            )
        elif (
            not width_aware
            and end_angle_deg >= config.close_contact_max_angle_deg
        ):
            reason = (
                f"{backend_name} final close angle {end_angle_deg:.1f} deg is "
                "at or beyond the configured legacy contact ceiling "
                f"{config.close_contact_max_angle_deg:.1f} deg while "
                "width-aware evidence is disabled; refusing the bare-hard-stop "
                "empty close"
            )
        elif not status.arrived and not config.accept_contact_without_arrival:
            reason = (
                f"{backend_name} reached a stable contact plateau at "
                f"{end_angle_deg:.1f} deg but the controller did not set the "
                "arrived flag and "
                "accept_contact_without_arrival=false"
            )
        else:
            if width_aware:
                angle_evidence = (
                    f"angle={end_angle_deg:.1f} deg; object-specific CAD travel "
                    "is authoritative and no fixed empty-close angle is used"
                )
            else:
                angle_evidence = (
                    f"angle={end_angle_deg:.1f} deg < "
                    f"{config.close_contact_max_angle_deg:.1f} deg"
                )
            reason = (
                f"{backend_name} stable contact evidence accepted: "
                f"travel={travel_deg:.1f} deg >= {minimum_travel_deg:.1f} deg, "
                f"{angle_evidence}, "
                f"arrived={status.arrived}, "
                f"stable_samples={stable_samples}"
            )
            log("[gripper:ms42dc] " + reason)
            return MS42DCCloseEvidence(
                width_m=float(width_m),
                minimum_travel_deg=float(minimum_travel_deg),
                start_angle_deg=float(start_angle_deg),
                end_angle_deg=end_angle_deg,
                travel_deg=travel_deg,
                arrived=status.arrived,
                contact_platform=True,
                stable_samples=stable_samples,
                accepted=True,
                reason=reason,
            )

        log("[gripper:ms42dc] rejected close evidence: " + reason)
        return _rejected_close_evidence(
            width_m=width_m,
            minimum_travel_deg=minimum_travel_deg,
            start_angle_deg=start_angle_deg,
            end_angle_deg=end_angle_deg,
            arrived=status.arrived,
            contact_platform=True,
            stable_samples=stable_samples,
            reason=reason,
        )

    end_angle_deg = (
        last_status.angle_deg
        if last_status is not None
        else float(start_angle_deg)
    )
    reason = (
        f"{backend_name} timed out before proving a fresh stable contact "
        f"plateau; saw_motion={saw_motion}, "
        f"stable_samples={stable_samples}/{required_samples}, "
        f"end_angle={end_angle_deg:.1f} deg"
    )
    log("[gripper:ms42dc] rejected close evidence: " + reason)
    return _rejected_close_evidence(
        width_m=width_m,
        minimum_travel_deg=minimum_travel_deg,
        start_angle_deg=start_angle_deg,
        end_angle_deg=end_angle_deg,
        arrived=bool(last_status.arrived) if last_status is not None else False,
        contact_platform=False,
        stable_samples=stable_samples,
        reason=reason,
    )


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
        self.last_close_evidence: MS42DCCloseEvidence | None = None
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
        action = "open" if direction == MS42DC_OPEN_DIR else "close"
        angle = (
            self.config.open_angle_deg
            if action == "open"
            else self.config.close_angle_deg
        )
        speed_rad_s = ms42dc_motion_speed_rad_s(self.config, action)
        if self.config.invert_direction:
            direction = MS42DC_CLOSE_DIR if direction == MS42DC_OPEN_DIR else MS42DC_OPEN_DIR
        return encode_ms42dc_frame(
            device_id=self.config.device_id,
            mode=self.config.motion_mode,
            direction=direction,
            subdivide=self.config.subdivide,
            angle_wire=int(round(angle * 10.0)),
            speed_wire=int(round(speed_rad_s * 10.0)),
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

    def _wait_for_arrival(
        self,
        *,
        require_fresh_motion: bool = True,
        timeout_s: float | None = None,
        maximum_arrival_angle_deg: float | None = None,
    ) -> MS42DCStatus | None:
        """Block until a stable, plausible 到位标志 sequence is observed.

        The controller can briefly return the previous motion's arrived flag
        immediately after a new command is written.  Accepting that stale flag
        made the arm lift while the jaws were still closing.  A reversal
        therefore has to expose at least one in-motion status.  In every case,
        completion requires consecutive arrived frames with near-zero speed
        and a stable angle.  Commands that request the already-commanded state
        may legitimately produce no motion at a hard stop, but only after the
        same multi-frame stability proof.
        """

        wait_s = (
            float(self.config.motion_timeout_s)
            if timeout_s is None
            else max(0.0, float(timeout_s))
        )
        deadline = time.monotonic() + wait_s
        saw_motion = False
        stale_warned = False
        angle_limit_warned = False
        stable_samples = 0
        previous_stable_angle: float | None = None
        while time.monotonic() < deadline:
            status = self.read_status()
            if status is None:
                if self.config.require_feedback is False:
                    self._log(
                        "[gripper:ms42dc] controller gave no feedback; "
                        "not waiting for arrival"
                    )
                    return None
                stable_samples = 0
                previous_stable_angle = None
            elif not status.arrived:
                saw_motion = True
                stable_samples = 0
                previous_stable_angle = None
            else:
                motion_evidence_ok = saw_motion or not require_fresh_motion
                speed_ok = (
                    abs(status.speed_rad_s)
                    <= self.config.arrival_speed_tolerance_rad_s
                )
                angle_limit_ok = (
                    maximum_arrival_angle_deg is None
                    or status.angle_deg <= float(maximum_arrival_angle_deg)
                )
                angle_ok = (
                    previous_stable_angle is None
                    or abs(status.angle_deg - previous_stable_angle)
                    <= self.config.arrival_angle_tolerance_deg
                )
                if not motion_evidence_ok:
                    stable_samples = 0
                    previous_stable_angle = None
                    if not stale_warned:
                        self._log(
                            "[gripper:ms42dc] ignoring pre-command arrival status; "
                            "waiting for fresh motion"
                        )
                        stale_warned = True
                elif not speed_ok:
                    stable_samples = 0
                    previous_stable_angle = None
                elif not angle_limit_ok:
                    stable_samples = 0
                    previous_stable_angle = None
                    if not angle_limit_warned:
                        self._log(
                            "[gripper:ms42dc] arrived at angle="
                            f"{status.angle_deg:.1f} deg, above required open "
                            f"hard stop <= {float(maximum_arrival_angle_deg):.1f} "
                            "deg; waiting for verified release"
                        )
                        angle_limit_warned = True
                elif not angle_ok:
                    # Start a new stable window at this frame rather than
                    # combining it with the preceding, inconsistent reading.
                    stable_samples = 1
                    previous_stable_angle = status.angle_deg
                else:
                    stable_samples += 1
                    previous_stable_angle = status.angle_deg
                    if stable_samples >= self.config.arrival_stable_samples:
                        self._log(
                            "[gripper:ms42dc] stable arrival after "
                            f"{stable_samples} samples, angle={status.angle_deg:.1f} "
                            f"deg, speed={status.speed_rad_s:.2f} rad/s"
                        )
                        return status
            time.sleep(0.05)
        self._log(
            "[gripper:ms42dc] motion did not report a stable arrival before timeout"
        )
        return None

    def _move(self, direction: int, state: GripperState, name: str) -> None:
        previous_state = self._state
        frame = self._motion_frame(direction)
        self._write_frame(frame)
        self._state = GripperState.MOVING
        self.events.append(name)
        self._log(f"[gripper:ms42dc] {name}: {frame.hex(' ')}")
        # Reversals and commands from UNKNOWN need fresh in-motion evidence so
        # a stale arrived=True reply cannot race the command.  Repeating the
        # already-confirmed state may be a legitimate no-motion hard stop, but
        # it still has to satisfy the multi-frame stable-plateau check.
        require_open_hard_stop = (
            state == GripperState.OPEN
            and self.config.require_open_hard_stop_evidence
        )
        status = self._wait_for_arrival(
            require_fresh_motion=previous_state != state,
            maximum_arrival_angle_deg=(
                self.config.open_hard_stop_max_angle_deg
                if require_open_hard_stop
                else None
            ),
        )
        if self.config.require_feedback and (status is None or not status.arrived):
            self._state = GripperState.FAULT
            if require_open_hard_stop:
                raise RuntimeError(
                    "MS42DC did not verify the open hard stop during release; "
                    "the object may still be clamped, so retreat is blocked"
                )
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
        if self._state == GripperState.UNKNOWN:
            # At process startup the controller state is unknown.  If the jaws
            # are already parked at the calibrated open hard stop, an open
            # command legitimately produces no fresh in-motion status.  Probe
            # read-only first so that an already-open gripper does not block
            # the task until motion_timeout_s expires.  Do not apply this
            # exception to partial openings: those still send the command and
            # must prove fresh motion.
            status = self._wait_for_arrival(
                require_fresh_motion=False,
                timeout_s=min(1.0, float(self.config.motion_timeout_s)),
            )
            if (
                status is not None
                and status.arrived
                and status.angle_deg <= self.config.open_hard_stop_max_angle_deg
            ):
                self._state = GripperState.OPEN
                self._log(
                    "[gripper:ms42dc] already at verified open hard stop "
                    f"angle={status.angle_deg:.1f} deg; no motion command sent"
                )
                time.sleep(self.config.settle_time_s)
                return
        self._move(MS42DC_OPEN_DIR, GripperState.OPEN, "open")

    def close(self) -> None:
        self.last_close_evidence = None
        try:
            self._move(MS42DC_CLOSE_DIR, GripperState.CLOSED, "close")
        except Exception as exc:
            self.last_close_evidence = _rejected_close_evidence(
                width_m=-1.0,
                minimum_travel_deg=0.0,
                start_angle_deg=0.0,
                end_angle_deg=0.0,
                arrived=False,
                contact_platform=False,
                stable_samples=0,
                reason=f"MS42DC legacy close failed without evidence: {exc}",
            )
            self._state = GripperState.FAULT
            raise

    def close_for_width(self, width_m: float) -> None:
        """Close and prove a fresh stable contact with object-specific travel."""

        width_m = float(width_m)
        minimum_travel_deg = float("nan")
        start_angle_deg = 0.0
        self.last_close_evidence = None
        try:
            if self._state != GripperState.OPEN:
                raise RuntimeError(
                    "MS42DC width-aware close requires the gripper to be OPEN"
                )
            minimum_travel_deg = _minimum_close_travel_deg(self.config, width_m)
            start_status = self._wait_for_arrival(require_fresh_motion=False)
            if start_status is None or not start_status.arrived:
                raise RuntimeError(
                    "MS42DC did not provide a stable open-position status "
                    "before close"
                )
            start_angle_deg = start_status.angle_deg

            frame = self._motion_frame(MS42DC_CLOSE_DIR)
            self._write_frame(frame)
            self._state = GripperState.MOVING
            self.events.append("close")
            self._log(
                f"[gripper:ms42dc] close_for_width width={width_m:.6f} m: "
                f"{frame.hex(' ')}"
            )
            evidence = _sample_ms42dc_close_evidence(
                self.config,
                backend_name="MS42DC",
                width_m=width_m,
                minimum_travel_deg=minimum_travel_deg,
                start_angle_deg=start_angle_deg,
                read_status=self.read_status,
                log=self._log,
            )
            self.last_close_evidence = evidence
            if not evidence.accepted:
                raise RuntimeError(evidence.reason)
            self._state = GripperState.CLOSED
            time.sleep(self.config.settle_time_s)
        except Exception as exc:
            if self.last_close_evidence is None:
                end_angle_deg = (
                    self.last_status.angle_deg
                    if self.last_status is not None
                    else start_angle_deg
                )
                self.last_close_evidence = _rejected_close_evidence(
                    width_m=width_m,
                    minimum_travel_deg=minimum_travel_deg,
                    start_angle_deg=start_angle_deg,
                    end_angle_deg=end_angle_deg,
                    arrived=(
                        self.last_status.arrived
                        if self.last_status is not None
                        else False
                    ),
                    contact_platform=False,
                    stable_samples=0,
                    reason=f"MS42DC width-aware close failed: {exc}",
                )
            self._state = GripperState.FAULT
            raise

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
        self.last_close_evidence: MS42DCCloseEvidence | None = None
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
        timeout = max(0.0, float(self.config.ros_connect_timeout_s))
        getter = getattr(pub, "get_subscription_count", None)
        if callable(getter):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if int(getter()) >= 1:
                    self._control_ready = True
                    return
                time.sleep(0.05)
            matched = False
        else:
            waiter = getattr(pub, "wait_for_matching_subscriptions", None)
            matched = bool(waiter(timeout)) if callable(waiter) else True
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
        require_fresh_motion: bool = False,
    ) -> MS42DCStatus | None:
        """Request feedback and optionally wait for stable arrival.

        A single fresh ``/motor_state`` arrival flag is not enough.  When
        ``wait_for_arrival`` is requested, consecutive frames must also have
        near-zero speed and a stable angle.  A reversal can additionally set
        ``require_fresh_motion`` so the pre-command hard-stop arrival cannot be
        mistaken for completion.
        """

        timeout = (
            self.config.status_timeout_s if timeout_s is None else float(timeout_s)
        )
        deadline = time.monotonic() + max(0.0, timeout)
        next_query = 0.0
        before_sequence = self._status_sequence
        saw_motion = False
        stale_warned = False
        stable_samples = 0
        previous_stable_angle: float | None = None
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
                if not wait_for_arrival:
                    return status
                if not status.arrived:
                    saw_motion = True
                    stable_samples = 0
                    previous_stable_angle = None
                else:
                    motion_evidence_ok = saw_motion or not require_fresh_motion
                    speed_ok = (
                        abs(status.speed_rad_s)
                        <= self.config.arrival_speed_tolerance_rad_s
                    )
                    angle_ok = (
                        previous_stable_angle is None
                        or abs(status.angle_deg - previous_stable_angle)
                        <= self.config.arrival_angle_tolerance_deg
                    )
                    if not motion_evidence_ok:
                        stable_samples = 0
                        previous_stable_angle = None
                        if not stale_warned:
                            self._log(
                                "[gripper:ms42dc_ros] ignoring pre-command arrival "
                                "status; waiting for fresh motion"
                            )
                            stale_warned = True
                    elif not speed_ok:
                        stable_samples = 0
                        previous_stable_angle = None
                    elif not angle_ok:
                        stable_samples = 1
                        previous_stable_angle = status.angle_deg
                    else:
                        stable_samples += 1
                        previous_stable_angle = status.angle_deg
                        if stable_samples >= self.config.arrival_stable_samples:
                            self._log(
                                "[gripper:ms42dc_ros] stable arrival after "
                                f"{stable_samples} samples, "
                                f"angle={status.angle_deg:.1f} deg, "
                                f"speed={status.speed_rad_s:.2f} rad/s"
                            )
                            return status
        self._log(
            "[gripper:ms42dc_ros] no stable arrival on /motor_state before timeout"
        )
        return None

    def _move(self, action: str, state: GripperState) -> None:
        previous_state = self._state
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
                require_fresh_motion=previous_state != state,
            )
            if status is None or not status.arrived:
                self._state = GripperState.FAULT
                raise RuntimeError(
                    "MS42DC did not report a stable 到位标志 sequence on "
                    "/motor_state; "
                    "check power, /dev/motor_serial and the USB DIP switch"
                )
        self._state = state
        time.sleep(self.config.settle_time_s)

    def open(self) -> None:
        self._move("open", GripperState.OPEN)

    def close(self) -> None:
        self.last_close_evidence = None
        try:
            self._move("close", GripperState.CLOSED)
        except Exception as exc:
            self.last_close_evidence = _rejected_close_evidence(
                width_m=-1.0,
                minimum_travel_deg=0.0,
                start_angle_deg=0.0,
                end_angle_deg=0.0,
                arrived=False,
                contact_platform=False,
                stable_samples=0,
                reason=(
                    "MS42DC ROS legacy close failed without evidence: "
                    f"{exc}"
                ),
            )
            self._state = GripperState.FAULT
            raise

    def close_for_width(self, width_m: float) -> None:
        """Close via ROS and independently prove fresh stable contact."""

        width_m = float(width_m)
        minimum_travel_deg = float("nan")
        start_angle_deg = 0.0
        self.last_close_evidence = None
        try:
            if self._state != GripperState.OPEN:
                raise RuntimeError(
                    "MS42DC ROS width-aware close requires the gripper to be OPEN"
                )
            minimum_travel_deg = _minimum_close_travel_deg(self.config, width_m)
            start_status = self.read_status(
                timeout_s=self.config.motion_timeout_s,
                wait_for_arrival=True,
                require_fresh_motion=False,
            )
            if start_status is None or not start_status.arrived:
                raise RuntimeError(
                    "MS42DC ROS did not provide a stable open-position status "
                    "before close"
                )
            start_angle_deg = start_status.angle_deg

            fields = ms42dc_ros_motion_fields(self.config, "close")
            self._wait_for_control_subscriber()
            self._state = GripperState.MOVING
            self._publish_fields(fields)
            self.events.append("close")
            self._log(
                f"[gripper:ms42dc_ros] close_for_width width={width_m:.6f} m: "
                f"dir={fields['dir']} angle_wire={fields['angle']} "
                f"speed_wire={fields['speed']}"
            )
            evidence = _sample_ms42dc_close_evidence(
                self.config,
                backend_name="MS42DC ROS",
                width_m=width_m,
                minimum_travel_deg=minimum_travel_deg,
                start_angle_deg=start_angle_deg,
                read_status=lambda: self.read_status(
                    timeout_s=self.config.status_timeout_s
                ),
                log=self._log,
            )
            self.last_close_evidence = evidence
            if not evidence.accepted:
                raise RuntimeError(evidence.reason)
            self._state = GripperState.CLOSED
            time.sleep(self.config.settle_time_s)
        except Exception as exc:
            if self.last_close_evidence is None:
                end_angle_deg = (
                    self.last_status.angle_deg
                    if self.last_status is not None
                    else start_angle_deg
                )
                self.last_close_evidence = _rejected_close_evidence(
                    width_m=width_m,
                    minimum_travel_deg=minimum_travel_deg,
                    start_angle_deg=start_angle_deg,
                    end_angle_deg=end_angle_deg,
                    arrived=(
                        self.last_status.arrived
                        if self.last_status is not None
                        else False
                    ),
                    contact_platform=False,
                    stable_samples=0,
                    reason=f"MS42DC ROS width-aware close failed: {exc}",
                )
            self._state = GripperState.FAULT
            raise

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
    """Accept a hold only when the width-aware close recorded proof.

    No independent arrival resampling is performed here.  The close command's
    evidence is the single source of truth and includes fresh motion, a stable
    near-zero-speed plateau and CAD-derived object-specific travel.  A fixed
    final angle is not used in width-aware mode.
    """

    def __init__(self, gripper, config: GripperConfig, log=None) -> None:
        self._gripper = gripper
        self._config = config
        self._log = log or (lambda message: None)

    def verify(self, detection: Detection, plan: MotionPlan) -> tuple[bool, str]:
        evidence = getattr(self._gripper, "last_close_evidence", None)
        if evidence is None:
            return (
                False,
                "no width-aware close evidence is available after closing",
            )
        if not bool(getattr(evidence, "accepted", False)):
            reason = str(
                getattr(evidence, "reason", "close evidence was rejected")
            )
            return False, f"width-aware close evidence rejected: {reason}"
        message = (
            "width-aware close evidence accepted at "
            f"{float(evidence.end_angle_deg):.1f} deg with "
            f"{float(evidence.travel_deg):.1f} deg travel"
        )
        self._log(f"[gripper:ms42dc] {message}")
        return True, message


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
