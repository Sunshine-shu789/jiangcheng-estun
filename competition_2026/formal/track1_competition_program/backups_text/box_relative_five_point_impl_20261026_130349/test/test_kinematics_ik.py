import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from track1_grasp.kinematics import (
    JOINT_LOW,
    controller_fk,
    controller_jacobian,
    rotation_angle_error,
    rotation_matrix_to_vector,
    rpy_to_matrix,
    solve_orientation_ik,
    solve_pose_ik,
    solve_pose_ik_multistart,
    validate_rotation_matrix,
    weighted_jacobian_singular_values,
)


BASE_JOINTS = np.radians(
    [94.191, 71.955, -16.289, -34.463, 88.164, 0.248]
)
ROBOTINFO_CAPTURE_FILE = (
    Path(__file__).resolve().parents[3]
    / "config"
    / "task_b_point_captures_20261020.yaml"
)


def _target_from_joints(joints):
    transform = controller_fk(joints)
    return transform[:3, 3].copy(), transform[:3, :3].copy()


def _recorded_robotinfo_captures():
    payload = yaml.safe_load(ROBOTINFO_CAPTURE_FILE.read_text(encoding="utf-8"))
    return payload["captures"]


def test_controller_fk_matches_recorded_robotinfo_postures():
    captures = _recorded_robotinfo_captures()
    expected_names = {
        "pick_1_observation",
        "pick_2_observation",
        "pick_3_observation",
        "pick_4_observation",
        "pick_5_observation",
    }
    assert set(captures) == expected_names

    failures = []
    for name in sorted(captures):
        capture = captures[name]
        joints = np.asarray(capture["joint_positions_rad"], dtype=float)
        observed_position = np.asarray(capture["tcp_xyz_m"], dtype=float)
        observed_rotation = rpy_to_matrix(capture["tcp_rpy_rad"])
        calculated = controller_fk(joints)

        position_error_m = float(
            np.linalg.norm(calculated[:3, 3] - observed_position)
        )
        orientation_error_rad = rotation_angle_error(
            calculated[:3, :3],
            observed_rotation,
        )
        if position_error_m >= 1e-6 or orientation_error_rad >= 1e-6:
            failures.append(
                f"{name}: position={position_error_m * 1000:.9f} mm, "
                f"orientation={orientation_error_rad * 1000:.9f} mrad"
            )

    assert not failures, "recorded /RobotInfo FK mismatch:\n" + "\n".join(failures)


def test_controller_jacobian_matches_central_difference():
    joints = np.radians([20.0, 35.0, -25.0, 40.0, 55.0, -30.0])
    analytic = controller_jacobian(joints)
    finite_difference = np.zeros_like(analytic)
    epsilon = 1e-6

    for index in range(6):
        plus = joints.copy()
        minus = joints.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        transform_plus = controller_fk(plus)
        transform_minus = controller_fk(minus)
        finite_difference[:3, index] = (
            transform_plus[:3, 3] - transform_minus[:3, 3]
        ) / (2.0 * epsilon)
        relative_rotation = (
            transform_plus[:3, :3] @ transform_minus[:3, :3].T
        )
        finite_difference[3:, index] = (
            rotation_matrix_to_vector(relative_rotation) / (2.0 * epsilon)
        )

    assert np.max(np.abs(analytic - finite_difference)) < 1e-8


@pytest.mark.parametrize(
    ("matrix", "message"),
    (
        (np.diag([1.1, 1.0, 1.0]), "not orthonormal"),
        (np.diag([1.0, 1.0, -1.0]), "determinant \\+1"),
        (
            np.asarray(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, np.nan]]
            ),
            "non-finite",
        ),
        (np.eye(2), "must be 3x3"),
    ),
)
def test_ik_rejects_malformed_target_rotation(matrix, message):
    target_position, _target_rotation = _target_from_joints(BASE_JOINTS)

    with pytest.raises(ValueError, match=message):
        solve_pose_ik(BASE_JOINTS, target_position, matrix)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"damping": 0.0}, "damping"),
        ({"damping": math.nan}, "damping"),
        ({"max_iterations": 0}, "max_iterations"),
        ({"max_iterations": True}, "max_iterations"),
        ({"position_tolerance_m": 0.0}, "position tolerance"),
        ({"rotation_tolerance_rad": math.inf}, "rotation tolerance"),
        ({"joint_limit_margin_rad": -0.1}, "joint limit margin"),
    ),
)
def test_pose_ik_rejects_invalid_numerical_parameters(overrides, message):
    target_position, target_rotation = _target_from_joints(BASE_JOINTS)
    parameters = {
        "max_iterations": 10,
        "damping": 0.05,
        "position_tolerance_m": 2e-4,
        "rotation_tolerance_rad": 2e-4,
        "joint_limit_margin_rad": 0.0,
    }
    parameters.update(overrides)

    with pytest.raises(ValueError, match=message):
        solve_pose_ik(
            BASE_JOINTS,
            target_position,
            target_rotation,
            **parameters,
        )


def test_pose_ik_rejects_seed_inside_configured_limit_margin():
    margin = math.radians(2.0)
    seed = BASE_JOINTS.copy()
    seed[0] = JOINT_LOW[0] + 0.5 * margin
    target_position, target_rotation = _target_from_joints(BASE_JOINTS)

    with pytest.raises(ValueError, match="joint-limit margin"):
        solve_pose_ik(
            seed,
            target_position,
            target_rotation,
            joint_limit_margin_rad=margin,
        )


def test_multistart_rejects_invalid_fallback_and_seed_parameters():
    target_position, target_rotation = _target_from_joints(BASE_JOINTS)

    with pytest.raises(ValueError, match="allow_unbounded_fallback"):
        solve_pose_ik_multistart(
            BASE_JOINTS,
            target_position,
            target_rotation,
            allow_unbounded_fallback=1,
        )
    with pytest.raises(ValueError, match="accept_delta_rad"):
        solve_pose_ik_multistart(
            BASE_JOINTS,
            target_position,
            target_rotation,
            accept_delta_rad=-1.0,
        )
    with pytest.raises(ValueError, match="at least one damping"):
        solve_pose_ik_multistart(
            BASE_JOINTS,
            target_position,
            target_rotation,
            damping_candidates=(),
        )
    with pytest.raises(ValueError, match="seed offset 0"):
        solve_pose_ik_multistart(
            BASE_JOINTS,
            target_position,
            target_rotation,
            seed_offsets_rad=((0.0,) * 5,),
        )
    with pytest.raises(ValueError, match="non-finite"):
        solve_pose_ik_multistart(
            BASE_JOINTS,
            target_position,
            target_rotation,
            seed_offsets_rad=((0.0,) * 5 + (math.nan,),),
        )


def test_multistart_falls_back_to_offset_seed_and_rechecks_final_fk(monkeypatch):
    initial = BASE_JOINTS.copy()
    target_joints = initial + np.radians(
        [0.20, -0.15, 0.10, -0.12, 0.08, -0.05]
    )
    target_position, target_rotation = _target_from_joints(target_joints)
    calls = []

    def fake_single_start(seed, _position, _rotation, *, damping, **_kwargs):
        calls.append((np.asarray(seed, dtype=float).copy(), float(damping)))
        if np.allclose(seed, initial):
            raise RuntimeError("primary seed intended failure")
        return target_joints.copy()

    monkeypatch.setattr(
        "track1_grasp.kinematics.solve_pose_ik",
        fake_single_start,
    )

    solution = solve_pose_ik_multistart(
        initial,
        target_position,
        target_rotation,
        accept_delta_rad=math.radians(0.5),
        damping_candidates=(0.05, 0.02),
        seed_offsets_rad=(target_joints - initial,),
        return_diagnostics=True,
    )

    assert solution.seed_index == 1
    assert np.allclose(solution.seed_joints, target_joints)
    assert np.allclose(solution.initial_joints, initial)
    assert solution.position_error_m < 1e-9
    assert solution.orientation_error_rad < 1e-9
    assert solution.max_joint_delta_rad <= math.radians(0.5) + 1e-12
    assert len(calls) == 3


def test_multistart_enforces_step_bound_unless_refiner_explicitly_opts_out(
    monkeypatch,
):
    initial = BASE_JOINTS.copy()
    coarse_joints = initial + np.radians(
        [0.8, -0.7, 0.6, -0.5, 0.4, -0.3]
    )
    target_position, target_rotation = _target_from_joints(coarse_joints)

    def fake_single_start(_seed, _position, _rotation, **_kwargs):
        return coarse_joints.copy()

    monkeypatch.setattr(
        "track1_grasp.kinematics.solve_pose_ik",
        fake_single_start,
    )
    arguments = {
        "accept_delta_rad": math.radians(0.5),
        "damping_candidates": (0.05,),
        "seed_offsets_rad": (),
    }

    with pytest.raises(RuntimeError, match="joint-step bound"):
        solve_pose_ik_multistart(
            initial,
            target_position,
            target_rotation,
            **arguments,
        )

    coarse = solve_pose_ik_multistart(
        initial,
        target_position,
        target_rotation,
        allow_unbounded_fallback=True,
        return_diagnostics=True,
        **arguments,
    )
    assert coarse.max_joint_delta_rad > math.radians(0.5)
    assert coarse.position_error_m < 1e-9
    assert coarse.orientation_error_rad < 1e-9


def test_multistart_reports_a_candidate_that_reaches_joint_limit(monkeypatch):
    margin = math.radians(2.0)
    lower = JOINT_LOW[0] + margin
    initial = BASE_JOINTS.copy()
    initial[0] = lower + math.radians(0.20)
    target_joints = initial.copy()
    target_joints[0] = lower
    target_position, target_rotation = _target_from_joints(target_joints)

    monkeypatch.setattr(
        "track1_grasp.kinematics.solve_pose_ik",
        lambda _seed, _position, _rotation, **_kwargs: target_joints.copy(),
    )

    solution = solve_pose_ik_multistart(
        initial,
        target_position,
        target_rotation,
        accept_delta_rad=math.radians(0.5),
        damping_candidates=(0.05,),
        seed_offsets_rad=(),
        joint_limit_margin_rad=margin,
        return_diagnostics=True,
    )

    assert solution.at_joint_limit is True
    assert solution.joints[0] == pytest.approx(lower)


def test_orientation_ik_recomputes_strict_final_rotation_error():
    target_joints = BASE_JOINTS + np.radians(
        [0.10, -0.10, 0.05, 0.10, -0.10, 0.20]
    )
    target_rotation = controller_fk(target_joints)[:3, :3]

    solved = solve_orientation_ik(
        BASE_JOINTS,
        target_rotation,
        orientation_tolerance_rad=2e-4,
    )
    solved_rotation = controller_fk(solved)[:3, :3]

    assert rotation_angle_error(solved_rotation, target_rotation) <= 2e-4


def test_near_singular_candidate_is_rejected_even_when_pose_converges():
    singular_joints = np.radians([90.0, 0.0, 0.0, 0.0, 90.0, 0.0])
    transform = controller_fk(singular_joints)
    target_position = transform[:3, 3].copy()
    target_position[1] += 0.010
    checked_singular_values = []

    def require_safe_singular_value(candidate):
        singular_value = float(
            np.min(weighted_jacobian_singular_values(candidate))
        )
        checked_singular_values.append(singular_value)
        return singular_value >= 0.011

    with pytest.raises(RuntimeError, match="passed validation"):
        solve_pose_ik_multistart(
            singular_joints,
            target_position,
            transform[:3, :3],
            accept_delta_rad=math.radians(2.0),
            damping_candidates=(0.03,),
            seed_offsets_rad=(),
            candidate_validator=require_safe_singular_value,
            minimum_singular_value=0.011,
        )

    assert checked_singular_values
    assert min(checked_singular_values) < 0.011


def test_reachable_near_neighbor_pose_round_trips_with_strict_contract():
    rng = np.random.default_rng(20261025)
    accept_delta = math.radians(0.5)
    position_tolerance = 2e-4
    rotation_tolerance = 2e-4
    solutions = []

    for _ in range(120):
        seed = BASE_JOINTS + rng.uniform(
            -math.radians(0.10),
            math.radians(0.10),
            6,
        )
        target_joints = BASE_JOINTS + rng.uniform(
            -math.radians(0.38),
            math.radians(0.38),
            6,
        )
        target_position, target_rotation = _target_from_joints(target_joints)
        solution = solve_pose_ik_multistart(
            seed,
            target_position,
            target_rotation,
            accept_delta_rad=accept_delta,
            position_tolerance_m=position_tolerance,
            rotation_tolerance_rad=rotation_tolerance,
            return_diagnostics=True,
        )
        solutions.append(solution)

    assert len(solutions) == 120
    assert max(item.position_error_m for item in solutions) <= position_tolerance
    assert max(item.orientation_error_rad for item in solutions) <= rotation_tolerance
    assert max(item.max_joint_delta_rad for item in solutions) <= accept_delta + 1e-12
    assert min(item.minimum_jacobian_singular_value for item in solutions) > 0.011


def test_recorded_robotinfo_near_neighbors_round_trip_with_strict_contract():
    captures = _recorded_robotinfo_captures()
    rng = np.random.default_rng(20261025)
    accept_delta = math.radians(0.5)
    position_tolerance = 2e-4
    rotation_tolerance = 2e-4
    minimum_singular_value = 0.011
    solutions = []

    for name in sorted(captures):
        measured_joints = np.asarray(
            captures[name]["joint_positions_rad"],
            dtype=float,
        )
        baseline_singular = float(
            np.min(weighted_jacobian_singular_values(measured_joints))
        )
        assert baseline_singular >= minimum_singular_value, name

        for _ in range(20):
            seed = measured_joints + rng.uniform(
                -math.radians(0.10),
                math.radians(0.10),
                6,
            )
            target_joints = measured_joints + rng.uniform(
                -math.radians(0.38),
                math.radians(0.38),
                6,
            )
            target_position, target_rotation = _target_from_joints(target_joints)
            solution = solve_pose_ik_multistart(
                seed,
                target_position,
                target_rotation,
                accept_delta_rad=accept_delta,
                position_tolerance_m=position_tolerance,
                rotation_tolerance_rad=rotation_tolerance,
                minimum_singular_value=minimum_singular_value,
                return_diagnostics=True,
            )
            solutions.append(solution)

    assert len(solutions) == 20 * len(captures)
    assert max(item.position_error_m for item in solutions) <= position_tolerance
    assert max(item.orientation_error_rad for item in solutions) <= rotation_tolerance
    assert max(item.max_joint_delta_rad for item in solutions) <= accept_delta + 1e-12
    assert min(
        item.minimum_jacobian_singular_value for item in solutions
    ) >= minimum_singular_value


def test_multistart_is_deterministic_for_the_same_problem():
    seed = BASE_JOINTS + np.radians([0.05, -0.04, 0.03, -0.02, 0.01, 0.0])
    target_joints = BASE_JOINTS + np.radians(
        [0.30, -0.25, 0.20, -0.15, 0.10, -0.05]
    )
    target_position, target_rotation = _target_from_joints(target_joints)
    arguments = {
        "accept_delta_rad": math.radians(0.5),
        "return_diagnostics": True,
    }

    first = solve_pose_ik_multistart(
        seed,
        target_position,
        target_rotation,
        **arguments,
    )
    second = solve_pose_ik_multistart(
        seed,
        target_position,
        target_rotation,
        **arguments,
    )

    assert np.array_equal(first.joints, second.joints)
    assert first.seed_index == second.seed_index
    assert first.damping == second.damping
    assert first.iterations == second.iterations
