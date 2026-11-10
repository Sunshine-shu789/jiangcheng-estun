from pathlib import Path

from track1_grasp.config import load_config
from track1_grasp.factory import build_pick_pipeline


ROOT = Path(__file__).resolve().parents[3]
BOX_RELATIVE_CONFIG = ROOT / "config" / "track1_task_b_box_relative_mock.yaml"
EXPECTED_POINTS = (
    "corner_nw",
    "corner_ne",
    "corner_se",
    "corner_sw",
    "center",
)
EXPECTED_BOX_POSES = (
    "fitted box centre=(0.0000, 0.0500, 0.0200) yaw=0.00 deg",
    "fitted box centre=(-0.0300, 0.0600, 0.0200) yaw=20.00 deg",
    "fitted box centre=(0.0400, 0.0700, 0.0200) yaw=-15.00 deg",
    "fitted box centre=(0.0200, 0.0300, 0.0200) yaw=35.00 deg",
    "fitted box centre=(-0.0200, 0.0800, 0.0200) yaw=-30.00 deg",
)


def test_formal_box_relative_mock_runs_five_dynamic_rounds():
    config = load_config(BOX_RELATIVE_CONFIG)
    config.validate()

    assert config.motion.dry_run is True
    assert config.motion.backend == "mock"
    assert config.gripper.backend == "mock"
    assert config.task_b.rounds == 5
    assert config.task_b.place_poses == ()
    assert config.adaptive_inverse.arbitrary_placement is False
    assert config.adaptive_inverse.box_relative_five_point.enabled is True
    assert Path(config.vision.scenario_file).is_file()

    logs: list[str] = []
    pipeline = build_pick_pipeline(config, log=logs.append)
    try:
        report = pipeline.run()
    finally:
        pipeline.close()

    assert (report.requested_picks, report.attempts, report.successes) == (5, 5, 5)
    assert [result.place_name for result in report.results] == [
        f"box_relative_round_{index}" for index in range(1, 6)
    ]
    assert all(result.success for result in report.results)
    assert pipeline._task_b_completed_xy == []

    box_pose_logs = [
        next(
            line
            for line in logs
            if f"box_relative_round_{index}" in line
            and "fitted box centre=" in line
        )
        for index in range(1, 6)
    ]
    for line, expected in zip(box_pose_logs, EXPECTED_BOX_POSES):
        assert expected in line

    assert sum("all five routes prechecked" in line for line in logs) == 5
    plan_logs = "\n".join(logs)
    for point in EXPECTED_POINTS:
        assert f"item match '{point}'" in plan_logs
    assert (
        "all five fixed box-local points completed from current-frame box poses"
        in plan_logs
    )

def test_box_relative_mock_is_wired_into_dynamic_launcher_and_desktop():
    launcher = ROOT / "bin" / "run_task_b_arbitrary"
    runner = ROOT / "bin" / "track1_task_runner"
    desktop = ROOT / "任务B_任意摆放抓取.desktop"
    node = ROOT / "app/track1_grasp/track1_grasp/nodes/track1_node.py"

    launcher_text = launcher.read_text(encoding="utf-8")
    runner_text = runner.read_text(encoding="utf-8")
    desktop_text = desktop.read_text(encoding="utf-8")
    node_text = node.read_text(encoding="utf-8")

    assert launcher.is_file()
    assert "track1_task_b_box_relative_mock.yaml" in launcher_text
    assert "TRACK1_CONFIG:-" in launcher_text
    assert "uses_box_relative_five_point()" in runner_text
    assert "uses_dynamic_task_b()" in runner_text
    assert "dynamic_task_b = bool(config.adaptive_inverse.enabled)" in runner_text
    assert "任务B 箱体动态五点抓取" in runner_text
    assert "箱体动态五点抓取离线试运行" in runner_text
    assert "box_relative_five_point.enabled" in node_text
    assert "重新拟合箱体位姿" in node_text
    assert "run_task_b_arbitrary" in desktop_text
    assert "任务B 箱体动态五点抓取（离线试运行）" in desktop_text
