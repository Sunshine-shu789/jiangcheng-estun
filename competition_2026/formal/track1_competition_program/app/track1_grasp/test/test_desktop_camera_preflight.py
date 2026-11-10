from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_cd_launcher_releases_known_camera_preview_before_motion():
    runner = ROOT / "bin/track1_task_runner"
    preflight = ROOT / "bin/track1_camera_preflight"
    source = runner.read_text(encoding="utf-8")
    preflight_source = preflight.read_text(encoding="utf-8")

    assert preflight.is_file()
    assert preflight.stat().st_mode & 0o111
    assert '"$ROOT/bin/track1_camera_preflight" --release' in source
    assert "D405 相机预检失败" in source
    assert source.index("if ! stop_old_camera_programs; then") < source.index(
        "if ! ensure_motor_node; then"
    )
    assert "track1_capture_web/live_camera_preview.py" in preflight_source
    assert "track1_grasp.nodes.search_approach_node" in preflight_source
    assert "RealSense Viewer" in preflight_source
    # Shells whose -c script contains a pattern must never be mistaken for the
    # Python camera process and terminated by the preflight itself.
    assert "*/python3.*" in preflight_source
    assert "readlink -f -- \"/proc/$pid/exe\"" in preflight_source


def test_desktop_launcher_does_not_report_task_exit_as_terminal_failure():
    launcher = ROOT / "bin/track1_desktop_launch"
    source = launcher.read_text(encoding="utf-8")

    assert "/usr/bin/gnome-terminal --wait" not in source
    assert "gnome-terminal \\\n" in source
    assert "无法打开任务终端" in source


def test_capture_controller_recognizes_either_data_disk_path_alias():
    controller = ROOT / "bin/collect_track1_dataset"
    source = controller.read_text(encoding="utf-8")

    assert 'SERVICE_MATCH="track1_capture_web/live_camera_preview.py"' in source
    assert 'grep -F -- "$SERVICE_MATCH"' in source
    assert 'grep -F -- "$SERVICE"' not in source
