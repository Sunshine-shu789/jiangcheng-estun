from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DESKTOP = ROOT / "任务C_无序抓取.desktop"
FORMAL_ROOT = "/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序"


def _desktop_entries() -> dict[str, str]:
    entries: dict[str, str] = {}
    for raw_line in DESKTOP.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "[")):
            continue
        key, value = line.split("=", 1)
        entries[key] = value
    return entries


def test_task_c_desktop_entry_directly_launches_formal_task_c() -> None:
    entries = _desktop_entries()
    expected_runner = f"{FORMAL_ROOT}/bin/run_task_c"

    assert entries["Exec"] == expected_runner
    assert entries["TryExec"] == expected_runner
    assert entries["Path"] == FORMAL_ROOT
    assert entries["Terminal"] == "true"
    assert "track1_desktop_launch" not in DESKTOP.read_text(encoding="utf-8")
    assert "/home/sunshine" not in DESKTOP.read_text(encoding="utf-8")
    assert (Path(FORMAL_ROOT) / "bin" / "run_task_c").is_file()
