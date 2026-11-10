from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
FORMAL_ROOT = "/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序"
USER_DESKTOP = Path("/home/sunshine/Desktop/赛道一比赛程序")
TASKS = (
    ("C", "任务C_无序抓取.desktop", "run_task_c"),
    ("D", "任务D_未知物体抓取.desktop", "run_task_d"),
)


def _desktop_entries(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "[")):
            continue
        key, value = line.split("=", 1)
        entries[key] = value
    return entries


@pytest.mark.parametrize(("task", "entry_name", "runner_name"), TASKS)
def test_formal_desktop_entry_directly_launches_task(
    task: str,
    entry_name: str,
    runner_name: str,
) -> None:
    desktop = ROOT / entry_name
    entries = _desktop_entries(desktop)
    expected_runner = f"{FORMAL_ROOT}/bin/{runner_name}"

    assert entries["Exec"] == expected_runner
    assert entries["TryExec"] == expected_runner
    assert entries["Path"] == FORMAL_ROOT
    assert entries["Terminal"] == "true"

    text = desktop.read_text(encoding="utf-8")
    assert "track1_desktop_launch" not in text
    assert "/home/sunshine" not in text
    assert "新加卷1" not in text
    assert task in entries["Name"]
    assert Path(expected_runner).is_file()
    assert Path(expected_runner).stat().st_mode & 0o111


@pytest.mark.parametrize(("task", "entry_name", "runner_name"), TASKS)
def test_user_desktop_copy_matches_formal_desktop_entry(
    task: str,
    entry_name: str,
    runner_name: str,
) -> None:
    formal_desktop = ROOT / entry_name
    user_desktop = USER_DESKTOP / entry_name

    assert user_desktop.is_file()
    assert user_desktop.read_bytes() == formal_desktop.read_bytes()
    entries = _desktop_entries(user_desktop)
    assert entries["Exec"] == f"{FORMAL_ROOT}/bin/{runner_name}"
    assert entries["TryExec"] == f"{FORMAL_ROOT}/bin/{runner_name}"
    assert entries["Path"] == FORMAL_ROOT
    assert entries["Terminal"] == "true"
