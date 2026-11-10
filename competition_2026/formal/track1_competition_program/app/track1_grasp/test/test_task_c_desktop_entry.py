from pathlib import Path
import shlex

import pytest


ROOT = Path(__file__).resolve().parents[3]
FORMAL_ROOT = "/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序"
SHARED_LAUNCHER = f"{FORMAL_ROOT}/bin/run_task_cd"
USER_DESKTOP = Path("/home/sunshine/Desktop/赛道一比赛程序")
TASKS = (
    ("C", "任务C_无序抓取.desktop"),
    ("D", "任务D_未知物体抓取.desktop"),
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


@pytest.mark.parametrize(("task", "entry_name"), TASKS)
def test_formal_cd_desktop_entries_share_one_launcher(
    task: str,
    entry_name: str,
) -> None:
    desktop = ROOT / entry_name
    entries = _desktop_entries(desktop)

    assert entries["Exec"] == f"{SHARED_LAUNCHER} {task}"
    assert entries["TryExec"] == SHARED_LAUNCHER
    assert entries["Path"] == FORMAL_ROOT
    assert entries["Terminal"] == "true"

    text = desktop.read_text(encoding="utf-8")
    assert "track1_desktop_launch" not in text
    assert "/home/sunshine" not in text
    assert "新加卷1" not in text
    assert task in entries["Name"]
    assert Path(SHARED_LAUNCHER).is_file()
    assert Path(SHARED_LAUNCHER).stat().st_mode & 0o111


def test_cd_desktop_entries_have_no_second_executable_path() -> None:
    executables = set()
    for _task, entry_name in TASKS:
        entries = _desktop_entries(ROOT / entry_name)
        executables.add(shlex.split(entries["Exec"])[0])
        assert entries["TryExec"] == SHARED_LAUNCHER
    assert executables == {SHARED_LAUNCHER}


@pytest.mark.parametrize(("task", "entry_name"), TASKS)
def test_user_desktop_copy_matches_formal_desktop_entry(
    task: str,
    entry_name: str,
) -> None:
    formal_desktop = ROOT / entry_name
    user_desktop = USER_DESKTOP / entry_name

    assert user_desktop.is_file()
    assert user_desktop.read_bytes() == formal_desktop.read_bytes()
    entries = _desktop_entries(user_desktop)
    assert entries["Exec"] == f"{SHARED_LAUNCHER} {task}"
    assert entries["TryExec"] == SHARED_LAUNCHER
    assert entries["Path"] == FORMAL_ROOT
    assert entries["Terminal"] == "true"


@pytest.mark.parametrize(("task", "wrapper_name"), (("C", "run_task_c"), ("D", "run_task_d")))
def test_compatibility_wrappers_delegate_to_shared_launcher(
    task: str,
    wrapper_name: str,
) -> None:
    wrapper = (ROOT / "bin" / wrapper_name).read_text(encoding="utf-8")
    assert 'exec "$ROOT/bin/run_task_cd" ' + task in wrapper


@pytest.mark.parametrize(
    ("task", "entry_name", "config_name"),
    (
        (
            "C",
            "test_task_c_height_single_live",
            "track1_task_c_v11_official_20261102.yaml",
        ),
        (
            "D",
            "test_task_d_height_single_live",
            "track1_task_d_v11_official_20261102.yaml",
        ),
    ),
)
def test_height_single_live_entries_use_formal_config_and_hard_stop(
    task: str,
    entry_name: str,
    config_name: str,
) -> None:
    entry = ROOT / "bin" / entry_name
    text = entry.read_text(encoding="utf-8")

    assert entry.stat().st_mode & 0o111
    assert f'config/{config_name}' in text
    assert (
        f'"$ROOT/bin/track1_task_runner" {task} '
        "--single-live-acceptance --no-pause"
    ) in text
    assert "TRACK1_CONFIG=\"$CONFIG\"" in text


def test_redundant_desktop_root_task_c_alias_is_absent() -> None:
    redundant = Path("/home/sunshine/Desktop/任务C_正式抓取.desktop")
    assert not redundant.exists()
    assert not redundant.is_symlink()
