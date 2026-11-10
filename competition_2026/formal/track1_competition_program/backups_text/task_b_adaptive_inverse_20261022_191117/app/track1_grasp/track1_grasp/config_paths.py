"""Resolve packaged configuration files from source trees and ros2 installs."""

from __future__ import annotations

from pathlib import Path


def package_config_path(filename: str) -> Path:
    """Return a package config path without assuming the current working dir.

    In a source checkout the file lives beside the Python package.  After an
    ``ament_python`` install the Python package is copied into site-packages,
    while YAML files are placed in the package share directory.
    """

    source_path = Path(__file__).resolve().parents[1] / "config" / filename
    if source_path.is_file():
        return source_path

    try:
        from ament_index_python.packages import get_package_share_directory

        installed_path = (
            Path(get_package_share_directory("track1_grasp")) / "config" / filename
        )
        if installed_path.is_file():
            return installed_path
    except Exception:
        # Keep the source fallback so callers get a useful file-not-found error
        # in environments where ROS 2 is not sourced.
        pass
    return source_path
