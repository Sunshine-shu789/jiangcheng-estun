"""Track 1 task A shared-runner entry point."""

from track1_grasp.launch_helpers import build_task_launch


def generate_launch_description():
    return build_task_launch(mode="A", default_max_picks=1)
