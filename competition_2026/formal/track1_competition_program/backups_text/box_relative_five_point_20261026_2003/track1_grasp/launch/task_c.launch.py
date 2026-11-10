"""Track 1 task C shared-runner entry point."""

from track1_grasp.launch_helpers import build_task_launch


def generate_launch_description():
    return build_task_launch(mode="C", default_max_picks=1)
