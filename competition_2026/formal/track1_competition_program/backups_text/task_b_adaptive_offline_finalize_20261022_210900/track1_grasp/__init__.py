"""Track 1 (赛道一) vision + arm + gripper orchestration framework.

Covers the whole grasping challenge:

* task A - detection / annotation / IoU demo (vision only)
* task B - one object picked and placed at five bin positions
* task C/D/E - unordered and ordered bin clearing
"""

from .annotator import Annotator, iou, match_frame, score_frame
from .config import Track1Config, apply_mode, load_config
from .domain import PickStatus, TaskMode, TaskPhase
from .pipeline import PickPipeline
from .vision_pipeline import VisionDemoPipeline

#: Historical alias kept for code written against the C/D/E version.
CDEConfig = Track1Config
PickPhase = TaskPhase

__all__ = [
    "Annotator",
    "CDEConfig",
    "PickPipeline",
    "PickPhase",
    "PickStatus",
    "TaskMode",
    "TaskPhase",
    "Track1Config",
    "VisionDemoPipeline",
    "apply_mode",
    "iou",
    "load_config",
    "match_frame",
    "score_frame",
]
