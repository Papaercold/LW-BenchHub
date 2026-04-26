from isaaclab.utils import configclass

from autosim import ActionAdapterCfg

from .g1_action_adapter import G1ActionAdapter


@configclass
class G1ActionAdapterCfg(ActionAdapterCfg):
    """Configuration for the G1 action adapter."""

    class_type: type = G1ActionAdapter

    base_x_joint_name: str = "base_x_joint"
    base_y_joint_name: str = "base_y_joint"
    base_yaw_joint_name: str = "base_yaw_joint"

    finger_close_angles: tuple = (
        # right hand stays open (left arm is the grasping arm)
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        # left hand closes
        -1.9, -2.0,   # left index_0, index_1
        -1.9, -2.0,   # left middle_0, middle_1
         0.8,  0.8,  1.8,  # left thumb_0, thumb_1, thumb_2
    )
    """Per-joint finger angles (rad) when gripper is closed."""
    finger_open_angles: tuple = (0.0,) * 14
    """Per-joint finger angles (rad) when gripper is open."""

    squat_settle_steps: int = 40
    """Steps to squat before planning an arm skill.
    10 (loco→squat transition) + 110 × 0.0016 drop = squat_cmd[0] ≈ 0.574 (moderate crouch).
    Brings shoulder from ~1.29m down to ~1.16m so arm can reach target at Z=0.72m."""

    pos_search_nudge_m: float = 0.04
    """Metres per nudge step toward the oven after squatting."""

    pos_search_nudge_count: int = 0
    """Number of nudge steps (total forward shift = nudge_m × count).
    Set to 0 to disable nudging. solve_ik_batch is NOT used (CUDA graph conflict)."""

    pos_search_settle_steps: int = 20
    """Squat-mode settle steps after the forward nudge."""
