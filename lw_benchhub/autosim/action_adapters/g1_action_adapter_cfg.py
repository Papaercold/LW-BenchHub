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

    finger_close_angles: tuple = (-1.2, -1.2, -1.2, -1.2, -1.0, -1.0, -1.0, -1.2, -1.2, -1.2, -1.2, -1.0, -1.0, -1.0)
    """Per-joint finger angles (rad) when gripper is closed."""
    finger_open_angles: tuple = (0.0,) * 14
    """Per-joint finger angles (rad) when gripper is open."""
