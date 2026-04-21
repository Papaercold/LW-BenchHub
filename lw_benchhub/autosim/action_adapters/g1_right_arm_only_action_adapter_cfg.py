from isaaclab.utils import configclass

from autosim import ActionAdapterCfg

from .g1_right_arm_only_action_adapter import G1RightArmOnlyActionAdapter


@configclass
class G1RightArmOnlyActionAdapterCfg(ActionAdapterCfg):
    """Configuration for the G1 right-arm-only action adapter (legs in stance)."""

    class_type: type = G1RightArmOnlyActionAdapter

    finger_close_angles: tuple = (
         1.2,  1.4,   # right index_0, index_1
         1.2,  1.4,   # right middle_0, middle_1
         0.5, -0.5, -1.2,  # right thumb_0, thumb_1, thumb_2
        -1.2, -1.4,   # left index_0, index_1
        -1.2, -1.4,   # left middle_0, middle_1
         0.5,  0.5,  1.2,  # left thumb_0, thumb_1, thumb_2
    )
    finger_open_angles: tuple = (0.0,) * 14
