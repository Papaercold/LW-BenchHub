from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.envs import ManagerBasedEnv

from autosim import ActionAdapterBase
from autosim.core.types import SkillOutput

if TYPE_CHECKING:
    from .g1_action_adapter_cfg import G1ActionAdapterCfg


class G1ActionAdapter(ActionAdapterBase):
    """Action adapter for the Unitree G1 robot (leg-locomotion autosim variant).

    Action vector layout:
        [0:4]   base locomotion command [vx, vy, vyaw, mode]
                mode: 0=loco, 1=squat/stance
        [4:11]  right arm joints (7 DoF, absolute position)
        [11:18] left  arm joints (7 DoF, absolute position)
        [18:32] fingers (right + left three-finger hands)
    """

    def __init__(self, cfg: G1ActionAdapterCfg):
        super().__init__(cfg)
        self.register_apply_method("moveto",  self._apply_moveto)
        self.register_apply_method("reach",   self._apply_reach)
        self.register_apply_method("lift",    self._apply_reach)
        self.register_apply_method("pull",    self._apply_reach)
        self.register_apply_method("grasp",   self._apply_gripper)
        self.register_apply_method("ungrasp", self._apply_gripper)

    # ------------------------------------------------------------------
    # Navigation (leg locomotion base command)
    # ------------------------------------------------------------------

    def _apply_moveto(self, skill_output: SkillOutput, env: ManagerBasedEnv) -> torch.Tensor:
        """Convert world-frame velocity [vx, vy, vyaw] to virtual-base delta positions."""
        vx, vy, vyaw = skill_output.action  # world frame

        robot = env.scene["robot"]
        world_pose = robot.data.root_pose_w[0]  # [x, y, z, qw, qx, qy, qz]
        w, x, y, z = world_pose[3:7]
        sin_yaw = 2 * (w * z + x * y)
        cos_yaw = 1 - 2 * (y ** 2 + z ** 2)

        vx_body =  vx * cos_yaw + vy * sin_yaw
        vy_body = -vx * sin_yaw + vy * cos_yaw

        last_action = env.action_manager.action
        action = last_action[0, :].clone()

        _MAX_CMD = 0.9
        action[0] = vx_body / _MAX_CMD
        action[1] = vy_body / _MAX_CMD
        action[2] = vyaw   / _MAX_CMD
        action[3] = 0.0  # mode=0: locomotion

        return action

    # ------------------------------------------------------------------
    # Arm motion (cuRobo joint trajectory playback)
    # ------------------------------------------------------------------

    def _apply_reach(self, skill_output: SkillOutput, env: ManagerBasedEnv) -> torch.Tensor:
        """Write cuRobo joint positions into the arm action terms."""
        target_joint_pos = skill_output.action

        last_action = env.action_manager.action
        action = last_action[0, :].clone()

        robot = env.scene["robot"]
        r_arm_ids, _ = robot.find_joints(env.action_manager.get_term("right_arm_action").cfg.joint_names)
        l_arm_ids, _ = robot.find_joints(env.action_manager.get_term("left_arm_action").cfg.joint_names)

        action[0] = 0.0
        action[1] = 0.0
        action[2] = 0.0
        action[3] = 1.0  # mode=1: squat/stance — keep legs fixed during arm motion
        action[4:11]  = target_joint_pos[r_arm_ids]
        action[11:18] = target_joint_pos[l_arm_ids]

        return action

    # ------------------------------------------------------------------
    # Gripper (three-finger open / close)
    # ------------------------------------------------------------------

    def _apply_gripper(self, skill_output: SkillOutput, env: ManagerBasedEnv) -> torch.Tensor:
        """Set all finger joints to the closed (grasp) or open (ungrasp) position."""
        gripper_signal = skill_output.action[0].item()

        angles = self.cfg.finger_close_angles if gripper_signal < 0 else self.cfg.finger_open_angles
        finger_angles = torch.tensor(angles, dtype=torch.float32, device=env.device)

        last_action = env.action_manager.action
        action = last_action[0, :].clone()
        action[0] = 0.0
        action[1] = 0.0
        action[2] = 0.0
        action[3] = 1.0  # mode=1: squat/stance

        robot = env.scene["robot"]
        finger_ids, _ = robot.find_joints(env.action_manager.get_term("gripper_action").cfg.joint_names)
        action[18:18 + len(finger_ids)] = finger_angles[:len(finger_ids)]

        return action
