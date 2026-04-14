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
        self.register_apply_method("push",    self._apply_reach)
        self.register_apply_method("pull",    self._apply_reach)
        self.register_apply_method("grasp",   self._apply_gripper)
        self.register_apply_method("ungrasp", self._apply_gripper)

    # ------------------------------------------------------------------
    # Navigation (leg locomotion base command)
    # ------------------------------------------------------------------

    def _apply_moveto(self, skill_output: SkillOutput, env: ManagerBasedEnv) -> torch.Tensor:
        """Convert world-frame velocity [vx, vy, vyaw] to virtual-base delta positions.

        G1ActionsCfg layout:
            base_action [0:3]   – JointPositionAction with scale=0.01 (delta)
            right_arm   [3:10]  – absolute
            left_arm    [10:17] – absolute
            gripper     [17:31] – absolute
        """
        vx, vy, vyaw = skill_output.action  # world frame

        robot = env.scene["robot"]
        world_pose = robot.data.root_pose_w[0]  # [x, y, z, qw, qx, qy, qz]
        w, x, y, z = world_pose[3:7]
        sin_yaw = 2 * (w * z + x * y)
        cos_yaw = 1 - 2 * (y ** 2 + z ** 2)
        yaw = torch.atan2(sin_yaw, cos_yaw)

        # Rotate world-frame velocity → robot body frame
        vx_body =  vx * cos_yaw + vy * sin_yaw
        vy_body = -vx * sin_yaw + vy * cos_yaw

        dt_control = env.cfg.sim.dt * env.cfg.decimation

        last_action = env.action_manager.action
        action = last_action[0, :].clone()

        # base_action uses scale=0.01: action_value = delta / 0.01
        action[0] = (vx_body * dt_control) / 0.01
        action[1] = (vy_body * dt_control) / 0.01
        action[2] = (vyaw   * dt_control) / 0.01

        return action

    # ------------------------------------------------------------------
    # Arm motion (cuRobo joint trajectory playback)
    # ------------------------------------------------------------------

    def _apply_reach(self, skill_output: SkillOutput, env: ManagerBasedEnv) -> torch.Tensor:
        """Write cuRobo joint positions into the arm action terms."""

        target_joint_pos = skill_output.action   # [N_sim_joints], full robot order
        robot = env.scene["robot"]
        current_joint_pos = robot.data.joint_pos[0]

        last_action = env.action_manager.action
        action = last_action[0, :].clone()

        # Resolve joint → robot-joint index mappings
        r_arm_ids, _ = robot.find_joints(env.action_manager.get_term("right_arm_action").cfg.joint_names)
        l_arm_ids, _ = robot.find_joints(env.action_manager.get_term("left_arm_action").cfg.joint_names)

        # Keep base steady for manipulation (zero delta → no base movement).
        action[:3] = 0.0

        # G1ActionsCfg action layout:
        #   [0:3]   base_action    (scale=0.01 delta, kept zero above)
        #   [3:10]  right_arm      (7-DoF absolute)
        #   [10:17] left_arm       (7-DoF absolute)
        #   [17:31] gripper        (14 joints absolute)
        action[3:10]  = target_joint_pos[r_arm_ids]
        action[10:17] = target_joint_pos[l_arm_ids]

        return action

    # ------------------------------------------------------------------
    # Gripper (three-finger open / close)
    # ------------------------------------------------------------------

    def _apply_gripper(self, skill_output: SkillOutput, env: ManagerBasedEnv) -> torch.Tensor:
        """Set all finger joints to the closed (grasp) or open (ungrasp) position."""

        gripper_signal = skill_output.action[0].item()  # -1.0 = grasp, +1.0 = ungrasp

        # Map signal → actual joint angle
        finger_angle = self.cfg.finger_close_angle if gripper_signal < 0 else self.cfg.finger_open_angle

        last_action = env.action_manager.action
        action = last_action[0, :].clone()
        action[:3] = 0.0  # keep base steady

        robot = env.scene["robot"]
        finger_ids, _ = robot.find_joints(env.action_manager.get_term("gripper_action").cfg.joint_names)

        # gripper_action starts at action index 17
        action[17:17 + len(finger_ids)] = finger_angle

        return action
