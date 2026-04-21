from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.envs import ManagerBasedEnv

from autosim import ActionAdapterBase
from autosim.core.types import SkillOutput

if TYPE_CHECKING:
    from .g1_right_arm_only_action_adapter_cfg import G1RightArmOnlyActionAdapterCfg


class G1RightArmOnlyActionAdapter(ActionAdapterBase):
    """Action adapter for G1 — right arm only, legs fixed in stance mode.

    Action vector layout (G1ActionsCfg):
        [0:4]   base locomotion command — action[3]=1.0 keeps squat/stance
        [4:11]  right arm joints (7 DoF, absolute position) — planned by cuRobo
        [11:18] left  arm joints (7 DoF, absolute position) — locked by cuRobo
        [18:32] fingers (right + left three-finger hands)

    moveto is not registered; the LLM will not generate navigation steps.
    """

    def __init__(self, cfg: G1RightArmOnlyActionAdapterCfg):
        super().__init__(cfg)
        self.register_apply_method("reach",   self._apply_reach)
        self.register_apply_method("lift",    self._apply_reach)
        self.register_apply_method("push",    self._apply_reach)
        self.register_apply_method("pull",    self._apply_reach)
        self.register_apply_method("grasp",   self._apply_gripper)
        self.register_apply_method("ungrasp", self._apply_gripper)

    # ------------------------------------------------------------------
    # Arm motion (cuRobo joint trajectory playback — left arm primary)
    # ------------------------------------------------------------------

    def _apply_reach(self, skill_output: SkillOutput, env: ManagerBasedEnv) -> torch.Tensor:
        """Write cuRobo joint positions into the arm action terms.

        cuRobo plans the right arm; left arm joints come back at their locked
        values and are written unchanged.
        """
        target_joint_pos = skill_output.action   # [N_sim_joints], full robot order
        robot = env.scene["robot"]

        last_action = env.action_manager.action
        action = last_action[0, :].clone()

        r_arm_ids, _ = robot.find_joints(env.action_manager.get_term("right_arm_action").cfg.joint_names)
        l_arm_ids, _ = robot.find_joints(env.action_manager.get_term("left_arm_action").cfg.joint_names)

        # G1ActionsCfg layout: [vx, vy, vyaw, mode, r_arm×7, l_arm×7, gripper×14]
        action[0] = 0.0
        action[1] = 0.0
        action[2] = 0.0
        action[3] = 1.0  # mode=1: squat/stance — legs fixed

        action[4:11]  = target_joint_pos[r_arm_ids]   # right arm — planned
        action[11:18] = target_joint_pos[l_arm_ids]   # left arm  — locked

        return action

    def _apply_gripper(self, skill_output: SkillOutput, env: ManagerBasedEnv) -> torch.Tensor:
        """Set right-hand finger joints to closed (grasp) or open (ungrasp)."""

        gripper_signal = skill_output.action[0].item()
        angles = self.cfg.finger_close_angles if gripper_signal < 0 else self.cfg.finger_open_angles
        finger_angles = torch.tensor(angles, dtype=torch.float32, device=env.device)

        robot = env.scene["robot"]
        last_action = env.action_manager.action
        action = last_action[0, :].clone()
        action[0] = 0.0
        action[1] = 0.0
        action[2] = 0.0
        action[3] = 1.0  # mode=1: squat/stance

        finger_ids, _ = robot.find_joints(env.action_manager.get_term("gripper_action").cfg.joint_names)
        action[18:18 + len(finger_ids)] = finger_angles[:len(finger_ids)]

        return action
