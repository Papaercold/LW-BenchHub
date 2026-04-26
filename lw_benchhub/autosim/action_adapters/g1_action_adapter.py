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

        self._dbg_reach_target_w: torch.Tensor | None = None
        self._dbg_reach_step: int = 0
        self._has_squatted: bool = False

    _ARM_SKILLS = frozenset({"reach", "lift", "push", "pull"})

    # ------------------------------------------------------------------
    # Pre-skill settling: squat + optional forward nudge
    # ------------------------------------------------------------------

    def pre_skill_hook(self, skill_type: str, env: ManagerBasedEnv, last_action: torch.Tensor) -> None:
        """Squat to a moderate height before any arm skill, then nudge forward.

        squat_cmd[0] starts at 0.75 (stand) and drops by -1.0 * max_cmd[0] / 250 = -0.0016
        per env.step. With squat_settle_steps=80: 10 transition + 70 steps → squat_cmd[0] ≈ 0.64
        (a slight crouch, ~10 cm pelvis drop).

        After squatting, for the reach skill the robot is teleported forward along its heading
        by pos_search_nudge_m * pos_search_nudge_count so it is closer to the target.
        solve_ik_batch is intentionally NOT used here because it conflicts with plan_motion's
        CUDA graph on CUDA < 12.0 (changing goal type error).
        """
        if skill_type not in self._ARM_SKILLS:
            return
        if self._has_squatted:
            return
        self._has_squatted = True

        robot = env.scene["robot"]
        r_arm_ids, _ = robot.find_joints(env.action_manager.get_term("right_arm_action").cfg.joint_names)
        l_arm_ids, _ = robot.find_joints(env.action_manager.get_term("left_arm_action").cfg.joint_names)

        settle_action = last_action.clone()
        settle_action[0, 0] = -1.0   # negative cmd drives squat_cmd[0] down from 0.75
        settle_action[0, 1] = 0.0
        settle_action[0, 2] = 0.0
        settle_action[0, 3] = 1.0    # mode=1: squat/stance
        settle_action[0, 4:11]  = robot.data.joint_pos[0, r_arm_ids]
        settle_action[0, 11:18] = robot.data.joint_pos[0, l_arm_ids]

        for _ in range(self.cfg.squat_settle_steps):
            env.step(settle_action)

        # Diagnostic: target position & quaternion in robot root frame
        import isaaclab.utils.math as PoseUtils
        root_pose = robot.data.root_pose_w[0:1]          # [1, 7]
        root_pos_w  = root_pose[:, :3]
        root_quat_w = root_pose[:, 3:]

        oven = env.scene["oven_main_group"]
        oven_pos  = oven.data.root_pos_w[0:1]
        oven_quat = oven.data.root_quat_w[0:1]
        offset_world_pos, offset_world_quat = PoseUtils.combine_frame_transforms(
            oven_pos, oven_quat,
            torch.tensor([[-0.176, -0.990,  0.100]], device=env.device),
            torch.tensor([[0.707, 0.0, 0.0, 0.707]], device=env.device),
        )
        tgt_pos_root, tgt_quat_root = PoseUtils.subtract_frame_transforms(
            root_pos_w, root_quat_w, offset_world_pos, offset_world_quat,
        )
        print(f"[G1ActionAdapter] root_pos_w          = {root_pos_w[0].cpu().numpy().round(4)}")
        print(f"[G1ActionAdapter] EE target world pos  = {offset_world_pos[0].cpu().numpy().round(4)}")
        print(f"[G1ActionAdapter] EE target world quat = {offset_world_quat[0].cpu().numpy().round(4)}")
        print(f"[G1ActionAdapter] EE target ROOT  pos  = {tgt_pos_root[0].cpu().numpy().round(4)}")
        print(f"[G1ActionAdapter] EE target ROOT  quat = {tgt_quat_root[0].cpu().numpy().round(4)}")

        # Store reach target for real-time console tracking in _apply_reach
        self._dbg_reach_target_w = offset_world_pos[0].detach().cpu()
        self._dbg_reach_step = 0

        # Forward nudge (reach only): teleport robot toward the oven after squatting.
        # solve_ik_batch cannot be used here (CUDA graph goal-type conflict with plan_motion).
        if skill_type != "reach" or self.cfg.pos_search_nudge_count == 0:
            return

        root_quat_w = robot.data.root_quat_w[0]
        w_q, x_q, y_q, z_q = root_quat_w[0].item(), root_quat_w[1].item(), root_quat_w[2].item(), root_quat_w[3].item()
        fwd_x = 1.0 - 2.0 * (y_q**2 + z_q**2)  # cos(yaw): local +x in world frame
        fwd_y = 2.0 * (w_q * z_q + x_q * y_q)   # sin(yaw)

        total_nudge = self.cfg.pos_search_nudge_m * self.cfg.pos_search_nudge_count
        pos = robot.data.root_pos_w[0].clone()
        pos[0] += fwd_x * total_nudge
        pos[1] += fwd_y * total_nudge

        pose = torch.zeros(1, 7, device=env.device)
        pose[0, :3] = pos
        pose[0, 3:]  = root_quat_w
        robot.write_root_pose_to_sim(pose)
        robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))

        for _ in range(self.cfg.pos_search_settle_steps):
            env.step(settle_action)

    # ------------------------------------------------------------------
    # Navigation (leg locomotion base command)
    # ------------------------------------------------------------------

    def _apply_moveto(self, skill_output: SkillOutput, env: ManagerBasedEnv) -> torch.Tensor:
        """Convert world-frame velocity [vx, vy, vyaw] to virtual-base delta positions."""
        self._has_squatted = False  # reset so squat fires once after next moveto
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

        self._dbg_reach_step += 1
        if self._dbg_reach_step % 10 == 1:
            ee_ids, _ = robot.find_bodies("left_wrist_yaw_link")
            ee_pos_w = robot.data.body_pos_w[0, ee_ids[0]].cpu()
            ee_str = ee_pos_w.numpy().round(3).tolist()
            if self._dbg_reach_target_w is not None:
                dist = (ee_pos_w - self._dbg_reach_target_w).norm().item()
                tgt_str = self._dbg_reach_target_w.numpy().round(3).tolist()
                print(f"[reach step={self._dbg_reach_step:4d}] EE={ee_str}  tgt={tgt_str}  dist={dist:.4f}m")
            else:
                print(f"[reach step={self._dbg_reach_step:4d}] EE={ee_str}")

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
