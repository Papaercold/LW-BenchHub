"""Runtime compatibility patches for external autosim package."""

from __future__ import annotations

import types

import isaaclab.utils.math as PoseUtils
import torch

from autosim.core.types import SkillGoal


def patch_env_extra_info() -> None:
    """Add backward/forward compatibility fields to EnvExtraInfo."""
    from autosim.core.types import EnvExtraInfo

    if hasattr(EnvExtraInfo, "_lw_patched_extra_reach"):
        return

    original_init = EnvExtraInfo.__init__
    original_reset = EnvExtraInfo.reset

    def _reset_extra_target_pose_iterators(self) -> None:
        data = getattr(self, "object_extra_reach_target_poses", {}) or {}
        self._object_extra_reach_target_poses_iterator_dict = {
            object_name: {
                ee_name: self._build_iterator(reach_target_poses)
                for ee_name, reach_target_poses in extra_targets.items()
            }
            for object_name, extra_targets in data.items()
        }

    def __init__(self, *args, object_extra_reach_target_poses=None, **kwargs):
        original_init(self, *args, **kwargs)
        self.object_extra_reach_target_poses = object_extra_reach_target_poses or {}
        _reset_extra_target_pose_iterators(self)

    def reset(self) -> None:
        original_reset(self)
        if not hasattr(self, "object_extra_reach_target_poses"):
            self.object_extra_reach_target_poses = {}
        _reset_extra_target_pose_iterators(self)

    def get_next_extra_reach_target_pose(self, object_name: str, ee_name: str):
        return next(self._object_extra_reach_target_poses_iterator_dict[object_name][ee_name])

    EnvExtraInfo.__init__ = __init__
    EnvExtraInfo.reset = reset
    EnvExtraInfo._reset_extra_target_pose_iterators = _reset_extra_target_pose_iterators
    EnvExtraInfo.get_next_extra_reach_target_pose = get_next_extra_reach_target_pose
    EnvExtraInfo._lw_patched_extra_reach = True


def patch_reach_skill() -> None:
    """Patch ReachSkill for mixed autosim versions.

    Some autosim versions include newer ReachSkill call-sites that rely on
    helper methods/fields not present in older class definitions.
    """
    from autosim.skills import reach as reach_mod
    from autosim.core.registration import SkillRegistry

    ReachSkill = reach_mod.ReachSkill

    if not hasattr(ReachSkill, "_lw_original_build_activate_joint_state"):
        ReachSkill._lw_original_build_activate_joint_state = ReachSkill._build_activate_joint_state

        def _build_activate_joint_state_compat(
            self, full_sim_joint_names, full_sim_q, full_sim_qd=None
        ):
            # Planner may include virtual base joints that do not exist in the
            # simulated leg-loco robot articulation. Fill missing joints with 0.
            activate_q = []
            activate_qd = [] if full_sim_qd is not None else None
            for joint_name in self._planner.target_joint_names:
                if joint_name in full_sim_joint_names:
                    sim_joint_idx = full_sim_joint_names.index(joint_name)
                    activate_q.append(full_sim_q[sim_joint_idx])
                    if full_sim_qd is not None and activate_qd is not None:
                        activate_qd.append(full_sim_qd[sim_joint_idx])
                else:
                    activate_q.append(torch.tensor(0.0, device=full_sim_q.device, dtype=full_sim_q.dtype))
                    if full_sim_qd is not None and activate_qd is not None:
                        activate_qd.append(
                            torch.tensor(0.0, device=full_sim_qd.device, dtype=full_sim_qd.dtype)
                        )
            activate_q_tensor = torch.stack(activate_q, dim=0)
            if activate_qd is None:
                return activate_q_tensor, None
            return activate_q_tensor, torch.stack(activate_qd, dim=0)

        ReachSkill._build_activate_joint_state = _build_activate_joint_state_compat

    if not hasattr(ReachSkill, "_lw_original_step"):
        ReachSkill._lw_original_step = ReachSkill.step

        def step_compat(self, state):
            self.visualize_debug_target_pose()

            traj_positions = self._trajectory.position
            if self._step_idx >= len(self._trajectory.position):
                traj_pos = traj_positions[-1]
                done = True
            else:
                traj_pos = traj_positions[self._step_idx]
                done = False
                self._step_idx += 1

            # Keep upstream corrective-reach behavior when available.
            if done and getattr(self, "_corrective_reach_done", False) is False and getattr(
                self.cfg.extra_cfg, "corrective_reach", False
            ):
                self._corrective_reach_done = True
                if hasattr(self, "_compute_corrective_goal"):
                    new_goal = self._compute_corrective_goal()
                    if new_goal is not None:
                        self._logger.info("corrective_reach: re-planning to corrected object pose")
                        self._step_idx = 0
                        plan_success = self.execute_plan(state, new_goal)
                        if plan_success:
                            done = False

            curobo_joint_names = self._trajectory.joint_names
            sim_joint_names = state.sim_joint_names
            joint_pos = state.robot_joint_pos.clone()
            for curobo_idx, curobo_joint_name in enumerate(curobo_joint_names):
                if curobo_joint_name not in sim_joint_names:
                    continue
                sim_idx = sim_joint_names.index(curobo_joint_name)
                joint_pos[sim_idx] = traj_pos[curobo_idx]

            from autosim.core.types import SkillOutput

            return SkillOutput(action=joint_pos, done=done, success=True, info={})

        ReachSkill.step = step_compat

    if not hasattr(ReachSkill, "_compute_goal_from_offset"):

        def _compute_goal_from_offset(
            self,
            env,
            robot_name: str,
            target_object: str,
            reach_offset: torch.Tensor,
            extra_offsets=None,
        ):
            try:
                object_pose_in_env = env.scene[target_object].data.root_pose_w
            except Exception:
                self._logger.warning(f"could not read pose for '{target_object}', skipping")
                return None

            object_pos_in_env = object_pose_in_env[:, :3]
            object_quat_in_env = object_pose_in_env[:, 3:]

            offset = reach_offset.to(env.device).unsqueeze(0)
            reach_target_pos_in_env, reach_target_quat_in_env = PoseUtils.combine_frame_transforms(
                object_pos_in_env, object_quat_in_env, offset[:, :3], offset[:, 3:]
            )
            self._target_poses["target_pose"] = torch.cat((reach_target_pos_in_env, reach_target_quat_in_env), dim=-1)

            robot = env.scene[robot_name]
            robot_root_pos_in_env = robot.data.root_pose_w[:, :3]
            robot_root_quat_in_env = robot.data.root_pose_w[:, 3:]

            reach_target_pos_in_robot_root, reach_target_quat_in_robot_root = PoseUtils.subtract_frame_transforms(
                robot_root_pos_in_env,
                robot_root_quat_in_env,
                reach_target_pos_in_env,
                reach_target_quat_in_env,
            )
            target_pose = torch.cat(
                (reach_target_pos_in_robot_root, reach_target_quat_in_robot_root), dim=-1
            ).squeeze(0)

            activate_q, _ = self._build_activate_joint_state(
                robot.data.joint_names, robot.data.joint_pos[0], robot.data.joint_vel[0]
            )
            # Current upstream offset-based extra-EEF API is inconsistent across
            # versions. Keep behavior stable by using the built-in policy.
            extra_target_poses = self._build_extra_target_poses(activate_q, target_pose, self._saved_env_extra_info)

            return SkillGoal(target_object=target_object, target_pose=target_pose, extra_target_poses=extra_target_poses)

        ReachSkill._compute_goal_from_offset = _compute_goal_from_offset

    original_extract = ReachSkill.extract_goal_from_info

    def extract_goal_from_info(self, skill_info, env, env_extra_info):
        target_object = skill_info.target_object
        reach_offset = env_extra_info.get_next_reach_target_pose(target_object).to(env.device)

        self._saved_env = env
        self._saved_env_extra_info = env_extra_info
        self._saved_robot_name = env_extra_info.robot_name
        self._saved_target_object = target_object
        self._saved_reach_offset = reach_offset
        self._saved_extra_offsets = None

        # Keep compatibility with older/newer autosim versions.
        if hasattr(self, "_compute_goal_from_offset"):
            return self._compute_goal_from_offset(env, env_extra_info.robot_name, target_object, reach_offset, None)
        return original_extract(self, skill_info, env, env_extra_info)

    ReachSkill.extract_goal_from_info = extract_goal_from_info

    original_init = ReachSkill.__init__

    def __init__(self, extra_cfg):
        original_init(self, extra_cfg)
        if not hasattr(self.cfg.extra_cfg, "corrective_reach"):
            self.cfg.extra_cfg.corrective_reach = False
        self._corrective_reach_done = False
        self._saved_env = None
        self._saved_env_extra_info = None
        self._saved_robot_name = "robot"
        self._saved_target_object = None
        self._saved_reach_offset = None
        self._saved_extra_offsets = None

    ReachSkill.__init__ = __init__

    original_reset = ReachSkill.reset

    def reset(self):
        original_reset(self)
        self._corrective_reach_done = False
        self._saved_env = None
        self._saved_env_extra_info = None
        self._saved_target_object = None
        self._saved_reach_offset = None
        self._saved_extra_offsets = None

    ReachSkill.reset = reset

    # Fallback: patch the factory path too, in case another ReachSkill class
    # object (from a different autosim import path) is instantiated.
    if not hasattr(SkillRegistry, "_lw_patched_create"):

        @classmethod
        def _patched_create(cls, name, extra_cfg):
            skill_cls = cls.get(name)
            skill = skill_cls(extra_cfg)
            if skill.__class__.__name__ == "ReachSkill":
                if not hasattr(skill, "_compute_goal_from_offset"):
                    skill._compute_goal_from_offset = types.MethodType(
                        ReachSkill._compute_goal_from_offset, skill
                    )
                if not hasattr(skill.cfg.extra_cfg, "corrective_reach"):
                    skill.cfg.extra_cfg.corrective_reach = False
                if not hasattr(skill, "_saved_env_extra_info"):
                    skill._saved_env_extra_info = None
                if not hasattr(skill, "_corrective_reach_done"):
                    skill._corrective_reach_done = False
            return skill

        SkillRegistry.create = _patched_create
        SkillRegistry._lw_patched_create = True


def patch_curobo_config() -> None:
    """Relax cuRobo start-state checks for mixed G1 planner/sim setups."""
    from curobo.wrap.reacher.motion_gen import MotionGenConfig

    if hasattr(MotionGenConfig, "_lw_original_load_from_robot_config"):
        return

    MotionGenConfig._lw_original_load_from_robot_config = MotionGenConfig.load_from_robot_config

    @classmethod
    def _patched_load_from_robot_config(cls, *args, **kwargs):
        # In our current hybrid setup planner robot model and sim articulation
        # differ on virtual base joints, which can trigger false-positive
        # INVALID_START_STATE_SELF_COLLISION. Relaxing self collision check
        # avoids immediate planner rejection at step 0.
        kwargs.setdefault("self_collision_check", False)
        kwargs.setdefault("self_collision_opt", False)
        return cls._lw_original_load_from_robot_config(*args, **kwargs)

    MotionGenConfig.load_from_robot_config = _patched_load_from_robot_config
