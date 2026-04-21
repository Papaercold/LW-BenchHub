import torch
from autosim.core.pipeline import AutoSimPipeline, AutoSimPipelineCfg
from autosim.core.registration import SkillRegistry
from autosim.core.types import PipelineOutput
from autosim.decomposers import LLMDecomposerCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import configclass

from lw_benchhub.autosim.prompt_utils import render_additional_prompt
from lw_benchhub.autosim.robot_profiles import (
    TaskRobotOverride,
    apply_robot_env_cfg,
    build_env_extra_info,
    configure_robot_runtime_settings,
    resolve_robot_settings,
)


def _x7s_skill_cfg(cfg) -> None:
    cfg.skills.moveto.extra_cfg.local_planner.max_linear_velocity  = 0.1
    cfg.skills.moveto.extra_cfg.local_planner.max_angular_velocity = 0.4
    cfg.skills.moveto.extra_cfg.local_planner.predict_time         = 0.4
    cfg.skills.moveto.extra_cfg.global_planner.safety_distance     = 0.8
    cfg.skills.moveto.extra_cfg.global_planner.proximity_weight    = 3.0
    cfg.skills.moveto.extra_cfg.waypoint_tolerance                 = 0.2
    cfg.skills.moveto.extra_cfg.goal_tolerance                     = 0.1
    cfg.skills.moveto.extra_cfg.yaw_tolerance                      = 0.008
    cfg.skills.moveto.extra_cfg.uws_dwa                            = False
    cfg.skills.moveto.extra_cfg.sampling_radius                    = 1.6
    cfg.skills.push.extra_cfg.move_offset = 0.36
    cfg.skills.push.extra_cfg.move_axis   = "+x"
    cfg.skills.lift.extra_cfg.move_offset = 0.15
    cfg.skills.lift.extra_cfg.move_axis   = "+z"
    cfg.max_steps = 1000


def _g1_skill_cfg(cfg) -> None:
    cfg.skills.moveto.extra_cfg.local_planner.max_linear_velocity  = 0.2
    cfg.skills.moveto.extra_cfg.local_planner.max_angular_velocity = 0.2
    cfg.skills.moveto.extra_cfg.local_planner.predict_time         = 0.4
    cfg.skills.moveto.extra_cfg.global_planner.safety_distance     = 0.5
    cfg.skills.moveto.extra_cfg.global_planner.proximity_weight    = 3.0
    cfg.skills.moveto.extra_cfg.waypoint_tolerance                 = 0.25
    cfg.skills.moveto.extra_cfg.goal_tolerance                     = 0.30
    cfg.skills.moveto.extra_cfg.yaw_tolerance                      = 0.01
    cfg.skills.moveto.extra_cfg.use_dwa                            = False
    cfg.skills.moveto.extra_cfg.sampling_radius                    = 0.87
    cfg.skills.push.extra_cfg.move_offset = 0.10
    cfg.skills.push.extra_cfg.move_axis   = "+x"
    cfg.skills.lift.extra_cfg.move_offset = 0.15
    cfg.skills.lift.extra_cfg.move_axis   = "+z"
    cfg.max_steps = 2000


TASK_ROBOT_OVERRIDES: dict[str, TaskRobotOverride] = {
    "x7s_joint_left": TaskRobotOverride(
        extra_target_link_names=("link20_tip",),
        reach_extra_target_mode="keep_relative_offset",
        object_reach_target_poses={
            "oven_main_group": [
                torch.tensor([-0.176, -0.739, -0.180, 0.707, -0.00, -0.00, 0.707]),
            ],
        },
        init_state_pos_delta=(-0.6, -1.2, 0.0),
        skill_cfg_fn=_x7s_skill_cfg,
    ),
    "g1_loco_left": TaskRobotOverride(
        object_reach_target_poses={
            "oven_main_group": [
                torch.tensor([-0.1687, -0.9214, -0.0407, 0.3762, 0.0, 0.0, 0.9264]),
            ],
        },
        init_state_pos_delta=(-0.21, -0.70, 0.01),
        init_state_rot=(0.707, 0.0, 0.0, 0.707),
        skill_cfg_fn=_g1_skill_cfg,
    ),
}


def get_task_robot_override(robot_profile: str) -> TaskRobotOverride:
    try:
        return TASK_ROBOT_OVERRIDES[robot_profile]
    except KeyError as exc:
        supported = ", ".join(TASK_ROBOT_OVERRIDES)
        raise ValueError(
            f"CloseOvenPipeline does not support robot profile '{robot_profile}'. Supported: {supported}"
        ) from exc


@configclass
class CloseOvenPipelineCfg(AutoSimPipelineCfg):
    robot_profile: str = "x7s_joint_left"
    decomposer: LLMDecomposerCfg = LLMDecomposerCfg()

    def __post_init__(self):
        resolved_robot = resolve_robot_settings(self.robot_profile, override=get_task_robot_override(self.robot_profile))
        configure_robot_runtime_settings(self, resolved_robot)

        if resolved_robot.override.skill_cfg_fn:
            resolved_robot.override.skill_cfg_fn(self)

        self.occupancy_map.floor_prim_suffix = "Scene/floor_room"
        self.motion_planner.world_ignore_subffixes = ["Scene/floor_room"]
        self.motion_planner.world_only_subffixes   = [
            "Scene/island_island_group",
            "Scene/island_panel_cab_right_island_group_1",
            "Scene/counter_main_main_group",
            "Scene/oven_main_group",
        ]


_DOOR_PROFILES = {"g1_loco_left"}
_DOOR_JOINT_PATH = "/World/envs/env_0/Scene/oven_main_group/Oven032_door/door_joint"


class CloseOvenPipeline(AutoSimPipeline):
    def __init__(self, cfg: AutoSimPipelineCfg):
        self._resolved_robot = resolve_robot_settings(
            cfg.robot_profile, override=get_task_robot_override(cfg.robot_profile)
        )
        super().__init__(cfg)

    def _set_door_drive(self, stiffness: float, damping: float, target_deg: float) -> None:
        try:
            import omni.usd
            from pxr import UsdPhysics
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(_DOOR_JOINT_PATH)
            if prim.IsValid() and (drive := UsdPhysics.DriveAPI.Get(prim, "angular")):
                drive.GetStiffnessAttr().Set(stiffness)
                drive.GetDampingAttr().Set(damping)
                if not (attr := drive.GetTargetPositionAttr()).IsValid():
                    attr = drive.CreateTargetPositionAttr()
                attr.Set(target_deg)
        except Exception as e:
            print(f"[CloseOven] door drive set failed: {e}")

    def reset_env(self):
        super().reset_env()
        if self._resolved_robot.profile.profile_id in _DOOR_PROFILES:
            self._env.cfg.isaaclab_arena_env.task._setup_scene(self._env)
            self._set_door_drive(stiffness=0.0, damping=1.0, target_deg=0.0)

    def execute_skill_sequence(self, decompose_result):
        if self._resolved_robot.profile.profile_id not in _DOOR_PROFILES:
            return super().execute_skill_sequence(decompose_result)

        self._check_skill_extra_cfg()
        self.reset_env()

        for subtask in decompose_result.subtasks:
            for skill_info in subtask.skills:
                if skill_info.skill_type == "push":
                    self._set_door_drive(stiffness=50.0, damping=5.0, target_deg=0.0)
                skill = SkillRegistry.create(
                    skill_info.skill_type, self.cfg.skills.get(skill_info.skill_type).extra_cfg
                )
                if self._action_adapter.should_skip_apply(skill):
                    continue
                goal = skill.extract_goal_from_info(skill_info, self._env, self._env_extra_info)
                success, steps = self._execute_single_skill(skill, goal)
                if not success:
                    raise ValueError(f"Skill {skill_info.skill_type} failed after {steps} steps.")

        self.reset_env()
        return PipelineOutput(success=True, generated_actions=self._generated_actions)

    def load_env(self) -> ManagerBasedEnv:
        import gymnasium as gym
        from lw_benchhub.utils.env import parse_env_cfg, ExecuteMode

        env_cfg = parse_env_cfg(
            scene_backend="robocasa",
            task_backend="robocasa",
            task_name="CloseOven",
            robot_name=self._resolved_robot.profile.robot_name,
            scene_name="robocasakitchen-2-2",
            robot_scale=1.0,
            device="cpu",
            num_envs=1,
            use_fabric=False,
            first_person_view=False,
            enable_cameras=False,
            execute_mode=ExecuteMode.TELEOP,
            usd_simplify=False,
            seed=42,
            sources=["objaverse", "lightwheel", "aigen_objs"],
            object_projects=[],
            rl_name=None,
            headless_mode=False,
            replay_cfgs={"add_camera_to_observation": True, "render_resolution": (640, 480)},
            resample_robot_placement_on_reset=False,
        )

        apply_robot_env_cfg(env_cfg, self._resolved_robot)
        env_cfg.terminations.time_out = None

        env_id = f"Robocasa-CloseOven-{self._resolved_robot.profile.robot_name}-v0"
        gym.register(
            id=env_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={},
            disable_env_checker=True,
        )

        env = gym.make(env_id, cfg=env_cfg).unwrapped

        if self._resolved_robot.profile.profile_id in _DOOR_PROFILES:
            env.cfg.isaaclab_arena_env.task._setup_scene(env)
            self._set_door_drive(stiffness=0.0, damping=1.0, target_deg=0.0)

        return env

    def get_env_extra_info(self):
        return build_env_extra_info(
            task_name="Robocasa-Task-CloseOven",
            objects=["oven_main_group"],
            additional_prompt_contents=(
                f"{render_additional_prompt()}\n\nWhen you close the oven, you should lift and then push the oven door to close it."
            ),
            resolved_robot=self._resolved_robot,
        )
