"""G1 autosim pipeline — open microwave door.

Robot:   G1 loco controller variant (leg locomotion + dual arms + hands)
Task:    Open microwave door
Scene:   RoboCasa kitchen  robocasakitchen-2-2
Planner: cuRobo right-arm planning (g1_autosim.yml)
Nav:     A* + leg locomotion controller
"""

from pathlib import Path

import torch
from autosim.core.pipeline import AutoSimPipeline, AutoSimPipelineCfg
from autosim.core.types import EnvExtraInfo
from autosim.decomposers import LLMDecomposerCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import configclass

from lw_benchhub.autosim.action_adapters.g1_action_adapter_cfg import G1ActionAdapterCfg

_CUROBO_ROOT = Path(__file__).parent.parent / "content"
_G1_URDF_DIR = Path(__file__).parent.parent / "content/assets/robot/g1"


@configclass
class G1OpenMicrowavePipelineCfg(AutoSimPipelineCfg):
    """Pipeline configuration for G1 RoboCasa kitchen microwave opening."""

    decomposer:     LLMDecomposerCfg   = LLMDecomposerCfg()
    action_adapter: G1ActionAdapterCfg = G1ActionAdapterCfg()

    def __post_init__(self):
        # ---- Navigation ----
        self.skills.moveto.extra_cfg.local_planner.max_linear_velocity  = 0.25
        self.skills.moveto.extra_cfg.local_planner.max_angular_velocity = 1.0
        self.skills.moveto.extra_cfg.local_planner.predict_time         = 0.4
        self.skills.moveto.extra_cfg.global_planner.safety_distance     = 0.5
        self.skills.moveto.extra_cfg.global_planner.proximity_weight    = 3.0
        self.skills.moveto.extra_cfg.waypoint_tolerance                 = 0.25
        self.skills.moveto.extra_cfg.goal_tolerance                     = 0.30
        self.skills.moveto.extra_cfg.yaw_tolerance                      = 0.01
        self.skills.moveto.extra_cfg.use_dwa                            = False
        self.skills.moveto.extra_cfg.sampling_radius                    = 0.8

        # ---- Occupancy map (RoboCasa kitchen floor) ----
        self.occupancy_map.floor_prim_suffix = "Scene/floor_room"

        # ---- cuRobo ----
        self.motion_planner.robot_config_file  = "g1_autosim.yml"
        self.motion_planner.curobo_config_path = str(_CUROBO_ROOT / "configs" / "robot")
        self.motion_planner.curobo_asset_path  = str(_G1_URDF_DIR)
        self.motion_planner.world_ignore_subffixes = ["Scene/floor_room"]
        self.motion_planner.world_only_subffixes   = ["Scene/microwave_main_group"]

        # ---- Pull skill (open microwave door by pulling) ----
        self.skills.pull.extra_cfg.move_offset = 0.25
        self.skills.pull.extra_cfg.move_axis   = "-x"

        self.max_steps = 1000


class G1OpenMicrowavePipeline(AutoSimPipeline):
    """Pipeline that opens a microwave door with the G1 right arm."""

    _TASK_NAME = "Robocasa-OpenMicrowave-G1-Autosim-v0"

    def __init__(self, cfg: AutoSimPipelineCfg):
        from lw_benchhub.autosim.compat_autosim import (
            patch_curobo_config,
            patch_env_extra_info,
            patch_reach_skill,
        )
        patch_env_extra_info()
        patch_reach_skill()
        patch_curobo_config()
        super().__init__(cfg)

    # ------------------------------------------------------------------
    # Environment loading
    # ------------------------------------------------------------------

    def load_env(self) -> ManagerBasedEnv:
        import gymnasium as gym
        from lw_benchhub.utils.env import parse_env_cfg, ExecuteMode
        from lw_benchhub.autosim.isaaclab_tasks.g1_autosim_cfg import (
            G1ActionsCfg,
            G1ObservationsCfg,
            G1EventCfg,
        )

        env_cfg = parse_env_cfg(
            scene_backend="robocasa",
            task_backend="robocasa",
            task_name="OpenMicrowave",
            robot_name="G1-Loco-Controller",
            scene_name="robocasakitchen-2-2",
            robot_scale=1.0,
            device="cpu",
            num_envs=1,
            use_fabric=False,
            first_person_view=False,
            enable_cameras=False,
            execute_mode=ExecuteMode.TRAIN,
            usd_simplify=False,
            seed=42,
            sources=["objaverse", "lightwheel", "aigen_objs"],
            object_projects=[],
            rl_name=None,
            headless_mode=False,
        )

        env_cfg.actions      = G1ActionsCfg()
        env_cfg.observations = G1ObservationsCfg()
        env_cfg.events       = G1EventCfg()

        # Disable built-in timeout — autosim manages episode endings.
        env_cfg.terminations.time_out = None

        gym.register(
            id=self._TASK_NAME,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={},
            disable_env_checker=True,
        )

        env = gym.make(self._TASK_NAME, cfg=env_cfg).unwrapped

        # env_cfg.events was replaced with an empty G1EventCfg to prevent
        # unwanted autosim resets, which removed the startup init_task event
        # that normally calls task.init_fixtures(env) → fixture.setup_env(env).
        # Call it explicitly so fixture controllers (e.g. Microwave) have
        # self._env set before update_state() is triggered.
        arena_env = env.cfg.isaaclab_arena_env
        arena_env.task.init_fixtures(env)

        return env

    # ------------------------------------------------------------------
    # Task metadata for the LLM decomposer
    # ------------------------------------------------------------------

    def get_env_extra_info(self) -> EnvExtraInfo:
        env_info = EnvExtraInfo(
            task_name="Robocasa-Task-OpenMicrowave",
            objects=list(self._env.scene.keys()),
            robot_name="robot",
            robot_base_link_name="pelvis",
            ee_link_name="right_wrist_yaw_link",
            object_reach_target_poses={
                "microwave_main_group": [
                    torch.tensor([0.05, -0.22, 0.12, 0.707, 0.0, 0.0, 0.707]),
                ],
            },
        )
        env_info.object_extra_reach_target_poses = {}
        return env_info
