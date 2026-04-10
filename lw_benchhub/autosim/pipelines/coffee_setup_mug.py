from autosim.core.pipeline import AutoSimPipeline, AutoSimPipelineCfg
from autosim.core.types import EnvExtraInfo
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import configclass

import torch

from autosim.decomposers import LLMDecomposerCfg
from jinja2 import Environment, FileSystemLoader
from pathlib import Path

from ..action_adapters.x7s_action_adapter_cfg import X7SActionAdapterCfg

jinja_env = Environment(
    loader=FileSystemLoader(str(Path(__file__).parent.parent)),
    autoescape=False,
)
prompt_template = jinja_env.get_template("additional_prompt.jinja")

curobo_content_root_path = Path(__file__).parent.parent / "content"


@configclass
class CoffeeSetupMugPipelineCfg(AutoSimPipelineCfg):
    """Configuration for the LWBenchhubAutosimPipeline."""

    decomposer: LLMDecomposerCfg = LLMDecomposerCfg()

    action_adapter: X7SActionAdapterCfg = X7SActionAdapterCfg()

    def __post_init__(self):
        self.skills.moveto.extra_cfg.local_planner.max_linear_velocity = 0.4
        self.skills.moveto.extra_cfg.local_planner.max_angular_velocity = 0.8
        self.skills.moveto.extra_cfg.local_planner.predict_time = 0.4
        self.skills.moveto.extra_cfg.global_planner.safety_distance = 0.8
        self.skills.moveto.extra_cfg.global_planner.proximity_weight = 3.0
        self.skills.moveto.extra_cfg.waypoint_tolerance = 0.2
        self.skills.moveto.extra_cfg.goal_tolerance = 0.07  # 0.25
        self.skills.moveto.extra_cfg.yaw_tolerance = 0.008  # 0.01
        self.skills.moveto.extra_cfg.uws_dwa = False
        self.max_steps = 1000

        self.skills.lift.extra_cfg.lift_offset = 0.20
        # self.skills.debug_target_pose()

        self.skills.reach.extra_cfg.extra_target_link_names = ["link20_tip"]
        self.skills.reach.extra_cfg.extra_target_mode = "keep_relative_offset"

        self.occupancy_map.floor_prim_suffix = "Scene/floor_room"

        self.motion_planner.robot_config_file = "x7s.yml"
        self.motion_planner.curobo_asset_path = str(curobo_content_root_path / "assets")
        self.motion_planner.curobo_config_path = str(curobo_content_root_path / "configs" / "robot")
        self.motion_planner.world_ignore_subffixes = ["Scene/floor_room"]
        self.motion_planner.world_only_subffixes = [
            "Scene/island_island_group",
            "Scene/island_panel_cab_right_island_group_1",
            "Scene/counter_main_main_group",
            "Scene/coffee_machine_main_group",
            "Scene/obj",
        ]


class CoffeeSetupMugPipeline(AutoSimPipeline):
    def __init__(self, cfg: AutoSimPipelineCfg):
        super().__init__(cfg)

    def reset_env(self):
        super().reset_env()

        obj = self._env.scene['obj']
        pose = torch.tensor([[1.9, -0.582, 0.950, 8.183e-01, -5.918e-04, 4.849e-03, 5.747e-01]], device=self._env.device)
        obj.write_root_pose_to_sim(pose)
        obj.reset()

    def load_env(self) -> ManagerBasedEnv:
        import gymnasium as gym
        from lw_benchhub.utils.env import parse_env_cfg, ExecuteMode

        env_cfg = parse_env_cfg(
            scene_backend="robocasa",
            task_backend="robocasa",
            task_name="CoffeeSetupMug",
            robot_name="X7S-Joint",
            scene_name="robocasakitchen-2-2",
            robot_scale=1.0,
            # device="cuda:0", num_envs=1, use_fabric=True,
            device="cpu", num_envs=1, use_fabric=False,
            first_person_view=False,
            enable_cameras=False,
            execute_mode=ExecuteMode.TELEOP,
            usd_simplify=False,
            seed=42,
            sources=['objaverse', 'lightwheel', 'aigen_objs'],
            object_projects=[],
            rl_name=None,
            headless_mode=False,
            replay_cfgs={"add_camera_to_observation": True,
                         "render_resolution": (640, 480)},
        )
        usd_path = env_cfg.scene.robot.spawn.usd_path
        env_cfg.scene.robot.spawn.usd_path = usd_path.replace("x7s.usd", "x7s_autosim.usd")
        task_name = "Robocasa-CoffeeSetupMug-X7S-Joint-v0"

        gym.register(
            id=task_name,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={},
            disable_env_checker=True
        )

        env_cfg.terminations.time_out = None

        env_cfg.scene.robot.init_state.pos[1] -= 0.8
        env_cfg.scene.robot.init_state.pos[2] += 0.3
        env_cfg.scene.robot.init_state.rot = (0.707, 0.0, 0.0, -0.707)

        env = gym.make(task_name, cfg=env_cfg, render_mode="rgb_array").unwrapped

        return env

    def get_env_extra_info(self) -> EnvExtraInfo:
        available_objects = self._env.scene.keys()
        return EnvExtraInfo(
            task_name="Robocasa-Task-CoffeeSetupMug",
            objects=available_objects,
            additional_prompt_contents=prompt_template.render(),
            robot_name="robot",
            robot_base_link_name="base_link",
            ee_link_name="left_hand_link",
            object_reach_target_poses={
                "obj": [
                    torch.tensor([-0.028, -0.01, 0.01, 0.985, -0.003, -0.004, 0.173]),
                ],
                "coffee_machine_main_group": [
                    torch.tensor([0.0, -0.12, -0.03, 0.707, 0.0, 0.0, 0.707]),
                ],
            },
        )
