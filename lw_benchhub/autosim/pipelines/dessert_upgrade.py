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
class DessertUpgradePipelineCfg(AutoSimPipelineCfg):
    """Configuration for the DessertUpgradePipeline."""

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
        self.skills.reach.extra_cfg.extra_target_mode = "keep_initial_relative_offset"

        self.occupancy_map.floor_prim_suffix = "Scene/floor_room"

        self.motion_planner.robot_config_file = "x7s.yml"
        self.motion_planner.curobo_asset_path = str(
            curobo_content_root_path / "assets")
        self.motion_planner.curobo_config_path = str(
            curobo_content_root_path / "configs" / "robot")
        self.motion_planner.world_ignore_subffixes = ["Scene/floor_room"]
        self.motion_planner.world_only_subffixes = [
            "Scene/counter_main_main_group",
            "Scene/counter_1_left_group",
            "Scene/counter_1_right_group",
        ]


class DessertUpgradePipeline(AutoSimPipeline):
    def __init__(self, cfg: AutoSimPipelineCfg):
        super().__init__(cfg)

    def load_env(self) -> ManagerBasedEnv:
        import gymnasium as gym
        from lw_benchhub.utils.env import parse_env_cfg, ExecuteMode

        import lw_benchhub_tasks.lightwheel_robocasa_tasks.multi_stage.serving_food.dessert_upgrade as du
        du.DessertUpgrade._get_obj_cfgs = patch_get_obj_cfgs

        env_cfg = parse_env_cfg(
            scene_backend="robocasa",
            task_backend="robocasa",
            task_name="DessertUpgrade",
            robot_name="X7S-Joint",
            scene_name="robocasakitchen-6-2",
            robot_scale=1.0,
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
            resample_robot_placement_on_reset=False,
        )
        usd_path = env_cfg.scene.robot.spawn.usd_path
        env_cfg.scene.robot.spawn.usd_path = usd_path.replace("x7s.usd", "x7s_autosim.usd")
        task_name = "Robocasa-DessertUpgrade-X7S-Joint-v0"

        gym.register(
            id=task_name,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={},
            disable_env_checker=True
        )

        env_cfg.terminations.time_out = None
        env_cfg.terminations.success = None

        env_cfg.scene.robot.init_state.pos[1] -= 1.6
        env_cfg.scene.robot.init_state.pos[0] -= 0.2

        env = gym.make(task_name, cfg=env_cfg,
                       render_mode="rgb_array").unwrapped

        return env

    def get_env_extra_info(self) -> EnvExtraInfo:
        available_objects = ["dessert1", "receptacle"]
        return EnvExtraInfo(
            task_name="Robocasa-Task-DessertUpgrade",
            objects=available_objects,
            additional_prompt_contents=f"{prompt_template.render()}\n\nOnly grasp the dessert1 and place it in the receptacle. After grasping the dessert1, you should lift it up.",
            robot_base_link_name="base_link",
            ee_link_name="left_hand_link",
            object_reach_target_poses={
                "dessert1": [
                    torch.tensor([0.003, -0.019, 0.025, 0.705, -0.002, 0.05, 0.707]),
                ],
                "receptacle": [
                    torch.tensor([0.003, -0.049, 0.085, 0.705, -0.002, 0.05, 0.707]),
                ],
            },
        )


def patch_get_obj_cfgs(self):
    cfgs = []
    cfgs.append(
        dict(
            name="receptacle",
            obj_groups="tray",
            asset_name="Tray033.usd",
            graspable=False,
            placement=dict(
                fixture=self.counter,
                sample_region_kwargs=dict(top_size=(1.0, 0.4)),
                size=(1, 0.35),
                pos=(0, -0.6),
                offset=(0.0, -0.0),
                rotation=(0.0, 0.0),
            ),
        )
    )

    cfgs.append(
        dict(
            name="dessert1",
            obj_groups=["cake"],
            asset_name="Cake005.usd",
            graspable=True,
            placement=dict(
                fixture=self.counter,
                size=(1, 0.15),
                pos=(0, -1),
                rotation=(0.0, 0.0),
            ),
        )
    )

    return cfgs
