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
class CheesyBreadPipelineCfg(AutoSimPipelineCfg):
    """Configuration for the LWBenchhubAutosimPipeline."""

    decomposer: LLMDecomposerCfg = LLMDecomposerCfg()

    action_adapter: X7SActionAdapterCfg = X7SActionAdapterCfg()

    def __post_init__(self):
        self.skills.moveto.extra_cfg.local_planner.max_linear_velocity = 3.5
        self.skills.moveto.extra_cfg.local_planner.max_angular_velocity = 3.0
        self.skills.moveto.extra_cfg.local_planner.predict_time = 0.4
        self.skills.moveto.extra_cfg.global_planner.safety_distance = 1.1  # 0.9
        self.skills.moveto.extra_cfg.global_planner.proximity_weight = 3.0
        self.skills.moveto.extra_cfg.waypoint_tolerance = 0.2
        self.skills.moveto.extra_cfg.goal_tolerance = 0.1  # 0.25
        self.skills.moveto.extra_cfg.yaw_tolerance = 0.005  # 0.01
        self.skills.moveto.extra_cfg.uws_dwa = False

        self.skills.reach.extra_cfg.extra_target_link_names = ["link20_tip"]

        self.occupancy_map.floor_prim_suffix = "Scene/floor_room"

        self.motion_planner.robot_config_file = "x7s.yml"
        self.motion_planner.curobo_asset_path = str(curobo_content_root_path / "assets")
        self.motion_planner.curobo_config_path = str(curobo_content_root_path / "configs" / "robot")
        self.motion_planner.world_ignore_subffixes = ["Scene/floor_room"]
        self.motion_planner.world_only_subffixes = [
            "Scene/bread",
            "Scene/cheese",
            "Scene/counter_main_main_group",
            "Scene/counter_1_front_group",
        ]


class CheesyBreadPipeline(AutoSimPipeline):
    def __init__(self, cfg: AutoSimPipelineCfg):
        super().__init__(cfg)

    def reset_env(self):
        super().reset_env()

        cheese = self._env.scene['cheese']
        pose = torch.tensor([[1.1, -0.635, 1.1, 1.0, 0.0, 0.0, 0.0]], device=self._env.device)
        cheese.write_root_pose_to_sim(pose)
        cheese.reset()

        bread = self._env.scene['bread']
        pose = torch.tensor([[2.8, -3.97, 1.1, 1.0, 0.0, 0.0, 0.0]], device=self._env.device)
        bread.write_root_pose_to_sim(pose)
        bread.reset()

        bread_container = self._env.scene['bread_container']
        pose = torch.tensor([[2.88, -4.0, 1.0, 1.0, 0.0, 0.0, 0.0]], device=self._env.device)
        bread_container.write_root_pose_to_sim(pose)
        bread_container.reset()

    def load_env(self) -> ManagerBasedEnv:
        import gymnasium as gym
        from lw_benchhub.utils.env import parse_env_cfg, ExecuteMode

        import lw_benchhub_tasks.lightwheel_robocasa_tasks.multi_stage.making_toast.cheesy_bread as cb
        cb.CheesyBread._get_obj_cfgs = patch_get_obj_cfgs

        env_cfg = parse_env_cfg(
            scene_backend="robocasa",
            task_backend="robocasa",
            task_name="CheesyBread",
            robot_name="X7S-Joint",
            scene_name="robocasakitchen-9-8",
            robot_scale=1.0,
            device="cpu", num_envs=1, use_fabric=False,
            # device="cuda:0", num_envs=1, use_fabric=True,
            first_person_view=False,
            enable_cameras=False,
            execute_mode=ExecuteMode.TRAIN,
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
        task_name = "Robocasa-CheesyBread-X7S-Joint-v0"

        gym.register(
            id=task_name,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={},
            disable_env_checker=True
        )

        env_cfg.terminations.time_out = None

        env_cfg.scene.robot.init_state.pos[1] -= 0.8

        env = gym.make(task_name, cfg=env_cfg,
                       render_mode="rgb_array").unwrapped

        return env

    def get_env_extra_info(self) -> EnvExtraInfo:
        available_objects = self._env.scene.keys()
        return EnvExtraInfo(
            task_name="Robocasa-Task-CheesyBread",
            objects=available_objects,
            additional_prompt_contents=f"{prompt_template.render()}\n\n After grasp the cheess, you need to lift it up and place it on the bread.",
            robot_name="robot",
            robot_base_link_name="base_link",
            ee_link_name="left_hand_link",
            object_reach_target_poses={
                "cheese": [
                    torch.tensor([0.003,
                                  -0.049,  # -0.051
                                  0.025,  # 0.033
                                  0.705,
                                  -0.002,
                                  0.05,
                                  0.707]),
                ],
                "bread": [
                    torch.tensor([-0.165,  # -0.159 +y
                                  0.017,  # 0.017 -z
                                  0.017,  # -x
                                  0.548,
                                  0.462,
                                  -0.479,
                                  -0.508]),
                ],
            },
        )


def patch_get_obj_cfgs(self):
    cfgs = []
    cfgs.append(
        dict(
            name="bread",
            obj_groups="bread_flat",
            asset_name="Bread010.usd",
            object_scale=1.5,
            placement=dict(
                fixture=self.counter,
                size=(0.5, 0.7),
                pos=(0, -1.0),
                # try_to_place_in="cutting_board",
            ),
        )
    )
    cfgs.append(
        dict(
            name="cheese",
            obj_groups="cheese",
            asset_name="Cheese003.usd",
            init_robot_here=True,
            placement=dict(
                # ref_obj="bread_container",
                fixture=self.counter,
                size=(1.0, 0.3),
                pos=(0, -1.0),
            ),
        )
    )

    cfgs.append(
        dict(
            name="bread_container",
            obj_groups="cutting_board",
            asset_name="CuttingBoard025.usd",
            object_scale=1.0,
            placement=dict(
                fixture=self.counter,
                size=(0.5, 0.5),
                pos=(0.2, -0.8),
            ),
        )
    )

    # Distractor on the counter
    cfgs.append(
        dict(
            name="distr_counter",
            obj_groups="all",
            placement=dict(fixture=self.counter, size=(1.0, 0.20), pos=(0, 1.0)),
        )
    )
    return cfgs
