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
class KettleBoilingPipelineCfg(AutoSimPipelineCfg):
    """Configuration for the LWBenchhubAutosimPipeline."""

    decomposer: LLMDecomposerCfg = LLMDecomposerCfg()

    action_adapter: X7SActionAdapterCfg = X7SActionAdapterCfg()

    def __post_init__(self):
        self.skills.moveto.extra_cfg.local_planner.max_linear_velocity = 0.4
        self.skills.moveto.extra_cfg.local_planner.max_angular_velocity = 0.3
        self.skills.moveto.extra_cfg.local_planner.predict_time = 0.4
        self.skills.moveto.extra_cfg.global_planner.safety_distance = 1.1
        self.skills.moveto.extra_cfg.global_planner.proximity_weight = 3.0
        self.skills.moveto.extra_cfg.waypoint_tolerance = 0.2
        self.skills.moveto.extra_cfg.goal_tolerance = 0.25
        self.skills.moveto.extra_cfg.yaw_tolerance = 0.01
        self.skills.moveto.extra_cfg.uws_dwa = False

        self.skills.lift.extra_cfg.move_offset = 0.15
        self.skills.lift.extra_cfg.move_axis = "+z"

        self.occupancy_map.floor_prim_suffix = "Scene/floor_room"

        self.max_steps = 800

        self.motion_planner.robot_config_file = "x7s.yml"
        self.motion_planner.curobo_asset_path = str(curobo_content_root_path / "assets")
        self.motion_planner.curobo_config_path = str(curobo_content_root_path / "configs" / "robot")
        self.motion_planner.world_ignore_subffixes = ["Scene/floor_room"]
        self.motion_planner.world_only_subffixes = [
            "Scene/obj",
            "Scene/stovetop_main_group",
            "Scene/counter_main_main_group",
            "Scene/counter_1_main_group",
        ]


class KettleBoilingPipeline(AutoSimPipeline):
    def __init__(self, cfg: AutoSimPipelineCfg):
        super().__init__(cfg)

    def reset_env(self):
        super().reset_env()

        obj = self._env.scene['obj']
        pose = torch.tensor([[2.0, -0.56, 1.1, 0.0, 0.0, 0.0, 1.0]], device=self._env.device)
        obj.write_root_pose_to_sim(pose)
        obj.reset()

        stove_distr = self._env.scene['stove_distr']
        pose = stove_distr.data.root_pose_w.clone()
        pose[:, 3:] = torch.tensor([0.707, 0.0, 0.0, 0.707], device=self._env.device)
        pose[:, 0] += 0.3
        pose[:, 1] += 0.2
        stove_distr.write_root_pose_to_sim(pose)
        stove_distr.reset()

    def load_env(self) -> ManagerBasedEnv:
        import gymnasium as gym
        from lw_benchhub.utils.env import parse_env_cfg, ExecuteMode

        import lw_benchhub_tasks.lightwheel_robocasa_tasks.multi_stage.brewing.kettle_boiling as kb
        kb.KettleBoiling._get_obj_cfgs = patch_get_obj_cfgs

        env_cfg = parse_env_cfg(
            scene_backend="robocasa",
            task_backend="robocasa",
            task_name="KettleBoiling",
            robot_name="X7S-Joint",
            scene_name="robocasakitchen-7-3",
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
        task_name = "Robocasa-KettleBoiling-X7S-Joint-v0"

        gym.register(
            id=task_name,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={},
            disable_env_checker=True
        )

        env_cfg.terminations.time_out = None

        env_cfg.scene.robot.init_state.pos[1] -= 0.8
        env_cfg.scene.robot.init_state.pos[2] += 0.01

        env = gym.make(task_name, cfg=env_cfg,
                       render_mode="rgb_array").unwrapped

        return env

    def get_env_extra_info(self) -> EnvExtraInfo:
        available_objects = self._env.scene.keys()
        return EnvExtraInfo(
            task_name="Robocasa-Task-KettleBoiling",
            objects=available_objects,
            additional_prompt_contents=prompt_template.render(),
            robot_name="robot",
            robot_base_link_name="base_link",
            ee_link_name="left_hand_link",
            object_reach_target_poses={
                "obj": [
                    torch.tensor([-0.029,  # -0.06 -X
                                  0.105,  # 0.095 -Y
                                  0.10,  # 0.153
                                  0.708,
                                  -0.008,
                                  -0.003,
                                  -0.707]),
                ],
                "stovetop_main_group": [
                    torch.tensor([-0.0,  # +X -0.035
                                  -0.045,  # -0.04 +Y
                                  0.24,  # 0.232 +Z
                                  0.707,
                                  0.,
                                  0.,
                                  0.707]),
                ],
            },
            object_extra_reach_target_poses={
                "obj": {
                    "link20_tip": [torch.tensor([-0.329,  # -X
                                                0.105,  # 0.105 -Y
                                                0.10,
                                                0.708,
                                                -0.008,
                                                -0.003,
                                                -0.707]),]
                },
                "stovetop_main_group": {
                    "link20_tip": [torch.tensor([0.3,  # +X
                                                -0.045,  # +Y
                                                0.24,  # +Z
                                                0.707,
                                                0.,
                                                0.,
                                                0.707]),]
                },
            }
        )


def patch_get_obj_cfgs(self):
    cfgs = []
    cfgs.append(
        dict(
            name="obj",
            obj_groups=("kettle_non_electric"),
            asset_name="Kettle073.usd",  # 57
            graspable=True,
            placement=dict(
                fixture=self.counter,
                sample_region_kwargs=dict(
                    ref=self.stove,
                ),
                size=(0.35, 0.35),
                pos=("ref", -1),
            ),
        )
    )

    cfgs.append(
        dict(
            name="stove_distr",
            # obj_groups=("pan", "pot"),
            obj_groups=("pan"),
            asset_name="Pan023.usd",
            placement=dict(
                fixture=self.stove,
                # ensure_object_boundary_in_range=False because the pans handle is a part of the
                # bounding box making it hard to place it if set to True
                ensure_object_boundary_in_range=False,
            ),
        )
    )

    return cfgs
