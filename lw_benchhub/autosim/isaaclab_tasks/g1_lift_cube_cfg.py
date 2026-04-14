"""Isaac Lab scene + env config for G1 cube-lift autosim.

Scene layout (units: metres, world frame):
  - G1 robot  : spawned at (0, 0, 0.8) floating via virtual base joints
  - Table      : at (0.6, 0, 0), packing table (≈ 1.0 m tall, non-instanceable)
  - Cube       : at (0.6, 0, 1.025)  ← on top of the table
  - Ground plane: z = -1.05 (below table)
"""

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import lw_benchhub.core.mdp as lw_benchhub_mdp
from lw_benchhub.core.robots.unitree.assets_cfg import G1_CFG


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

@configclass
class G1LiftCubeSceneCfg(InteractiveSceneCfg):
    """Scene with G1 robot, a table, and a cube."""

    robot: ArticulationCfg = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Table the robot stands in front of
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.6, 0.0, 0.0],
            rot=[0.707, 0.0, 0.0, 0.707],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
        ),
    )

    # Target object — load from local USDA that has PhysicsRigidBodyAPI already applied.
    # Using UsdFileCfg (same pattern as the Franka lift-cube reference) avoids the
    # CuboidCfg / define_rigid_body_properties issue in this IsaacLab version.
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.6, 0.0, 1.025],
            rot=[1, 0, 0, 0],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=MassPropertiesCfg(mass=0.1),
        ),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


# ---------------------------------------------------------------------------
# Actions  (names must match what G1ActionAdapter expects)
# ---------------------------------------------------------------------------

@configclass
class G1ActionsCfg:
    """Action terms for G1 autosim.

    Action vector layout:
        [0:4]   base_action   (loco cmd: vx, vy, vyaw, mode)
        [4:11]  right_arm     (absolute joint position)
        [11:18] left_arm      (absolute joint position)
        [18:32] gripper       (14 finger joints, absolute position)
    """

    base_action: lw_benchhub_mdp.LegPositionActionCfg = lw_benchhub_mdp.LegPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_hip_pitch_joint", "right_hip_pitch_joint", "left_hip_roll_joint",
            "right_hip_roll_joint", "left_hip_yaw_joint", "right_hip_yaw_joint",
            "left_knee_joint", "right_knee_joint", "left_ankle_pitch_joint",
            "right_ankle_pitch_joint", "left_ankle_roll_joint", "right_ankle_roll_joint",
        ],
        body_name="base",
        scale=1.0,
        loco_config="g1_loco.yaml",
        squat_config="g1_squat.yaml",
    )

    right_arm_action: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
        scale=1.0,
        use_default_offset=False,
    )

    left_arm_action: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
        ],
        scale=1.0,
        use_default_offset=False,
    )

    gripper_action: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "right_hand_index_0_joint",  "right_hand_index_1_joint",
            "right_hand_middle_0_joint", "right_hand_middle_1_joint",
            "right_hand_thumb_0_joint",  "right_hand_thumb_1_joint",  "right_hand_thumb_2_joint",
            "left_hand_index_0_joint",   "left_hand_index_1_joint",
            "left_hand_middle_0_joint",  "left_hand_middle_1_joint",
            "left_hand_thumb_0_joint",   "left_hand_thumb_1_joint",   "left_hand_thumb_2_joint",
        ],
        scale=1.0,
        use_default_offset=False,
    )


# ---------------------------------------------------------------------------
# Observations / Terminations / Events (minimal)
# ---------------------------------------------------------------------------

@configclass
class G1ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actions   = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


def _cube_lifted(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.scene["cube"].data.root_pos_w[:, 2] > 0.85


@configclass
class G1RewardsCfg:
    pass


@configclass
class G1EventCfg:
    pass


@configclass
class G1TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success  = DoneTerm(func=_cube_lifted)


# ---------------------------------------------------------------------------
# Top-level env config
# ---------------------------------------------------------------------------

@configclass
class G1LiftCubeEnvCfg(ManagerBasedRLEnvCfg):
    scene: G1LiftCubeSceneCfg = G1LiftCubeSceneCfg(
        num_envs=1, env_spacing=3.0, replicate_physics=False
    )
    observations: G1ObservationsCfg = G1ObservationsCfg()
    actions:      G1ActionsCfg      = G1ActionsCfg()
    terminations: G1TerminationsCfg = G1TerminationsCfg()
    rewards: G1RewardsCfg = G1RewardsCfg()
    events:  G1EventCfg   = G1EventCfg()

    def __post_init__(self):
        self.decimation      = 2
        self.episode_length_s = 60.0
        self.sim.dt           = 0.01   # 100 Hz physics
        self.sim.render_interval = 2
        self.sim.use_fabric   = False  # disable fabric for compatibility

        self.sim.physx.bounce_threshold_velocity          = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity  = 16 * 1024
        self.sim.physx.friction_correlation_distance       = 0.00625
