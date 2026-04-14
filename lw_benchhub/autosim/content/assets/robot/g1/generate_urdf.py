"""
Generate G1_autosim.urdf by prepending virtual base joints to the original
g1_29dof_with_hand.urdf. The virtual joints (base_x, base_y, base_yaw) mirror
the same approach used in X7S, allowing A*+P-controller navigation to drive the
robot via joint position commands in Isaac Lab.

Usage:
    python generate_urdf.py
Output:
    G1_autosim.urdf (in the same directory as this script)
"""

from pathlib import Path

ORIGINAL_URDF = (
    Path(__file__).parent.parent.parent.parent.parent.parent
    / "core/mdp/actions/wbc_policy/robot_model/g1/g1_29dof_with_hand.urdf"
)
OUTPUT_URDF = Path(__file__).parent / "G1_autosim.urdf"

VIRTUAL_BASE_PREAMBLE = """\
  <!-- ===== Virtual base joints (autosim addition) ===== -->
  <!-- These three joints replace the original floating_base_joint and allow  -->
  <!-- Isaac Lab to move the whole robot via joint position commands, matching  -->
  <!-- the same pattern used in X7S.                                           -->

  <link name="world_link"/>

  <link name="base_x_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.01"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="base_x_joint" type="prismatic">
    <parent link="world_link"/>
    <child link="base_x_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-50" upper="50" effort="100" velocity="1000"/>
  </joint>

  <link name="base_y_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.01"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="base_y_joint" type="prismatic">
    <parent link="base_x_link"/>
    <child link="base_y_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-50" upper="50" effort="100" velocity="1000"/>
  </joint>

  <link name="base_yaw_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.01"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="base_yaw_joint" type="revolute">
    <parent link="base_y_link"/>
    <child link="base_yaw_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-6.28159" upper="6.28159" effort="100" velocity="1000"/>
  </joint>

  <joint name="base_to_pelvis" type="fixed">
    <parent link="base_yaw_link"/>
    <child link="pelvis"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>

  <!-- ===== End of virtual base joints ===== -->
"""


def generate():
    original = ORIGINAL_URDF.read_text(encoding="utf-8")

    # Remove the mujoco block (not needed by cuRobo / Isaac Lab URDF loader)
    import re
    original = re.sub(r"\s*<mujoco>.*?</mujoco>", "", original, flags=re.DOTALL)

    # Remove the commented-out floating_base_joint block so the file is clean
    original = re.sub(r"<!--.*?-->", "", original, flags=re.DOTALL)

    # Find the insertion point: right before the first <link name="pelvis">
    insert_marker = '<link name="pelvis">'
    idx = original.find(insert_marker)
    if idx == -1:
        raise RuntimeError("Could not find '<link name=\"pelvis\">' in the original URDF.")

    output = original[:idx] + VIRTUAL_BASE_PREAMBLE + original[idx:]

    # Rename robot tag for clarity
    output = output.replace('name="g1_29dof_with_hand"', 'name="g1_autosim"', 1)

    OUTPUT_URDF.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_URDF.write_text(output, encoding="utf-8")
    print(f"Written: {OUTPUT_URDF}")
    print(f"Source:  {ORIGINAL_URDF}")


if __name__ == "__main__":
    generate()
