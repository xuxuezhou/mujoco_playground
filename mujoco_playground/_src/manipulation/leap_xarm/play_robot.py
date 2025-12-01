"""Quick viewer to load and inspect the LEAP hand + XArm model."""

from pathlib import Path
from typing import Dict, Any
from etils import epath

import mujoco
import mujoco.viewer as viewer

from mujoco_playground._src import mjx_env

# robot_path = "/home/xxz/code/mujoco_playground/mujoco_playground/_src/manipulation/leap_xarm/xmls/leap_rh_mjx.xml"
robot_path = "/home/xxz/code/mujoco_playground/mujoco_playground/_src/manipulation/leap_xarm/xmls/leap_xarm_rh_mjx.xml"

def build_assets() -> Dict[str, bytes]:
  """Collects all asset files needed by the XML."""
  assets: Dict[str, Any] = {}
  # Ensure menagerie exists for LEAP hand meshes.
  mjx_env.ensure_menagerie_exists()

  xml_dir = Path(robot_path).parent
  # Menagerie LEAP hand assets.
  mjx_env.update_assets(
      assets, mjx_env.MENAGERIE_PATH / "leap_hand" / "assets"
  )
  # Local XMLs, textures and meshes.
  mjx_env.update_assets(assets, xml_dir, "*.xml")
  mjx_env.update_assets(assets, xml_dir, "*.stl")  # XArm link meshes are STL files.
  mjx_env.update_assets(assets, xml_dir / "reorientation_cube_textures")
  mjx_env.update_assets(assets, xml_dir / "meshes")
  return assets


def load_callback(model=None, data=None):
  del model, data  # Unused by viewer loader.
  xml_path = epath.Path(robot_path)
  robot_xml_text = xml_path.read_text()
  assets = build_assets()
  # Make robot XML available to <include file="..."> by name.
  robot_basename = Path(robot_path).name
  assets[robot_basename] = robot_xml_text.encode("utf-8")

  # Minimal wrapper scene with floor and light, including the robot.
  wrapper_xml = f"""
<mujoco model="leap_xarm_scene">
  <option timestep="0.002" integrator="Euler" gravity="0 0 0" />
  <visual>
    <quality shadowsize="4096"/>
  </visual>
  <include file="{robot_basename}"/>
  <worldbody>
    <light name="top" pos="0 0 3" dir="0 0 -1" diffuse="1 1 1" specular="0.1 0.1 0.1"/>
  </worldbody>
</mujoco>
""".strip()

  mj_model = mujoco.MjModel.from_xml_string(wrapper_xml, assets=assets)
  mj_data = mujoco.MjData(mj_model)
  return mj_model, mj_data


if __name__ == "__main__":
  viewer.launch(loader=load_callback)
