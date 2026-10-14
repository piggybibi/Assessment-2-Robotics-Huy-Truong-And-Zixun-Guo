YaskawaGP7 - Yaskawa Motoman GP7 (6-axis, 927 mm reach, 7 kg payload) for ir_support

INSTALL
  Unzip so the folder ends up here (the same place the UR3e etc. live):
      <your venv>\Lib\site-packages\ir_support\robots\YaskawaGP7\
  It must contain YaskawaGP7.py, __init__.py and the seven YaskawaGP7Link0..6.dae meshes.

USE
  from ir_support.robots.YaskawaGP7 import YaskawaGP7
  r = YaskawaGP7()                  # q = 0: arm stands up, forearm and flange point +x; r.home_q = tool down
  r.base = SE3(1.0, 0, 0.75) * SE3.Rz(pi)
  r.add_to_env(env); r.q = r.home_q

NOTES
  GP7_DH_NOTES.txt   - how the DH table was derived from the URDF
  gp7_dh_check.py    - optional script that checks the DH model against the URDF kinematics
  The tool z axis points out of the flange.

Meshes: motoman_gp7_support (ROS-Industrial), BSD-3-Clause (see MESH_LICENSE_Motoman_BSD3.txt).
