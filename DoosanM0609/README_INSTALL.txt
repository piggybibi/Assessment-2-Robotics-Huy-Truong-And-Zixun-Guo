DoosanM0609 - Doosan Robotics M0609 (6-axis cobot, 900 mm reach) for ir_support

INSTALL
  Unzip so the folder ends up here (the same place the UR3e etc. live):
      <your venv>\Lib\site-packages\ir_support\robots\DoosanM0609\
  It must contain DoosanM0609.py, __init__.py and the seven DoosanM0609Link0..6.dae meshes.

USE
  from ir_support.robots.DoosanM0609 import DoosanM0609
  r = DoosanM0609()                 # q = 0 is the arm pointing straight up, r.home_q = forearm forward, tool down
  r.base = SE3(0, 0, 0.75)          # base = the surface the robot bolts to
  r.add_to_env(env); r.q = r.home_q

OPTIONAL
  To import it as  from ir_support.robots import DoosanM0609  add
      from .DoosanM0609 import DoosanM0609
  to ir_support\robots\__init__.py (and "DoosanM0609" to __all__).

Meshes: Doosan Robotics dsr_description2, BSD-3-Clause (see MESH_LICENSE_Doosan_BSD3.txt).
