Robot Café Drink Line

A Robotics Toolbox for Python and Swift simulation of three different 6-axis arms running an automated café. A Doosan M0609 takes a cup from the rack and fills it at the coffee machine, a UR3e presses a lid onto it and passes it across, and a Yaskawa GP7 places the finished drink in a free pickup slot. The arms work as a pipeline, so several drinks are in progress at once. One full run of three drinks takes about 57 seconds of simulated time.

A simulated 2-D laser scanner tracks a customer walking up to the counter. Each arm runs normally, slows to 35% speed when the customer is within 0.80 m, and stops within 0.30 m. It resumes 1.5 s after the customer has stayed clear. Amber and red floor zones show where slowing and stopping happen, and each arm has a status lamp.

A Workcell Control window (a separate desktop window) lets you start, pause, resume and stop the café, adjust the speed, and trigger an emergency stop with Esc. You can also take any one arm into manual mode and jog its joints or its tool point in world X, Y and Z, while the other arms keep working.

A separate scanner demo (scanner_demo.py) shows the laser sensor on its own, with a live top view and range plot. You can move the customer by hand or let them walk in automatically.

The Doosan M0609 and Yaskawa GP7 models were built from the manufacturers’ URDF files (BSD-3-Clause meshes). The UR3e comes from the course’s ir_support package.
