# Assessment-2-Robotics-Huy-Truong-And-Zixun-Guo
Robot Café Drink Line

A Robotics Toolbox for Python and Swift simulation of three different 6-axis arms running an automated café. A Doosan M0609 takes a cup from the rack and fills it at the coffee machine, a UR3e presses on the lid, and a Yaskawa GP7 places the finished drink in a free pickup slot. The arms work as a pipeline.

A simulated laser scanner tracks a customer walking up to the counter. Each arm runs normally, slows to 35% speed, or stops, depending on how close the customer is, and resumes after they have stayed clear for 1.5 s. Amber and red floor zones show where slowing and stopping happen.
