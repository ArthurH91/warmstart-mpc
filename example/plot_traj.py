import random
import numpy as np
from wrapper_meshcat import MeshcatWrapper
from wrapper_panda import PandaWrapper


a = np.load("/home/arthur/Desktop/Code/warmstart-mpc/results_wall_10.npy", allow_pickle=True)
print(a[1])

q0 = a[1][0]
qgoal = a[1][1]
Q = a[random.randint(0,len(a))][2]

# # Creating the robot
robot_wrapper = PandaWrapper(capsule=True, auto_col=True)
rmodel, cmodel, vmodel = robot_wrapper()
rdata = rmodel.createData()
cdata = cmodel.createData()
# Generating the meshcat visualizer
MeshcatVis = MeshcatWrapper()
vis = MeshcatVis.visualize(
    robot_model=rmodel, robot_visual_model=cmodel, robot_collision_model=cmodel
)

for q in Q:    
    vis[0].display(q)
    input()
