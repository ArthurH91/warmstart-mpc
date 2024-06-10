import matplotlib.pylab as plt
import numpy as np
from wrapper_meshcat import MeshcatWrapper
from wrapper_panda import PandaWrapper
import pinocchio as pin
a = np.load("/home/arthur/Desktop/Code/warmstart-mpc/example/results_ball_5000.npy", allow_pickle=True)
import random
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
# q = np.array([ 1.06747267,  1.44892299, -0.10145964 ,-2.42389347 , 2.60903241  ,3.45138352,
# -2.04166928])
# vis[0].display(q)

# pin.computeCollisions(rmodel, rdata, cmodel, cdata, q, False)
# for k in range(len(cmodel.collisionPairs)):
#     cr = cdata.collisionResults[k]
#     cp = cmodel.collisionPairs[k]
#     print(
#         "collision pair:",
#         cmodel.geometryObjects[cp.first].name,
#         ",",
#         cmodel.geometryObjects[cp.second].name,
#         "- collision:",
#         "Yes" if cr.isCollision() else "No",
#     )
# q = pin.randomConfiguration(rmodel)
# vis[0].display(pin.randomConfiguration(rmodel))
