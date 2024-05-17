import time
import numpy as np

import pinocchio as pin
import hppfcl

from wrapper_meshcat import MeshcatWrapper
from wrapper_panda import PandaWrapper
from ocp import OCPPandaReachingColWithMultipleCol

from wrapper_meshcat import BLUE, YELLOW_FULL
### PARAMETERS
# Number of nodes of the trajectory
T = 20
# Time step between each node
dt = 0.01

# Creating the robot
robot_wrapper = PandaWrapper(capsule=False)
rmodel, cmodel, vmodel = robot_wrapper()
rdata = rmodel.createData()
cdata = cmodel.createData()

### CREATING THE TARGET
TARGET_POSE = pin.SE3(pin.utils.rotate("x", np.pi), np.array([0, 0, 1.55]))
TARGET_POSE.translation = np.array([0, -0.4, 1.5])

### CREATING THE OBSTACLE
OBSTACLE_RADIUS = 1.5e-1
OBSTACLE_POSE = pin.SE3.Identity()
OBSTACLE_POSE.translation = np.array([0.25, -0.4, 1.5])
OBSTACLE = hppfcl.Sphere(OBSTACLE_RADIUS)

OBSTACLE_GEOM_OBJECT = pin.GeometryObject(
    "obstacle",
    0,
    0,
    OBSTACLE,
    OBSTACLE_POSE,
)
OBSTACLE_GEOM_OBJECT.meshColor = BLUE

IG_OBSTACLE = cmodel.addGeometryObject(OBSTACLE_GEOM_OBJECT)

### INITIAL CONFIG OF THE ROBOT
INITIAL_CONFIG = pin.neutral(rmodel)

### ADDING THE COLLISION PAIR BETWEEN A LINK OF THE ROBOT & THE OBSTACLE
cmodel.geometryObjects[
    cmodel.getGeometryId("panda2_leftfinger_0")
].meshColor = YELLOW_FULL
cmodel.geometryObjects[
    cmodel.getGeometryId("panda2_link5_sc_4")
].meshColor = YELLOW_FULL

cmodel.addCollisionPair(
    pin.CollisionPair(cmodel.getGeometryId("panda2_leftfinger_0"), IG_OBSTACLE)
)
cmodel.addCollisionPair(
    pin.CollisionPair(cmodel.getGeometryId("panda2_rightfinger_0"), IG_OBSTACLE)
)
cmodel.addCollisionPair(
    pin.CollisionPair(cmodel.getGeometryId("panda2_link5_sc_4"), IG_OBSTACLE)
)
cdata = cmodel.createData()

# Generating the meshcat visualizer
MeshcatVis = MeshcatWrapper()
vis, meshcatVis = MeshcatVis.visualize(
    TARGET_POSE,
    robot_model=rmodel,
    robot_collision_model=cmodel,
    robot_visual_model=vmodel,
)

### INITIAL X0
q0 = INITIAL_CONFIG
x0 = np.concatenate([q0, pin.utils.zero(rmodel.nv)])

### CREATING THE PROBLEM WITHOUT WARM START
problem = OCPPandaReachingColWithMultipleCol(
    rmodel,
    cmodel,
    TARGET_POSE,
    OBSTACLE_POSE,
    OBSTACLE_RADIUS,
    T,
    dt,
    x0,
    WEIGHT_GRIPPER_POSE=100,
    WEIGHT_xREG=1e-2,
    WEIGHT_uREG=1e-4,
    SAFETY_THRESHOLD=2.5e-3,
)
ddp = problem()

XS_init = [x0] * (T+1)
# US_init = [np.zeros(rmodel.nv)] * T
US_init = ddp.problem.quasiStatic(XS_init[:-1])

# Solving the problem
ddp.solve(XS_init, US_init)

print("End of the computation, press enter to display the traj if requested.")
### DISPLAYING THE TRAJ
while True:
    vis.display(INITIAL_CONFIG)
    input()
    for xs in ddp.xs:
        vis.display(np.array(xs[:7].tolist()))
        time.sleep(1e-1)
    input()
    print("replay")