import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch

import time 
import pinocchio as pin
from model import Net
from training import NumpyDataset


from wrapper_meshcat import MeshcatWrapper
from wrapper_panda import PandaWrapper
from scenes import Scene

def inverse_kinematics(
        target_pose, initial_guess=None, max_iters=10000, tol=1e-4
    ):
        """
        Solve the inverse kinematics problem for a given robot and target pose.

        Args:
        target_pose (pin.SE3): Desired end-effector pose (as a pin.SE3 object)
        initial_guess (np.ndarray): Initial guess for the joint configuration (optional)
        max_iters (int): Maximum number of iterations
        tol (float): Tolerance for convergence

        Returns:
        q_sol (np.ndarray): Joint configuration that achieves the target pose
        """

        rdata = rmodel.createData()
        end_effector_id = rmodel.getFrameId("panda2_leftfinger")
        if initial_guess is None:
            q = pin.neutral(
                rmodel
            )  # Use the neutral configuration as the initial guess
        else:
            q = initial_guess

        for i in range(max_iters):
            # Compute the current end-effector pose
            pin.forwardKinematics(rmodel, rdata, q)
            pin.updateFramePlacements(rmodel, rdata)
            current_pose = rdata.oMf[end_effector_id]

            # Compute the error between current and target poses
            error = pin.log6(current_pose.inverse() * target_pose).vector
            if np.linalg.norm(error) < tol:
                print(f"Converged in {i} iterations.")
                return q

            # Compute the Jacobian of the end effector
            J = pin.computeFrameJacobian(rmodel, rdata, q, end_effector_id)

            # Compute the change in joint configuration using the pseudo-inverse of the Jacobian
            dq = np.linalg.pinv(J) @ error

            # Update the joint configuration
            q = pin.integrate(rmodel, q, dq)

        raise RuntimeError("Inverse kinematics did not converge")

# Paths to the model and data
model_path = "/home/arthur/Desktop/Code/warmstart-mpc/trained_model_box_6000.pth"

# Load data from .npy file
data = np.load(
    "/home/arthur/Desktop/Code/warmstart-mpc/example/results_box_6000.npy",
    allow_pickle=True,
)
T = len(data[0, 2])
nq = len(data[0, 0])
net = Net(nq, T)

# Load the model state
if os.path.exists(model_path):
    net.load_state_dict(torch.load(model_path))
    print("Model loaded successfully.")
else:
    print("Model file does not exist.")
    exit()

# Create dataset and dataloader
dataset = NumpyDataset(data)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Set the model to evaluation mode
net.eval()

# Initialize lists to store predictions and actual values
predictions = []
actuals = []
diff = []

# Make predictions and compare with actual values
with torch.no_grad():
    for inputs, actual in dataloader:
        output = net(inputs)
        diff.append((output.numpy()[0] - actual.numpy()[0])**2)
        predictions.append(output)
        actuals.append(actual)



### PARAMETERS
# Number of nodes of the trajectory
T = 20
# Time step between each node
dt = 0.01

# Creating the robot
robot_wrapper = PandaWrapper(auto_col=True, capsule=True)
rmodel, cmodel, vmodel = robot_wrapper()

# Creating the scene
scene = Scene("ball")
rmodel, cmodel, TARGET_POSE1, TARGET_POSE2, q0 = scene.create_scene_from_urdf(rmodel, cmodel)
# Generating the meshcat visualizer
# qgoal = inverse_kinematics(TARGET_POSE2)
MeshcatVis = MeshcatWrapper()
vis, meshcatVis = MeshcatVis.visualize(
    TARGET_POSE2,
    robot_model=rmodel,
    robot_collision_model=cmodel,
    robot_visual_model=vmodel,
)

### INITIAL X0
random_int = 95
# q0 = data[random_int][0] 
# qgoal = data[random_int][1]
# input_traj = data[random_int][2]
# inputs = np.concatenate((q0, qgoal))

print("q0")
vis.display(q0)
print("qgoal")
input()
qgoal = np.array([-0.35420209,  0.53709731,  0.93901117, -0.78226258,  0.5047614 ,
        1.31105294,  0.02461606])
vis.display(qgoal)
inputs = np.concatenate((q0, qgoal))
input()

with torch.no_grad():
    output = net(torch.tensor(inputs, dtype=torch.float32))

# print("visualisation of the input given trajectory")


# vis.display(q0)
# input()
# for xs in input_traj:
#     vis.display(np.array(xs[:7].tolist()))
#     input()
while True:
    print("visualisation of the NN given trajectory")
    vis.display(q0)
    input()
    for xs in output.numpy()[0]:
        vis.display(xs)
        input()
    print("replay")
