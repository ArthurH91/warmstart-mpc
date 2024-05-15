import pinocchio as pin
import sys
from os.path import dirname, join, abspath
import example_robot_data as robex
import ompl
from pinocchio.visualize import MeshcatVisualizer
import matplotlib.pyplot as plt

robot = robex.load("ur5")
model = robot.model
collision_model = robot.collision_model
visual_model = robot.visual_model


 
viz = MeshcatVisualizer(model, collision_model, visual_model)
 
# Start a new MeshCat server and client.
# Note: the server can also be started separately using the "meshcat-server" command in a terminal:
# this enables the server to remain active after the current script ends.
#
# Option open=True pens the visualizer.
# Note: the visualizer can also be opened seperately by visiting the provided URL.
try:
    viz.initViewer(open=True)
except ImportError as err:
    print(
        "Error while initializing the viewer. It seems you should install Python meshcat"
    )
    print(err)
    sys.exit(0)
 
# Load the robot in the viewer.
viz.loadViewerModel()
 
# Display a robot configuration.
q0 = pin.neutral(model)
viz.display(q0)
viz.displayVisuals(True)

input()