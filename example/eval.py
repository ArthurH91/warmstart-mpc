import os
import numpy as np
import torch


from model import Net


from wrapper_meshcat import MeshcatWrapper
from wrapper_panda import PandaWrapper
from scenes import Scene


class Eval:
    "Evaluation class that takes the two configurations in input and returns a trajectory linking both of the configurations."

    def __init__(self, model_path: str, data_path: str) -> None:
        """Instantiate the class that takes the two configurations in input and returns a trajectory linking both of the configurations.

        Args:
            model_path (str): path of the trained model.
            data_path (str): path of the generated data.

        Raises:
            NameError: Wrong model path.
        """
        data = np.load(
            data_path,
            allow_pickle=True,
        )
        
        self._T = len(data[0, 2])
        self._nq = len(data[0, 0])
        self._net = Net(self._nq, self._T)
        
        # Load the model state
        if os.path.exists(model_path):
            self._net.load_state_dict(torch.load(model_path))
            print("Model loaded successfully.")
        else:
            raise NameError("Model file does not exist.")
               
        # Set the model to evaluation mode
        self._net.eval()
        
    def generate_trajectory(self, q_init, q_goal):
        inputs = np.concatenate((q_init, q_goal))
        with torch.no_grad():
            output = self._net(torch.tensor(inputs, dtype=torch.float32))
        output = output.numpy()[0]
        x = [np.concatenate((output[k], np.zeros(self._nq))) for k in range(len(output))]
        return x
        
        
if __name__ == "__main__":

    model_path = "/home/arthur/Desktop/Code/warmstart-mpc/example/models/trained_model_box_6000.pth"
    data_path = "/home/arthur/Desktop/Code/warmstart-mpc/example/results/results_box_6000.npy"

    eval = Eval(model_path, data_path)
    
    ### PARAMETERS
    # Number of nodes of the trajectory
    T = 20
    # Time step between each node
    dt = 0.01

    # Creating the robot
    robot_wrapper = PandaWrapper(auto_col=True, capsule=True)
    rmodel, cmodel, vmodel = robot_wrapper()

    # Creating the scene
    scene = Scene("box")
    rmodel, cmodel, TARGET_POSE1, TARGET_POSE2, q_init = scene.create_scene_from_urdf(
        rmodel, cmodel
    )
    # Generating the meshcat visualizer
    MeshcatVis = MeshcatWrapper()
    vis, meshcatVis = MeshcatVis.visualize(
        TARGET_POSE2,
        robot_model=rmodel,
        robot_collision_model=cmodel,
        robot_visual_model=vmodel,
    )

    print("q0")
    vis.display(q_init)
    print("q_goal")
    input()
    q_goal =np.array([-0.46356800079345684, 0.2820479965209959, 0.05794600009918227, -2.351319994330406, -0.794600009918227, 2.0937000320106747, 0.0]
    )
    vis.display(q_goal)
    
    output = eval.generate_trajectory(q_init, q_goal)

    print("visualisation of the input given trajectory")

    input()
    while True:
        print("visualisation of the NN given trajectory")
        vis.display(q_init)
        input()
        for xs in output:
            vis.display(xs)
            input()
        print("replay")
