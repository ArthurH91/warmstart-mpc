import numpy as np
import hppfcl
import pinocchio as pin

import meshcat
import meshcat.geometry as g



RED = np.array([249, 136, 126, 125]) / 255
RED_FULL = np.array([249, 136, 126, 255]) / 255

GREEN = np.array([170, 236, 149, 125]) / 255
GREEN_FULL = np.array([170, 236, 149, 255]) / 255

BLUE = np.array([144, 169, 183, 125]) / 255
BLUE_FULL = np.array([144, 169, 183, 255]) / 255

YELLOW = np.array([1, 1, 0, 0.5])
YELLOW_FULL = np.array([1, 1, 0, 1.])

BLACK = np.array([0, 0, 0, 0.5])
BLACK_FULL = np.array([0, 0, 0, 1.])

def get_transform(T_: hppfcl.Transform3f):
    """Returns a np.ndarray instead of a pin.SE3 or a hppfcl.Transform3f

    Args:
        T_ (hppfcl.Transform3f): transformation to change into a np.ndarray. Can be a pin.SE3 as well

    Raises:
        NotADirectoryError: _description_

    Returns:
        _type_: _description_
    """
    T = np.eye(4)
    if isinstance(T_, hppfcl.Transform3f):
        T[:3, :3] = T_.getRotation()
        T[:3, 3] = T_.getTranslation()
    elif isinstance(T_, pin.SE3):
        T[:3, :3] = T_.rotation
        T[:3, 3] = T_.translation
    else:
        raise NotADirectoryError
    return T

class MeshcatWrapper:
    """Wrapper displaying a robot and a target in a meshcat server."""

    def __init__(self, grid=False, axes=False):
        """Wrapper displaying a robot and a target in a meshcat server.

        Parameters
        ----------
        grid : bool, optional
            Boolean describing whether the grid will be displayed or not, by default False
        axes : bool, optional
            Boolean describing whether the axes will be displayed or not, by default False
        """

        self._grid = grid
        self._axes = axes

    def visualize(
        self,
        TARGET=None,
        OBSTACLE=None,
        RADII_TARGET=5e-2,
        OBSTACLE_DIM=([5e-1, 5e-1, 5e-2]),
        obstacle_type="sphere",
        robot_model=None,
        robot_collision_model=None,
        robot_visual_model=None,
    ):
        """Returns the visualiser, displaying the robot and the target if they are in input.

        Parameters
        ----------
        TARGET : pin.SE3
            pin.SE3 describing the position of the target
        RADII_TARGET : float, optional
            radii of the target which is a ball, by default 5e-2
        robot : robot, optional
            robot from example robot data for instance, by default None

        Returns
        -------
        vis : MeshcatVisualizer
            visualizer from Meshcat
        """

        # Creation of the visualizer,
        self.viewer = self.create_visualizer()

        if TARGET is not None:
            self._renderSphere("target", dim = RADII_TARGET, pose = TARGET)

        if OBSTACLE is not None:
            if type(OBSTACLE) == tuple:
                number_obstacle = len(OBSTACLE)
                # Creating the obstacle
                for k in range(number_obstacle):
                    if obstacle_type == "box":
                        self._renderBox("obstacle" + str(k), OBSTACLE_DIM[k], OBSTACLE[k])
                    if obstacle_type == "sphere":
                        self._renderSphere("obstacle" + str(k), OBSTACLE_DIM[k], OBSTACLE[k])
            else:
                if obstacle_type == "box":
                        self._renderBox("obstacle", OBSTACLE_DIM, OBSTACLE)
                if obstacle_type == "sphere":
                        self._renderSphere("obstacle", OBSTACLE_DIM, OBSTACLE)
        elif (
            robot_model is not None
            and robot_collision_model is not None
            and robot_visual_model is not None
        ):
            self._rmodel = robot_model
            self._rcmodel = robot_collision_model
            self._rvmodel = robot_visual_model

        Viewer = pin.visualize.MeshcatVisualizer

        if (
            robot_model is not None
            and robot_collision_model is not None
            and robot_visual_model is not None
        ):
            self.viewer_pin = Viewer(
                robot_model, robot_collision_model, robot_visual_model
            )
        self.viewer_pin.initViewer(
            viewer=meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
        )

        self.viewer_pin.loadViewerModel()
        self.viewer_pin.displayCollisions(True)

        return self.viewer_pin, self.viewer

    def create_visualizer(self):
        """Creation of an empty visualizer.

        Returns
        -------
        vis : Meshcat.Visualizer
            visualizer from meshcat
        """
        self.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
        self.viewer.delete()
        if not self._grid:
            self.viewer["/Grid"].set_property("visible", False)
        if not self._axes:
            self.viewer["/Axes"].set_property("visible", False)
        return self.viewer

    def _renderSphere(self, e_name: str, dim: np.ndarray, pose : pin.SE3, color=GREEN):
        """Displaying a sphere in a meshcat server.

        Parameters
        ----------
        e_name : str
            name of the object displayed
        color : np.ndarray, optional
            array describing the color of the target, by default np.array([1., 1., 1., 1.]) (ie white)
        """
        # Setting the object in the viewer
        self.viewer[e_name].set_object(
            g.Sphere(dim), self._meshcat_material(*color)
        )
        T = get_transform(pose)

        # Applying the transformation to the object
        self.viewer[e_name].set_transform(T)

    def _renderBox(self, e_name: str, dim : np.array, pose : pin.SE3, color=YELLOW):
        """Displaying a sphere in a meshcat server.

        Parameters
        ----------
        e_name : str
            name of the object displayed
        color : np.ndarray, optional
            array describing the color of the target, by default np.array([1., 1., 1., 1.]) (ie white)
        """
        # Setting the object in the viewer
        self.viewer[e_name].set_object(
            g.Box(dim), self._meshcat_material(*color)
        )

        # Obtaining its position in the right format
        T = get_transform(pose)

        # Applying the transformation to the object
        self.viewer[e_name].set_transform(T)

    def _meshcat_material(self, r, g, b, a):
        """Converting RGBA color to meshcat material.

        Parameters
        ----------
        r : _type_
            _description_
        g : _type_
            _description_
        b : _type_
            _description_
        a : _type_
            _description_

        Returns
        -------
        material : meshcat.geometry.MeshPhongMaterial()
            material for meshcat
        """
        material = meshcat.geometry.MeshPhongMaterial()
        material.color = int(r * 255) * 256**2 + int(g * 255) * 256 + int(b * 255)
        material.opacity = a
        return material

    def applyConfiguration(self, name, placement):
        if isinstance(placement, list) or isinstance(placement, tuple):
            placement = np.array(placement)
        if isinstance(placement, pin.SE3):
            R, p = placement.rotation, placement.translation
            T = np.r_[np.c_[R, p], [[0, 0, 0, 1]]]
        elif isinstance(placement, np.ndarray):
            if placement.shape == (7,):  # XYZ-quat
                R = pin.Quaternion(np.reshape(placement[3:], [4, 1])).matrix()
                p = placement[:3]
                T = np.r_[np.c_[R, p], [[0, 0, 0, 1]]]
            else:
                print("Error, np.shape of placement is not accepted")
                return False
        else:
            print("Error format of placement is not accepted")
            return False
        self.viewer[name].set_transform(T)


if __name__ == "__main__":
    from wrapper_panda import PandaWrapper

    # Creating the robot
    robot_wrapper = PandaWrapper()
    rmodel, cmodel, vmodel = robot_wrapper()
    rdata = rmodel.createData()
    cdata = cmodel.createData()
        # Generating the meshcat visualizer
    MeshcatVis = MeshcatWrapper()
    vis = MeshcatVis.visualize(robot_model=rmodel, robot_visual_model=vmodel, robot_collision_model=cmodel)
    vis[0].display(pin.neutral(rmodel))