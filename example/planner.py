from typing import Tuple
from os.path import dirname, join, abspath

import numpy as np

from hpp.corbaserver import Client, Robot, ProblemSolver
from hpp.gepetto import ViewerFactory

import pinocchio as pin

from scenes import Scene


class Planner:
    def __init__(
        self, rmodel: pin.Model, cmodel: pin.GeometryModel, scene: Scene, T: int
    ) -> None:
        """Instatiate a motion planning class taking the pinocchio model and the geometry model.

        Args:
            rmodel (pin.Model): pinocchio model of the robot.
            cmodel (pin.GeometryModel): collision model of the robot
            scene (Scene): scene describing the environement.
            T (int): number of nodes describing the trajectory.
        """
        # Models of the robot
        self._rmodel = rmodel
        self._cmodel = cmodel

        # Scene describing the problem
        self._scene = scene
        self._end_effector_id = self._rmodel.getFrameId("panda2_leftfinger")

        # Number of nodes describing the trajectory
        self._T = T

    def _create_planning_scene(self):
        """Creates the planning scene, sets the viewer factory and the problem solver.
        """

        # Registers the urdf paths for the robot and the srdf
        obstacle_urdf, urdf_robot_path, srdf_robot_path = self._get_urdf_srdf_paths()
        Robot.urdfFilename = urdf_robot_path
        Robot.srdfFilename = srdf_robot_path

        # Sets the problem anew
        Client().problem.resetProblem()

        # Creates the robot, the problem solver and the factory
        robot = Robot("panda", rootJointType="anchor")
        self._ps = ProblemSolver(robot)
        vf = ViewerFactory(self._ps)

        # Loads the URDF of the obstacle
        vf.loadObstacleModel(obstacle_urdf, self._scene._name_scene)
        
        # For each obstacle, move the obstacle to its given pose in the collision model
        for obstacle in self._cmodel.geometryObjects:
            if "obstacle" in obstacle.name:
                name = join(self._scene._name_scene, obstacle.name)
                pose = self._scene.obstacle_pose
                pos = self._ps.getObstaclePosition(name)
                pose_obs = pin.SE3ToXYZQUAT(pose)
                for i, p in enumerate(pos[:3]):
                    p += pose_obs[i]
                vf.moveObstacle(name, pos)
        self._v = vf.createViewer(collisionURDF=True)

    def _setup_planner(self):
        """Setting up the planner by create the scene and finding proper q_init and q_goal.
        """
        
        # Setting up the problem solver and the viewer Factory
        self._create_planning_scene()
        
        # col, col1 = True, True
        # while col == True and col1 == True:
            # Generating feasible goal and init pose
        self._q_init = self._generate_feasible_configurations_array()
        self._q_goal = self._generate_feasible_configurations_array()

            # # Creating the data models
            # rdata = self._rmodel.createData()
            # cdata = self._cmodel.createData()
            
            # # Checking whether there is collision for the  
            # col = pin.computeCollisions(
            #     self._rmodel, rdata, self._cmodel, cdata, self._q_init, True
            # )
            # col1 = pin.computeCollisions(
            #     self._rmodel, rdata, self._cmodel, cdata, self._q_goal, True
            # )
        
        # Transforming the np.ndarray of pinocchio into a list with the gripper included for hpp
        q_init_list = self._q_init.tolist() + [0] + [0]
        q_goal_list = self._q_goal.tolist() + [0] + [0]
        self._v(q_init_list)
        
        # Setting up the path planner
        self._ps.selectPathPlanner("BiRRT*")
        self._ps.setMaxIterPathPlanning(100)
        self._ps.setInitialConfig(q_init_list)
        self._ps.addGoalConfig(q_goal_list)

    def solve_and_optimize(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve and optimize the solution found by the planner.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Returns the initial configuration, the goal configuration and the trajectory segmented in T configurations.
        """
        self._setup_planner()
        self._ps.solve()
        self._ps.getAvailable("pathoptimizer")
        self._ps.addPathOptimizer("RandomShortcut")
        self._ps.solve()
        path_length = self._ps.pathLength(2)
        X = [
            self._ps.configAtParam(0, i * path_length / self._T)[:7]
            for i in range(self._T)
        ]
        return self._q_init, self._q_goal, np.array(X)

    def _generate_feasible_configurations_array(self) -> np.ndarray:
        """Genereate a random feasible configuration of the robot. 

        Returns:
            np.ndarray: configuration of the robot
        """
        col = True
        while col:
            q = np.zeros(self._rmodel.nq)
            for i, qi in enumerate(q):
                q[i] = np.random.uniform(self._rmodel.lowerPositionLimit[i], self._rmodel.upperPositionLimit[i], 1)
            col = self._check_collisions(q)
        return q
    
    def _check_collisions(self, q: np.ndarray):
        """Check the collisions for a given configuration array.

        Args:
            q (np.ndarray): configuration array
        Returns:
            col (bool): True if no collision
        """

        rdata = self._rmodel.createData()
        cdata = self._cmodel.createData()
        pin.forwardKinematics(self._rmodel, rdata,q)
        pin.updateGeometryPlacements(self._rmodel, rdata, self._cmodel, cdata, q)
        col = pin.computeCollisions(self._rmodel, rdata, self._cmodel, cdata, q, True)
        return col

    def _inverse_kinematics(
        self, target_pose, initial_guess=None, max_iters=1000, tol=1e-6
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

        rdata = self._rmodel.createData()

        if initial_guess is None:
            q = pin.neutral(
                self._rmodel
            )  # Use the neutral configuration as the initial guess
        else:
            q = initial_guess

        for i in range(max_iters):
            # Compute the current end-effector pose
            pin.forwardKinematics(self._rmodel, rdata, q)
            pin.updateFramePlacements(self._rmodel, rdata)
            current_pose = rdata.oMf[self._end_effector_id]

            # Compute the error between current and target poses
            error = pin.log6(current_pose.inverse() * target_pose).vector
            if np.linalg.norm(error) < tol:
                print(f"Converged in {i} iterations.")
                return q

            # Compute the Jacobian of the end effector
            J = pin.computeFrameJacobian(self._rmodel, rdata, q, self._end_effector_id)

            # Compute the change in joint configuration using the pseudo-inverse of the Jacobian
            dq = np.linalg.pinv(J) @ error

            # Update the joint configuration
            q = pin.integrate(self._rmodel, q, dq)

        raise RuntimeError("Inverse kinematics did not converge")

    def _get_urdf_srdf_paths(self):
        """Return the URDF path of the obstacle and the robot and the SRDF path from the robot.

        Returns:
            tuple: tuple of strings.
        """
        pinocchio_model_dir = dirname(dirname(((str(abspath(__file__))))))
        model_path = join(pinocchio_model_dir, "robot_description")
        self._mesh_dir = join(model_path, "meshes")
        urdf_filename = "franka2.urdf"
        srdf_filename = "demo.srdf"
        urdf_robot_path = join(join(model_path, "urdf"), urdf_filename)
        srdf_robot_path = join(join(model_path, "srdf"), srdf_filename)

        obstacle_urdf = self._scene.urdf_model_path
        return obstacle_urdf, urdf_robot_path, srdf_robot_path
