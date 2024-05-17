import numpy as np
import pinocchio as pin

import hppfcl


class Scene:
    def __init__(self) -> None:
        pass

    def create_scene(self, rmodel: pin.Model, cmodel: pin.Model, name_scene: str):
        """Create a scene amond the ones : "box"

        Args:
            rmodel (pin.Model): robot model
            cmodel (pin.Model): collision model of the robot
            name_scene (str): name of the scene
        """

        self._name_scene = name_scene
        self._cmodel = cmodel
        self._rmodel = rmodel

        if name_scene == "box":
            OBSTACLE_HEIGHT = 0.85
            OBSTACLE_X = 2.0e-1
            OBSTACLE_Y = 0.5e-2
            OBSTACLE_Z = 0.5
            obstacles = [
                (
                    "obstacle1",
                    hppfcl.Box(OBSTACLE_X, OBSTACLE_Y, OBSTACLE_Z),
                    pin.SE3(
                        pin.utils.rotate("y", np.pi / 2),
                        np.array([-0.0, 0.0, OBSTACLE_HEIGHT]),
                    ),
                ),
                (
                    "obstacle2",
                    hppfcl.Box(OBSTACLE_X, OBSTACLE_Y, OBSTACLE_Z),
                    pin.SE3(
                        pin.utils.rotate("y", np.pi / 2),
                        np.array([-0.0, 0.45, OBSTACLE_HEIGHT]),
                    ),
                ),
                (
                    "obstacle3",
                    hppfcl.Box(OBSTACLE_X, OBSTACLE_Y, OBSTACLE_Z),
                    pin.SE3(
                        pin.utils.rotate("y", np.pi / 2)
                        @ pin.utils.rotate("x", np.pi / 2),
                        np.array([0.25, 0.225, OBSTACLE_HEIGHT]),
                    ),
                ),
                (
                    "obstacle4",
                    hppfcl.Box(OBSTACLE_X, OBSTACLE_Y, OBSTACLE_Z),
                    pin.SE3(
                        pin.utils.rotate("y", np.pi / 2)
                        @ pin.utils.rotate("x", np.pi / 2),
                        np.array([-0.25, 0.225, OBSTACLE_HEIGHT]),
                    ),
                ),
            ]
        else:
            raise NotImplementedError(f"The input {name_scene} is not implemented.")
        
        # Adding all the obstacles to the geom model
        for obstacle in obstacles:
            name = obstacle[0]
            shape = obstacle[1]
            pose = obstacle[2]
            geom_obj = pin.GeometryObject(
                name,
                0,
                0,
                shape,
                pose,
            )
            self._cmodel.addGeometryObject(geom_obj)
        self._add_collision_pairs()
        return self._cmodel

    def _add_collision_pairs(self):
        if self._name_scene == "box":
            obstacles = [
                "support_link_0",
                "obstacle1",
                "obstacle2",
                "obstacle3",
                "obstacle4",
            ]
            shapes_avoiding_collision = [
                "panda2_link7_sc_4",
                "panda2_link7_sc_1",
                "panda2_link6_sc_2",
                "panda2_link5_sc_3",
                "panda2_link5_sc_4",
                "panda2_rightfinger_0",
                "panda2_leftfinger_0",
            ]

        for shape in shapes_avoiding_collision:
            for obstacle in obstacles:
                self._cmodel.addCollisionPair(
                    pin.CollisionPair(
                        self._cmodel.getGeometryId(shape),
                        self._cmodel.getGeometryId(obstacle),
                    )
                )


if __name__ == "__main__":
    from wrapper_meshcat import MeshcatWrapper
    from wrapper_panda import PandaWrapper

    # Creating the robot
    robot_wrapper = PandaWrapper(capsule=False, obstacles=None)
    rmodel, cmodel, vmodel = robot_wrapper()

    scene = Scene()
    cmodel = scene.create_scene(rmodel, cmodel, "box")

    rdata = rmodel.createData()
    cdata = cmodel.createData()
    # Generating the meshcat visualizer
    MeshcatVis = MeshcatWrapper()
    vis = MeshcatVis.visualize(
        robot_model=rmodel, robot_visual_model=cmodel, robot_collision_model=cmodel
    )
    # vis[0].display(pin.randomConfiguration(rmodel))
    vis[0].display(np.array([0.5] * 7))

    pin.computeCollisions(rmodel, rdata, cmodel, cdata, pin.neutral(rmodel), False)
    for k in range(len(cmodel.collisionPairs)):
        cr = cdata.collisionResults[k]
        cp = cmodel.collisionPairs[k]
        print(
            "collision pair:",
            cp.first,
            ",",
            cp.second,
            "- collision:",
            "Yes" if cr.isCollision() else "No",
        )
