import pinocchio as pin
from pinocchio.utils import zero
import numpy as np

def inverse_kinematics(rmodel, end_effector_id, target_pose, initial_guess=None, max_iters=1000, tol=1e-6):
    """
    Solve the inverse kinematics problem for a given robot and target pose.

    :param robot: Pinocchio robot model
    :param end_effector_id: Index of the end effector in the robot model
    :param target_pose: Desired end-effector pose (as a pin.SE3 object)
    :param initial_guess: Initial guess for the joint configuration (optional)
    :param max_iters: Maximum number of iterations
    :param tol: Tolerance for convergence
    :return: Joint configuration that achieves the target pose
    """
    
    rdata = rmodel.createData()
    if initial_guess is None:
        q = pin.neutral(rmodel)  # Use the neutral configuration as the initial guess
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

# Example usage
if __name__ == "__main__":
    from wrapper_panda import PandaWrapper

    # Creating the robot
    robot_wrapper = PandaWrapper(capsule=True)
    rmodel, cmodel, vmodel = robot_wrapper()
    # Define the target pose for the end effector
    target_pose = pin.SE3(np.eye(3), np.array([0.0, 0.0, 1.5]))  # Example target pose

    # Get the end effector ID
    end_effector_id = rmodel.getFrameId("panda2_leftfinger")  # Replace with your end effector link name

    # Solve the IK problem
    try:
        q_sol = inverse_kinematics(rmodel, end_effector_id, target_pose)
        print("Found solution:", q_sol)
    except RuntimeError as e:
        print(e)
    
    rdata = rmodel.createData()
    pin.framesForwardKinematics(rmodel, rdata, q_sol)
    print(rdata.oMf[rmodel.getFrameId("panda2_leftfinger")])