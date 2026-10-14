"""Bimanual whole-body inverse kinematics for the reduced Unitree G1 model."""
from importlib.resources import files

from .model import Joint, RobotModel, SphereObstacle
from .solver import BalanceRegion, GraspPair, IKResult, WholeBodyIK


def g1_robot() -> RobotModel:
    """Load the fixed-pelvis, 16-DoF Unitree G1 29-DoF rev_1_0 reduction."""
    return RobotModel.from_json(files(__package__).joinpath("default_robot.json"))


def default_robot() -> RobotModel:
    """Load the included G1 model (waist yaw/pitch plus seven joints per arm)."""
    return g1_robot()


__all__ = ["BalanceRegion", "GraspPair", "IKResult", "Joint", "RobotModel",
           "SphereObstacle", "WholeBodyIK", "default_robot", "g1_robot"]
