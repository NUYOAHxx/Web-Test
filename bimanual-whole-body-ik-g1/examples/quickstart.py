"""Quickstart: solve one bimanual object-carrying pose in simulation."""
from bimanual_wbik import GraspPair, WholeBodyIK, default_robot
from bimanual_wbik.math3d import transform

robot = default_robot()
q = robot.nominal.copy()  # 可替换为硬件测得的 G1 当前关节状态
solver = WholeBodyIK(robot)
# In hardware, use the calibrated object pose and measured grasp transforms.
object_home = GraspPair.infer_object_pose(robot, q)
grasps = GraspPair.from_current(robot, q)
target = object_home @ transform(xyz=(0.04, 0.02, 0.02), rpy=(0.0, 0.0, 0.04))
result = solver.solve_object_carry(q, target, grasps)
print(result.as_dict(robot.names))
if not result.success:
    raise SystemExit("IK did not reach the target within tolerance")
