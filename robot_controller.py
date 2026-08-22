"""Deterministic PyBullet robot controller: IK, joint interpolation, gripper,
constraint-based grasping, and the high-level PICK / PLACE / MOVE_TO /
MOVE_OBJECT primitives.

No LLM ever touches this file. Every method here is plain, reproducible
Python + PyBullet calls -- grasp success/failure is a controller-logic
decision (did we attach a constraint or not), never a physics outcome, so
the same script produces the same result every run.
"""
import math

import pybullet as p

ARM_JOINTS = [0, 1, 2, 3, 4, 5, 6]
FINGER_JOINTS = [9, 10]
EE_LINK = 11  # panda_grasptarget -- the point between the fingertips

HOME_POSE = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

# Explicit joint limits/ranges/rest-pose for the IK null-space solver --
# without these, calculateInverseKinematics's DLS solver can converge to a
# poor local minimum (barely moving from the seed pose) even for reachable
# targets. Values from the URDF's own joint limits; rest pose = HOME_POSE.
IK_LOWER_LIMITS = [-2.967, -1.833, -2.967, -3.142, -2.967, -0.087, -2.967]
IK_UPPER_LIMITS = [2.967, 1.833, 2.967, 0.0, 2.967, 3.822, 2.967]
IK_JOINT_RANGES = [u - l for l, u in zip(IK_LOWER_LIMITS, IK_UPPER_LIMITS)]
IK_REST_POSES = HOME_POSE

# Each finger's joint range is 0-0.04m, so max physical opening is 0.08m.
# Ingredients are up to ~0.064m wide -- OPEN_WIDTH must use the true max or
# the fingers still overlap/pinch the object even when "open", dragging it
# along by contact penetration after release (confirmed empirically).
OPEN_WIDTH = 0.08
CLOSED_WIDTH = 0.0

DOWN_ORN = p.getQuaternionFromEuler([math.pi, 0, 0])


def _rotated_orn(yaw_rad):
    return p.getQuaternionFromEuler([math.pi, 0, yaw_rad])


class RobotController:
    def __init__(self, sim):
        self.sim = sim
        self.robot_id = sim.robot_id
        self.grasp_constraint_id = None
        self.grasped_object = None
        self.frame_hook = None  # optional callable(), invoked periodically during motion (e.g. for video recording)
        self.move_to_joints(HOME_POSE, steps=150)
        self.set_gripper(OPEN_WIDTH, steps=15)

    # -- low level -----------------------------------------------------

    def _current_arm_joints(self):
        return [p.getJointState(self.robot_id, j)[0] for j in ARM_JOINTS]

    def move_to_joints(self, target_joint_positions, steps=150, waypoints=15):
        # PyBullet's default POSITION_CONTROL gain (kp=0.1) can't track a
        # target that moves every single timestep -- it perpetually lags.
        # Instead: hold each of a handful of interpolated waypoints for
        # several physics steps (with an explicit, stronger gain) so the
        # PD controller actually converges before advancing, while still
        # producing visibly smooth intermediate motion for the camera feed.
        start = self._current_arm_joints()
        hold = max(1, steps // waypoints)
        for wp in range(1, waypoints + 1):
            t = wp / waypoints
            interp = [s + (g - s) * t for s, g in zip(start, target_joint_positions)]
            for _ in range(hold):
                p.setJointMotorControlArray(
                    self.robot_id, ARM_JOINTS, p.POSITION_CONTROL,
                    targetPositions=interp, forces=[87] * len(ARM_JOINTS),
                    positionGains=[0.3] * len(ARM_JOINTS),
                    velocityGains=[1.0] * len(ARM_JOINTS),
                )
                self._hold_gripper_and_step()
            if self.frame_hook:
                self.frame_hook()

    def _hold_gripper_and_step(self):
        # keep whatever the finger target currently is while the arm moves
        self.sim.step(1)

    def move_ee_to(self, target_pos, target_orn=DOWN_ORN, steps=60):
        joint_positions = p.calculateInverseKinematics(
            self.robot_id, EE_LINK, target_pos, target_orn,
            lowerLimits=IK_LOWER_LIMITS, upperLimits=IK_UPPER_LIMITS,
            jointRanges=IK_JOINT_RANGES, restPoses=IK_REST_POSES,
            maxNumIterations=200, residualThreshold=1e-4,
        )
        self.move_to_joints(list(joint_positions[:7]), steps=steps)

    def set_gripper(self, width, steps=20):
        half = width / 2
        for _ in range(steps):
            p.setJointMotorControlArray(
                self.robot_id, FINGER_JOINTS, p.POSITION_CONTROL,
                targetPositions=[half, half], forces=[20, 20],
            )
            self.sim.step(1)

    def ee_position(self):
        state = p.getLinkState(self.robot_id, EE_LINK)
        return state[4]

    # -- grasp mechanics -------------------------------------------------

    def _attach(self, object_id):
        ee_pos, ee_orn = p.getLinkState(self.robot_id, EE_LINK)[4:6]
        obj_pos, obj_orn = p.getBasePositionAndOrientation(object_id)
        inv_ee_pos, inv_ee_orn = p.invertTransform(ee_pos, ee_orn)
        rel_pos, rel_orn = p.multiplyTransforms(inv_ee_pos, inv_ee_orn, obj_pos, obj_orn)
        cid = p.createConstraint(
            parentBodyUniqueId=self.robot_id, parentLinkIndex=EE_LINK,
            childBodyUniqueId=object_id, childLinkIndex=-1,
            jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
            parentFramePosition=rel_pos, parentFrameOrientation=rel_orn,
            childFramePosition=[0, 0, 0], childFrameOrientation=[0, 0, 0, 1],
        )
        self.grasp_constraint_id = cid
        self.grasped_object = object_id

    def _release(self):
        if self.grasp_constraint_id is not None:
            p.removeConstraint(self.grasp_constraint_id)
        self.grasp_constraint_id = None
        self.grasped_object = None
        # Snap the fingers open instantly (not via motor control) so there
        # is zero frame where the released object is still pinched between
        # closing/opening fingers -- otherwise friction drags it along with
        # the very next arm move even though the constraint is gone.
        for j in FINGER_JOINTS:
            p.resetJointState(self.robot_id, j, OPEN_WIDTH / 2)

    # -- high level actions ----------------------------------------------

    def pick(self, object_id, object_pos, strategy):
        """strategy is a dict: {approach: 'top_down'|'top_down_corrected',
        pre_grasp_offset: [dx,dy,dz], grasp_offset: [dx,dy,dz],
        wrist_yaw: float, attach: bool}. `attach=False` deliberately seeds
        a missed grasp: fingers close on empty air, nothing is attached."""
        orn = _rotated_orn(strategy.get("wrist_yaw", 0.0))
        pre = [object_pos[i] + strategy["pre_grasp_offset"][i] for i in range(3)]
        grasp = [object_pos[i] + strategy["grasp_offset"][i] for i in range(3)]

        self.set_gripper(OPEN_WIDTH, steps=15)
        self.move_ee_to(pre, orn, steps=strategy.get("steps", 150))
        self.move_ee_to(grasp, orn, steps=strategy.get("steps", 150))
        self.set_gripper(CLOSED_WIDTH, steps=25)

        if strategy.get("attach", True):
            self._attach(object_id)

        lift = [grasp[0], grasp[1], grasp[2] + 0.15]
        self.move_ee_to(lift, orn, steps=100)
        self.sim.step(20)

        return {
            "ee_position_after_lift": self.ee_position(),
            "attached": self.grasped_object == object_id,
        }

    def place(self, tray_slot_pos):
        orn = DOWN_ORN
        pre = [tray_slot_pos[0], tray_slot_pos[1], tray_slot_pos[2] + 0.15]
        down = [tray_slot_pos[0], tray_slot_pos[1], tray_slot_pos[2] + 0.03]

        self.move_ee_to(pre, orn, steps=150)
        self.move_ee_to(down, orn, steps=150)
        self._release()
        self.set_gripper(OPEN_WIDTH, steps=15)
        retreat = [down[0], down[1], down[2] + 0.15]
        self.move_ee_to(retreat, orn, steps=100)
        self.sim.step(40)  # let the object settle
        self.move_to_joints(HOME_POSE, steps=150)

    def move_to(self, position, orn=DOWN_ORN, steps=150):
        self.move_ee_to(position, orn, steps=steps)
