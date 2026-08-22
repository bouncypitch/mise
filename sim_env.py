"""PyBullet tabletop scene for Mise: robot arm, prep tray, ingredients, obstacles.

Deterministic geometry only (primitive shapes) so the verifier can check
object state against exact, known bounds -- no reliance on mesh collision
inspection or friction-based outcomes.
"""
import base64
import io
import os

import pybullet as p
import pybullet_data
from PIL import Image

TABLE_TOP_Z = 0.62
TABLE_HALF_EXTENTS = [0.55, 0.45, 0.02]
TABLE_CENTER = [0.45, 0.0, TABLE_TOP_Z - TABLE_HALF_EXTENTS[2]]

# Mounted on top of the table (like a real tabletop-mounted arm) -- if the
# base sat at floor level while the table slab occupied the space above it,
# the arm's own links would crash into the table on every reach-forward
# motion (confirmed empirically: contact forces in the tens of thousands
# of Newtons). Basing it at table height keeps the whole workspace above
# the table surface.
ROBOT_BASE_POS = [0.0, 0.0, TABLE_TOP_Z]

TRAY_CENTER_XY = [0.15, -0.30]
TRAY_HALF_EXTENTS = [0.14, 0.11, 0.01]
TRAY_TOP_Z = TABLE_TOP_Z + TRAY_HALF_EXTENTS[2]

OBSTACLES = [
    {"name": "obstacle_1", "pos": [0.55, 0.05, TABLE_TOP_Z + 0.04], "half_extents": [0.02, 0.02, 0.04], "color": [0.3, 0.3, 0.3, 1]},
    {"name": "obstacle_2", "pos": [0.30, 0.28, TABLE_TOP_Z + 0.05], "half_extents": [0.03, 0.03, 0.05], "color": [0.2, 0.2, 0.2, 1]},
]

# Ingredient layout: kept within a ~0.3-0.6m radius of the robot base, spread
# out enough that grasp approaches don't collide with each other.
INGREDIENTS = [
    {"name": "curry_paste",   "shape": "box",      "size": 0.028, "color": [0.55, 0.65, 0.15, 1], "pos": [0.45, -0.18, TABLE_TOP_Z + 0.028]},
    {"name": "coconut_milk",  "shape": "cylinder",  "size": 0.032, "height": 0.09, "color": [0.95, 0.95, 0.90, 1], "pos": [0.55, -0.02, TABLE_TOP_Z + 0.045]},
    {"name": "chicken",       "shape": "box",      "size": 0.032, "color": [0.85, 0.65, 0.55, 1], "pos": [0.50, 0.18, TABLE_TOP_Z + 0.032]},
    {"name": "basil",         "shape": "sphere",    "size": 0.022, "color": [0.15, 0.55, 0.20, 1], "pos": [0.38, 0.05, TABLE_TOP_Z + 0.022]},
    {"name": "zucchini",      "shape": "cylinder",  "size": 0.020, "height": 0.11, "color": [0.30, 0.60, 0.25, 1], "pos": [0.40, 0.30, TABLE_TOP_Z + 0.055]},
]

CAMERA_TARGET = [0.35, 0.0, TABLE_TOP_Z + 0.15]
CAMERA_DISTANCE = 1.5
CAMERA_YAW = 50
CAMERA_PITCH = -30
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


class SimEnv:
    def __init__(self, gui=False):
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setPhysicsEngineParameter(fixedTimeStep=1.0 / 240.0)

        self.plane_id = p.loadURDF("plane.urdf")
        self.table_id = self._make_box(
            TABLE_CENTER, TABLE_HALF_EXTENTS, [0.55, 0.4, 0.3, 1], mass=0.0
        )
        self.tray_id = self._make_box(
            [TRAY_CENTER_XY[0], TRAY_CENTER_XY[1], TABLE_TOP_Z + TRAY_HALF_EXTENTS[2] / 2],
            [TRAY_HALF_EXTENTS[0], TRAY_HALF_EXTENTS[1], TRAY_HALF_EXTENTS[2] / 2],
            [0.75, 0.55, 0.25, 1],
            mass=0.0,
        )

        panda_urdf = os.path.join(pybullet_data.getDataPath(), "franka_panda", "panda.urdf")
        self.robot_id = p.loadURDF(panda_urdf, ROBOT_BASE_POS, useFixedBase=True)
        # The URDF's default all-zero joint pose interpenetrates the table;
        # reset (not motor-drive) straight to a safe retracted pose before
        # any physics stepping so the initial settle loop below can't
        # trigger a contact-resolution explosion.
        for j, pos in zip([0, 1, 2, 3, 4, 5, 6], [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]):
            p.resetJointState(self.robot_id, j, pos)
        for j in [9, 10]:
            p.resetJointState(self.robot_id, j, 0.04 / 2)

        self.obstacle_ids = {}
        for obs in OBSTACLES:
            oid = self._make_box(obs["pos"], obs["half_extents"], obs["color"], mass=0.0)
            self.obstacle_ids[obs["name"]] = oid

        self.ingredient_ids = {}
        for ing in INGREDIENTS:
            if ing["shape"] == "box":
                oid = self._make_box(ing["pos"], [ing["size"]] * 3, ing["color"], mass=0.05)
            elif ing["shape"] == "sphere":
                oid = self._make_sphere(ing["pos"], ing["size"], ing["color"], mass=0.03)
            elif ing["shape"] == "cylinder":
                oid = self._make_cylinder(ing["pos"], ing["size"], ing["height"], ing["color"], mass=0.05)
            else:
                raise ValueError(f"unknown shape {ing['shape']}")
            self.ingredient_ids[ing["name"]] = oid

        for _ in range(60):
            p.stepSimulation()

    def _make_box(self, pos, half_extents, color, mass):
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
        return p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col,
                                  baseVisualShapeIndex=vis, basePosition=pos)

    def _make_sphere(self, pos, radius, color, mass):
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
        return p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col,
                                  baseVisualShapeIndex=vis, basePosition=pos)

    def _make_cylinder(self, pos, radius, height, color, mass):
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
        return p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col,
                                  baseVisualShapeIndex=vis, basePosition=pos)

    def get_camera_frame(self):
        """Render the scene and return (raw_rgb_array, jpeg_bytes, base64_str)."""
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=CAMERA_TARGET,
            distance=CAMERA_DISTANCE,
            yaw=CAMERA_YAW,
            pitch=CAMERA_PITCH,
            roll=0,
            upAxisIndex=2,
        )
        proj = p.computeProjectionMatrixFOV(
            fov=55, aspect=CAMERA_WIDTH / CAMERA_HEIGHT, nearVal=0.05, farVal=3.0
        )
        _, _, rgb, _, _ = p.getCameraImage(
            CAMERA_WIDTH, CAMERA_HEIGHT, view, proj,
            renderer=p.ER_TINY_RENDERER,
        )
        img = Image.fromarray(rgb[:, :, :3].astype("uint8"), "RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        jpeg_bytes = buf.getvalue()
        b64 = base64.b64encode(jpeg_bytes).decode("ascii")
        return rgb, jpeg_bytes, b64

    def object_position(self, body_id):
        pos, _ = p.getBasePositionAndOrientation(body_id)
        return pos

    def step(self, n=1):
        for _ in range(n):
            p.stepSimulation()

    def disconnect(self):
        p.disconnect(self.client)


if __name__ == "__main__":
    env = SimEnv(gui=False)
    print("bodies in scene:", p.getNumBodies())
    print("robot joints:", p.getNumJoints(env.robot_id))
    print("ingredient ids:", env.ingredient_ids)
    print("obstacle ids:", env.obstacle_ids)
    _, jpeg_bytes, b64 = env.get_camera_frame()
    out_path = os.path.join(os.path.dirname(__file__), "checkpoints", "scene_test.jpg")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(jpeg_bytes)
    print("wrote", out_path, "bytes:", len(jpeg_bytes))
    assert len(jpeg_bytes) > 1000, "camera frame looks empty"
    env.disconnect()
    print("M0 DONE")
