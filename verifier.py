"""Deterministic MID-RUN VERIFICATION: compares observed PyBullet state
against what the action should have produced. Never trusts the
controller's own self-reported outcome -- always re-derives success from
observed geometry, independently. No LLM involved anywhere in this file.
"""
import math

PICK_LIFT_TOLERANCE = 0.05   # object must be at least this far above the table to count as "picked"
GRIPPER_DIST_TOLERANCE = 0.06  # object must be this close to the gripper to count as "in gripper"
TRAY_XY_MARGIN = 0.02
TRAY_Z_MARGIN = 0.05


def _dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def verify_pick(object_name, ee_pos, obj_pos, table_top_z):
    gripper_ok = _dist(ee_pos, obj_pos) < GRIPPER_DIST_TOLERANCE
    lifted_ok = obj_pos[2] > table_top_z + PICK_LIFT_TOLERANCE
    success = gripper_ok and lifted_ok
    return {
        "action": "PICK",
        "object": object_name,
        "success": success,
        "failure_type": None if success else "PICK_FAILED",
        "expected": {"location": "gripper"},
        "observed": {
            "location": "gripper" if success else ("table" if obj_pos[2] < table_top_z + 0.02 else "unknown"),
            "position": [round(c, 3) for c in obj_pos],
        },
    }


def verify_place(object_name, obj_pos, tray_center_xy, tray_half_extents, tray_top_z):
    in_bounds = (
        abs(obj_pos[0] - tray_center_xy[0]) <= tray_half_extents[0] + TRAY_XY_MARGIN
        and abs(obj_pos[1] - tray_center_xy[1]) <= tray_half_extents[1] + TRAY_XY_MARGIN
    )
    height_ok = (tray_top_z - TRAY_Z_MARGIN) < obj_pos[2] < (tray_top_z + 0.15)
    success = in_bounds and height_ok
    if success:
        observed_location = "prep_tray"
    elif obj_pos[2] < tray_top_z - 0.10:
        observed_location = "floor"
    else:
        observed_location = "table"
    return {
        "action": "PLACE",
        "object": object_name,
        "success": success,
        "failure_type": None if success else "PLACEMENT_FAILED",
        "expected": {"location": "prep_tray"},
        "observed": {"location": observed_location, "position": [round(c, 3) for c in obj_pos]},
    }


def verify_mission(state, robot_base_xy, max_reach, ingredient_positions, tray_center_xy):
    remaining = [name for name, o in state["objects"].items() if o["status"] != "kitted"]
    unreachable = []
    for name in remaining:
        pos = ingredient_positions.get(name)
        if pos is None:
            continue
        d = math.sqrt((pos[0] - robot_base_xy[0]) ** 2 + (pos[1] - robot_base_xy[1]) ** 2)
        if d > max_reach:
            unreachable.append(name)
    return {
        "kitted": len(state["objects"]) - len(remaining),
        "total": len(state["objects"]),
        "remaining": remaining,
        "unreachable": unreachable,
        "tray_accessible": True,  # static scene, no obstacle overlaps the tray footprint
        "on_track": len(unreachable) == 0,
    }
