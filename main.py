"""Mise -- Long-Horizon Physical Agent.

GOAL -> PLAN -> ACT -> OBSERVE -> VERIFY -> DIAGNOSE -> REPLAN -> ACT -> VERIFY -> COMPLETE

Run: python main.py ["your meal goal"]
Dashboard: http://127.0.0.1:5050 (starts automatically unless --no-ui)
"""
import sys
import time

from sim_env import (
    SimEnv, TRAY_TOP_Z, TRAY_CENTER_XY, TRAY_HALF_EXTENTS, TABLE_TOP_Z, ROBOT_BASE_POS,
    INGREDIENTS,
)
from robot_controller import RobotController
import world_model as wm
import verifier as vf
from llm_planner import get_planner

TOP_DOWN_DEFAULT = {
    "approach": "TOP_DOWN_DEFAULT", "pre_grasp_offset": [0, 0, 0.15],
    "grasp_offset": [0, 0, 0.0], "wrist_yaw": 0.0, "attach": True, "steps": 150,
}

# Deterministic, reproducible seeded failure -- see robot_controller.py's
# module docstring for why grasp success is a controller decision, and
# README for why this is honest, demo-legible drift rather than a scripted
# retry: the recovery strategy is genuinely different (side grasp, rotated
# wrist), not the same action repeated.
FAILURE_SCHEDULE = {"basil": {"attempt": 1, "mode": "missed_grasp"}}
BAD_STRATEGY = {
    "approach": "MISSED_GRASP_SEEDED", "pre_grasp_offset": [0.05, 0, 0.15],
    "grasp_offset": [0.05, 0, 0.0], "wrist_yaw": 0.0, "attach": False, "steps": 150,
}

SLOT_OFFSETS = [(-0.07, -0.05), (0.07, -0.05), (-0.07, 0.05), (0.07, 0.05), (0.0, 0.0)]
MAX_REACH = 0.75


def run_mission(goal_text, ui=None, pace_seconds=0.0):
    env = SimEnv(gui=False)
    ctrl = RobotController(env)
    planner = get_planner()

    def push_ui(current_action=None, last_verification=None, status=None):
        if ui:
            _, _, b64 = env.get_camera_frame()
            ui.update_frame(b64)
            ui.update_state(state, current_action, last_verification, status)
            if pace_seconds:
                time.sleep(pace_seconds)

    mission = planner.interpret_goal(goal_text)
    names = mission["ingredients"]
    ingredient_positions = {ing["name"]: ing["pos"] for ing in INGREDIENTS}

    state = wm.create_initial_state(mission["mission_goal"], names)
    wm.log_event(state, "Goal interpreted")
    wm.log_event(state, f"Mission created: {names}")
    wm.checkpoint(state)
    push_ui(current_action="Mission created", status="RUNNING")

    for i, name in enumerate(names):
        obj_id = env.ingredient_ids[name]
        strategy = dict(TOP_DOWN_DEFAULT)

        while True:
            attempt_number = state["objects"][name]["attempts"] + 1
            sched = FAILURE_SCHEDULE.get(name)
            if sched and attempt_number == sched["attempt"]:
                use_strategy = BAD_STRATEGY
                wm.log_event(state, f"(seeded) forcing {sched['mode']} on {name} attempt {attempt_number}")
            else:
                use_strategy = strategy

            push_ui(current_action=f"Pick {name} (attempt {attempt_number}, {use_strategy['approach']})", status="RUNNING")
            pre_pos = env.object_position(obj_id)
            ctrl.pick(obj_id, pre_pos, use_strategy)
            obs_pos = env.object_position(obj_id)
            v = vf.verify_pick(name, ctrl.ee_position(), obs_pos, TABLE_TOP_Z)

            if v["success"]:
                wm.record_action_result(state, "PICK", name, v)
                wm.log_event(state, f"{name} PICK -> PASS (attempt {attempt_number}, strategy {use_strategy['approach']})")
                wm.checkpoint(state)
                push_ui(current_action=f"Picked {name}", last_verification=v, status="RUNNING")
                break

            diagnosis = planner.diagnose_failure(v, {"mode": sched["mode"] if sched else "unknown"})
            wm.record_action_result(state, "PICK", name, v, drift_type=diagnosis["drift_type"])
            wm.log_event(state, f"MID-RUN VERIFICATION FAILURE: {v['failure_type']} on {name}")
            wm.log_event(state, f"Diagnosis: {diagnosis['cause']} [drift_type={diagnosis['drift_type']}]")
            wm.checkpoint(state)
            push_ui(current_action=f"Pick {name} failed", last_verification=v, status="EXECUTION_FAILURE")

            new_strategy, note = planner.replan(state, diagnosis, name)
            wm.bump_plan_version(state, note)
            wm.log_event(state, f"Replanning -> {note}")
            wm.checkpoint(state)
            push_ui(current_action=f"Replanning {name}: {note}", status="REPLANNING")
            strategy = new_strategy

        dx, dy = SLOT_OFFSETS[i]
        slot = [TRAY_CENTER_XY[0] + dx, TRAY_CENTER_XY[1] + dy, TRAY_TOP_Z]
        push_ui(current_action=f"Place {name}", status="RUNNING")
        ctrl.place(slot)
        obs_pos = env.object_position(obj_id)
        v = vf.verify_place(name, obs_pos, TRAY_CENTER_XY, TRAY_HALF_EXTENTS, TRAY_TOP_Z)
        wm.record_action_result(state, "PLACE", name, v)
        wm.log_event(state, f"{name} PLACE -> {'PASS' if v['success'] else v['failure_type']}")
        wm.checkpoint(state)
        push_ui(current_action=f"Placed {name}", last_verification=v, status="RUNNING")

    complete = wm.mission_complete(state, len(names))
    push_ui(current_action="Mission complete", status="VERIFIED" if complete else "EXECUTION_FAILURE")

    mv = vf.verify_mission(state, ROBOT_BASE_POS[:2], MAX_REACH, ingredient_positions, TRAY_CENTER_XY)
    print_final_report(state, names, mv)
    return env, state


def print_final_report(state, names, mission_verify):
    kitted = wm.kitted_count(state)
    hz = wm.hazard_rate(state)
    status = "VERIFIED" if wm.mission_complete(state, len(names)) else "INCOMPLETE"

    print("\n" + "=" * 60)
    print("MISSION COMPLETE" if status == "VERIFIED" else "MISSION INCOMPLETE")
    print(state["mission_goal"])
    print("=" * 60)

    print("\n-- OUTCOME-LEVEL --")
    print(f"Ingredients kitted: {kitted}/{len(names)}")
    print(f"Actions taken: {state['actions_taken']}")
    print(f"Failures recovered: {state['failed_attempts']}")
    print(f"Replans: {state['replans']}")
    print(f"Hazard rate (failed/attempted actions): {hz}")
    print(f"Mission status: {status}")

    print("\n-- TRAJECTORY-LEVEL --")
    print(f"Plan version: {state['plan_version']}")
    for p in state["plan_history"]:
        print(f"  v{p['version']}: {p['note']}")
    print(f"Remaining tasks: {state['remaining_tasks'] or 'none'}")
    print(f"Reachability check: unreachable={mission_verify['unreachable']}  tray_accessible={mission_verify['tray_accessible']}")

    print("\n-- COMPONENT-LEVEL --")
    print(f"{'step':>4}  {'action':<6} {'object':<14} {'result':<16} drift_type")
    for name, o in state["objects"].items():
        pass
    for f in state["failures"]:
        print(f"{f['step']:>4}  {f['action']:<6} {f['object']:<14} {f['failure_type']:<16} {f['drift_type']}")
    for t in state["completed_tasks"]:
        print(f"{'':>4}  {t['action']:<6} {t['object']:<14} {'PASS':<16} -")
    print()


if __name__ == "__main__":
    use_ui = "--no-ui" not in sys.argv
    fast = "--fast" in sys.argv
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    goal = positional[0] if positional else "I want a low-carb Thai green curry"

    ui = None
    if use_ui:
        import ui_server
        ui_server.start_server_in_background()
        ui = ui_server
        print("Dashboard: http://127.0.0.1:5050")
        time.sleep(1.0)

    env, state = run_mission(goal, ui=ui, pace_seconds=0.0 if fast else 2.0)

    if use_ui:
        print("Demo finished. Dashboard still running at http://127.0.0.1:5050 -- Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    env.disconnect()
