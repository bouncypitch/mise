"""Explicit, externalized world state -- the agent's memory of the mission.

Not LLM conversation history: a plain JSON-serializable dict, checkpointed
to disk after every action. This is the "H2-flavored" piece of the
architecture -- state that could in principle be resumed from disk, even
though this demo doesn't exercise an actual crash/resume.
"""
import json
import os

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "state_checkpoint.json")


def create_initial_state(mission_goal, ingredient_names):
    return {
        "mission_goal": mission_goal,
        "plan_version": 1,
        "plan_history": [{"version": 1, "note": "initial plan"}],
        "objects": {
            name: {"location": "table", "status": "pending", "attempts": 0}
            for name in ingredient_names
        },
        "completed_tasks": [],
        "remaining_tasks": [
            task
            for name in ingredient_names
            for task in ({"action": "PICK", "object": name}, {"action": "PLACE", "object": name})
        ],
        "failures": [],
        "event_log": [],
        "actions_taken": 0,
        "failed_attempts": 0,
        "replans": 0,
    }


def log_event(state, message):
    state["event_log"].append({"t": len(state["event_log"]), "msg": message})


def record_action_result(state, action, object_name, verification, drift_type=None):
    state["actions_taken"] += 1
    task = {"action": action, "object": object_name}
    if task in state["remaining_tasks"]:
        state["remaining_tasks"].remove(task)

    if verification["success"]:
        state["completed_tasks"].append(task)
        if action == "PICK":
            state["objects"][object_name]["location"] = "gripper"
        elif action == "PLACE":
            state["objects"][object_name]["location"] = "prep_tray"
            state["objects"][object_name]["status"] = "kitted"
    else:
        state["failed_attempts"] += 1
        state["objects"][object_name]["attempts"] += 1
        state["failures"].append({
            "step": state["actions_taken"],
            "object": object_name,
            "action": action,
            "failure_type": verification["failure_type"],
            "expected": verification["expected"],
            "observed": verification["observed"],
            "drift_type": drift_type,
        })
        # not resolved yet -- put the task back so it can be retried after replan
        state["remaining_tasks"].insert(0, task)

    state["objects"][object_name]["last_attempts"] = state["objects"][object_name]["attempts"]
    return state


def bump_plan_version(state, note):
    state["plan_version"] += 1
    state["replans"] += 1
    state["plan_history"].append({"version": state["plan_version"], "note": note})


def hazard_rate(state):
    if state["actions_taken"] == 0:
        return 0.0
    return round(state["failed_attempts"] / state["actions_taken"], 3)


def kitted_count(state):
    return sum(1 for o in state["objects"].values() if o["status"] == "kitted")


def mission_complete(state, total_ingredients):
    return kitted_count(state) == total_ingredients and len(state["remaining_tasks"]) == 0


def checkpoint(state, path=CHECKPOINT_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)
