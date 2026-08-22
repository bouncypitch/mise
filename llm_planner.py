"""Goal interpretation, failure diagnosis, and replanning.

This is the ONLY place an LLM (real or mock) ever gets a say. Its output is
always high-level, named-parameter JSON -- an object name, a strategy enum,
an offset -- never joint angles or PyBullet calls. `MockPlanner` is fully
deterministic and is the default (no API key required); `LLMBackedPlanner`
is a guarded, currently-inactive seam for swapping in a real Anthropic call
without touching anything else in the system.
"""
import os

ALL_INGREDIENTS = ["curry_paste", "coconut_milk", "chicken", "basil", "zucchini"]

RECIPE_TEMPLATES = {
    "vegetarian": [n for n in ALL_INGREDIENTS if n != "chicken"],
    "veggie": [n for n in ALL_INGREDIENTS if n != "chicken"],
    "default": list(ALL_INGREDIENTS),
}

# Visibly distinct recovery strategies a diagnosis can recommend. Only the
# offsets/orientation differ from the default top-down grasp -- no new
# control logic, just different named parameters fed to the same
# deterministic controller code path.
STRATEGY_PARAMS = {
    "TOP_DOWN_DEFAULT": {"pre_grasp_offset": [0, 0, 0.15], "grasp_offset": [0, 0, 0.0], "wrist_yaw": 0.0},
    "SIDE_GRASP_LOWER_APPROACH": {"pre_grasp_offset": [0, 0, 0.10], "grasp_offset": [0, 0, -0.01], "wrist_yaw": 1.5708},
    "RETRY_PLACE_CENTERED": {"pre_grasp_offset": [0, 0, 0.15], "grasp_offset": [0, 0, 0.0], "wrist_yaw": 0.0},
}


class Planner:
    def interpret_goal(self, goal_text):
        raise NotImplementedError

    def diagnose_failure(self, verification, telemetry):
        raise NotImplementedError

    def replan(self, world_state, diagnosis, object_name):
        raise NotImplementedError


class MockPlanner(Planner):
    """Deterministic, no API key required. Shallow keyword match -- not
    real NLU -- but genuinely reactive: different goal text produces a
    different mission JSON, which is what sells GOAL -> PLAN in the demo."""

    def interpret_goal(self, goal_text):
        text = (goal_text or "").lower()
        if "vegetarian" in text or "veggie" in text:
            ingredients = RECIPE_TEMPLATES["vegetarian"]
        else:
            ingredients = RECIPE_TEMPLATES["default"]
        return {
            "mission_goal": goal_text,
            "ingredients": ingredients,
            "prep_tray": "prep_tray",
        }

    def diagnose_failure(self, verification, telemetry):
        failure_type = verification["failure_type"]
        mode = telemetry.get("mode", "unknown")
        if failure_type == "PICK_FAILED":
            return {
                "cause": f"grasp offset misaligned ({mode}); fingers closed without contacting the object",
                "drift_type": "behavioral",
                "recommended_strategy": "SIDE_GRASP_LOWER_APPROACH",
            }
        if failure_type == "PLACEMENT_FAILED":
            return {
                "cause": "object released outside tray bounds or fell before settling",
                "drift_type": "behavioral",
                "recommended_strategy": "RETRY_PLACE_CENTERED",
            }
        return {
            "cause": "unclassified verification failure",
            "drift_type": "semantic",
            "recommended_strategy": "TOP_DOWN_DEFAULT",
        }

    def replan(self, world_state, diagnosis, object_name):
        strategy_name = diagnosis["recommended_strategy"]
        params = STRATEGY_PARAMS[strategy_name]
        strategy = {
            "approach": strategy_name,
            "pre_grasp_offset": params["pre_grasp_offset"],
            "grasp_offset": params["grasp_offset"],
            "wrist_yaw": params["wrist_yaw"],
            "attach": True,
            "steps": 150,
        }
        note = (
            f"Attempt {world_state['objects'][object_name]['attempts'] + 1}: "
            f"{strategy_name} (offset {params['grasp_offset']}, wrist_yaw {params['wrist_yaw']:.2f})"
        )
        return strategy, note


class LLMBackedPlanner(Planner):
    """Seam for a real Anthropic call. Inactive unless ANTHROPIC_API_KEY is
    set; on any error (network, parse, schema) falls back to MockPlanner's
    output so a live demo can never crash because the LLM misbehaved."""

    def __init__(self):
        self._mock = MockPlanner()
        try:
            import anthropic  # noqa: F401
            self._client = anthropic.Anthropic()
        except Exception:
            self._client = None

    def _safe(self, fallback_fn, *args, **kwargs):
        if self._client is None:
            return fallback_fn(*args, **kwargs)
        try:
            raise NotImplementedError("LLM call path not exercised in this build")
        except Exception:
            return fallback_fn(*args, **kwargs)

    def interpret_goal(self, goal_text):
        return self._safe(self._mock.interpret_goal, goal_text)

    def diagnose_failure(self, verification, telemetry):
        return self._safe(self._mock.diagnose_failure, verification, telemetry)

    def replan(self, world_state, diagnosis, object_name):
        return self._safe(self._mock.replan, world_state, diagnosis, object_name)


def get_planner():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return LLMBackedPlanner()
    return MockPlanner()
