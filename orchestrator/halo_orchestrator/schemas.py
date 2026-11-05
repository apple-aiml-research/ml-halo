#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Guided-JSON schemas for the mission orchestrator (forces schema-valid model output)."""
from __future__ import annotations

_MAX_STEPS = 8


def _step_item(skill_names) -> dict:
    """One plan step, shared by t1_plan_schema and agent_action_schema (set_plan)."""
    return {
        "type": "object",
        "properties": {
            "id":    {"type": "string", "maxLength": 32},
            "skill": {"type": "string", "enum": list(skill_names)},
        },
        "required": ["id", "skill"],
        "additionalProperties": False,
    }


def t1_plan_schema(skill_names) -> dict:
    """T1 plan-authoring schema: `reply` then compact `steps` (id, skill).

    Field order matters (guided JSON decodes in order): committing to `reply` conditions the plan.
    Length caps bound token cost; `skill` is a registry enum.
    """
    return {
        "type": "object",
        "properties": {
            # Required on every request; the only output for a conversational one (empty steps).
            "reply": {"type": "string", "maxLength": 400},
            # minItems 0: a conversational request must be able to produce no motion plan.
            "steps": {"type": "array", "minItems": 0, "maxItems": _MAX_STEPS,
                      "items": _step_item(skill_names)},
        },
        "required": ["reply", "steps"],
        "additionalProperties": False,
    }


def agent_action_schema(skill_names) -> dict:
    """Single-agent action schema: pick ONE tool in `action` and fill only its argument field(s).

      * set_plan       -> `steps` (same shape as t1_plan_schema.steps)
      * complete_step  -> `success` (false = finished but failed)
      * retry_step     -> re-run the current step
      * insert_step    -> `skill` (run first, then resume the current step)
      * ask_operator   -> `question`
      * finish_mission -> whole task done
    """
    skill_names = list(skill_names)
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string",
                       "enum": ["set_plan", "complete_step", "retry_step", "insert_step",
                                "ask_operator", "finish_mission"]},
            "steps": {"type": "array", "minItems": 0, "maxItems": _MAX_STEPS,   # set_plan
                      "items": _step_item(skill_names)},
            "success":  {"type": "boolean"},                       # complete_step
            "skill":    {"type": "string", "enum": skill_names},   # insert_step
            "question": {"type": "string", "maxLength": 160},      # ask_operator
            "say":      {"type": "string", "maxLength": 120},
        },
        "required": ["action", "say"],
        "additionalProperties": False,
    }
