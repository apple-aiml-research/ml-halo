#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Mission-orchestrator prompt templates, rendered from the skill registry (no skill names here)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from . import registry as _registry

# Authored prompt text is folded to ASCII: non-ASCII costs more tokens per character, which can
# overrun the output budget and truncate the JSON.
_TYPOGRAPHIC_ASCII = {
    "\u2014": "--", "\u2013": "-", "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u00a0": " ",
}


def _ascii(text: str) -> str:
    """Fold typographic punctuation to ASCII and drop the rest of non-ASCII.

    Only for text we author; never apply it to the operator's mission string.
    """
    for k, v in _TYPOGRAPHIC_ASCII.items():
        text = text.replace(k, v)
    return text.encode("ascii", "ignore").decode("ascii")


_FUNCTIONS_PATH = Path(__file__).resolve().parent / "functions.json"


def load_functions(reg) -> list:
    """Load functions.json, the call-style overlay on skills.yaml.

    Raises on a function name the registry cannot route.
    """
    p = _FUNCTIONS_PATH
    doc = json.loads(p.read_text())
    fns = [f for f in (doc.get("functions") or []) if isinstance(f, dict) and f.get("name")]
    known = set(_registry.names(reg))
    unknown = [f["name"] for f in fns if f["name"] not in known]
    if unknown:
        raise ValueError(
            f"{p.name} declares function(s) {unknown} that are not skills in skills.yaml "
            f"(known: {sorted(known)}). The name is what the plan is routed by, so a function the "
            f"registry cannot resolve would produce an unexecutable step.")
    return fns


def _functions_block(reg, names) -> str:
    """Render the T1 function catalogue, limited to the skills in `names`."""
    want = list(names)
    wanted = set(want)
    fns = [f for f in load_functions(reg=reg) if f["name"] in wanted]
    covered = {f["name"] for f in fns}
    # Skills without a functions.json entry still get a minimal registry-derived entry.
    by_name = {s.name: s for s in reg}
    extra = [n for n in want if n not in covered and n in by_name]
    if not fns and not extra:
        return ""
    out = ["FUNCTIONS (callable skills; select a SEQUENCE of these):"]
    _tag = (lambda nm: "   [FOLLOW: living/moving target -- a settle does NOT finish it]"
            if _registry.target_kind_of(reg, nm) == "living" else "")
    for f in fns:
        out.append(f"  {f['name']}()" + _tag(f['name']))
        out.append(f"      {str(f.get('description','')).strip()}")
        if f.get("completion_cue"):
            out.append(f"      done when: {str(f['completion_cue']).strip()}")
    for n in extra:
        s = by_name[n]
        out.append(f"  {n}()" + _tag(n))
        pre = " ".join(s.when_applicable.split())
        if pre:
            out.append(f"      {pre}")
        out.append(f"      done when: {_registry.completion_appearance_of(reg, n).strip()}")
    return "\n".join(out)


# The ORCHESTRATOR tier's opening.
_ORCHESTRATOR_ROLE = (
    "You are a humanoid robot with general-purpose whole-body motion capabilities.\n"
    "Your job is to help humans accomplish their requests by:\n"
    "\n"
    "1. UNDERSTANDING the request clearly.\n"
    "2. PLANNING a task list - decompose the request into an ordered sequence\n"
    "   of steps, each mapped to a specific function/skill call where applicable.\n"
    "3. EXECUTING by chaining function calls in the correct order,\n"
    "   passing outputs from one step as inputs to the next.\n"
    "4. REPLYING to the human in ONE short sentence (under ~20 words) saying\n"
    "   what you will do, in plain conversational language. No numbers, distances,\n"
    "   heights or measurements in the reply."
)


_ORCHESTRATOR_JOB = (
    "Rules:\n"
    "- Always reply to the human - never silently execute without explanation.\n"
    "- If a request requires multiple steps, build a numbered task list first,\n"
    "  then execute each step in order.\n"
    "- If only a single step is needed, skip the task list and just do it.\n"
    "- If a request is conversational (no action needed), reply directly\n"
    "  without invoking any function calls.\n"
    "- Never fabricate results - only report what was actually returned\n"
    "  by a tool or function call."
)
# Rules map onto the schema: reply -> `reply`; task list -> `steps`; conversational -> `steps: []`.


def _join(sections) -> str:
    return _ascii("\n\n".join(s for s in sections if s))


def build_t1_system_prompt(subtasks: Sequence[str], reg) -> str:
    """T1 orchestrator system prompt; per-call context lives in the user message."""
    return _join([
        _ORCHESTRATOR_ROLE,
        _ORCHESTRATOR_JOB,
        _functions_block(reg, subtasks),
    ])


_AGENT_JOB = (
    "You act by calling ONE tool per decision. TOOLS:\n"
    "  set_plan(steps)        - author OR re-author the ordered plan (a list of skill steps). Use at\n"
    "                           the start of a task, or when the goal itself has changed.\n"
    "  complete_step(success) - the CURRENT step finished -> advance. success=true if it reached its\n"
    "                           goal, false if it is finished but failed.\n"
    "  retry_step()           - the current step stopped too early and its goal is still reachable\n"
    "                           straight ahead -> run the SAME step again from here.\n"
    "  insert_step(skill)     - the current step is BLOCKED and needs a different motion FIRST (e.g. a\n"
    "                           narrow doorway ahead -> insert_step(\"sidewalk\")). `skill` runs FIRST;\n"
    "                           the blocked step is PAUSED and RESUMES automatically once it finishes.\n"
    "  ask_operator(question) - you are UNCERTAIN (wrong target? which door? stuck?) OR you LOST\n"
    "                           SIGHT of the target you were following -> ASK the human instead of\n"
    "                           guessing or walking blindly. The robot holds until they answer.\n"
    "  finish_mission()       - the WHOLE task is done.\n"
    "When to call which, once the robot has STOPPED on the current step: prefer retry_step over\n"
    "complete_step while the target is still ahead and not yet reached; prefer insert_step when the\n"
    "path is physically blocked and a short different motion clears it; prefer ask_operator over\n"
    "guessing when the scene is ambiguous. TARGET KIND changes what a settle MEANS: a step that goes\n"
    "to a STATIC object (box, table, chair, desk, bin) is FINISHED once the robot has arrived and\n"
    "stopped -> complete_step; a step that FOLLOWS a LIVING target (a person, dog, animal) is NEVER\n"
    "finished on your own judgment -> KEEP FOLLOWING with retry_step EVEN IF it is not moving right\n"
    "now (it can move again) AS LONG AS you can still SEE it. If you can NO LONGER see the target in\n"
    "the current view (it left the frame / went around a corner or through a door and you cannot see\n"
    "where), do NOT keep walking blindly -> ask_operator where it went. On a follow you may ONLY:\n"
    "retry_step (keep following while the target is visible), ask_operator (uncertain or lost sight),\n"
    "or insert_step(\"sidewalk\") to continue THROUGH a narrow gap/doorway you can SEE the target went\n"
    "through (the follow resumes automatically after). You may NOT complete_step / finish_mission /\n"
    "set_plan to abandon a follow; it ends ONLY when the operator says so. A\n"
    "LOW or HORIZONTAL body pose during a crawl/crouch is EXPECTED - it is NOT a fall. Always fill\n"
    "`say` with one short spoken line; act autonomously and do not ask approval for routine steps.\n"
    "TWO HARD SAFETY RULES for CHANGING subtasks: (1) it is only safe to complete a step and move to\n"
    "the next when the robot is STANDING again (base_height back near upright, ~0.7 m) - while it is\n"
    "still low or mid-motion (crawling/crouching/turning), do NOT complete_step; retry_step or wait\n"
    "until it stands. (2) You may mark a step complete_step ONLY on a decision triggered by the\n"
    "motion SETTLING (the robot has stopped) - if this check fired while the robot is still MOVING\n"
    "(an idle monitor), never complete_step; only observe (retry_step), insert_step, or ask_operator.\n"
    "All text English, ASCII only, no CJK."
)


def build_agent_system_prompt(subtasks: Sequence[str], reg) -> str:
    """Unified agent system prompt: one agent authors the plan and picks a tool at each settle.

    Per-call context lives in the user message so this stays a byte-stable cached prefix."""
    return _join([
        _ORCHESTRATOR_ROLE,
        _AGENT_JOB,
        _functions_block(reg, subtasks),
        _SENSOR_GUIDANCE,
    ])


_SENSOR_GUIDANCE = (
    "READING SENSORS: judge each subtask's goal from the CURRENT image, using the pelvis height "
    "(base_height) and its trajectory as support. Your past turns were intentions, not confirmed "
    "outcomes - only the current frame proves what the robot actually did."
)
# Joint activity is deliberately omitted: it describes the generator's plan, not the robot.
