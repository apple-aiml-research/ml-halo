#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""MissionOrchestrator: mission state plus the T1 plan-authoring and per-step agent decisions."""
from __future__ import annotations
import time
from dataclasses import replace
from typing import Any, Optional

from .interface import VLMInterface
from .plan import Plan, Step
from . import prompts
from . import registry as _registry

from .schemas import t1_plan_schema, agent_action_schema


# Output budgets, set well above need: guided JSON cut at max_tokens is an unparseable prefix.
# The agent budget also covers reasoning tokens a thinking model spends (stripped from `raw`).
T1_MAX_TOKENS = 1536     # a full plan is ~100-400 tokens
AGENT_MAX_TOKENS = 3072


class MissionOrchestrator:

    def _steps_from_wire(self, raw_steps):
        """Model plan steps (`skill`, `id`) -> Steps, first one in_progress."""
        raw_steps = [s for s in (raw_steps or []) if s.get("skill") in self.subtasks]
        return tuple(
            Step(id=str(s.get("id") or f"s{i}"), subtask=s["skill"], title=self._title(s["skill"]),
                 status=("in_progress" if i == 0 else "pending"))
            for i, s in enumerate(raw_steps))

    def _title(self, skill) -> str:
        """Step title = the generator prompt the step runs (mission-YAML prompt, else registry)."""
        sk = str(skill or "")
        if sk in self.step_prompts:
            return self.step_prompts[sk]
        if sk in _registry.names(self._reg):
            return _registry.prompt_of(self._reg, sk) or sk
        return sk

    def __init__(self, vlm: VLMInterface, mission: str, subtasks):
        self.vlm = vlm
        self.mission = mission
        self._reg = _registry.load()
        self.subtasks = list(subtasks)
        # Set by the node: per-step generator prompts (mission YAML), used for step titles.
        self.step_prompts: dict = {}
        self.system_t1 = prompts.build_t1_system_prompt(self.subtasks, reg=self._reg)
        self.system_agent = prompts.build_agent_system_prompt(self.subtasks, reg=self._reg)

    def set_mission(self, mission: str) -> None:
        """(Re)set the mission; the system prompts do not depend on it."""
        self.mission = mission

    def deliberate(self, prev_plan: Plan, image: Any = None,
                   frames: Optional[list] = None,
                   operator_instruction: Optional[str] = None,
                   history_turns: Optional[list] = None) -> Plan:
        """T1: author the plan; returns a new `Plan` (version + 1) and raises on failure.

        Runs on the background worker concurrently with the tick, so it must not read
        tick-mutated state."""
        # Instructions live in the (cached) system prompt; the user turn carries only per-call context.
        parts = []
        if self.mission:
            parts.append(str(self.mission))
        if prev_plan.steps:
            parts.append(f"PREVIOUS PLAN (revision {prev_plan.revision}):")
            for s in prev_plan.steps:
                parts.append(f"  {s.id} [{s.status}] {s.title} (skill {s.subtask})")
        # No "no previous plan" line: naming an absent plan primes the model to invent one.
        if operator_instruction:
            parts.append(f"OPERATOR INSTRUCTION: {operator_instruction}")
        utext = "\n".join(parts)
        content = self.vlm.user_content(utext, image=image, frames=frames)
        msgs = [{"role": "system", "content": self.system_t1}]
        # Past turns go as native messages.
        for t in (history_turns or []):
            msgs.append({"role": ("assistant" if t.get("role") == "assistant" else "user"),
                         "content": str(t.get("text", ""))})
        msgs.append({"role": "user", "content": content})
        out = self.vlm.decide_messages(msgs, schema=t1_plan_schema(self.subtasks),
                                       max_tokens=T1_MAX_TOKENS,
                                       timeout_s=self.vlm.timeout_s_long)
        d = out.get("data")
        if not d:
            raise RuntimeError(
                f"T1 deliberate: output TRUNCATED at max_tokens={T1_MAX_TOKENS} "
                f"(guided JSON emitted a valid prefix; raise T1_MAX_TOKENS)"
                if out.get("truncated") else "T1 deliberate: parse-fail or VLM error")

        now = time.time()
        steps = self._steps_from_wire(d.get("steps"))
        return Plan(
            run_id=prev_plan.run_id,
            version=prev_plan.version + 1,
            revision=prev_plan.revision + 1,
            stamp=now,
            mission=self.mission,
            # For a conversational request with no steps this is the only output.
            reply=str(d.get("reply") or ""),
            current=0,
            # The agent decisions own completion.
            done=False,
            steps=steps,
        )

    def decide_at_settle(self, plan, image=None, frames=None, running_subtask=None,
                         operator_note=None, at_settle=True, pose_text=None,
                         stall_count=0, progress_m=None) -> dict:
        """Agent decision (one VLM call) at a settle or operator utterance: pick one tool and
        apply its pure Plan transition. complete_step / retry_step / insert_step / finish_mission /
        set_plan edit the plan; ask_operator is returned for the node to apply."""
        now = time.time()
        idx = plan.current if plan.steps and 0 <= plan.current < len(plan.steps) else 0
        cur = running_subtask or (plan.steps[idx].subtask if plan.steps else None)
        checklist = "\n".join(f"  {i + 1}. [{s.status}] {s.subtask} — {s.title}"
                              for i, s in enumerate(plan.steps))
        parts = [f"Task: {self.mission}", f"Plan:\n{checklist}"]
        if at_settle:
            parts.append(f"The robot has STOPPED MOVING while running the current step: '{cur}'. "
                         f"Decide the next action by calling ONE tool.")
        else:
            # The robot keeps running while the agent judges completion.
            parts.append(f"The robot is RUNNING the current step: '{cur}' (it may still be moving). "
                         f"Judge from the view whether this step's goal is ACHIEVED: call "
                         f"complete_step if it is, retry_step to let it keep going, or intervene "
                         f"(insert_step / ask_operator / set_plan).")
        # Being in position for the next step also ends this one.
        _nxt = next((s.subtask for s in plan.steps[idx + 1:] if s.status == "pending"), None)
        if _nxt:
            _ntgt = _registry.target_of(self._reg, _nxt)
            parts.append(
                f"NEXT step: '{_nxt}'" + (f" (acts on the {_ntgt})" if _ntgt else "") + ". "
                f"If the robot is already in a good position to START it"
                + (f" (the {_ntgt} is close and in view)" if _ntgt else "")
                + ", the current step has done its job: call complete_step.")
        # A living-target follow must not advance on a settle; a static-object go-to should.
        _tk = _registry.target_kind_of(self._reg, cur) if cur else "object"
        if _tk == "living":
            parts.append(
                "This step FOLLOWS a LIVING target (a person, dog, or animal). While you can SEE the "
                "target in the current view, KEEP FOLLOWING — call retry_step — EVEN IF it is not "
                "moving right now (it can move again). You may NOT complete_step, finish_mission, or "
                "set_plan to abandon this follow on your own. "
                "BUT IF the target is NO LONGER VISIBLE in the current ego view (you lost sight of it "
                "— it went out of frame, around a corner, or through a door and you cannot see where), "
                "do NOT keep walking blindly with retry_step: call ask_operator to ask where it went "
                "(e.g. \"I lost sight of you — which way did you go?\"). The exception is when you can "
                "SEE the gap it went through: if the straight follow is BLOCKED by a narrow "
                "gap/doorway the target just went through, insert_step(\"sidewalk\") to get through and "
                "CONTINUE following (the follow resumes automatically once you are through). "
                "The follow ends only when the operator says so.")
            # Stall readout (a fact, not a gate): a non-visual cue to consider the side-step.
            if stall_count and stall_count >= 3 and (progress_m is None or progress_m < 0.15):
                _pm = "~0.0 m" if progress_m is None else f"~{progress_m:.2f} m"
                parts.append(
                    f"[orchestrator] You have settled here {stall_count} times in a row with {_pm} "
                    "progress toward the target since this step last advanced -- the path ahead "
                    "appears BLOCKED, not merely a target that paused. If a doorway or narrow gap is "
                    "ahead, call insert_step(\"sidewalk\") to get through and continue the follow; "
                    "otherwise keep following with retry_step.")
        else:
            if _registry.pass_through_of(self._reg, cur):
                parts.append(
                    "This is a PASS-THROUGH step: the robot crawls/crouches UNDER the desk/obstacle "
                    "and comes OUT the far side — it does NOT arrive-and-stop at it. You have "
                    "already gone low and are now on/near the far side. Call complete_step("
                    "success=true) — the crawl is DONE once you have emerged past the obstacle. Do "
                    "NOT call retry_step: retrying RE-RUNS the whole crawl and marches the robot far "
                    "past the target (overshoot). Only retry if you clearly have NOT gone under yet.")
            else:
                parts.append("This step goes to a STATIC object: once the robot has ARRIVED and stopped, "
                             "the step is done — call complete_step to advance to the next step.")
        if operator_note:
            parts.append(f"The operator just said: {operator_note}")
        if pose_text:
            parts.append(f"Robot pose (raw): {pose_text}")
        utext = "\n".join(parts)
        content = self.vlm.user_content(utext, image=image, frames=frames)
        msgs = [{"role": "system", "content": self.system_agent},
                {"role": "user", "content": content}]
        out = self.vlm.decide_messages(msgs, schema=agent_action_schema(self.subtasks),
                                          max_tokens=AGENT_MAX_TOKENS)
        d = out.get("data")
        base = {"plan": plan, "action": None, "say": "", "skill": None, "success": None,
                "question": None}
        if not d:
            return base
        action = d["action"]
        # A follow may not be abandoned without the operator: coerce complete/finish/set_plan to
        # retry_step. insert_step is allowed (a recovery maneuver that resumes the follow).
        if _tk == "living" and not operator_note and action in (
                "complete_step", "finish_mission", "set_plan"):
            action = "retry_step"
        base.update(action=action, say=d.get("say", ""), skill=d.get("skill"),
                    success=d.get("success"), question=d.get("question"))
        if action == "complete_step":
            plan = plan.close_current() if d.get("success", True) \
                else plan.fail_current()
        elif action == "retry_step":
            plan = plan.reopen(idx)
        elif action == "insert_step":
            sk = d.get("skill")
            if sk and sk in self.subtasks:
                step = Step(id=f"ins{int(now * 1000) % 10_000_000}", subtask=sk, title=self._title(sk),
                            status="pending")
                plan = plan.insert_step(idx, step)
        elif action == "finish_mission":
            plan = plan.close_all()
        elif action == "set_plan":
            now2 = time.time()
            new_steps = self._steps_from_wire(d.get("steps"))
            if new_steps:
                plan = replace(plan, steps=new_steps, current=0, done=False,
                               version=plan.version + 1, revision=plan.revision + 1, stamp=now2)
        # ask_operator: the node applies the side effect; the plan is unchanged.
        base["plan"] = plan
        return base
