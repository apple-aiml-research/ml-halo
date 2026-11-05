#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Declarative skill registry loader (`skills.yaml`; file order = id).

`enum_for_run` only fixes, once at startup, which skill names the model may choose from. Nothing
here validates or corrects a model-chosen subtask; a skill with no loaded checkpoint is an
execution fact for the generator, not a planning gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import yaml

_DEFAULT_PATH = Path(__file__).resolve().parent / "skills.yaml"

# A skill with no `completion_appearance` gets this neutral template, never an error.
_NEUTRAL_COMPLETION_TEMPLATE = (
    "no specific visual cue is documented for this skill; judge completion from the "
    "mission and the current ego view."
)


@dataclass(frozen=True)
class Skill:
    id: int
    name: str
    generator_prompt: str
    when_applicable: str
    completion_appearance: Optional[str] = None
    # "object": a settle finishes the step; "living": the target can move, so a settle does not.
    target_kind: str = "object"       # object | living
    # Object the skill acts on (None = not object-bound); named in the agent's next-step hint.
    target: Optional[str] = None
    # Goes through/under the object and out the far side: complete once emerged, do not retry.
    pass_through: bool = False


Registry = tuple[Skill, ...]

# Published directly by code (operator /stop, unanswered question, no mission yet).
HOLD = "hold"


def load() -> Registry:
    """Load the registry from the sibling `skills.yaml`."""
    p = _DEFAULT_PATH
    with open(p, "r") as f:
        doc = yaml.safe_load(f) or {}
    entries = doc.get("skills") or []
    skills = []
    for i, raw in enumerate(entries):
        skills.append(Skill(
            id=i,
            name=raw["name"],
            generator_prompt=raw["generator_prompt"],
            when_applicable=raw.get("when_applicable", ""),
            completion_appearance=raw.get("completion_appearance"),
            target_kind=str(raw.get("target_kind", "object")),
            pass_through=bool(raw.get("pass_through", False)),
            target=(str(raw["target"]) if raw.get("target") else None),
        ))
    return tuple(skills)


def names(reg: Registry) -> list[str]:
    """Skill names, in file order (== id order)."""
    return [s.name for s in reg]


def _find(reg: Registry, name: str) -> Optional[Skill]:
    for s in reg:
        if s.name == name:
            return s
    return None


def _get(reg: Registry, name: str) -> Skill:
    s = _find(reg, name)
    if s is None:
        raise KeyError(f"unknown skill: {name!r}")
    return s


def id_of(reg: Registry, name: str) -> int:
    return _get(reg, name).id


def prompt_of(reg: Registry, name: str) -> str:
    return _get(reg, name).generator_prompt


def target_kind_of(reg: Registry, name: str) -> str:
    """"object" or "living" (see Skill.target_kind). Unknown skill -> "object", never raises."""
    s = _find(reg, name)
    return s.target_kind if s else "object"


def pass_through_of(reg: Registry, name: str) -> bool:
    """True for a skill that passes through/under an obstacle. Unknown skill -> False."""
    s = _find(reg, name)
    return bool(s and s.pass_through)


def completion_appearance_of(reg: Registry, name: str) -> str:
    """A missing `completion_appearance` yields a neutral template, never an error."""
    return _get(reg, name).completion_appearance or _NEUTRAL_COMPLETION_TEMPLATE


def enum_for_run(loaded: Optional[Sequence[str]], reg: Optional[Registry] = None) -> list[str]:
    """Guided-JSON skill enum for this run, resolved once at startup: the `loaded` skills (task
    `subtasks:` keys) in registry order, else every registry entry."""
    if reg is None:
        reg = load()
    all_names = names(reg)
    if loaded:
        return [n for n in all_names if n in loaded]
    return list(all_names)


def target_of(reg: Registry, name: str) -> Optional[str]:
    """Object a skill acts on (or None)."""
    s = _find(reg, name)
    return s.target if s else None
