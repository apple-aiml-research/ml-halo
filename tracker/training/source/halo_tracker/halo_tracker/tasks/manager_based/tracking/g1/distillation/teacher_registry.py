#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Teacher-checkpoint registry for multi-teacher distillation and multi-critic PostRL.

Each PreRL teacher is trained in isolation on one dataset, e.g.::

  robot_only       data/g1_processed/robot_only            Halo-...-Robot_Only-PreRL
  object_pushbox   data/g1_processed/robot-object/pushbox  Halo-...-Robot_Object-PreRL
  labsofa          data/g1_processed/robot-terrain/labsofa Halo-...-Robot_Terrain-PreRL

``task_type`` strings (the YAML keys) are arbitrary and any number of teachers can be registered;
classification is per teacher via the match rules in the registry YAML (see ``TeacherSpec``).

Teacher routing is per MOTION, not per env: a flat-terrain env may accept both robot-only and push-box
motions, so a per-env static teacher index would be wrong. ``build_motion_teacher_index`` produces a
``(num_motions,)`` lookup table, and ``rsl_rl_ext.routing.route`` gathers each env's current motion id
through it every rollout step.

Adding a new task_type: drop the URDF family under ``<repo>/assets/objects/``; register it in
``mdp/commands.py`` (``OBJECT_KEYS_FROM_FILENAME`` or ``TERRAIN_PREFIX_CATEGORIES``); add the entity
field to the scene cfg; add a YAML entry with ``filename_prefixes`` / ``filename_substrings`` /
``path_substrings`` so the classifier routes to the new teacher.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import torch


# ── Path resolution ────────────────────────────────────────────────────

def _resolve_project_path(path: str | os.PathLike) -> str:
    """Anchor relative paths to ``PROJ_DIR`` (the ``tracker/`` directory); absolute paths pass through.

    ``PROJ_DIR`` is imported lazily from ``..`` to avoid a circular import with ``g1/__init__.py`` (which
    imports this package). Used for ``TeacherSpec.checkpoint_path`` / ``motion_folder`` so the YAML
    carries only project-relative paths and works on any machine that preserves the repo layout.
    """
    p = os.fspath(path)
    if not p or os.path.isabs(p):
        return p
    from .. import PROJ_DIR  # lazy, breaks the g1.__init__ ↔ distillation cycle
    return os.path.join(str(PROJ_DIR), p)


# ── Default classifier (used when no per-teacher patterns are provided) ──

def _default_classify_motion_file(npz_path: str) -> str:
    """Return a coarse category from the filename (the fallback when no per-teacher rule matches).

      * bound to an object (``push_v2_*``)       → ``object``
      * bound to a terrain URDF (``labsofa_*``)  → ``terrain``
      * everything else                          → ``robot_only``

    The result is used only if a registry entry has that exact name.
    """
    # Local imports keep this module importable without IsaacLab installed.
    from .. import ASSETS_DIR
    from ..mdp.commands import (
        _resolve_motion_terrain_variant,
        resolve_object_type,
    )

    object_dir = Path(ASSETS_DIR) / "objects"
    has_terrain_urdf = _resolve_motion_terrain_variant(npz_path, object_dir) is not None
    has_object = resolve_object_type(npz_path) is not None

    if has_object:
        return "object"
    if has_terrain_urdf:
        return "terrain"
    return "robot_only"


# ── Data classes ─────────────────────────────────────────────────────────

@dataclass
class TeacherSpec:
    """One PreRL teacher's on-disk artifacts, env entry, and match rules.

    ``filename_prefixes`` / ``filename_substrings`` / ``filename_excludes`` drive per-teacher
    classification (matched against the file STEM). When all pattern lists are empty, the teacher is only
    selected by the default classifier (its name must then be one of the default categories).

    ``path_substrings`` matches against the FULL npz path (not the stem) — use when the filename alone
    doesn't disambiguate but the parent directory does (e.g. generic ``jumps1_subject1.npz`` under
    ``robot_only/lafan_acrobatic/``: set ``path_substrings: ["robot_only/lafan_acrobatic"]``).

    ``priority`` breaks ties when multiple teachers' patterns match (higher wins), so specific teachers
    should outrank generic ones.

    ``checkpoint_path`` / ``motion_folder`` accept absolute paths or paths relative to the ``tracker/``
    directory; ``__post_init__`` resolves both to absolute paths, independent of the working directory.
    """

    task_type: str
    checkpoint_path: str
    # Informational (how the teacher was trained; shown in routing warnings). Routing uses only the match rules.
    task_id: str = ""
    motion_folder: str = ""
    filename_prefixes: list[str] = field(default_factory=list)
    filename_substrings: list[str] = field(default_factory=list)
    filename_excludes: list[str] = field(default_factory=list)
    path_substrings: list[str] = field(default_factory=list)
    priority: int = 0

    def __post_init__(self) -> None:
        # Anchor relative paths to PROJ_DIR (portable YAML across machines); absolute paths unchanged.
        self.checkpoint_path = _resolve_project_path(self.checkpoint_path)
        self.motion_folder = _resolve_project_path(self.motion_folder)


@dataclass
class TeacherRegistry:
    """Ordered map of task_type → TeacherSpec.

    Ordering is frozen at construction (SORTED by teacher name) and exposed via ``teacher_type_list``; a
    motion's teacher / critic index is its task_type's position in that list.

    ``default_task_type`` (the YAML's ``_default``) is the safety-net teacher used when a motion matches no
    per-teacher pattern AND the default classifier returns a task_type not in the registry. Without it, an
    uncovered motion folder fails startup with ``KeyError``; with it, unmapped motions route to the designated
    teacher and a startup warning lists the folders that fell through.
    """

    specs: dict[str, TeacherSpec]
    default_task_type: str | None = None
    teacher_type_list: list[str] = field(init=False)
    _default_routed_paths: set[str] = field(init=False, default_factory=set)

    def __post_init__(self) -> None:
        # Sorted by name (not YAML order), so the teacher index is the same in the distillation run, the PostRL run
        # and on every DDP rank.
        self.teacher_type_list = sorted(self.specs)
        # Validate all checkpoints exist.
        for t, spec in self.specs.items():
            if not os.path.isfile(spec.checkpoint_path):
                raise FileNotFoundError(
                    f"TeacherRegistry[{t}]: checkpoint not found at {spec.checkpoint_path}"
                )
        # Validate default_task_type is registered if set.
        if self.default_task_type is not None and self.default_task_type not in self.specs:
            raise KeyError(
                f"TeacherRegistry._default={self.default_task_type!r} but that task_type is "
                f"not in the registry. Registered: {list(self.specs)}"
            )

    # ── Loader ───────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str) -> "TeacherRegistry":
        """Load from a YAML manifest (example: ``tracker/training/configs/teacher_registry_scene.yaml``).

        Top-level keys not starting with ``_`` are task_types whose entries hold the ``TeacherSpec`` fields; an
        unknown field raises ``TypeError``. The optional top-level ``_default`` names the fallback teacher for
        motions that no rule and no built-in classifier matches.
        """
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        default_task_type: str | None = data.get("_default") if isinstance(data, dict) else None
        specs: dict[str, TeacherSpec] = {}
        for task_type, entry in data.items():
            if task_type.startswith("_"):
                continue  # reserved keys such as "_default"
            specs[task_type] = TeacherSpec(task_type=task_type, **entry)
        return cls(specs=specs, default_task_type=default_task_type)

    # ── Serialization ────────────────────────────────────────────────────

    def to_manifest(self) -> dict[str, dict]:
        """Serialize to a plain dict suitable for embedding in a checkpoint."""
        return {t: asdict(spec) for t, spec in self.specs.items()}

    # ── Accessors ────────────────────────────────────────────────────────

    def get(self, task_type: str) -> TeacherSpec:
        try:
            return self.specs[task_type]
        except KeyError as e:
            raise KeyError(
                f"TeacherRegistry has no entry for task_type {task_type!r}; "
                f"available: {list(self.specs)}"
            ) from e

    def index_of(self, task_type: str) -> int:
        """Position of this teacher in ``teacher_type_list`` (its routing index)."""
        return self.teacher_type_list.index(task_type)

    # ── Classifier ───────────────────────────────────────────────────────

    def _match_spec(self, spec: "TeacherSpec", fname: str, npz_path: str) -> bool:
        """True iff ``spec``'s per-teacher rules match (AND across non-empty rule types).

        ``filename_*`` match the file STEM; ``path_substrings`` the full path. A spec with no rules never
        matches here (reachable only via the default-classifier fallback).
        """
        has_rules = spec.filename_prefixes or spec.filename_substrings or spec.path_substrings
        if not has_rules:
            return False
        if any(sub in fname for sub in spec.filename_excludes):
            return False
        if spec.filename_prefixes and not any(fname.startswith(p) for p in spec.filename_prefixes):
            return False
        if spec.filename_substrings and not any(s in fname for s in spec.filename_substrings):
            return False
        if spec.path_substrings and not any(s in npz_path for s in spec.path_substrings):
            return False
        return True

    def classify_motion_file(self, npz_path: str) -> str:
        """Return the task_type that owns this motion.

        Priority order:
          1. Per-teacher rules, specs sorted by ``priority`` descending — first match wins (AND semantics
             across non-empty rule types; see :meth:`_match_spec`).
          2. Default classifier fallback (``_default_classify_motion_file``), accepted only if its
             task_type is in the registry.
          3. ``default_task_type`` safety net if set.

        Raises ``KeyError`` if none produces a registered task_type (loud failure by design: silent
        default-routing hides routing mistakes).
        """
        fname = Path(npz_path).stem

        # Sort by priority descending, then insertion order (stable).
        ordered = sorted(
            self.specs.items(),
            key=lambda kv: -kv[1].priority,
        )

        for task_type, spec in ordered:
            if self._match_spec(spec, fname, npz_path):
                return task_type

        # Fallback to default classifier.
        t = _default_classify_motion_file(npz_path)
        if t in self.specs:
            return t

        # Registry-wide safety net: route unmapped motions to ``_default`` when set. Track parent dirs so
        # startup can print a deduplicated summary instead of one line per NPZ.
        if self.default_task_type is not None:
            self._default_routed_paths.add(os.path.dirname(npz_path))
            return self.default_task_type

        raise KeyError(
            f"motion file {npz_path!r} did not match any registered teacher. "
            f"Default classifier returned {t!r} which is not in registry "
            f"(registered: {list(self.specs)}). Fixes: (a) add filename_prefixes / "
            f"filename_substrings / path_substrings to an existing TeacherSpec, "
            f"or (b) set a top-level ``_default: <task_type>`` in the YAML to "
            f"route unmapped motions to a fallback teacher."
        )

    def build_motion_teacher_index(
        self,
        motion_files: list[str],
        device: str = "cpu",
    ) -> "torch.Tensor":
        """Return per-motion teacher index ``(num_motions,)``.

        Classifies every motion filename and looks up the teacher's position in ``teacher_type_list``. The
        result is a static lookup table that ``rsl_rl_ext.routing.route`` indexes with each env's current
        motion id every step.
        """
        import torch

        # Reset per-call bookkeeping so a re-call doesn't double-log.
        self._default_routed_paths = set()

        indices: list[int] = []
        for npz_path in motion_files:
            task_type = self.classify_motion_file(npz_path)
            indices.append(self.index_of(task_type))

        if self._default_routed_paths:
            teachers = "\n".join(
                f"      - {t}: {self.specs[t].motion_folder}" for t in self.teacher_type_list
            )
            print(
                f"[TeacherRegistry] WARNING: {len(self._default_routed_paths)} motion folder(s) matched no "
                f"explicit teacher rule and fell back to _default={self.default_task_type!r}:\n"
                f"    unmatched folders: {sorted(self._default_routed_paths)}\n"
                f"    registered teachers (task_type: dataset):\n{teachers}\n"
                f"    -> add path_substrings / filename_* to a teacher to route these explicitly."
            )
        return torch.tensor(indices, dtype=torch.long, device=device)

    def summarize_env_distribution(
        self,
        motion_files: list[str],
        env_compatible_motions: Sequence["torch.Tensor"],
    ) -> dict[str, int]:
        """Count envs whose compatible-motion bucket intersects each task_type (informational only).

        An env can appear in multiple counts if its bucket spans task_types (e.g. a flat-terrain env
        accepts both robot-only and push-box motions).
        """
        counts: dict[str, int] = {t: 0 for t in self.specs}
        for bucket in env_compatible_motions:
            seen: set[str] = set()
            for m in bucket.tolist():
                seen.add(self.classify_motion_file(motion_files[m]))
            for t in seen:
                counts[t] = counts.get(t, 0) + 1
        return counts
