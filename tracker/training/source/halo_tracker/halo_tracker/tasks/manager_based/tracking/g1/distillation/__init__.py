#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Teacher-checkpoint registry shared by multi-teacher distillation and multi-critic PostRL."""

from .teacher_registry import TeacherRegistry, TeacherSpec

__all__ = ["TeacherRegistry", "TeacherSpec"]
