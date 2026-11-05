#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Session-level mission history that T1 reads when it (re-)plans: what each request asked and what
the robot replied. Text only; records, decides nothing."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

# Caps so the prompt cannot grow without bound over a long session.
MAX_MISSIONS = 6            # oldest missions drop out of the history
MAX_MISSION_CHARS = 200     # the operator's own words, truncated for the prompt
MAX_REPLY_CHARS = 300       # what the robot said back (its own words)


@dataclass(frozen=True)
class MissionRecord:
    """One operator request and what became of it."""
    text: str                          # what they asked for, verbatim
    reply: str = ""                    # what the robot SAID back (T1's own words to the operator)


@dataclass
class MissionHistory:
    """The session's missions, oldest first. Mutated only from the tick thread.

    The T1 worker must never read it directly: build turns() on the tick thread instead.
    """
    missions: list[MissionRecord] = field(default_factory=list)

    def open_mission(self, text: str) -> None:
        """The operator asked for something new."""
        self.missions.append(MissionRecord(text=text))
        while len(self.missions) > MAX_MISSIONS:
            self.missions.pop(0)

    def attach_reply(self, reply: str) -> None:
        """Record T1's reply for the current mission, so follow-ups can refer to it."""
        if not self.missions or not reply:
            return
        self.missions[-1] = replace(self.missions[-1], reply=reply)

    def turns(self) -> list:
        """Past missions as user/assistant role-text turns (excludes the newest, in-flight request).

        Rendered on the tick thread.
        """
        out = []
        for m in self.missions[:-1]:          # everything except the current (newest) request
            if not m.text:
                continue                       # no user turn -> don't emit an orphan assistant turn
            out.append({"role": "user", "text": m.text[:MAX_MISSION_CHARS]})
            out.append({"role": "assistant",
                        "text": m.reply[:MAX_REPLY_CHARS] or "(no reply recorded)"})
        return out
