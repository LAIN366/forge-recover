"""Append-only JSONL event logging for manipulation experiments."""

from dataclasses import asdict, is_dataclass
from enum import Enum
import json
from pathlib import Path
import time


class OperationMonitor:
    def __init__(self, log_path=None, append=False):
        self.log_path = Path(log_path).expanduser() if log_path else None
        self.stream = None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            self.stream = self.log_path.open(mode, encoding="utf-8")

    @staticmethod
    def _json_default(value):
        if isinstance(value, Enum):
            return value.value
        if is_dataclass(value):
            return asdict(value)
        if hasattr(value, "tolist"):
            return value.tolist()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    def write(self, record_type, payload):
        entry = {
            "record_type": record_type,
            "wall_time": time.time(),
            "payload": payload,
        }
        if self.stream:
            self.stream.write(json.dumps(
                entry, ensure_ascii=False, default=self._json_default
            ) + "\n")
            self.stream.flush()
        return entry

    def record(self, stage, cube_position, tool_position, both_contacts,
               event=None, extra=None):
        sample = {
            "time": time.time(),
            "stage": stage,
            "cube_position": [float(value) for value in cube_position],
            "tool_position": [float(value) for value in tool_position],
            "both_contacts": bool(both_contacts),
            "event": event,
        }
        if extra:
            sample.update(extra)
        self.write("legacy_sample", sample)
        return sample

    def record_observation(self, observation):
        return self.write("observation", observation)

    def record_diagnosis(self, report):
        return self.write("diagnosis", report)

    def record_recovery_plan(self, plan):
        return self.write("recovery_plan", plan)

    def record_event(self, name, **details):
        return self.write("event", {"name": name, **details})

    def record_summary(self, **summary):
        return self.write("summary", summary)

    def close(self):
        if self.stream:
            self.stream.close()
            self.stream = None
