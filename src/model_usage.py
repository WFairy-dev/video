from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class ModelUsageRecorder:
    def __init__(self, output_path: str | Path, stage: str) -> None:
        self.output_path = Path(output_path)
        self.stage = stage
        self.records: list[dict[str, Any]] = []

    def record_chat_completion(
        self,
        response: Any,
        *,
        step: str,
        request_name: str,
        model: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        response_data = response_to_dict(response)
        usage = response_data.get("usage") if isinstance(response_data, dict) else None
        if not isinstance(usage, dict):
            usage = {}

        record = {
            "stage": self.stage,
            "step": step,
            "request_name": request_name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "response_id": response_data.get("id"),
            "model_requested": model,
            "model_returned": response_data.get("model"),
            "provider": response_data.get("provider"),
            "finish_reason": first_choice_value(response_data, "finish_reason"),
            "native_finish_reason": first_choice_value(response_data, "native_finish_reason"),
            "prompt_tokens": number_or_zero(usage.get("prompt_tokens")),
            "completion_tokens": number_or_zero(usage.get("completion_tokens")),
            "total_tokens": number_or_zero(usage.get("total_tokens")),
            "cost": number_or_zero(usage.get("cost")),
            "usage": usage,
            "metadata": metadata or {},
        }
        self.records.append(record)
        self.write()

    def write(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "stage": self.stage,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "summary": self.summary(),
            "records": self.records,
        }
        self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def summary(self) -> dict[str, Any]:
        by_step: dict[str, dict[str, Any]] = {}
        total = empty_summary()

        for record in self.records:
            step = str(record.get("step") or "unknown")
            step_summary = by_step.setdefault(step, empty_summary())
            add_record_to_summary(step_summary, record)
            add_record_to_summary(total, record)

        return {
            "request_count": total["request_count"],
            "prompt_tokens": total["prompt_tokens"],
            "completion_tokens": total["completion_tokens"],
            "total_tokens": total["total_tokens"],
            "cost": round(total["cost"], 8),
            "by_step": {
                step: {
                    "request_count": item["request_count"],
                    "prompt_tokens": item["prompt_tokens"],
                    "completion_tokens": item["completion_tokens"],
                    "total_tokens": item["total_tokens"],
                    "cost": round(item["cost"], 8),
                }
                for step, item in sorted(by_step.items())
            },
        }


def response_to_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    if hasattr(response, "model_dump_json"):
        return json.loads(response.model_dump_json())
    return {}


def first_choice_value(response_data: dict[str, Any], key: str) -> Any:
    choices = response_data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    return first.get(key)


def empty_summary() -> dict[str, Any]:
    return {
        "request_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
    }


def add_record_to_summary(summary: dict[str, Any], record: dict[str, Any]) -> None:
    summary["request_count"] += 1
    summary["prompt_tokens"] += int(number_or_zero(record.get("prompt_tokens")))
    summary["completion_tokens"] += int(number_or_zero(record.get("completion_tokens")))
    summary["total_tokens"] += int(number_or_zero(record.get("total_tokens")))
    summary["cost"] += float(number_or_zero(record.get("cost")))


def number_or_zero(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0
