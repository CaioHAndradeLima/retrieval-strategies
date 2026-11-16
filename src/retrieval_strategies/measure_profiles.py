from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class MeasureProfile:
    name: str
    description: str
    strategies: tuple[str, ...]
    gates: dict[str, dict[str, float]]
    scenario_gates: dict[str, float] | None = None


def load_measure_profiles(path: Path) -> dict[str, MeasureProfile]:
    raw = json.loads(path.read_text())
    profiles: dict[str, MeasureProfile] = {}
    for name, body in raw["profiles"].items():
        profiles[name] = MeasureProfile(
            name=name,
            description=body.get("description", ""),
            strategies=tuple(body.get("strategies", [])),
            gates=body.get("gates", {}),
            scenario_gates=body.get("scenario_gates"),
        )
    return profiles


def _check_min(metrics: dict, field: str, minimum: float, scope: str, failures: list, checks: list) -> None:
    value = metrics.get(field)
    if value is None:
        failures.append(f"{scope} missing {field}")
        return
    ok = value >= minimum
    checks.append({"scope": scope, "metric": field, "expected_min": minimum, "actual": value, "passed": ok})
    if not ok:
        failures.append(f"{scope}.{field}={value} < {minimum}")


def _check_max(metrics: dict, field: str, maximum: float, scope: str, failures: list, checks: list) -> None:
    value = metrics.get(field)
    if value is None:
        failures.append(f"{scope} missing {field}")
        return
    ok = value <= maximum
    checks.append({"scope": scope, "metric": field, "expected_max": maximum, "actual": value, "passed": ok})
    if not ok:
        failures.append(f"{scope}.{field}={value} > {maximum}")


def evaluate_gates(report: dict, profile: MeasureProfile) -> dict[str, Any]:
    failures: list[str] = []
    checks: list[dict] = []
    summary = report.get("summary", {})

    for strategy, thresholds in profile.gates.items():
        metrics = summary.get(strategy, {})
        if not metrics:
            failures.append(f"missing summary for strategy {strategy}")
            continue
        for key, bound in thresholds.items():
            if key.startswith("min_"):
                field = key[4:]
                _check_min(metrics, field, bound, strategy, failures, checks)
            elif key.startswith("max_"):
                field = key[4:]
                _check_max(metrics, field, bound, strategy, failures, checks)

    scenarios = report.get("scenarios", {})
    for key, minimum in (profile.scenario_gates or {}).items():
        if not key.startswith("min_"):
            continue
        field = key[4:]
        if field in scenarios:
            _check_min(scenarios, field, minimum, "scenarios", failures, checks)
        elif field == "hybrid_recall_at_k" and "hybrid_recall_at_k" in scenarios:
            _check_min(scenarios, "hybrid_recall_at_k", minimum, "scenarios", failures, checks)
        elif field == "metadata_filter_accuracy" and scenarios.get("metadata"):
            _check_min(scenarios["metadata"], "pass_rate", minimum, "scenarios.metadata", failures, checks)
        elif field == "tenant_isolation_rate" and scenarios.get("tenant"):
            _check_min(scenarios["tenant"], "pass_rate", minimum, "scenarios.tenant", failures, checks)

    return {"profile": profile.name, "passed": not failures, "failures": failures, "checks": checks}