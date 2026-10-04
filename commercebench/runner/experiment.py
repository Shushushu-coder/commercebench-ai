"""Minimal experiment runner: CaseSpec + manifest + system -> RunTrace.

The runner only executes the system and records facts. It does not
evaluate, retry, persist, or call external services.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from commercebench.contracts.case import CaseSpec
from commercebench.contracts.common import SCHEMA_VERSION
from commercebench.contracts.errors import ContractValidationError
from commercebench.contracts.experiment import ExperimentManifest
from commercebench.contracts.trace import RunTrace
from commercebench.systems.base import SystemUnderTest


def run_case(
    case: CaseSpec,
    manifest: ExperimentManifest,
    system: SystemUnderTest,
) -> RunTrace:
    """Run one case once and return its RunTrace.

    Raises ContractValidationError if the case does not belong to the
    experiment (benchmark version mismatch or case_id not listed).
    """
    if case.benchmark_version != manifest.benchmark_version:
        raise ContractValidationError(
            f"case benchmark_version {case.benchmark_version!r} does not match "
            f"manifest benchmark_version {manifest.benchmark_version!r}"
        )
    if case.case_id not in manifest.case_ids:
        raise ContractValidationError(
            f"case_id {case.case_id!r} is not listed in manifest.case_ids"
        )

    started_at = datetime.now(timezone.utc)
    output = system.generate(case.user_input)
    finished_at = datetime.now(timezone.utc)
    latency_ms = (finished_at - started_at).total_seconds() * 1000.0

    return RunTrace(
        schema_version=SCHEMA_VERSION,
        run_id=uuid.uuid4().hex,
        experiment_id=manifest.experiment_id,
        case_id=case.case_id,
        system_id=system.system_id,
        system_version=system.system_version,
        config_fingerprint=manifest.config_fingerprint(),
        input_text=case.user_input,
        output_text=output.output_text,
        started_at=started_at.isoformat(),
        finished_at=finished_at.isoformat(),
        latency_ms=latency_ms,
        retrieval_events=output.retrieval_events,
        tool_events=output.tool_events,
        memory_events=output.memory_events,
        usage=output.usage,
        runtime_metadata=dict(output.runtime_metadata),
    )


__all__ = ["run_case"]
