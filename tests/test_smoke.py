"""Smoke test: verify the package imports and exposes a version."""

from pathlib import Path

import commercebench
from commercebench.contracts import CaseSpec, EvaluationResult, ExperimentManifest, RunTrace
from commercebench.evaluation import DeterministicEvaluator
from commercebench.runner import run_case
from commercebench.systems import DeterministicSystem

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples" / "phase0"


def test_package_imports():
    assert commercebench.__version__ == "0.1.0"


def test_phase0_end_to_end_pipeline():
    """Case JSON -> manifest -> system -> trace -> evaluate -> result.

    Includes RunTrace and EvaluationResult JSON round-trips, and asserts
    the deterministic Phase 0 pipeline passes.
    """
    case = CaseSpec.from_json((EXAMPLES_DIR / "case.json").read_text(encoding="utf-8"))
    manifest = ExperimentManifest.from_json(
        (EXAMPLES_DIR / "experiment.json").read_text(encoding="utf-8")
    )

    system = DeterministicSystem(
        responses={case.user_input: case.expected_response}
    )
    trace = run_case(case, manifest, system)
    trace = RunTrace.from_json(trace.to_json())

    evaluator = DeterministicEvaluator()
    result = evaluator.evaluate(case, trace)
    result = EvaluationResult.from_json(result.to_json())

    assert result.passed is True
    assert result.failure_categories == ()
