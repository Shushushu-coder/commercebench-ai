"""Live OpenAI dialogue baseline runner (Phase 3E).

Reproducible entry point for the first real-LLM dialogue baseline::

    python -m commercebench.runner.openai_dialogue \
        --manifest examples/dialogue_v0/experiments/dialogue_openai_v0.json \
        --model <explicitly-verified-model> \
        --out-dir <trace-output-dir>

Hard requirements — no ad-hoc Python snippets needed later:

- an explicit model: either declared as a real value in the
  manifest's ``dialogue_model.model`` or injected via ``--model``
  (the checked-in manifest carries ``__LIVE_MODEL_REQUIRED__`` and can
  never reach a provider as-is);
- ``OPENAI_API_KEY`` available at execution time (resolved lazily by
  the transport — never read at import or argument parsing);
- a manifest carrying ``dialogue`` + ``dialogue_model`` identities.

Every case in ``manifest.case_ids`` runs through the real
``DialogueHarness``/``CommerceToolSimulator`` stack via
``run_dialogue_case`` and is evaluated offline; per-case
``*.trace.json`` / ``*.evaluation.json`` files are written for later
offline replay. No retries, no provider session state.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from commercebench.contracts.dialogue import (
    DialogueCaseSpec,
    DialogueRunTrace,
)
from commercebench.contracts.errors import ContractValidationError
from commercebench.contracts.evaluation import EvaluationResult
from commercebench.contracts.experiment import (
    DialogueModelIdentity,
    ExperimentManifest,
)
from commercebench.dialogue.harness import run_dialogue_case
from commercebench.evaluation.dialogue import DialogueEvaluator
from commercebench.providers.errors import (
    ProviderCredentialError,
    ProviderError,
)
from commercebench.providers.openai_responses import OpenAIResponsesClient
from commercebench.systems.openai import (
    LIVE_MODEL_PLACEHOLDER,
    OpenAIDialogueSystem,
    load_prompt_source,
)

DEFAULT_CASE_DIR = (
    Path(__file__).resolve().parents[2] / "examples" / "dialogue_v0" / "cases"
)


def resolve_dialogue_model(
    manifest: ExperimentManifest, model: Optional[str]
) -> DialogueModelIdentity:
    """Bind the final explicit model into the declared identity.

    ``--model`` is required when the manifest carries the placeholder;
    a real manifest value is used verbatim; providing both with
    different values is rejected — no silent override, no fallback.
    """
    identity = manifest.dialogue_model
    if identity is None:
        raise ContractValidationError(
            "manifest.dialogue_model is required for the live OpenAI "
            "dialogue baseline"
        )
    declared = identity.model
    if declared == LIVE_MODEL_PLACEHOLDER:
        if not model:
            raise ContractValidationError(
                "manifest model is a placeholder: --model is required "
                "(explicit model, no default)"
            )
        return dataclasses.replace(identity, model=model)
    if model and model != declared:
        raise ContractValidationError(
            f"--model {model!r} conflicts with manifest model "
            f"{declared!r}"
        )
    return identity


def build_openai_system(
    manifest: ExperimentManifest,
    *,
    model: Optional[str] = None,
    prompt_path: Optional[Any] = None,
    client: Optional[Any] = None,
) -> OpenAIDialogueSystem:
    """Construct the adapter from the manifest identity (model
    resolved by :func:`resolve_dialogue_model`) and a prompt source
    (the checked-in baseline prompt unless ``prompt_path`` overrides).
    """
    identity = resolve_dialogue_model(manifest, model)
    prompt_source = load_prompt_source(prompt_path)
    return OpenAIDialogueSystem(
        identity=identity,
        prompt_source=prompt_source,
        client=client if client is not None else OpenAIResponsesClient(),
    )


def run_openai_dialogue_baseline(
    cases: Sequence[DialogueCaseSpec],
    manifest: ExperimentManifest,
    *,
    model: Optional[str] = None,
    prompt_path: Optional[Any] = None,
    client: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Run every case through the real adapter + harness + evaluator.

    Returns ``[{"case_id", "trace", "evaluation"}]`` in case order.
    The credential is only needed here — inside ``respond`` — and a
    missing one surfaces as :class:`ProviderCredentialError`.
    """
    system = build_openai_system(
        manifest, model=model, prompt_path=prompt_path, client=client
    )
    evaluator = DialogueEvaluator()
    results: List[Dict[str, Any]] = []
    for case in cases:
        trace = run_dialogue_case(case, manifest, system)
        evaluation = evaluator.evaluate(case, trace)
        results.append(
            {
                "case_id": case.case_id,
                "trace": trace,
                "evaluation": evaluation,
            }
        )
    return results


def _load_cases(
    manifest: ExperimentManifest, case_dir: Path
) -> List[DialogueCaseSpec]:
    cases: List[DialogueCaseSpec] = []
    for case_id in manifest.case_ids:
        path = case_dir / f"{case_id}.json"
        if not path.is_file():
            raise ContractValidationError(
                f"case file not found for {case_id!r}: {path}"
            )
        cases.append(
            DialogueCaseSpec.from_json(path.read_text(encoding="utf-8"))
        )
    return cases


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m commercebench.runner.openai_dialogue",
        description=(
            "Live OpenAI Responses dialogue baseline. Requires an "
            "explicit model and OPENAI_API_KEY."
        ),
    )
    parser.add_argument("--manifest", required=True, help="manifest JSON path")
    parser.add_argument(
        "--model",
        default=None,
        help="explicit model id (required when the manifest carries "
        "the placeholder)",
    )
    parser.add_argument(
        "--case-dir",
        default=str(DEFAULT_CASE_DIR),
        help="directory holding <case_id>.json dialogue cases",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="prompt template path (default: checked-in baseline prompt)",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="output directory for *.trace.json / *.evaluation.json",
    )
    args = parser.parse_args(argv)

    manifest = ExperimentManifest.from_json(
        Path(args.manifest).read_text(encoding="utf-8")
    )
    cases = _load_cases(manifest, Path(args.case_dir))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        results = run_openai_dialogue_baseline(
            cases,
            manifest,
            model=args.model,
            prompt_path=args.prompt,
        )
    except ProviderCredentialError as exc:
        print(f"credential missing: {exc}", file=sys.stderr)
        return 3
    except ProviderError as exc:
        print(f"provider error: {exc}", file=sys.stderr)
        return 2

    for result in results:
        case_id = result["case_id"]
        trace: DialogueRunTrace = result["trace"]
        evaluation: EvaluationResult = result["evaluation"]
        (out_dir / f"{case_id}.trace.json").write_text(
            trace.to_json(), encoding="utf-8"
        )
        (out_dir / f"{case_id}.evaluation.json").write_text(
            json.dumps(evaluation.to_dict(), ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"{case_id}: termination={trace.termination_reason} "
            f"passed={evaluation.passed} "
            f"failures={list(evaluation.failure_categories)}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CASE_DIR",
    "build_openai_system",
    "main",
    "resolve_dialogue_model",
    "run_openai_dialogue_baseline",
]
