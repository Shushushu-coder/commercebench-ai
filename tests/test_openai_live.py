"""Phase 3E live OpenAI smoke tests — OPT-IN ONLY.

These exercise the real OpenAI Responses transport. They are skipped
unless both ``OPENAI_API_KEY`` and ``COMMERCEBENCH_OPENAI_MODEL`` are
set; the default suite never runs them. They assert contract-level
structure (a well-formed turn output, a real tool loop), not model
quality — the formal A/B/C live baseline is a separate gate.
"""

from __future__ import annotations

import os
import pathlib

import pytest

from commercebench.contracts.dialogue import (
    DialogueCaseSpec,
    DialogueHistoryEntry,
    DialogueTurnInput,
)
from commercebench.dialogue.harness import run_dialogue_case
from commercebench.dialogue.tools import TOOL_DESCRIPTORS
from commercebench.runner.openai_dialogue import (
    run_openai_dialogue_baseline,
)
from commercebench.contracts.experiment import ExperimentManifest
from commercebench.providers.openai_responses import OpenAIResponsesClient
from commercebench.systems.openai import (
    OpenAIDialogueSystem,
    openai_dialogue_identity,
)

DIALOGUE_DIR = (
    pathlib.Path(__file__).resolve().parents[1] / "examples" / "dialogue_v0"
)

MODEL = os.environ.get("COMMERCEBENCH_OPENAI_MODEL")
API_KEY = os.environ.get("OPENAI_API_KEY")

pytestmark = [
    pytest.mark.live_llm,
    pytest.mark.skipif(
        not API_KEY or not MODEL,
        reason="OPENAI_API_KEY / COMMERCEBENCH_OPENAI_MODEL not configured",
    ),
]


def _load_case(name):
    return DialogueCaseSpec.from_json(
        (DIALOGUE_DIR / "cases" / name).read_text(encoding="utf-8")
    )


def _load_manifest():
    return ExperimentManifest.from_json(
        (DIALOGUE_DIR / "experiments" / "dialogue_openai_v0.json")
        .read_text(encoding="utf-8")
    )


class TestLiveTextSmoke:
    def test_single_text_turn(self):
        system = OpenAIDialogueSystem(
            identity=openai_dialogue_identity(
                model=MODEL,
                generation_parameters={"max_output_tokens": 256},
            ),
            prompt_path=pathlib.Path(
                DIALOGUE_DIR / "prompts" / "openai_support_v0.txt"
            ),
            client=OpenAIResponsesClient(),
        )
        turn_input = DialogueTurnInput(
            turn_id="t1",
            user_input="你好，我想退掉订单A1001",
            history=(
                DialogueHistoryEntry(
                    role="user", content="你好，我想退掉订单A1001"
                ),
            ),
            state_view={
                "orders": {
                    "A1001": {"status": "delivered", "returnable": True}
                },
                "returns": {},
            },
            available_tools=tuple(TOOL_DESCRIPTORS),
        )
        output = system.respond(turn_input)
        assert isinstance(output.output_text, str)
        assert output.usage.input_tokens is None or (
            output.usage.input_tokens >= 0
        )


class TestLiveToolSmoke:
    def test_tool_call_round_trip(self):
        system = OpenAIDialogueSystem(
            identity=openai_dialogue_identity(
                model=MODEL,
                generation_parameters={
                    "max_output_tokens": 1024,
                    "temperature": 0.0,
                },
            ),
            prompt_path=pathlib.Path(
                DIALOGUE_DIR / "prompts" / "openai_support_v0.txt"
            ),
            client=OpenAIResponsesClient(),
        )
        case = _load_case("dialogue-entity-001.json")
        manifest = _load_manifest()
        trace = run_dialogue_case(case, manifest, system)
        assert trace.case_id == case.case_id
        assert trace.turns, "no turns executed"
        # contract-level only: the run produced honest trace evidence
        assert trace.termination_reason in (
            "completed",
            "user_done",
            "max_turns",
            "tool_limit",
            "system_error",
        )


class TestLiveBaselineRunner:
    def test_abc_baseline_runs(self):
        manifest = _load_manifest()
        cases = [
            _load_case(f"{case_id}.json")
            for case_id in manifest.case_ids
        ]
        results = run_openai_dialogue_baseline(
            cases, manifest, model=MODEL
        )
        assert len(results) == len(manifest.case_ids)
        for result in results:
            assert result["trace"].case_id == result["case_id"]
            assert result["evaluation"].metrics
