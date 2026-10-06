"""DialogueHarness: deterministic multi-turn orchestration (Phase 3B).

The harness — not the system — owns:

- canonical business state (deep-copied per turn boundary)
- the visible conversation history (application-managed, explicit)
- conditional user-turn release (scripted deterministic simulator)
- bounded inner tool loop and bounded max_turns
- termination decisions and the DialogueRunTrace

System boundary: ``DialogueSystemUnderTest.respond`` receives a fully
constructed ``DialogueTurnInput`` and returns text + tags + proposed
tool calls. Proposed calls are executed through the deterministic
tool simulator; only executed calls can change state. A failed call
leaves state unchanged; an invalid call does NOT terminate the
dialogue — it is evaluator evidence, not a crash.

Termination reasons: ``completed`` (expected_final_state satisfied
and all declared tool obligations met), ``user_done`` (no further
turn can activate), ``max_turns`` (user-turn budget exhausted),
``tool_limit`` (per-turn tool-call budget exceeded), ``system_error``
(unrecoverable exception inside ``respond``).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from commercebench.contracts.common import SCHEMA_VERSION
from commercebench.contracts.dialogue import (
    DialogueCaseSpec,
    DialogueHistoryEntry,
    DialogueRunTrace,
    DialogueStepTrace,
    DialogueToolDescriptor,
    DialogueTurnInput,
    DialogueTurnOutput,
    DialogueTurnSpec,
    DialogueTurnTrace,
    TERMINATION_COMPLETED,
    TERMINATION_MAX_TURNS,
    TERMINATION_SYSTEM_ERROR,
    TERMINATION_TOOL_LIMIT,
    TERMINATION_USER_DONE,
)
from commercebench.contracts.errors import ContractValidationError
from commercebench.contracts.experiment import ExperimentManifest
from commercebench.contracts.trace import ToolEvent, UsageStats

from .identity import runtime_identity_metadata
from .policy import (
    TOOL_POLICY_DIRECT,
    evaluate_tool_call,
    validate_tool_policy,
)
from .state import (
    delta_obligations_satisfied,
    json_state_copy,
    json_value_copy,
    json_value_equal,
    state_subset_matches,
)
from .tools import (
    CommerceToolSimulator,
    TOOL_DESCRIPTORS,
    TOOL_LOOKUP_ORDER,
)

DEFAULT_MAX_TURNS = 6
DEFAULT_MAX_TOOL_CALLS_PER_TURN = 3
DEFAULT_HISTORY_POLICY = "full"
DEFAULT_TOOL_POLICY = TOOL_POLICY_DIRECT


def _tool_event_snapshot(event: ToolEvent) -> ToolEvent:
    """Deep-copied ToolEvent: nested ``arguments``/``result`` payloads
    are independent objects so no two persisted/shared views of one
    call alias the same mutable dicts."""
    return ToolEvent(
        tool_name=event.tool_name,
        arguments=json_value_copy(event.arguments),
        result=json_value_copy(event.result),
        status=event.status,
        error=event.error,
    )


def _history_entry_snapshot(entry: DialogueHistoryEntry) -> DialogueHistoryEntry:
    """Deep-copied history entry: mutable ``content`` and every nested
    ``tool_events`` payload become independent objects."""
    return DialogueHistoryEntry(
        role=entry.role,
        content=json_value_copy(entry.content),
        tool_events=tuple(
            _tool_event_snapshot(event) for event in entry.tool_events
        ),
    )


class DialogueHarness:
    """Deterministic multi-turn harness for one experiment config."""

    def __init__(
        self,
        config: Optional[Any] = None,
        tool_simulator: Optional[CommerceToolSimulator] = None,
    ) -> None:
        # ``config`` is a DialogueHarnessConfig (typed lazily to keep
        # this module importable without the experiment module).
        self._config = config
        parameters = dict(config.parameters) if config is not None else {}
        if config is not None and not hasattr(config, "parameters"):
            raise ContractValidationError(
                "harness config must expose parameters"
            )
        self.max_turns = int(parameters.get("max_turns", DEFAULT_MAX_TURNS))
        self.max_tool_calls_per_turn = int(
            parameters.get(
                "max_tool_calls_per_turn", DEFAULT_MAX_TOOL_CALLS_PER_TURN
            )
        )
        self.history_policy = parameters.get(
            "history_policy", DEFAULT_HISTORY_POLICY
        )
        if self.history_policy != "full":
            raise ContractValidationError(
                f"unsupported history_policy {self.history_policy!r}"
            )
        self.tool_policy = validate_tool_policy(
            parameters.get("tool_policy", DEFAULT_TOOL_POLICY)
        )
        simulator_identity = parameters.get("tool_simulator") or {}
        simulator = tool_simulator or CommerceToolSimulator()
        if simulator_identity:
            if (
                simulator_identity.get("id") != simulator.simulator_id
                or simulator_identity.get("version")
                != simulator.simulator_version
            ):
                raise ContractValidationError(
                    "manifest tool_simulator identity does not match the "
                    "provided simulator"
                )
        self.tool_simulator = simulator

    # -- public API ---------------------------------------------------------

    def run(
        self,
        case: DialogueCaseSpec,
        manifest: ExperimentManifest,
        system: Any,
    ) -> DialogueRunTrace:
        """Run one dialogue case end-to-end and return its trace.

        Fail closed before any turn executes: the manifest must carry a
        ``dialogue`` config and this harness must have been built from
        exactly that config — one source of truth for the fingerprinted
        orchestration parameters.
        """
        self._validate_harness_binding(manifest)
        self._validate_case_against_manifest(case, manifest)
        state = json_state_copy(case.initial_state)
        initial_state = json_state_copy(case.initial_state)
        history: List[DialogueHistoryEntry] = []
        consumed: Set[str] = set()
        successful_tools: Set[str] = set()
        successful_lookups: Set[str] = set()
        emitted_tags: Set[str] = set()
        turn_traces: List[DialogueTurnTrace] = []
        total_usage = _usage_zero()
        termination = TERMINATION_USER_DONE
        last_output = ""

        started_at = datetime.now(timezone.utc)

        while True:
            if len(turn_traces) >= self.max_turns:
                termination = TERMINATION_MAX_TURNS
                break
            selected, evidence = self._select_turn(
                case, consumed, successful_tools, emitted_tags
            )
            if selected is None:
                termination = TERMINATION_USER_DONE
                break
            consumed.add(selected.turn_id)

            turn_result = self._run_turn(
                case=case,
                spec=selected,
                turn_index=len(turn_traces) + 1,
                activation_evidence=evidence,
                state=state,
                history=history,
                successful_lookups=successful_lookups,
                system=system,
            )
            if turn_result[0] is None:
                # system_error: turn abandoned, no turn trace
                termination = turn_result[1]
                break
            (
                turn_trace,
                state,
                turn_usage,
                turn_successful_tools,
                turn_tags,
                last_output,
                tool_limit_hit,
            ) = turn_result
            assert turn_trace is not None
            turn_traces.append(turn_trace)
            total_usage = _usage_add(total_usage, turn_usage)
            successful_tools.update(turn_successful_tools)
            emitted_tags.update(turn_tags)
            if tool_limit_hit:
                termination = TERMINATION_TOOL_LIMIT
                break
            if self._is_complete(case, state, turn_traces):
                termination = TERMINATION_COMPLETED
                break
            if len(consumed) == len(case.turns):
                termination = TERMINATION_USER_DONE
                break

        finished_at = datetime.now(timezone.utc)
        return DialogueRunTrace(
            schema_version=SCHEMA_VERSION,
            run_id=uuid.uuid4().hex,
            experiment_id=manifest.experiment_id,
            case_id=case.case_id,
            system_id=system.system_id,
            system_version=system.system_version,
            config_fingerprint=manifest.config_fingerprint(),
            turns=tuple(turn_traces),
            initial_state=initial_state,
            final_state=json_state_copy(state),
            termination_reason=termination,
            output_text=last_output,
            started_at=started_at.isoformat(),
            finished_at=finished_at.isoformat(),
            latency_ms=(
                finished_at - started_at
            ).total_seconds() * 1000.0,
            usage=total_usage,
            runtime_metadata={
                "harness": {
                    "harness_id": getattr(self._config, "harness_id", None),
                    "harness_version": getattr(
                        self._config, "harness_version", None
                    ),
                    "max_turns": self.max_turns,
                    "max_tool_calls_per_turn": self.max_tool_calls_per_turn,
                    "history_policy": self.history_policy,
                    "tool_policy": self.tool_policy,
                    "tool_simulator": {
                        "id": self.tool_simulator.simulator_id,
                        "version": self.tool_simulator.simulator_version,
                    },
                },
                "identity": runtime_identity_metadata(
                    manifest.dialogue_model
                ),
                "consumed_turn_ids": [t.turn_id for t in turn_traces],
            },
        )

    # -- turn selection (deterministic user simulator) ----------------------

    def _select_turn(
        self,
        case: DialogueCaseSpec,
        consumed: Set[str],
        successful_tools: Set[str],
        emitted_tags: Set[str],
    ) -> Tuple[Optional[DialogueTurnSpec], Dict[str, Any]]:
        """First unconsumed turn whose activation is satisfied.

        Ordered scan semantics (fixed, Phase 3D documented): the case's
        ``turns`` are scanned in declaration order and the first
        unconsumed turn whose activation is satisfied fires. A later
        turn with default activation therefore releases *before* an
        earlier gated turn whose condition is not yet met — gating
        skips a turn for now, it does not block the sequence.

        ``activation`` ``None``/``{}`` is the default trigger: it
        requires no prior signal, so it fires whenever reached in case
        order. ``on_tool`` requires a *successful* execution of that
        tool in an earlier turn; ``on_assistant_tag`` requires the tag
        in an earlier assistant step.
        """
        for spec in case.turns:
            if spec.turn_id in consumed:
                continue
            activation = spec.activation
            if not activation:
                return spec, {"type": "default"}
            if "on_tool" in activation:
                tool = activation["on_tool"]
                if tool in successful_tools:
                    return spec, {"type": "tool", "value": tool}
                continue
            if "on_assistant_tag" in activation:
                tag = activation["on_assistant_tag"]
                if tag in emitted_tags:
                    return spec, {"type": "assistant_tag", "value": tag}
                continue
        return None, {}

    # -- single turn ---------------------------------------------------------

    def _run_turn(
        self,
        case: DialogueCaseSpec,
        spec: DialogueTurnSpec,
        turn_index: int,
        activation_evidence: Dict[str, Any],
        state: Dict[str, Any],
        history: List[DialogueHistoryEntry],
        successful_lookups: Set[str],
        system: Any,
    ):
        """Execute one user turn: system invocation + bounded tool loop.

        Returns ``(None, "system_error")`` on unrecoverable system
        failure, otherwise a tuple with the finished turn trace.

        Integrity boundaries (Phase 3D): the system's ``DialogueTurnInput``
        holds only deep-copied snapshots of the stored history and
        tool descriptors; every executed ``ToolEvent`` is recorded via
        independent snapshots for the step trace, the history, and the
        turn-level aggregate. Nothing the system can mutate is shared
        with persisted evidence or future inputs.
        """
        state_before = json_state_copy(state)
        history.append(
            DialogueHistoryEntry(role="user", content=spec.user_input)
        )
        steps: List[DialogueStepTrace] = []
        executed_events: List[ToolEvent] = []
        turn_usage = _usage_zero()
        turn_successful: Set[str] = set()
        turn_tags: Set[str] = set()
        tool_calls_executed = 0
        tool_limit_hit = False
        last_output = ""
        step_index = 0
        turn_start = datetime.now(timezone.utc)

        while True:
            turn_input = DialogueTurnInput(
                turn_id=spec.turn_id,
                user_input=spec.user_input,
                history=tuple(
                    _history_entry_snapshot(entry) for entry in history
                ),
                state_view=json_state_copy(state),
                available_tools=tuple(
                    DialogueToolDescriptor.from_dict(d.to_dict())
                    for d in TOOL_DESCRIPTORS
                ),
            )
            digest = turn_input.input_context_digest()
            step_start = datetime.now(timezone.utc)
            try:
                output = system.respond(turn_input)
                output_text_is_str = isinstance(
                    output, DialogueTurnOutput
                ) and isinstance(output.output_text, str)
            except Exception:
                return None, TERMINATION_SYSTEM_ERROR
            step_end = datetime.now(timezone.utc)
            if not output_text_is_str:
                return None, TERMINATION_SYSTEM_ERROR
            turn_usage = _usage_add(turn_usage, output.usage)
            turn_tags.update(output.assistant_tags)
            last_output = output.output_text

            remaining_budget = (
                self.max_tool_calls_per_turn - tool_calls_executed
            )
            to_execute = list(output.tool_calls[:remaining_budget])
            dropped = list(output.tool_calls[remaining_budget:])
            new_events: List[ToolEvent] = []
            for call in to_execute:
                # Policy gate (Phase 3D): a rejected call is recorded
                # with status="error" and never reaches the simulator,
                # so canonical state cannot change.
                rejection = evaluate_tool_call(
                    self.tool_policy,
                    call.tool_name,
                    call.arguments,
                    successful_lookups,
                )
                if rejection is not None:
                    new_events.append(
                        ToolEvent(
                            tool_name=call.tool_name,
                            arguments=json_value_copy(call.arguments),
                            result=rejection,
                            status="error",
                            error=rejection["error_code"],
                        )
                    )
                    tool_calls_executed += 1
                    continue
                execution = self.tool_simulator.call(
                    call.tool_name, call.arguments, state
                )
                state = execution.new_state
                new_events.append(
                    ToolEvent(
                        tool_name=call.tool_name,
                        arguments=json_value_copy(call.arguments),
                        result=json_value_copy(execution.result),
                        status=execution.status,
                        error=execution.error,
                    )
                )
                tool_calls_executed += 1
                if execution.status == "ok":
                    turn_successful.add(call.tool_name)
                    if (
                        call.tool_name == TOOL_LOOKUP_ORDER
                        and isinstance(
                            call.arguments.get("order_id"), str
                        )
                    ):
                        successful_lookups.add(
                            call.arguments["order_id"]
                        )
            executed_events.extend(new_events)

            step_metadata = json_value_copy(output.runtime_metadata)
            if dropped:
                step_metadata["tool_calls_not_executed"] = len(dropped)
                step_metadata["terminated_by"] = "tool_limit"
            steps.append(
                DialogueStepTrace(
                    step_index=step_index,
                    output_text=output.output_text,
                    assistant_tags=tuple(output.assistant_tags),
                    tool_events=tuple(
                        _tool_event_snapshot(e) for e in new_events
                    ),
                    history_digest=digest,
                    usage=output.usage,
                    latency_ms=(
                        step_end - step_start
                    ).total_seconds() * 1000.0,
                    runtime_metadata=step_metadata,
                )
            )
            step_index += 1

            history.append(
                DialogueHistoryEntry(
                    role="assistant",
                    content=output.output_text,
                    tool_events=tuple(
                        _tool_event_snapshot(e) for e in new_events
                    ),
                )
            )
            if new_events:
                history.append(
                    DialogueHistoryEntry(
                        role="tool",
                        content=[
                            _tool_event_snapshot(e).to_dict()
                            for e in new_events
                        ],
                    )
                )
            if dropped:
                tool_limit_hit = True
                break
            if not output.tool_calls:
                break
            # All proposed calls executed and budget remains: loop.

        turn_trace = DialogueTurnTrace(
            turn_id=spec.turn_id,
            turn_index=turn_index,
            user_input=spec.user_input,
            state_before=state_before,
            state_after=json_state_copy(state),
            output_text=last_output,
            steps=tuple(steps),
            assistant_tags=tuple(sorted(turn_tags)),
            tool_events=tuple(executed_events),
            activation_evidence=activation_evidence,
            usage=turn_usage,
            latency_ms=(
                datetime.now(timezone.utc) - turn_start
            ).total_seconds() * 1000.0,
            runtime_metadata={},
        )
        return (
            turn_trace,
            state,
            turn_usage,
            turn_successful,
            turn_tags,
            last_output,
            tool_limit_hit,
        )

    # -- completion -----------------------------------------------------------

    def _is_complete(
        self,
        case: DialogueCaseSpec,
        state: Dict[str, Any],
        turn_traces: List[DialogueTurnTrace],
    ) -> bool:
        if not state_subset_matches(case.expected_final_state, state):
            return False
        traces_by_turn: Dict[str, DialogueTurnTrace] = {
            t.turn_id: t for t in turn_traces
        }
        events_by_turn: Dict[str, List[ToolEvent]] = {
            t.turn_id: list(t.tool_events) for t in turn_traces
        }
        for spec in case.turns:
            # Phase 3D: ``expected_state_delta`` means "this turn must
            # newly cause this change" — the same predicate the
            # evaluator scores, so ``completed`` can no longer mask an
            # unsatisfied delta obligation.
            if spec.expected_state_delta:
                trace = traces_by_turn.get(spec.turn_id)
                if trace is None or not delta_obligations_satisfied(
                    spec.expected_state_delta,
                    trace.state_before,
                    trace.state_after,
                ):
                    return False
            if not spec.expected_tool_calls:
                continue
            executed = events_by_turn.get(spec.turn_id, [])
            consumed_positions: Set[int] = set()
            for expected in spec.expected_tool_calls:
                matched = False
                for position, event in enumerate(executed):
                    if position in consumed_positions:
                        continue
                    if event.tool_name != expected.tool_name:
                        continue
                    if event.status != "ok":
                        continue
                    if expected.arguments is not None and not (
                        json_value_equal(event.arguments, expected.arguments)
                    ):
                        continue
                    consumed_positions.add(position)
                    matched = True
                    break
                if not matched:
                    return False
        return True

    # -- manifest binding -----------------------------------------------------

    @staticmethod
    def _validate_case_against_manifest(
        case: DialogueCaseSpec, manifest: ExperimentManifest
    ) -> None:
        if case.benchmark_version != manifest.benchmark_version:
            raise ContractValidationError(
                f"case benchmark_version {case.benchmark_version!r} does "
                f"not match manifest {manifest.benchmark_version!r}"
            )
        if case.case_id not in manifest.case_ids:
            raise ContractValidationError(
                f"case_id {case.case_id!r} is not listed in "
                "manifest.case_ids"
            )


    def _validate_harness_binding(
        self, manifest: ExperimentManifest
    ) -> None:
        """Phase 3D: the executed harness must be exactly the
        fingerprinted ``manifest.dialogue`` config — canonical
        ``to_dict`` equality, fail closed before any case work."""
        if manifest.dialogue is None:
            raise ContractValidationError(
                "manifest.dialogue is required for a formal dialogue "
                "run; no hidden harness defaults are applied"
            )
        if self._config is None:
            raise ContractValidationError(
                "DialogueHarness(config=None) cannot execute a formal "
                "dialogue run; build it from manifest.dialogue"
            )
        if not hasattr(self._config, "to_dict"):
            raise ContractValidationError(
                "harness config must be a DialogueHarnessConfig"
            )
        if not json_value_equal(
            self._config.to_dict(), manifest.dialogue.to_dict()
        ):
            raise ContractValidationError(
                "harness config does not match manifest.dialogue; the "
                "fingerprinted configuration is the only permitted "
                "source of orchestration parameters"
            )


def run_dialogue_case(
    case: DialogueCaseSpec,
    manifest: ExperimentManifest,
    system: Any,
) -> DialogueRunTrace:
    """Formal entry point: build the harness from the fingerprinted
    ``manifest.dialogue`` (the single source of truth) and run one
    case to its trace. ``manifest.dialogue=None`` fails closed."""
    harness = DialogueHarness(config=manifest.dialogue)
    return harness.run(case, manifest, system)


def _usage_zero() -> UsageStats:
    return UsageStats(input_tokens=0, output_tokens=0, total_tokens=0)


def _usage_add(left: UsageStats, right: UsageStats) -> UsageStats:
    def _sum(a: Any, b: Any) -> Optional[int]:
        if a is None and b is None:
            return None
        return int(a or 0) + int(b or 0)

    return UsageStats(
        input_tokens=_sum(left.input_tokens, right.input_tokens),
        output_tokens=_sum(left.output_tokens, right.output_tokens),
        total_tokens=_sum(left.total_tokens, right.total_tokens),
    )


__all__ = [
    "DEFAULT_HISTORY_POLICY",
    "DEFAULT_MAX_TOOL_CALLS_PER_TURN",
    "DEFAULT_MAX_TURNS",
    "DEFAULT_TOOL_POLICY",
    "DialogueHarness",
    "TOOL_DESCRIPTORS",
    "run_dialogue_case",
]
