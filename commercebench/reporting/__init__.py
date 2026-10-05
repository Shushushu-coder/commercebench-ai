"""Experiment reporting, comparison, and regression analysis."""

from .reranker_analysis import (
    REPORT_VERSION,
    RankMovement,
    RerankerCaseAnalysis,
    analyze_reranker_case,
    render_reranker_case_summary,
)
from .failure_attribution import (
    DiagnosticObservationKind,
    FailureAttribution,
    FailureAttributionReport,
    FailureEvidence,
    FailureStage,
    RetrievalMissAttribution,
    RetrievalMissOrigin,
    analyze_failure_attribution,
    derive_comparison_observations,
    render_failure_attribution_summary,
)
from .reranker_comparison import (
    RerankerComparison,
    RerankerComparisonSummary,
    compare_reranker_runs,
    summarize_reranker_comparison,
)

__all__ = [
    "REPORT_VERSION",
    "DiagnosticObservationKind",
    "FailureAttribution",
    "FailureAttributionReport",
    "FailureEvidence",
    "FailureStage",
    "RankMovement",
    "RerankerCaseAnalysis",
    "RerankerComparison",
    "RerankerComparisonSummary",
    "RetrievalMissAttribution",
    "RetrievalMissOrigin",
    "analyze_failure_attribution",
    "analyze_reranker_case",
    "compare_reranker_runs",
    "derive_comparison_observations",
    "render_failure_attribution_summary",
    "render_reranker_case_summary",
    "summarize_reranker_comparison",
]
