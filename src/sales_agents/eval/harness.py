"""Run an agent function over labelled leads and score it against gold labels.

A failed call (timeout, invalid output) is recorded as a missing prediction, so
it counts against the agent and never shrinks the denominator.
"""

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from sales_agents.eval.metrics import confusion_matrix, precision_recall
from sales_agents.schemas import LABELS, Label, LabelledLead, Lead, QualificationResult


class TokenCapExceeded(RuntimeError):
    pass


@dataclass
class EvalReport:
    n: int
    precision: float
    recall: float
    matrix: dict[tuple[Label, Label | None], int]
    errors: list[tuple[str, str]] = field(default_factory=list)
    cost: int = 0  # tokens actually used, when the caller can count them
    latency_p50: float = 0.0  # seconds per lead
    estimated_tokens: int = 0


def run_eval(
    cases: list[LabelledLead],
    fn: Callable[[Lead], QualificationResult],
    *,
    token_cap: int,
    estimated_tokens: int = 0,
    count_tokens: Callable[[], int] | None = None,
) -> EvalReport:
    if estimated_tokens > token_cap:
        raise TokenCapExceeded(
            f"estimated {estimated_tokens} tokens exceeds the cap of {token_cap}; "
            "run a smaller subset"
        )

    predictions: list[Label | None] = []
    errors: list[tuple[str, str]] = []
    latencies: list[float] = []
    for case in cases:
        start = time.perf_counter()
        try:
            result = fn(case.lead)
            if not isinstance(result, QualificationResult):
                raise TypeError(f"expected QualificationResult, got {type(result).__name__}")
            predictions.append(result.label)
        except Exception as exc:  # any failure is a wrong prediction, not a dropped case
            predictions.append(None)
            errors.append((case.lead.lead_id, f"{type(exc).__name__}: {exc}"))
        latencies.append(time.perf_counter() - start)

    gold = [c.gold for c in cases]
    precision, recall = precision_recall(gold, predictions)
    return EvalReport(
        n=len(cases),
        precision=precision,
        recall=recall,
        matrix=confusion_matrix(gold, predictions),
        errors=errors,
        cost=count_tokens() if count_tokens else 0,
        latency_p50=statistics.median(latencies) if latencies else 0.0,
        estimated_tokens=estimated_tokens,
    )


def format_table(rows: dict[str, EvalReport]) -> str:
    lines = [
        f"{'agent':<20}{'n':>5}{'precision(hot)':>16}{'recall(hot)':>13}{'errors':>8}{'tokens':>10}{'p50 s/lead':>12}"
    ]
    for name, r in rows.items():
        lines.append(
            f"{name:<20}{r.n:>5}{r.precision:>16.3f}{r.recall:>13.3f}{len(r.errors):>8}{r.cost:>10}{r.latency_p50:>12.4f}"
        )
    for name, r in rows.items():
        lines.append("")
        lines.append(f"{name}: confusion matrix (rows = gold, columns = predicted)")
        lines.append(f"{'':<8}" + "".join(f"{col:>8}" for col in (*LABELS, "error")))
        for true in LABELS:
            cells = "".join(f"{r.matrix[(true, p)]:>8}" for p in (*LABELS, None))
            lines.append(f"{true:<8}{cells}")
    return "\n".join(lines)


if __name__ == "__main__":
    from pathlib import Path

    from sales_agents.agents.baseline import make_rule_baseline
    from sales_agents.config import get_settings
    from sales_agents.datagen import load_leads

    cases = load_leads(Path("datasets/leads_seed42.jsonl"))
    report = run_eval(cases, make_rule_baseline(cases), token_cap=get_settings().eval_token_cap)
    print(format_table({"rule baseline": report}))
