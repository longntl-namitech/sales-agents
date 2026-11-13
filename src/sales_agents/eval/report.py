"""The results table. Every number in it comes from a recorded run.

Live tests call `save_measurement` after a real run, which keeps one row per name in
results/measurements.json. `build_report` turns those rows into a markdown table.
Nothing here computes a metric or accepts a number typed by hand.

    uv run python -m sales_agents.eval.report
"""

import json
from dataclasses import asdict, dataclass
from datetime import date as _date
from pathlib import Path

from sales_agents.eval.harness import EvalReport

RESULTS_PATH = Path(__file__).parents[3] / "results" / "measurements.json"


@dataclass
class Measurement:
    name: str  # the row's identity, saving the same name again replaces it
    use_case: str
    n: int
    metrics: dict[str, float]
    tokens: int
    latency_p50: float | None  # seconds per case, None when it was not measured
    date: str
    note: str = ""


def from_eval_report(name: str, use_case: str, report: EvalReport, *, note: str = "") -> Measurement:
    """A row for a run of the lead harness: hot precision and recall, how many calls failed, tokens, latency."""
    return Measurement(
        name=name,
        use_case=use_case,
        n=report.n,
        metrics={"precision (hot)": report.precision, "recall (hot)": report.recall, "errors": float(len(report.errors))},
        tokens=report.cost,
        latency_p50=report.latency_p50,
        date=_date.today().isoformat(),
        note=note,
    )


def load_measurements(path: Path = RESULTS_PATH) -> list[Measurement]:
    if not path.exists():
        return []
    return [Measurement(**row) for row in json.loads(path.read_text())]


def save_measurement(measurement: Measurement, path: Path = RESULTS_PATH) -> None:
    rows = load_measurements(path)
    for i, row in enumerate(rows):
        if row.name == measurement.name:
            rows[i] = measurement  # replace in place, so the table does not reshuffle
            break
    else:
        rows.append(measurement)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(m) for m in rows], indent=2) + "\n")


def build_report(measurements: list[Measurement]) -> str:
    lines = ["| Use case | Agent | n | Metrics | Tokens | p50 s/case | Note |", "|---|---|---|---|---|---|---|"]
    for m in measurements:
        metrics = ", ".join(f"{k} {v:.3f}" for k, v in m.metrics.items())
        latency = "n/a" if m.latency_p50 is None else f"{m.latency_p50:.2f}"
        lines.append(f"| {m.use_case} | {m.name} | {m.n} | {metrics} | {m.tokens:,} | {latency} | {m.note} |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(build_report(load_measurements()))
