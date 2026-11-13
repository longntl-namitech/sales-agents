from sales_agents.eval.harness import EvalReport
from sales_agents.eval.report import Measurement, build_report, from_eval_report, load_measurements, save_measurement


def measurement(name="Qualifier", **overrides):
    fields = dict(
        name=name,
        use_case="Lead qualification",
        n=60,
        metrics={"precision (hot)": 0.952, "recall (hot)": 1.0},
        tokens=28204,
        latency_p50=0.85,
        date="2026-11-13",
        note="with enrichment",
    )
    return Measurement(**{**fields, **overrides})


def test_report_is_a_markdown_table_of_the_measured_numbers():
    table = build_report([measurement()])

    assert table.splitlines()[0].startswith("| Use case")
    assert "precision (hot) 0.952" in table and "28,204" in table and "0.85" in table


def test_an_unmeasured_latency_is_shown_as_not_available_and_survives_saving(tmp_path):
    path = tmp_path / "m.json"
    save_measurement(measurement(latency_p50=None), path)

    row = build_report(load_measurements(path)).splitlines()[2]

    assert "n/a" in row and "0.00" not in row


def test_report_has_one_row_per_measurement():
    rows = build_report([measurement("a"), measurement("b")]).splitlines()

    assert len(rows) == 2 + 2  # header, separator, two rows


def test_saving_the_same_name_replaces_the_old_row_in_place(tmp_path):
    path = tmp_path / "m.json"

    save_measurement(measurement(tokens=1), path)
    save_measurement(measurement("other"), path)
    save_measurement(measurement(tokens=2), path)

    saved = load_measurements(path)
    assert [m.name for m in saved] == ["Qualifier", "other"]
    assert saved[0].tokens == 2


def test_measurement_from_an_eval_report():
    report = EvalReport(
        n=60, precision=0.952, recall=1.0, matrix={}, errors=[("lead-1", "boom")], cost=28204, latency_p50=0.85
    )

    m = from_eval_report("Qualifier", "Lead qualification", report, note="with enrichment")

    assert m.n == 60 and m.tokens == 28204 and m.latency_p50 == 0.85
    assert m.metrics == {"precision (hot)": 0.952, "recall (hot)": 1.0, "errors": 1.0}


def test_a_missing_file_means_no_measurements(tmp_path):
    assert load_measurements(tmp_path / "none.json") == []
