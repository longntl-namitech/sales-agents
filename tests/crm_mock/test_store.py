from sales_agents.crm_mock.store import CrmStore


def test_same_lead_twice_updates_not_duplicates():
    store = CrmStore()

    store.upsert_lead("lead-1", "warm", "first pass")
    store.upsert_lead("lead-1", "hot", "second pass")

    records = store.list_records()
    assert len(records) == 1
    assert records[0]["label"] == "hot" and records[0]["notes"] == "second pass"


def test_recording_outreach_twice_keeps_one_record_with_the_latest_text():
    store = CrmStore()

    store.record_outreach("lead-1", "first draft")
    store.record_outreach("lead-1", "second draft")

    assert [r["outreach"] for r in store.list_records()] == ["second draft"]


def test_recording_outreach_keeps_the_label_set_earlier():
    store = CrmStore()
    store.upsert_lead("lead-1", "hot", "gates pass")

    store.record_outreach("lead-1", "a draft")

    record = store.get_lead("lead-1")
    assert record["label"] == "hot" and record["outreach"] == "a draft"


def test_unknown_lead_is_none():
    assert CrmStore().get_lead("nope") is None


def test_state_persists_in_a_file_between_connections(tmp_path):
    path = tmp_path / "crm.sqlite3"
    CrmStore(path).upsert_lead("lead-1", "hot", "x")

    assert CrmStore(path).get_lead("lead-1")["label"] == "hot"
