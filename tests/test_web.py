import json
from pathlib import Path

from fastapi.testclient import TestClient
from test_cli import DATASET, GOOD, LEAD, models

from sales_agents.crm_mock.store import CrmStore
from sales_agents.tools.crm import store_tools
from sales_agents.web import create_app


def client():
    return TestClient(create_app(dataset=DATASET, raw_tools=store_tools(CrmStore()), models=models()))


def events(response):
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def test_leads_are_listed():
    leads = client().get("/api/leads").json()

    assert any(item["lead_id"] == LEAD.lead_id and item["company_name"] == LEAD.company_name for item in leads)


def test_the_page_is_served():
    assert "<title>" in client().get("/").text


def test_a_run_shows_each_agent_then_pauses_for_approval_with_the_draft():
    c = client()

    run = c.post("/api/runs", json={"lead_id": LEAD.lead_id})
    seen = events(run)

    agents = [e["agent"] for e in seen if e["type"] == "step"]
    assert agents[:3] == ["researcher", "qualifier", "crm_agent"]
    assert seen[-1]["type"] == "approval" and seen[-1]["draft_text"] == GOOD


def test_approving_records_the_draft_and_ends_with_the_crm_record():
    c = client()
    run_id = events(c.post("/api/runs", json={"lead_id": LEAD.lead_id}))[0]["run_id"]

    seen = events(c.post(f"/api/runs/{run_id}/decision", json={"type": "approve"}))

    assert seen[-1]["type"] == "result"
    assert seen[-1]["record"]["label"] == "hot" and seen[-1]["record"]["outreach"] == GOOD


def test_rejecting_leaves_the_record_without_outreach():
    c = client()
    run_id = events(c.post("/api/runs", json={"lead_id": LEAD.lead_id}))[0]["run_id"]

    seen = events(c.post(f"/api/runs/{run_id}/decision", json={"type": "reject", "message": "no"}))

    assert seen[-1]["type"] == "result" and not seen[-1]["record"].get("outreach")


def test_an_edited_draft_is_what_the_crm_records():
    c = client()
    run_id = events(c.post("/api/runs", json={"lead_id": LEAD.lead_id}))[0]["run_id"]
    edited = GOOD.replace("Hi Dana", "Hello Dana")

    seen = events(c.post(f"/api/runs/{run_id}/decision", json={"type": "edit", "text": edited}))

    assert seen[-1]["record"]["outreach"] == edited


def test_an_unknown_lead_is_a_404():
    assert client().post("/api/runs", json={"lead_id": "nope"}).status_code == 404


def test_an_unknown_run_is_a_404():
    assert client().post("/api/runs/zzz/decision", json={"type": "approve"}).status_code == 404


def test_the_approval_is_sent_once_even_though_the_interrupt_surfaces_twice():
    seen = events(client().post("/api/runs", json={"lead_id": LEAD.lead_id}))

    assert sum(e["type"] == "approval" for e in seen) == 1
