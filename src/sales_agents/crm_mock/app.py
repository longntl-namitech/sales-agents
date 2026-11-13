"""The mock CRM as a small HTTP service (D2: a mock, with HubSpot as a later swap)."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sales_agents.crm_mock.store import CrmStore


class LeadBody(BaseModel):
    label: str
    notes: str = ""


class OutreachBody(BaseModel):
    draft_text: str


def create_app(store: CrmStore) -> FastAPI:
    app = FastAPI(title="Mock CRM")

    @app.put("/leads/{lead_id}")
    def put_lead(lead_id: str, body: LeadBody) -> dict:
        return store.upsert_lead(lead_id, body.label, body.notes)

    @app.post("/leads/{lead_id}/outreach")
    def post_outreach(lead_id: str, body: OutreachBody) -> dict:
        return store.record_outreach(lead_id, body.draft_text)

    @app.get("/leads")
    def list_leads() -> list[dict]:
        return store.list_records()

    @app.get("/leads/{lead_id}")
    def get_lead(lead_id: str) -> dict:
        record = store.get_lead(lead_id)
        if record is None:
            raise HTTPException(status_code=404, detail="no such lead")
        return record

    return app
