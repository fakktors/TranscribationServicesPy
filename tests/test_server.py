from dataclasses import replace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("mcp")

from fastapi.testclient import TestClient  # noqa: E402

from transcribation.server.app import create_app  # noqa: E402

from .conftest import URL, VIDEO_ID  # noqa: E402


@pytest.fixture
def client(settings, fake_stages):
    with TestClient(create_app(settings)) as c:
        yield c


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_submit_wait_and_fetch(client):
    r = client.post("/v1/transcripts?wait=10", json={"url": URL})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "done" and body["id"] == VIDEO_ID
    assert body["links"]["markdown"].endswith(f"/v1/transcripts/{VIDEO_ID}.md")

    md = client.get(f"/v1/transcripts/{VIDEO_ID}.md")
    assert md.status_code == 200 and md.text.startswith("# Test video")
    assert client.get(f"/v1/transcripts/{VIDEO_ID}.json").json()["video_id"] == VIDEO_ID
    assert client.get("/v1/transcripts").json()[0]["id"] == VIDEO_ID

    # second submit is served from cache
    again = client.post("/v1/transcripts", json={"url": URL})
    assert again.status_code == 200 and again.json()["status"] == "done"


def test_rejects_foreign_hosts(client):
    r = client.post("/v1/transcripts", json={"url": "http://169.254.169.254/latest"})
    assert r.status_code == 422


def test_unknown_and_invalid_ids(client):
    assert client.get("/v1/transcripts/nope").status_code == 404
    assert client.get("/v1/transcripts/a.b.c").status_code in (400, 404)


def test_bearer_auth(settings, fake_stages):
    app = create_app(replace(settings, api_token="s3cret"))
    with TestClient(app) as c:
        assert c.get("/healthz").status_code == 200
        assert c.get("/v1/transcripts").status_code == 401
        assert c.get("/v1/transcripts", headers={"Authorization": "Bearer nope"}).status_code == 401
        assert c.get("/v1/transcripts", headers={"Authorization": "Bearer s3cret"}).status_code == 200
        assert c.post("/mcp", json={}).status_code == 401


def _mcp_call(c, method, params=None, id_=1):
    r = c.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": id_, "method": method, "params": params or {}},
        headers={"Accept": "application/json, text/event-stream", "MCP-Protocol-Version": "2025-06-18"},
    )
    assert r.status_code == 200, r.text
    return r.json()


def test_mcp_tools(client):
    init = _mcp_call(
        client,
        "initialize",
        {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "0"},
        },
    )
    assert init["result"]["serverInfo"]["name"] == "transcribation"

    tools = _mcp_call(client, "tools/list", id_=2)["result"]["tools"]
    assert {t["name"] for t in tools} >= {"transcribe_video", "get_transcript", "list_transcripts"}

    res = _mcp_call(
        client, "tools/call", {"name": "transcribe_video", "arguments": {"url": URL, "wait_seconds": 10}}, 3
    )
    text = res["result"]["content"][0]["text"]
    assert "status: done" in text and "# Test video" in text

    page = _mcp_call(
        client,
        "tools/call",
        {
            "name": "get_transcript",
            "arguments": {"id": VIDEO_ID, "offset": 0, "max_chars": 1000},
        },
        4,
    )["result"]["content"][0]["text"]
    assert "chars: 0-" in page

    bad = _mcp_call(client, "tools/call", {"name": "get_transcript", "arguments": {"id": "../../etc"}}, 5)
    assert "invalid_id" in bad["result"]["content"][0]["text"]
