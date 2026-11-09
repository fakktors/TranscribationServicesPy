"""HTTP service: REST API (/v1), MCP endpoint (/mcp) and a health check.

Run with ``transcribation-server`` (or ``uvicorn transcribation.server.app:create_app --factory``).
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import logging
import os
import re
import shutil
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, Field

from .. import __version__
from ..asr import resolve_backend_name, resolve_model_name
from ..config import Settings, hf_token
from ..diarize import diarization_available
from ..download import is_url_allowed, normalize_url
from ..pipeline import Options
from .jobs import DONE, JobManager
from .mcp_tools import build_mcp

log = logging.getLogger(__name__)

_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


class TranscribeRequest(BaseModel):
    url: str = Field(description="YouTube URL or 11-character video id")
    language: str | None = Field(None, description="ISO language code; omit to auto-detect")
    prompt: str | None = Field(None, description="comma-separated vocabulary hint")
    diarize: bool = True
    allow_long: bool = False
    force: bool = Field(False, description="discard cached results and start over")


class _BearerAuth:
    """Pure ASGI middleware (does not buffer streaming MCP responses)."""

    def __init__(self, app, token: str | None, open_paths: tuple[str, ...] = ("/healthz",)):
        self.app, self.token, self.open_paths = app, token, open_paths

    async def __call__(self, scope, receive, send):
        if self.token and scope["type"] == "http" and scope["path"] not in self.open_paths:
            headers = dict(scope.get("headers") or [])
            auth = headers.get(b"authorization", b"").decode("latin-1")
            supplied = auth[7:] if auth.lower().startswith("bearer ") else ""
            if not hmac.compare_digest(supplied.encode(), self.token.encode()):
                resp = JSONResponse(
                    {"detail": "missing or invalid bearer token"},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
                await resp(scope, receive, send)
                return
        await self.app(scope, receive, send)


def create_app(settings: Settings | None = None, manager: JobManager | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    manager = manager or JobManager(settings)
    mcp = build_mcp(manager)
    mcp_app = mcp.streamable_http_app(
        streamable_http_path="/mcp",
        stateless_http=True,
        json_response=True,
        # Host/Origin checks are for browser-facing localhost servers; this service
        # is protected by the bearer token and normally sits behind a proxy.
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        settings.work_dir.mkdir(parents=True, exist_ok=True)
        settings.transcripts_dir.mkdir(parents=True, exist_ok=True)
        if not settings.api_token:
            log.warning("TRANSCRIBE_API_TOKEN is not set — the API is open to anyone who can reach it")
        manager.start()
        async with mcp.session_manager.run():
            yield
        manager.stop()

    app = FastAPI(title="transcribation", version=__version__, lifespan=lifespan)
    app.state.manager = manager

    def _require(job_id: str) -> dict:
        if not _ID_RE.fullmatch(job_id):
            raise HTTPException(400, "invalid id")
        desc = manager.describe(job_id)
        if desc is None:
            raise HTTPException(404, f"unknown id {job_id!r}")
        return desc

    def _with_links(desc: dict, request: Request) -> dict:
        if desc["status"] == DONE:
            base = str(request.base_url).rstrip("/")
            desc["links"] = {
                "markdown": f"{base}/v1/transcripts/{desc['id']}.md",
                "json": f"{base}/v1/transcripts/{desc['id']}.json",
            }
        return desc

    @app.get("/healthz", tags=["meta"])
    def healthz() -> dict:
        return {
            "status": "ok",
            "version": __version__,
            "asr_backend": resolve_backend_name(settings),
            "asr_model": resolve_model_name(settings),
            "diarization": diarization_available() and hf_token() is not None,
            "jobs_in_flight": manager.queue_size(),
        }

    @app.post("/v1/transcripts", tags=["transcripts"])
    async def submit(
        body: TranscribeRequest,
        request: Request,
        wait: int = Query(0, ge=0, description="seconds to wait for completion before answering"),
    ) -> JSONResponse:
        url = normalize_url(body.url)
        if not is_url_allowed(url, settings.allowed_hosts):
            raise HTTPException(422, f"URL host not allowed (allowed: {', '.join(settings.allowed_hosts)})")
        try:
            opts = Options(
                url=url, language=body.language, prompt=body.prompt, diarize=body.diarize, allow_long=body.allow_long
            )
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        job = manager.submit(opts, force=body.force)
        if wait:
            await asyncio.to_thread(job.event.wait, min(wait, settings.max_wait))
        desc = _with_links(manager.describe(job.id), request)
        return JSONResponse(desc, status_code=200 if desc["status"] in (DONE, "error") else 202)

    @app.get("/v1/transcripts", tags=["transcripts"])
    def list_transcripts(limit: int = Query(50, ge=1, le=500)) -> list[dict]:
        return manager.list_transcripts(limit=limit)

    @app.get("/v1/transcripts/{job_id}.md", response_class=PlainTextResponse, tags=["transcripts"])
    def get_markdown(job_id: str):
        return _file(job_id, "md", "text/markdown; charset=utf-8")

    @app.get("/v1/transcripts/{job_id}.json", tags=["transcripts"])
    def get_json(job_id: str):
        return _file(job_id, "json", "application/json")

    def _file(job_id: str, ext: Literal["md", "json"], media_type: str):
        desc = _require(job_id)
        path = settings.transcripts_dir / f"{job_id}.{ext}"
        if desc["status"] != DONE or not path.exists():
            raise HTTPException(409, f"transcript not ready (status: {desc['status']})")
        return FileResponse(path, media_type=media_type)

    @app.get("/v1/transcripts/{job_id}", tags=["transcripts"])
    async def get_status(
        job_id: str, request: Request, wait: int = Query(0, ge=0, description="seconds to wait for completion")
    ) -> dict:
        job = manager.get(job_id)
        if job is not None and wait:
            await asyncio.to_thread(job.event.wait, min(wait, settings.max_wait))
        return _with_links(_require(job_id), request)

    @app.delete("/v1/transcripts/{job_id}", status_code=204, tags=["transcripts"])
    def delete(job_id: str) -> None:
        desc = _require(job_id)
        if desc["status"] in ("queued", "running"):
            raise HTTPException(409, "job is in progress")
        shutil.rmtree(settings.work_dir / job_id, ignore_errors=True)
        for ext in ("md", "json"):
            (settings.transcripts_dir / f"{job_id}.{ext}").unlink(missing_ok=True)

    # MCP Streamable HTTP endpoint at /mcp (mounted last: API routes take precedence)
    app.mount("/", mcp_app)
    if settings.api_token:
        app.add_middleware(_BearerAuth, token=settings.api_token)
    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(
        "transcribation.server.app:create_app",
        factory=True,
        host=os.environ.get("TRANSCRIBE_HOST", "0.0.0.0"),
        port=int(os.environ.get("TRANSCRIBE_PORT", "8000")),
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
