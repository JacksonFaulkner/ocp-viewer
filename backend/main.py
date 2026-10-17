from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse

from app import config, db, ingest
from app.api import router as analysis_router

app = FastAPI(title="OCP video analysis", description="Search and browse analysed Open Compute Project talks; run the YouTube ingest.")
app.include_router(analysis_router)

_running = False
_INDEX = Path(__file__).resolve().parents[1] / "static" / "index.html"


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(_INDEX, headers={"Cache-Control": "no-store"})


@app.get("/health")
def health():
    return {"ok": True}


# ---- ingest (yt-dlp -> S3); state lives in SQLite ----

@app.get("/ingest/videos")
def ingest_videos(status: str | None = None, limit: int = 100, offset: int = 0):
    q, args = "SELECT * FROM videos", []
    if status:
        q += " WHERE status = ?"
        args.append(status)
    q += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
    with db.conn() as c:
        return [dict(r) for r in c.execute(q, (*args, limit, offset))]


@app.get("/ingest/stats")
def ingest_stats():
    with db.conn() as c:
        rows = c.execute("SELECT status, COUNT(*) n FROM videos GROUP BY status").fetchall()
    return {r["status"]: r["n"] for r in rows}


def _job(limit: int | None, retry_failed: bool) -> None:
    global _running
    try:
        ingest.run(limit, retry_failed)
    finally:
        _running = False


@app.post("/ingest", status_code=202)
def start_ingest(bg: BackgroundTasks, limit: int | None = None, retry_failed: bool = False):
    global _running
    if not config.S3_BUCKET:
        raise HTTPException(500, "S3_BUCKET not set")
    if _running:
        raise HTTPException(409, "ingest already running")
    _running = True
    bg.add_task(_job, limit, retry_failed)
    return {"started": True}
