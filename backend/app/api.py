"""Read API over the analysis data in Postgres."""
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app import chat as chat_mod
from app import clips, pg, search

router = APIRouter()

Kind = Literal["video", "segment", "claim", "transcript"]


def _add(where: list, args: list, cond: str, val) -> None:
    where.append(cond)
    args.append(val)


def _rows(cur) -> list[dict]:
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


@router.get("/search")
def search_chunks(
    q: str = Query(..., min_length=2, description="Natural-language or exact-term query"),
    k: int = Query(10, ge=1, le=50),
    kind: list[Kind] | None = Query(None, description="Restrict to chunk kinds; repeat for several"),
):
    """Hybrid search (full-text + vector, rank-fused). Each hit carries a YouTube link that jumps to the moment."""
    hits = search.search(q, k, kind)
    return {"query": q, "count": len(hits), "results": [
        {"video_id": h["video_id"], "title": h["title"], "kind": h["kind"], "start_sec": h["start_sec"], "end_sec": h["end_sec"],
         "text": h["text"], "meta": h["meta"], "link": h["link"], "score": round(h["score"], 5),
         "semantic_rank": h["sem_rank"], "lexical_rank": h["lex_rank"]} for h in hits]}


@router.get("/videos")
def list_videos(
    q: str | None = Query(None, description="Title contains"),
    speaker: str | None = Query(None, description="Speaker name contains"),
    entity: str | None = Query(None, description="Mentions this entity (name contains)"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    where, args = [], []
    if q:
        _add(where, args, "v.title ILIKE %s", f"%{q}%")
    if speaker:
        _add(where, args, "EXISTS (SELECT 1 FROM speakers s WHERE s.video_id = v.id AND s.name ILIKE %s)", f"%{speaker}%")
    if entity:
        _add(where, args, "EXISTS (SELECT 1 FROM video_entities ve JOIN entities e ON e.id = ve.entity_id WHERE ve.video_id = v.id AND e.name ILIKE %s)", f"%{entity}%")
    sql = ("SELECT v.id, v.title, v.url, v.duration_sec, v.upload_date, v.talk_type, v.summary, v.topics FROM videos v"
           + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY v.upload_date DESC NULLS LAST, v.title LIMIT %s OFFSET %s")
    with pg.connect() as c:
        return _rows(c.execute(sql, (*args, limit, offset)))


@router.get("/videos/{video_id}")
def get_video(video_id: str):
    with pg.connect() as c:
        v = _rows(c.execute("SELECT * FROM videos WHERE id = %s", (video_id,)))
        if not v:
            raise HTTPException(404, "video not found")

        def q(sql: str) -> list[dict]:
            return _rows(c.execute(sql, (video_id,)))

        return {
            **v[0],
            "speakers": q("SELECT name, affiliation, role, evidence FROM speakers WHERE video_id = %s"),
            "entities": q("SELECT e.id, e.name, e.kind, ve.segment_idxs FROM video_entities ve JOIN entities e ON e.id = ve.entity_id WHERE ve.video_id = %s ORDER BY e.name"),
            "source_documents": q("SELECT title, url, evidence FROM source_documents WHERE video_id = %s"),
            "quality_flags": q("SELECT location, issue FROM quality_flags WHERE video_id = %s"),
            "segments": q("SELECT idx, start_sec, end_sec, title, summary, on_screen_text, keep, merged_idxs, note FROM segments WHERE video_id = %s ORDER BY idx"),
            "claims": q("SELECT id, claim, claim_type, quote, start_sec, end_sec, segment_idxs, source_cited_on_slide, confidence, quote_verified FROM claims WHERE video_id = %s ORDER BY start_sec"),
        }


@router.get("/claims")
def list_claims(
    q: str | None = Query(None, description="Claim text contains"),
    video_id: str | None = None,
    claim_type: Literal["fact", "quantitative", "forecast", "opinion", "recommendation"] | None = None,
    confidence: Literal["high", "medium", "low"] | None = None,
    verified: bool | None = Query(None, description="Only claims whose quote was found in the transcript or slide text"),
    entity: str | None = Query(None, description="Claim is in a video that mentions this entity (name contains)"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    where, args = [], []
    if q:
        _add(where, args, "c.claim ILIKE %s", f"%{q}%")
    if video_id:
        _add(where, args, "c.video_id = %s", video_id)
    if claim_type:
        _add(where, args, "c.claim_type = %s", claim_type)
    if confidence:
        _add(where, args, "c.confidence = %s", confidence)
    if verified is not None:
        _add(where, args, "c.quote_verified IS NOT DISTINCT FROM %s", verified)
    if entity:
        _add(where, args, "EXISTS (SELECT 1 FROM video_entities ve JOIN entities e ON e.id = ve.entity_id WHERE ve.video_id = c.video_id AND e.name ILIKE %s)", f"%{entity}%")
    sql = ("SELECT c.id, c.video_id, v.title, c.claim, c.claim_type, c.quote, c.start_sec, c.end_sec, c.confidence, c.quote_verified, "
           "v.url || '&t=' || floor(coalesce(c.start_sec, 0))::int || 's' AS link FROM claims c JOIN videos v ON v.id = c.video_id"
           + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY c.video_id, c.start_sec LIMIT %s OFFSET %s")
    with pg.connect() as c:
        return _rows(c.execute(sql, (*args, limit, offset)))


@router.get("/entities")
def list_entities(q: str | None = Query(None, description="Name contains"), limit: int = Query(50, ge=1, le=200)):
    sql = ("SELECT e.id, e.name, e.kind, count(DISTINCT ve.video_id) AS videos FROM entities e LEFT JOIN video_entities ve ON ve.entity_id = e.id"
           + (" WHERE e.name ILIKE %s" if q else "") + " GROUP BY e.id ORDER BY videos DESC, e.name LIMIT %s")
    with pg.connect() as c:
        return _rows(c.execute(sql, ((f"%{q}%",) if q else ()) + (limit,)))


@router.get("/entities/{entity_id}")
def get_entity(entity_id: int):
    with pg.connect() as c:
        e = _rows(c.execute("SELECT id, name, kind FROM entities WHERE id = %s", (entity_id,)))
        if not e:
            raise HTTPException(404, "entity not found")
        vids = _rows(c.execute(
            "SELECT v.id, v.title, v.url, ve.raw_name, ve.segment_idxs FROM video_entities ve JOIN videos v ON v.id = ve.video_id WHERE ve.entity_id = %s ORDER BY v.title",
            (entity_id,)))
        aliases = [r[0] for r in c.execute("SELECT alias FROM entity_aliases WHERE entity_id = %s", (entity_id,)).fetchall()]
        return {**e[0], "aliases": aliases, "videos": vids}


@router.get("/stats")
def stats():
    tables = ("videos", "speakers", "entities", "video_entities", "segments", "claims", "transcript_chunks", "chunks")
    with pg.connect() as c:
        out = {t: c.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in tables}
        out["chunks_embedded"] = c.execute("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL").fetchone()[0]
        out["chunks_by_kind"] = dict(c.execute("SELECT kind, count(*) FROM chunks GROUP BY kind").fetchall())
    return out


# ---- clips + chat ----

@router.get("/videos/{video_id}/media")
def video_media(
    video_id: str,
    start: float = Query(0, ge=0),
    end: float | None = Query(None, ge=0, description="Defaults to start + 60"),
    expires: int = Query(3600, ge=60, le=43200, description="Signed-URL lifetime in seconds (capped by the server's credential lifetime)"),
):
    """Playable URL for a time range: presigned S3 link to the real mp4 (with #t=start,end), plus YouTube fallbacks."""
    v = clips.video_rows([video_id]).get(video_id)
    if not v:
        raise HTTPException(404, "video not found")
    end = end if end is not None else start + 60
    if end <= start:
        raise HTTPException(422, "end must be greater than start")
    if v["duration_sec"] and start >= v["duration_sec"]:
        raise HTTPException(422, f"start is past the end of the video ({v['duration_sec']}s)")
    return {"video_id": video_id, "title": v["title"], "start_sec": start, "end_sec": end, **clips.media_urls(v, start, end, expires)}


@router.get("/videos/{video_id}/transcript")
def video_transcript(video_id: str, start: float = Query(0, ge=0), end: float | None = Query(None, ge=0)):
    """Caption text and slide/segment descriptions for a time range (whole video if end is omitted)."""
    with pg.connect() as c:
        if not c.execute("SELECT 1 FROM videos WHERE id = %s", (video_id,)).fetchone():
            raise HTTPException(404, "video not found")
        hi = end if end is not None else 10**7
        return {"video_id": video_id, "start_sec": start, "end_sec": end,
                "transcript": clips.transcript_for(c, video_id, start, hi), "segments": clips.slides_for(c, video_id, start, hi)}


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=2, max_length=2000)
    history: list[Turn] = Field(default_factory=list, description="Prior turns, oldest first; the client keeps the conversation")
    max_clips: int = Field(4, ge=1, le=5)


@router.post("/chat")
def chat(req: ChatRequest):
    """Answer a question from the indexed talks and return the most useful timestamped clips, each with a playable URL."""
    try:
        return chat_mod.chat(req.message, [t.model_dump() for t in req.history], req.max_clips)
    except RuntimeError as e:
        raise HTTPException(502, str(e)) from e
