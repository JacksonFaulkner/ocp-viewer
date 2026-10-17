"""Question answering over the talk corpus, with timestamped video clips as evidence."""
import time

import boto3
import jsonschema
from botocore.config import Config

from app import clips, config, pg, search

CHAT_MODEL = "us.anthropic.claude-sonnet-4-6"
HISTORY_TURNS = 6

ANSWER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "found": {"type": "boolean", "description": "false if the candidate clips do not actually answer the question"},
        "answer": {"type": "string", "description": "Markdown answer grounded only in the candidate clips; cite clips like [1] or [2][3]"},
        "clips": {"type": "array", "maxItems": 5, "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"clip_id": {"type": "integer"}, "why_watch": {"type": "string", "description": "One sentence: what the viewer will see or learn in this clip"}},
            "required": ["clip_id", "why_watch"]}},
    },
    "required": ["found", "answer", "clips"],
}
ANSWER_TOOL = {"name": "answer", "description": "Answer the user's question and choose the clips worth watching.", "inputSchema": {"json": ANSWER_SCHEMA}}

SYSTEM = """You are a research assistant for a library of Open Compute Project conference talks.
You get the user's question and numbered candidate clips: each is a stretch of one talk, with a summary, on-screen text and transcript.
Rules:
- Answer using only the candidate clips. If they do not answer the question, set found=false, say so plainly, and return no clips.
- Cite clips inline as [n]. Do not state facts that no clip supports. Speech transcripts can contain recognition errors; prefer the on-screen text for names and numbers and say when sources conflict.
- Choose the 1-5 clips most worth watching, best first, and prefer clips from different talks unless one talk clearly has the best material. Give each a one-sentence why_watch.
- Be concise: a short direct answer first, then supporting detail if useful."""


def _client():
    return boto3.client("bedrock-runtime", region_name=config.AWS_REGION, config=Config(read_timeout=120, retries={"max_attempts": 2}))


def _text(resp) -> str:
    return "".join(b.get("text", "") for b in resp["output"]["message"]["content"]).strip()


def rewrite_query(brt, message: str, history: list[dict]) -> str:
    """Make a follow-up question standalone so retrieval does not depend on the chat history."""
    if not history:
        return message
    convo = "\n".join(f"{t['role']}: {t['content'][:600]}" for t in history[-HISTORY_TURNS:])
    resp = brt.converse(modelId=CHAT_MODEL, messages=[{"role": "user", "content": [{"text": (
        f"Conversation so far:\n{convo}\n\nNew question: {message}\n\n"
        "Rewrite the new question as one standalone search query that includes any needed context. Output only the query.")}]}],
        inferenceConfig={"maxTokens": 120, "temperature": 0})
    return _text(resp).strip('"') or message


def _candidate_text(c, cand: dict) -> str:
    slides = clips.slides_for(c, cand["video_id"], cand["start"], cand["end"])
    words = clips.transcript_for(c, cand["video_id"], cand["start"], cand["end"])
    parts = [f"Talk: {cand['title']} ({int(cand['start'])}s-{int(cand['end'])}s)"]
    for s in slides[:4]:
        parts.append(f"Slide/segment: {s['title']}. {s['summary']}" + (f" On screen: {s['on_screen_text'][:400]}" if s["on_screen_text"] else ""))
    claim_hits = [h["text"] for h in cand["hits"] if h["kind"] == "claim"]
    if claim_hits:
        parts.append("Extracted claims: " + " | ".join(claim_hits[:5]))
    if words:
        parts.append("Transcript: " + " ".join(w["text"] for w in words)[:1800])
    return "\n".join(parts)


def chat(message: str, history: list[dict] | None = None, max_clips: int = 4, expires: int = 3600) -> dict:
    t0 = time.monotonic()
    history = history or []
    brt = _client()
    query = rewrite_query(brt, message, history)
    hits = search.search(query, k=30, kinds=["segment", "claim", "transcript"])
    vids = clips.video_rows(sorted({h["video_id"] for h in hits}))
    cands = clips.build_candidates(hits, {k: v["duration_sec"] for k, v in vids.items()})
    base = {"query_used": query, "candidates": len(cands)}
    if not cands:
        return {"answer": "I couldn't find anything relevant in the indexed talks.", "found": False, "clips": [], **base, "secs": round(time.monotonic() - t0, 1)}

    with pg.connect() as c:
        for i, cand in enumerate(cands, 1):
            cand["id"] = i
            cand["text"] = _candidate_text(c, cand)
    prompt = f"Question: {message}\n\nCandidate clips:\n\n" + "\n\n".join(f"[{c['id']}] {c['text']}" for c in cands)
    msgs = [{"role": t["role"], "content": [{"text": t["content"]}]} for t in history[-HISTORY_TURNS:]] + [{"role": "user", "content": [{"text": prompt}]}]

    out, err = None, ""
    for _ in range(2):
        resp = brt.converse(modelId=CHAT_MODEL, system=[{"text": SYSTEM}], messages=msgs,
                            toolConfig={"tools": [{"toolSpec": ANSWER_TOOL}], "toolChoice": {"tool": {"name": "answer"}}},
                            inferenceConfig={"maxTokens": 2000, "temperature": 0.2})
        cand_out = next((b["toolUse"]["input"] for b in resp["output"]["message"]["content"] if "toolUse" in b), None)
        try:
            jsonschema.validate(cand_out, ANSWER_SCHEMA)
            bad = [x["clip_id"] for x in cand_out["clips"] if not 1 <= x["clip_id"] <= len(cands)]
            if bad:
                raise jsonschema.ValidationError(f"unknown clip_id {bad}; valid ids are 1-{len(cands)}")
            out = cand_out
            break
        except jsonschema.ValidationError as e:
            err = e.message[:200]
            msgs[-1] = {"role": "user", "content": [{"text": prompt + f"\n\nYour previous answer was invalid: {err}. Call the tool again."}]}
    if out is None:
        raise RuntimeError(f"model returned an invalid answer: {err}")

    chosen, seen = [], set()
    with pg.connect() as c:
        for pick in out["clips"]:
            if pick["clip_id"] in seen or len(chosen) >= max_clips:
                continue
            seen.add(pick["clip_id"])
            cand = cands[pick["clip_id"] - 1]
            v = vids[cand["video_id"]]
            chosen.append({
                "id": cand["id"], "video_id": cand["video_id"], "title": cand["title"],
                "start_sec": round(cand["start"], 1), "end_sec": round(cand["end"], 1),
                "why_watch": pick["why_watch"],
                "transcript": " ".join(w["text"] for w in clips.transcript_for(c, cand["video_id"], cand["start"], cand["end"])),
                **clips.media_urls(v, cand["start"], cand["end"], expires),
            })
    return {"answer": out["answer"], "found": out["found"], "clips": chosen, **base, "secs": round(time.monotonic() - t0, 1)}
