"""Two-stage video analysis. Stage 1: Pegasus 1.5 describes 3-minute clips. Stage 2: Kimi K3 makes the judgment calls."""
import json
import re
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import boto3
import jsonschema
import yt_dlp
from botocore.config import Config

from app import config

PEGASUS = "global.twelvelabs.pegasus-1-5-v1:0"  # same price as us.; 24 req/min quota vs 12
STAGE2_MODEL = "us.moonshotai.kimi-k3"
CLIP_SEC = 180
PEGASUS_USD_PER_SEC = 0.00049
PEGASUS_USD_PER_M_OUT = 7.50
# $/M tokens, regional standard. Bedrock auto-caches prompts on some models: `inputTokens` is then only the uncached tail,
# the rest comes back as cacheWrite / cacheRead. Kimi cache-write is 1.25x input; cache-read (0.1x input) is an assumption,
# since only the flex/priority rates are in the price list. Nova 2 Lite cache rates are assumed to follow the same pattern.
MODEL_USD_PER_M = {
    "us.moonshotai.kimi-k3": {"input": 3.30, "cache_write": 4.125, "cache_read": 0.33, "output": 16.50},
    "us.amazon.nova-2-lite-v1:0": {"input": 0.33, "cache_write": 0.33, "cache_read": 0.033, "output": 2.75},
    "us.anthropic.claude-sonnet-4-6": {"input": 3.30, "cache_write": 4.125, "cache_read": 0.33, "output": 16.50},
    # Haiku 4.5: input 1.10 regional is from the price list; output 5.50 is assumed from the 1:5 list ratio
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": {"input": 1.10, "cache_write": 1.375, "cache_read": 0.11, "output": 5.50},
}
SCHEMA_VERSION = 2

SEGMENT_DEFS = [{
    "id": "section",
    "description": "A distinct stretch of the video covering one idea or slide; typically 20s-2min. Describe only what is said and shown; do not guess names or intent.",
    "fields": [
        {"name": "title", "type": "string", "description": "Short heading"},
        {"name": "summary", "type": "string", "description": "2-3 sentences, self-contained, covering what the speaker says"},
        {"name": "on_screen_text", "type": "string", "description": "Text, labels, numbers, chart axes and footnotes, names and URLs visible on screen, copied exactly; empty if none"},
    ],
}]

_IDX = {"type": "array", "items": {"type": "integer"}}
DOC_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "3-5 sentences"},
        "talk_type": {"type": "string", "enum": ["keynote", "technical talk", "panel", "demo", "announcement", "ceremony", "welcome", "other"]},
        "session": {"type": "object", "properties": {
            "title": {"type": "string"}, "event": {"type": "string"}, "track": {"type": "string"}}, "required": ["title"]},
        "speakers": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "affiliation": {"type": "string"}, "role": {"type": "string"},
            "evidence": {"type": "string", "description": "Where the name came from: on-screen card, description, spoken"}},
            "required": ["name", "evidence"]}},
        "topics": {"type": "array", "items": {"type": "string"}, "description": "3-8 specific topics"},
        "entities": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string", "description": "Canonical spelling"},
            "kind": {"type": "string", "enum": ["organization", "product", "architecture", "specification", "project", "hardware", "other"]},
            "segment_idxs": _IDX}, "required": ["name", "kind", "segment_idxs"]}},
        "claims": {"type": "array", "description": "Atomic claims, one assertion each", "items": {"type": "object", "properties": {
            "claim": {"type": "string"},
            "claim_type": {"type": "string", "enum": ["fact", "quantitative", "forecast", "opinion", "recommendation"]},
            "quote": {"type": "string", "description": "Exact words copied from the transcript or on-screen text that support the claim; empty if none"},
            "start_sec": {"type": "number"}, "end_sec": {"type": "number"},
            "segment_idxs": _IDX,
            "source_cited_on_slide": {"type": "string", "description": "Footnote/source text shown for this claim; empty if none"},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]}},
            "required": ["claim", "claim_type", "quote", "start_sec", "end_sec", "segment_idxs", "confidence"]}},
        "source_documents": {"type": "array", "items": {"type": "object", "properties": {
            "title": {"type": "string"}, "url": {"type": "string"}, "evidence": {"type": "string"}}, "required": ["title", "evidence"]}},
        "quality_flags": {"type": "array", "items": {"type": "object", "properties": {
            "where": {"type": "string"}, "issue": {"type": "string"}}, "required": ["where", "issue"]}},
        "segments": {"type": "array", "description": "One entry per input segment", "items": {"type": "object", "properties": {
            "idx": {"type": "integer"},
            "keep": {"type": "boolean", "description": "false for title cards, filler, or segments that look hallucinated"},
            "merge_with_prev": {"type": "boolean", "description": "true if this continues the previous segment's idea (e.g. split by a clip boundary)"},
            "note": {"type": "string"}}, "required": ["idx", "keep", "merge_with_prev"]}},
    },
    "required": ["summary", "talk_type", "session", "speakers", "topics", "entities", "claims", "segments"],
}
def _forbid_extra(node):
    if isinstance(node, dict):
        if node.get("type") == "object":
            node["additionalProperties"] = False
        for v in node.values():
            _forbid_extra(v)
    elif isinstance(node, list):
        for v in node:
            _forbid_extra(v)


_forbid_extra(DOC_SCHEMA)
DOC_TOOL = {"name": "record_document", "description": "Record the document-level analysis of the talk.", "inputSchema": {"json": DOC_SCHEMA}}

SYSTEM = """You analyze conference talks from the Open Compute Project (OCP) YouTube channel for a searchable knowledge base.
Inputs: the YouTube title, description and chapters; timestamped segment descriptions produced by a video model (it can misread or invent details); and, when available, a timestamped caption transcript.
Rules:
- Use only what the inputs support. Never invent speaker names; if no source gives one, return an empty speakers list.
- Claims must be atomic (one assertion each) and typed: fact, quantitative (contains a number), forecast, opinion, or recommendation. Cover the substantive claims across the whole talk: aim for roughly 1-2 per minute of video (at least 8, at most 40).
- Output only the fields in the schema; never add extra fields. Keep every string under 300 characters.
- quote must be copied exactly from the transcript or a segment's on_screen_text. If you cannot copy one exactly, leave it empty and lower confidence.
- start_sec/end_sec must lie within the supporting segments' time ranges.
- Use canonical spellings for entities (OpenTelemetry, not Open Telemetry). Do NOT list the Open Compute Project, the OCP Global Summit or YouTube as entities. Entity kinds: organization, product, architecture (a design approach like 800 VDC), specification (a published standard or spec), project, hardware, other.
- Put any white paper, report, deck or URL the speaker shows or mentions in source_documents.
- Add quality_flags for suspicious text (likely OCR or model slips, e.g. units that look wrong), details present in a segment summary but not on screen or in the transcript, and conflicts between inputs.
- For segments: keep=false for title/thank-you cards and unreliable segments; merge_with_prev=true when a segment continues the previous one (clips are cut every ~3 minutes, so ideas are often split at the cut)."""

PEGASUS_CONCURRENCY = 3  # account quota: 3 concurrent InvokeModel requests
_pegasus_slots = threading.BoundedSemaphore(PEGASUS_CONCURRENCY)
_captions_lock = threading.Lock()


def clients():
    sess = boto3.Session(region_name=config.AWS_REGION)
    cfg = Config(read_timeout=900, retries={"max_attempts": 2, "mode": "standard"}, max_pool_connections=32)
    return sess.client("s3"), sess.client("bedrock-runtime", config=cfg), sess.client("sts").get_caller_identity()["Account"]


def probe_duration(path: Path) -> float:
    out = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)])
    return float(out)


def make_clips(src: Path, outdir: Path) -> list[tuple[Path, float, float]]:
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-c", "copy", "-map", "0", "-f", "segment",
                    "-segment_time", str(CLIP_SEC), "-reset_timestamps", "1", str(outdir / "clip_%03d.mp4")], check=True)
    clips, offset = [], 0.0
    for p in sorted(outdir.glob("clip_*.mp4")):
        d = probe_duration(p)
        clips.append((p, offset, d))
        offset += d
    return clips


def _ts(sec: float) -> str:
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"


def fetch_captions(vid: str, workdir: Path) -> list[dict]:
    """Auto/manual English captions -> [{start_sec, text}] in ~15s windows. Empty list if unavailable."""
    opts = {"skip_download": True, "writesubtitles": True, "writeautomaticsub": True, "subtitleslangs": ["en", "en-orig"],
            "subtitlesformat": "vtt", "outtmpl": str(workdir / "%(id)s"), "quiet": True, "noprogress": True, "retries": 3}
    with _captions_lock:  # YouTube rate-limits subtitle requests; go one at a time
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={vid}"])
        except Exception:
            pass  # a failed extra track must not discard a track that did download
        time.sleep(1.5)
    files = sorted(workdir.glob(f"{vid}*.vtt"), key=lambda f: (f.name != f"{vid}.en.vtt", f.name))
    if not files:
        return []
    cues, last = [], ""
    t_re = re.compile(r"(\d+):(\d+):(\d+)\.(\d+)\s+-->")
    start = None
    for line in files[0].read_text(errors="ignore").splitlines():
        m = t_re.match(line)
        if m:
            h, mi, s, ms = map(int, m.groups())
            start = h * 3600 + mi * 60 + s + ms / 1000
            continue
        text = re.sub(r"<[^>]+>", "", line).strip()
        if start is None or not text or text.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")) or text == last:
            continue
        cues.append((start, text))
        last = text
    out, cur, cur_t = [], [], None
    for t, text in cues:
        if cur_t is None:
            cur_t = t
        if t - cur_t >= 15 and cur:
            out.append({"start_sec": round(cur_t, 1), "text": " ".join(cur)})
            cur, cur_t = [], t
        cur.append(text)
    if cur:
        out.append({"start_sec": round(cur_t, 1), "text": " ".join(cur)})
    return out


def run_pegasus(brt, uri: str, account: str) -> dict:
    body = {"analysis_mode": "time_based_metadata", "temperature": 0, "maxOutputTokens": 8192, "segment_definitions": SEGMENT_DEFS,
            "mediaSource": {"s3Location": {"uri": uri, "bucketOwner": account}}}
    for attempt in range(8):
        with _pegasus_slots:
            t0 = time.monotonic()
            try:
                r = json.loads(brt.invoke_model(modelId=PEGASUS, body=json.dumps(body))["body"].read())
            except brt.exceptions.ThrottlingException:
                r = None
        if r is not None:
            r["_secs"] = time.monotonic() - t0
            return r
        time.sleep(min(15 * (attempt + 1), 90))  # outside the slot so others can proceed
    raise RuntimeError("Pegasus throttled after 8 attempts")


def stage1(vid: str, s3, brt, account: str, pool: ThreadPoolExecutor) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / f"{vid}.mp4"
        s3.download_file(config.S3_BUCKET, f"{config.S3_PREFIX}/{vid}/{vid}.mp4", str(src))
        total = probe_duration(src)
        clips = make_clips(src, tmp)
        keys = []
        for p, _, _ in clips:
            k = f"{config.S3_PREFIX.rsplit('/', 1)[0]}/clips/{vid}/{p.name}"
            s3.upload_file(str(p), config.S3_BUCKET, k)
            keys.append(k)
    futs = [pool.submit(run_pegasus, brt, f"s3://{config.S3_BUCKET}/{k}", account) for k in keys]
    results = [f.result() for f in futs]
    segments, out_tok, errs = [], 0, []
    for (_, off, _), r in zip(clips, results):
        out_tok += r.get("output_tokens") or 0
        errs += r.get("definition_errors") or []
        for s in r.get("segments", []):
            segments.append({"start_sec": round(s["start_sec"] + off, 1), "end_sec": round(s["end_sec"] + off, 1),
                             "title": s.get("title", ""), "summary": s.get("summary", ""), "on_screen_text": s.get("on_screen_text", "")})
    segments.sort(key=lambda s: s["start_sec"])
    for i, s in enumerate(segments):
        s["idx"] = i
    billed = sum(d for _, _, d in clips)
    usd = billed * PEGASUS_USD_PER_SEC + out_tok / 1e6 * PEGASUS_USD_PER_M_OUT
    return {"total_sec": total, "segments": segments, "meta": {
        "model": PEGASUS, "clips": len(clips), "billed_video_sec": round(billed), "output_tokens": out_tok,
        "definition_errors": errs, "usd": round(usd, 4), "max_clip_secs": round(max(r["_secs"] for r in results), 1)}}


def load_info(s3, vid: str) -> dict:
    try:
        obj = s3.get_object(Bucket=config.S3_BUCKET, Key=f"{config.S3_PREFIX}/{vid}/{vid}.info.json")
        i = json.loads(obj["Body"].read())
    except Exception:
        return {}
    return {"title": i.get("title"), "description": (i.get("description") or "")[:4000], "upload_date": i.get("upload_date"),
            "chapters": [{"start_sec": c["start_time"], "title": c["title"]} for c in i.get("chapters") or []],
            "tags": (i.get("tags") or [])[:20]}


def build_prompt(row: dict, info: dict, total: float, segments: list[dict], captions: list[dict]) -> str:
    parts = [f"YouTube title: {row['title']}", f"Duration: {total:.0f}s", f"Upload date: {info.get('upload_date')}",
             f"Description:\n{info.get('description') or '(none)'}"]
    if info.get("chapters"):
        parts.append("Chapters:\n" + "\n".join(f"{_ts(c['start_sec'])} {c['title']}" for c in info["chapters"]))
    parts.append("Segments (video model):\n" + json.dumps(
        [{k: s[k] for k in ("idx", "start_sec", "end_sec", "title", "summary", "on_screen_text")} for s in segments], indent=1))
    if captions:
        parts.append("Caption transcript:\n" + "\n".join(f"[{c['start_sec']:.0f}s] {c['text']}" for c in captions))
    else:
        parts.append("Caption transcript: (not available)")
    return "\n\n".join(parts)


def stage2(brt, prompt: str, model: str = STAGE2_MODEL, retries: int = 2, min_claims: int = 5,
           blocks: list | None = None, system: str | None = None, extra_fields: dict | None = None) -> tuple[dict, dict]:
    """`blocks` (text + image content blocks) replaces the plain-text prompt, e.g. to show the model frames directly."""
    base = blocks if blocks is not None else [{"text": prompt}]
    msgs = [{"role": "user", "content": base}]
    usage = {"input_tokens": 0, "cache_write_tokens": 0, "cache_read_tokens": 0, "output_tokens": 0, "attempts": 0, "secs": 0.0}
    last_err = ""
    for attempt in range(retries + 1):
        if last_err:
            msgs = [{"role": "user", "content": base + [{"text": f"Your previous answer failed validation: {last_err}\nCall the tool again with a corrected, schema-valid answer."}]}]
        t0 = time.monotonic()
        resp = brt.converse(modelId=model, system=[{"text": system or SYSTEM}], messages=msgs,
                            toolConfig={"tools": [{"toolSpec": DOC_TOOL}], "toolChoice": {"tool": {"name": "record_document"}}},
                            inferenceConfig={"maxTokens": 24000, **({} if "kimi" in model else {"temperature": 0})},
                            **({"additionalModelRequestFields": extra_fields} if extra_fields else {}))
        usage["secs"] += time.monotonic() - t0
        usage["attempts"] += 1
        usage["input_tokens"] += resp["usage"]["inputTokens"]
        usage["cache_write_tokens"] += resp["usage"].get("cacheWriteInputTokens", 0)
        usage["cache_read_tokens"] += resp["usage"].get("cacheReadInputTokens", 0)
        usage["output_tokens"] += resp["usage"]["outputTokens"]
        doc = next((b["toolUse"]["input"] for b in resp["output"]["message"]["content"] if "toolUse" in b), None)
        try:
            if doc is None:
                raise jsonschema.ValidationError("no tool call returned")
            jsonschema.validate(doc, DOC_SCHEMA)
            doc.setdefault("source_documents", [])  # optional: an empty list is a valid answer
            doc.setdefault("quality_flags", [])
            if len(doc["claims"]) < min_claims:
                raise jsonschema.ValidationError(f"only {len(doc['claims'])} claims; this talk needs at least {min_claims} (about 1 per minute)")
            usage["secs"] = round(usage["secs"], 1)
            k = MODEL_USD_PER_M.get(model, MODEL_USD_PER_M[STAGE2_MODEL])
            usage["usd_est"] = round((usage["input_tokens"] * k["input"] + usage["cache_write_tokens"] * k["cache_write"]
                                      + usage["cache_read_tokens"] * k["cache_read"] + usage["output_tokens"] * k["output"]) / 1e6, 4)
            return doc, usage
        except jsonschema.ValidationError as e:
            last_err = e.message[:300]
    raise RuntimeError(f"stage2 failed validation after {retries + 1} attempts: {last_err}")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def verify_quotes(doc: dict, segments: list[dict], captions: list[dict]) -> None:
    hay = _norm(" ".join(s["on_screen_text"] + " " + s["summary"] for s in segments) + " " + " ".join(c["text"] for c in captions))
    for c in doc["claims"]:
        q = _norm(c.get("quote", ""))
        c["quote_verified"] = bool(q) and q in hay


def merge_segments(segments: list[dict], decisions: list[dict]) -> list[dict]:
    dec = {d["idx"]: d for d in decisions}
    out = []
    for s in segments:
        d = dec.get(s["idx"], {"keep": True, "merge_with_prev": False})
        s = {**s, "keep": d["keep"], "note": d.get("note", ""), "merged_idxs": [s["idx"]]}
        if d["merge_with_prev"] and out and out[-1]["keep"] and s["keep"]:
            p = out[-1]
            p["end_sec"] = s["end_sec"]
            p["summary"] = f"{p['summary']} {s['summary']}"
            if s["on_screen_text"] and s["on_screen_text"] not in p["on_screen_text"]:
                p["on_screen_text"] = f"{p['on_screen_text']} | {s['on_screen_text']}".strip(" |")
            p["merged_idxs"].append(s["idx"])
        else:
            out.append(s)
    return out


def analyze_video(row: dict, outdir: Path, s3, brt, account: str, pool: ThreadPoolExecutor, model: str = STAGE2_MODEL) -> dict:
    vid = row["id"]
    cache = outdir / f"{vid}.stage1.json"
    if cache.exists():
        s1 = json.loads(cache.read_text())
    else:
        s1 = stage1(vid, s3, brt, account, pool)
        cache.write_text(json.dumps(s1, indent=2))
    info = load_info(s3, vid)
    cap_file = outdir / f"{vid}.captions.json"
    if cap_file.exists():
        captions = json.loads(cap_file.read_text())
    else:
        with tempfile.TemporaryDirectory() as tmp:
            captions = fetch_captions(vid, Path(tmp))
        cap_file.write_text(json.dumps(captions))
    segments = s1["segments"]
    doc, usage = stage2(brt, build_prompt(row, info, s1["total_sec"], segments, captions), model,
                        min_claims=max(5, round(s1["total_sec"] / 60 * 0.6)))
    verify_quotes(doc, segments, captions)
    decisions = doc.pop("segments")
    merged = merge_segments(segments, decisions)
    record = {
        "schema_version": SCHEMA_VERSION,
        "video": {"id": vid, "url": row["url"], "title": row["title"], "duration_sec": round(s1["total_sec"]),
                  "upload_date": info.get("upload_date"), "chapters": info.get("chapters", []),
                  "s3_uri": f"s3://{config.S3_BUCKET}/{config.S3_PREFIX}/{vid}/{vid}.mp4", "view_count": row["view_count"],
                  "channel": "Open Compute Project"},
        "analysis": {"run_at": datetime.now(UTC).isoformat(timespec="seconds"), "stage1": s1["meta"],
                     "stage2": {"model": model, **usage}, "captions": bool(captions), "has_description": bool(info.get("description")),
                     "usd_total": round(s1["meta"]["usd"] + usage["usd_est"], 4)},
        "document": doc,
        "segments": merged,
    }
    (outdir / f"{vid}.record.json").write_text(json.dumps(record, indent=2))
    return record
