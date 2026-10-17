"""Two-stage POC. Stage 1: Pegasus 1.5 describes clips. Stage 2: Sonnet 5.5 makes the judgment calls.

    uv run python scripts/pipeline_poc.py vCsnd3GjpbQ
"""
import json
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import boto3
from botocore.config import Config

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import config, db  # noqa: E402

PEGASUS = "us.twelvelabs.pegasus-1-5-v1:0"
STAGE2_ARGS = [a for a in sys.argv[2:] if not a.startswith("--")]
SONNET = STAGE2_ARGS[0] if STAGE2_ARGS else "us.anthropic.claude-sonnet-5-5"
CLIP_SEC = 180

SEGMENT_DEFS = [{
    "id": "section",
    "description": "A distinct stretch of the video covering one idea or slide; typically 20s-2min. Describe only what is said and shown; do not guess names or intent.",
    "fields": [
        {"name": "title", "type": "string", "description": "Short heading"},
        {"name": "summary", "type": "string", "description": "2-3 sentences, self-contained, covering what the speaker says"},
        {"name": "on_screen_text", "type": "string", "description": "Text, labels, numbers and names visible on screen (slide titles, name cards, captions); empty if none"},
    ],
}]

DOC_TOOL = {
    "name": "record_document",
    "description": "Record the document-level analysis of the talk.",
    "inputSchema": {"json": {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "3-5 sentences"},
            "talk_type": {"type": "string", "enum": ["keynote", "technical talk", "panel", "demo", "announcement", "ceremony", "welcome", "other"]},
            "speakers": {"type": "array", "items": {"type": "object", "properties": {
                "name": {"type": "string"}, "affiliation": {"type": "string"},
                "evidence": {"type": "string", "description": "Where the name came from: on-screen, title, description"}},
                "required": ["name"]}},
            "topics": {"type": "array", "items": {"type": "string"}, "description": "3-8 specific topics"},
            "entities": {"type": "array", "items": {"type": "object", "properties": {
                "name": {"type": "string", "description": "Canonical spelling"},
                "kind": {"type": "string", "enum": ["company", "product", "standard", "project", "hardware", "other"]},
                "segment_idxs": {"type": "array", "items": {"type": "integer"}}},
                "required": ["name", "kind", "segment_idxs"]}},
            "key_claims": {"type": "array", "items": {"type": "object", "properties": {
                "claim": {"type": "string"},
                "segment_idxs": {"type": "array", "items": {"type": "integer"}}},
                "required": ["claim", "segment_idxs"]}},
            "segments": {"type": "array", "description": "One entry per input segment", "items": {"type": "object", "properties": {
                "idx": {"type": "integer"},
                "keep": {"type": "boolean", "description": "false for title cards, filler, or segments that look hallucinated"},
                "note": {"type": "string", "description": "Why dropped or flagged; empty otherwise"}},
                "required": ["idx", "keep"]}},
        },
        "required": ["summary", "talk_type", "speakers", "topics", "entities", "key_claims", "segments"],
    }},
}

SONNET_SYSTEM = """You analyze conference talks from the Open Compute Project (OCP) YouTube channel for a searchable knowledge base.
You receive timestamped segment descriptions produced by a video model, plus the video's YouTube title. The video model describes only what it sees and hears; it can misread or invent details.
Rules: use only what the segments and title support. Do not invent speaker names; if none are shown or stated, return an empty speakers list. Use canonical spellings for entities (e.g. OpenTelemetry, not Open Telemetry). Every entity and claim must cite the segment idxs that support it. Mark segments keep=false if they are only title cards/filler or look unreliable."""


def probe_duration(path: Path) -> float:
    out = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)])
    return float(out)


def make_clips(src: Path, outdir: Path) -> list[tuple[Path, float]]:
    """Split into ~CLIP_SEC clips (cut on keyframes). Returns (path, actual_start_offset)."""
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-c", "copy", "-map", "0", "-f", "segment",
                    "-segment_time", str(CLIP_SEC), "-reset_timestamps", "1", str(outdir / "clip_%03d.mp4")], check=True)
    clips, offset = [], 0.0
    for p in sorted(outdir.glob("clip_*.mp4")):
        clips.append((p, offset))
        offset += probe_duration(p)
    return clips


def run_pegasus(client, uri: str, account: str) -> dict:
    body = {"analysis_mode": "time_based_metadata", "temperature": 0, "maxOutputTokens": 8192,
            "segment_definitions": SEGMENT_DEFS,
            "mediaSource": {"s3Location": {"uri": uri, "bucketOwner": account}}}
    t0 = time.monotonic()
    r = json.loads(client.invoke_model(modelId=PEGASUS, body=json.dumps(body))["body"].read())
    r["_secs"] = time.monotonic() - t0
    return r


if __name__ == "__main__":
    vid = sys.argv[1]
    sess = boto3.Session(region_name=config.AWS_REGION)
    account = sess.client("sts").get_caller_identity()["Account"]
    s3 = sess.client("s3")
    brt = sess.client("bedrock-runtime", config=Config(read_timeout=600, retries={"max_attempts": 2}))
    out = ROOT / "data" / "poc"
    out.mkdir(parents=True, exist_ok=True)
    with db.conn() as c:
        row = dict(c.execute("SELECT * FROM videos WHERE id = ?", (vid,)).fetchone())

    # ---- stage 1 ----
    cache = out / f"{vid}.stage1.json"
    if "--reuse" in sys.argv and cache.exists():
        saved = json.loads(cache.read_text())
        total, stage1, segments = saved["total"], saved["stage1"], saved["segments"]
        print(f"reusing {cache.name}", flush=True)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / f"{vid}.mp4"
            s3.download_file(config.S3_BUCKET, f"{config.S3_PREFIX}/{vid}/{vid}.mp4", str(src))
            total = probe_duration(src)
            clips = make_clips(src, tmp)
            print(f"{vid}: {total:.0f}s -> {len(clips)} clips", flush=True)
            keys = []
            for p, _ in clips:
                k = f"{config.S3_PREFIX.rsplit('/', 1)[0]}/clips/{vid}/{p.name}"
                s3.upload_file(str(p), config.S3_BUCKET, k)
                keys.append(k)

        def one(i):
            return run_pegasus(brt, f"s3://{config.S3_BUCKET}/{keys[i]}", account)

        with ThreadPoolExecutor(len(clips)) as pool:
            results = list(pool.map(one, range(len(clips))))

        segments, in_tok, out_tok = [], 0, 0
        for (_, offset), r in zip(clips, results):
            in_tok += r.get("input_tokens") or 0
            out_tok += r.get("output_tokens") or 0
            print(f"  clip @{offset:.0f}s: {r['_secs']:.0f}s, {len(r.get('segments', []))} segs, stop={r.get('stopReason')}, errs={r.get('definition_errors')}", flush=True)
            for s in r.get("segments", []):
                segments.append({"start_sec": round(s["start_sec"] + offset, 1), "end_sec": round(s["end_sec"] + offset, 1),
                                 "title": s.get("title", ""), "summary": s.get("summary", ""), "on_screen_text": s.get("on_screen_text", "")})
        segments.sort(key=lambda s: s["start_sec"])
        for i, s in enumerate(segments):
            s["idx"] = i
        stage1 = {"model": PEGASUS, "input_tokens": in_tok, "output_tokens": out_tok, "clips": len(clips)}
        cache.write_text(json.dumps({"total": total, "stage1": stage1, "segments": segments}, indent=2))


    # ---- stage 2 ----
    prompt = f"YouTube title: {row['title']}\nDuration: {total:.0f}s\n\nSegments:\n" + json.dumps(
        [{k: s[k] for k in ("idx", "start_sec", "end_sec", "title", "summary", "on_screen_text")} for s in segments], indent=1)
    t0 = time.monotonic()
    resp = brt.converse(modelId=SONNET, system=[{"text": SONNET_SYSTEM}],
                        messages=[{"role": "user", "content": [{"text": prompt}]}],
                        toolConfig={"tools": [{"toolSpec": DOC_TOOL}], "toolChoice": {"tool": {"name": "record_document"}}},
                        inferenceConfig={"maxTokens": 12000})
    document = next(b["toolUse"]["input"] for b in resp["output"]["message"]["content"] if "toolUse" in b)
    stage2 = {"model": SONNET, "input_tokens": resp["usage"]["inputTokens"], "output_tokens": resp["usage"]["outputTokens"], "secs": round(time.monotonic() - t0, 1)}

    keep = {s["idx"]: s for s in document.pop("segments", [])}
    for s in segments:
        k = keep.get(s["idx"], {})
        s["keep"], s["note"] = k.get("keep", True), k.get("note", "")

    record = {
        "schema_version": 1,
        "video": {"id": vid, "url": row["url"], "title": row["title"], "duration_sec": round(total),
                  "s3_uri": f"s3://{config.S3_BUCKET}/{config.S3_PREFIX}/{vid}/{vid}.mp4", "view_count": row["view_count"],
                  "channel": "Open Compute Project"},
        "analysis": {"run_at": datetime.now(UTC).isoformat(timespec="seconds"), "stage1": stage1, "stage2": stage2},
        "document": document,
        "segments": segments,
    }
    (out / f"{vid}.record.json").write_text(json.dumps(record, indent=2))
    print(f"wrote data/poc/{vid}.record.json | stage2 {stage2}")
