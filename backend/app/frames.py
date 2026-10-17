"""Frame-based alternative to Pegasus: scene-detect slide changes with ffmpeg, then read each frame with a cheap vision model."""
import json
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import jsonschema
from botocore.config import Config

from app import config

VISION_MODEL = "us.amazon.nova-2-lite-v1:0"
VISION_USD_PER_M = (0.33, 2.75)  # regional standard, input / output
FRAMES_PER_CALL = 6
MAX_FRAMES = 120

FRAMES_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"frames": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "k": {"type": "integer", "description": "Frame number as given in the prompt"},
            "is_slide": {"type": "boolean", "description": "false if the frame is only a speaker camera, audience, or a plain title/thank-you card"},
            "slide_title": {"type": "string"},
            "on_screen_text": {"type": "string", "description": "All legible text copied exactly: titles, bullets, labels, numbers, units, chart axes, footnotes, names, URLs. Empty if none."},
            "description": {"type": "string", "description": "One sentence on the non-text visuals: diagram, chart, photo, hardware, people"}},
        "required": ["k", "is_slide", "slide_title", "on_screen_text", "description"]}}},
    "required": ["frames"],
}
TOOL = {"name": "record_frames", "description": "Record what each frame shows.", "inputSchema": {"json": FRAMES_SCHEMA}}
SYSTEM = ("You read frames from conference-talk videos (slides, sometimes with a small speaker camera). "
          "For each numbered frame, report only what is visible. Copy on-screen text exactly as written, including numbers and units; do not guess illegible text and do not add information that is not on screen.")


def extract_frames(src: Path, outdir: Path) -> list[tuple[float, Path]]:
    """First frame, then a frame whenever the picture changes (>=3s apart), and at least one per 30s.
    The threshold is low on purpose: dark slide-to-slide transitions score ~0.11, and 0.12 skipped a whole 44s slide."""
    vf = ("fps=1,select='isnan(prev_selected_t)+gte(t-prev_selected_t,30)+gt(scene,0.04)*gte(t-prev_selected_t,3)',"
          "showinfo,scale=1280:-2")
    r = subprocess.run(["ffmpeg", "-v", "info", "-i", str(src), "-vf", vf, "-fps_mode", "vfr", "-q:v", "4", str(outdir / "f_%04d.jpg")],
                       capture_output=True, text=True, check=True)
    times = [float(m) for m in re.findall(r"pts_time:([\d.]+)", r.stderr)]
    files = sorted(outdir.glob("f_*.jpg"))
    frames = list(zip(times, files, strict=False))
    if len(frames) > MAX_FRAMES:  # keep an even spread
        step = len(frames) / MAX_FRAMES
        frames = [frames[int(i * step)] for i in range(MAX_FRAMES)]
    return frames


def _describe_batch(brt, batch: list[tuple[float, Path]], base_idx: int, model: str) -> tuple[list[dict], int, int]:
    content = []
    for j, (t, path) in enumerate(batch):
        content.append({"text": f"Frame {base_idx + j} (at {int(t)}s):"})
        content.append({"image": {"format": "jpeg", "source": {"bytes": path.read_bytes()}}})
    content.append({"text": "Record every frame with the tool."})
    in_tok = out_tok = 0
    for attempt in range(2):
        resp = brt.converse(modelId=model, system=[{"text": SYSTEM}], messages=[{"role": "user", "content": content}],
                            toolConfig={"tools": [{"toolSpec": TOOL}], "toolChoice": {"tool": {"name": "record_frames"}}},
                            inferenceConfig={"maxTokens": 4000, "temperature": 0})
        in_tok += resp["usage"]["inputTokens"]
        out_tok += resp["usage"]["outputTokens"]
        out = next((b["toolUse"]["input"] for b in resp["output"]["message"]["content"] if "toolUse" in b), None)
        try:
            jsonschema.validate(out, FRAMES_SCHEMA)
            by_k = {f["k"]: f for f in out["frames"]}
            if all(base_idx + j in by_k for j in range(len(batch))):
                return [by_k[base_idx + j] for j in range(len(batch))], in_tok, out_tok
        except jsonschema.ValidationError:
            pass
    return [{"k": base_idx + j, "is_slide": False, "slide_title": "", "on_screen_text": "", "description": "", "error": True} for j in range(len(batch))], in_tok, out_tok


def analyze_frames(vid: str, outdir: Path, s3, brt, pool: ThreadPoolExecutor, model: str = VISION_MODEL) -> dict:
    cache = outdir / f"{vid}.frames.json"
    if cache.exists():
        return json.loads(cache.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / f"{vid}.mp4"
        t0 = time.monotonic()
        s3.download_file(config.S3_BUCKET, f"{config.S3_PREFIX}/{vid}/{vid}.mp4", str(src))
        dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(src)]))
        frames = extract_frames(src, tmp)
        t_extract = time.monotonic() - t0
        t1 = time.monotonic()
        batches = [frames[i:i + FRAMES_PER_CALL] for i in range(0, len(frames), FRAMES_PER_CALL)]
        futs = [pool.submit(_describe_batch, brt, b, i * FRAMES_PER_CALL, model) for i, b in enumerate(batches)]
        results = [f.result() for f in futs]
    out_frames, in_tok, out_tok = [], 0, 0
    for (batch, (descs, it, ot)) in zip(batches, results, strict=True):
        in_tok += it
        out_tok += ot
        for (t, _), d in zip(batch, descs, strict=True):
            out_frames.append({"t": round(t, 1), **{k: v for k, v in d.items() if k != "k"}})
    rec = {"video_id": vid, "duration_sec": round(dur), "frames": out_frames, "meta": {
        "model": model, "frames": len(out_frames), "errors": sum(1 for f in out_frames if f.get("error")),
        "input_tokens": in_tok, "output_tokens": out_tok,
        "usd": round((in_tok * VISION_USD_PER_M[0] + out_tok * VISION_USD_PER_M[1]) / 1e6, 4),
        "extract_secs": round(t_extract, 1), "describe_secs": round(time.monotonic() - t1, 1)}}
    cache.write_text(json.dumps(rec, indent=1))
    return rec


def clients():
    sess = boto3.Session(region_name=config.AWS_REGION)
    return (sess.client("s3"),
            sess.client("bedrock-runtime", config=Config(read_timeout=300, retries={"max_attempts": 4, "mode": "adaptive"}, max_pool_connections=32)))
