"""Overnight batch: 100 shortest downloaded videos. Nova 2 Lite reads frames; Kimi K3 (reasoning off) writes the analysis.

    uv run python -u scripts/night100.py            # run
    uv run python -u scripts/night100.py --dry      # selection + estimate only
"""
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import analyze, db, frames  # noqa: E402

N = 100
OUT = ROOT / "data" / "night100"
STAGE2 = analyze.STAGE2_MODEL
REASONING_OFF = {"reasoning": {"effort": "none"}}
EST_USD = 0.14  # observed per-video: Kimi reasoning-off ~0.12 + Nova frames ~0.02


def secs(d: str) -> int:
    t = 0
    for x in d.split(":"):
        t = t * 60 + int(x)
    return t


def select() -> list[dict]:
    with db.conn() as c:
        rows = [dict(r) for r in c.execute("SELECT * FROM videos WHERE status='done' AND duration IS NOT NULL")]
    rows.sort(key=lambda r: secs(r["duration"]))
    return rows[:N]


def one(row: dict, s3, brt, pool) -> str:
    vid = row["id"]
    rec_path = OUT / f"{vid}.record.json"
    if rec_path.exists():
        return "skip"
    fr = frames.analyze_frames(vid, OUT, s3, brt, pool)
    info = analyze.load_info(s3, vid)
    cap_file = OUT / f"{vid}.captions.json"
    if cap_file.exists():
        captions = json.loads(cap_file.read_text())
    else:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            captions = analyze.fetch_captions(vid, Path(tmp))
        cap_file.write_text(json.dumps(captions))
    total = fr["duration_sec"]
    segs = []
    for i, f in enumerate(fr["frames"]):
        end = fr["frames"][i + 1]["t"] if i + 1 < len(fr["frames"]) else total
        segs.append({"idx": i, "start_sec": f["t"], "end_sec": end, "title": f["slide_title"],
                     "summary": f["description"], "on_screen_text": f["on_screen_text"]})
    doc, use = analyze.stage2(brt, analyze.build_prompt(row, info, total, segs, captions), model=STAGE2,
                              min_claims=max(5, round(total / 60 * 0.6)), extra_fields=REASONING_OFF)
    analyze.verify_quotes(doc, segs, captions)
    decisions = doc.pop("segments")
    merged = analyze.merge_segments(segs, decisions)
    cost = fr["meta"]["usd"] + use["usd_est"]
    rec = {
        "schema_version": analyze.SCHEMA_VERSION,
        "video": {"id": vid, "url": row["url"], "title": row["title"], "duration_sec": round(total),
                  "upload_date": info.get("upload_date"), "chapters": info.get("chapters", []),
                  "s3_uri": f"s3://{analyze.config.S3_BUCKET}/{analyze.config.S3_PREFIX}/{vid}/{vid}.mp4",
                  "view_count": row["view_count"], "channel": "Open Compute Project"},
        "analysis": {"run_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "pipeline": "nova-frames->kimi-k3(reasoning none)",
                     "frames": fr["meta"], "stage2": {"model": STAGE2, "reasoning": "none", **use},
                     "captions": bool(captions), "usd_total": round(cost, 4)},
        "document": doc,
        "segments": merged,
    }
    rec_path.write_text(json.dumps(rec, indent=2))
    return f"ok claims={len(doc['claims'])} quotes={sum(c['quote_verified'] for c in doc['claims'])}/{len(doc['claims'])} captions={bool(captions)} usd={cost:.3f}"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rows = select()
    hrs = sum(secs(r["duration"]) for r in rows) / 3600
    print(f"selected {len(rows)} shortest videos: {hrs:.2f} h, {secs(rows[0]['duration'])}s to {secs(rows[-1]['duration'])}s; est ${len(rows) * EST_USD:.0f}", flush=True)
    if "--dry" in sys.argv:
        for r in rows[:5] + rows[-2:]:
            print(f"  {r['duration']:>8} {r['id']} {r['title'][:70]}")
        sys.exit(0)
    s3, brt = frames.clients()
    t0 = time.monotonic()
    done = err = 0
    with ThreadPoolExecutor(3) as vid_pool, ThreadPoolExecutor(8) as frame_pool:
        futs = {vid_pool.submit(one, r, s3, brt, frame_pool): r for r in rows}
        for f, r in futs.items():
            try:
                res = f.result()
                done += res != "skip"
                print(f"{res:4s} {r['id']} {r['duration']:>8} {r['title'][:50]}", flush=True)
            except Exception as e:
                err += 1
                print(f"ERR  {r['id']} {r['duration']:>8} {str(e)[:160]}", flush=True)
                traceback.print_exc(limit=1)
    print(f"done: ok={done} err={err} in {(time.monotonic() - t0) / 60:.0f} min", flush=True)
