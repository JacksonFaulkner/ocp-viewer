"""POC: analyze 20 videos on 800 VDC / LVDC power architecture.   uv run python scripts/poc20.py [n] [--model <id>] [--random]"""
import json
import random
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import analyze, db  # noqa: E402

N = int(next((a for a in sys.argv[1:] if a.isdigit()), 20))
MODEL = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else analyze.STAGE2_MODEL
OUT = ROOT / "data" / "poc20"
SKIP = {"N_4_1orETrc", "vCsnd3GjpbQ", "X9Hx8DR5QYw"}
# Hand-picked by title: talks about 800 VDC / LVDC data center power architecture.
VDC_IDS = [
    "4ywmp2XSuiU", "Qws5vRGyG7w", "mil6y8lCEsk", "0gpY6YYbkfo", "lBa8sIxwMMM", "fBUUyiFYBnE", "ujqG-hxlzKg",
    "SLiKs3rOUoo", "UHYvWpdItR4", "jxq_rbd3IJk", "8Bv-hrfCxRM", "uaQa-9xmfhs", "KdFpoLIdd4M", "fkOOQWfITIw",
    "FnoSV8JqWkE", "UONmRpabnf4", "MJXJHn1w_hg", "5yg8mEA3Zs8", "9LR07xtVJRI", "WD8sIdWJuqg",
]


def secs(d: str) -> int:
    t = 0
    for x in d.split(":"):
        t = t * 60 + int(x)
    return t


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with db.conn() as c:
        rows = [dict(r) for r in c.execute("SELECT * FROM videos WHERE status='done' AND duration IS NOT NULL")]
    if "--random" in sys.argv:
        pool_rows = [r for r in rows if 180 <= secs(r["duration"]) <= 2700 and r["id"] not in SKIP]
        random.Random(7).shuffle(pool_rows)
        picks = pool_rows[:N]
    else:
        by_id = {r["id"]: r for r in rows}
        picks = [by_id[i] for i in VDC_IDS[:N]]
    print(f"{len(picks)} videos, {sum(secs(r['duration']) for r in picks) / 3600:.1f} h, stage2={MODEL}", flush=True)
    s3, brt, account = analyze.clients()
    t0 = time.monotonic()
    summary = []
    with ThreadPoolExecutor(analyze.PEGASUS_CONCURRENCY) as clip_pool, ThreadPoolExecutor(4) as vid_pool:
        futs = {vid_pool.submit(analyze.analyze_video, r, OUT, s3, brt, account, clip_pool, MODEL): r for r in picks}
        for f, r in futs.items():
            try:
                rec = f.result()
                a, d = rec["analysis"], rec["document"]
                row = {"id": r["id"], "dur": r["duration"], "title": r["title"][:60], "segs": len(rec["segments"]),
                       "claims": len(d["claims"]), "quotes_ok": sum(c["quote_verified"] for c in d["claims"]),
                       "speakers": len(d["speakers"]), "entities": len(d["entities"]), "flags": len(d["quality_flags"]),
                       "captions": a["captions"], "attempts": a["stage2"]["attempts"], "usd": a["usd_total"]}
                summary.append(row)
                print("ok ", row, flush=True)
            except Exception as e:
                print("ERR", r["id"], r["title"][:50], "->", str(e)[:200], flush=True)
                traceback.print_exc(limit=2)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\ndone {len(summary)}/{len(picks)} in {time.monotonic() - t0:.0f}s, total ${sum(s['usd'] for s in summary):.2f}")
