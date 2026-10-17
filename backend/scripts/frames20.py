"""Frame-based slide reading for the 20 VDC videos.   uv run python scripts/frames20.py [id ...] [--model <id>]"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from poc20 import VDC_IDS  # noqa: E402

from app import frames  # noqa: E402

OUT = ROOT / "data" / "frames20"

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else frames.VISION_MODEL
    args = [a for a in args if a != model]
    ids = args or VDC_IDS
    s3, brt = frames.clients()
    t0 = time.monotonic()
    total = 0.0
    with ThreadPoolExecutor(8) as describe_pool, ThreadPoolExecutor(3) as vid_pool:
        futs = {vid_pool.submit(frames.analyze_frames, v, OUT, s3, brt, describe_pool, model): v for v in ids}
        for f, v in futs.items():
            try:
                m = f.result()["meta"]
                total += m["usd"]
                print(f"ok  {v} frames={m['frames']} errors={m['errors']} tok={m['input_tokens']}/{m['output_tokens']} ${m['usd']:.3f} extract={m['extract_secs']}s describe={m['describe_secs']}s", flush=True)
            except Exception as e:
                print(f"ERR {v}: {str(e)[:200]}", flush=True)
    print(f"done in {time.monotonic() - t0:.0f}s, ${total:.2f}")
