"""Run stage 2 on the SAME frames+captions input with a different model.   uv run python scripts/stage2_model_test.py <video_id> <model_id> <tag>"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import analyze, db  # noqa: E402

VID, MODEL, TAG = sys.argv[1], sys.argv[2], sys.argv[3]
EXTRA = json.loads(sys.argv[4]) if len(sys.argv) > 4 else None  # e.g. '{"reasoning": {"effort": "none"}}'


def main():
    s3, brt, _ = analyze.clients()
    with db.conn() as c:
        row = dict(c.execute("SELECT * FROM videos WHERE id = ?", (VID,)).fetchone())
    info = analyze.load_info(s3, VID)
    captions = json.loads((ROOT / "data/poc20" / f"{VID}.captions.json").read_text())
    fr = json.loads((ROOT / "data/frames20" / f"{VID}.frames.json").read_text())
    total = fr["duration_sec"]
    segs = []
    for i, f in enumerate(fr["frames"]):
        end = fr["frames"][i + 1]["t"] if i + 1 < len(fr["frames"]) else total
        segs.append({"idx": i, "start_sec": f["t"], "end_sec": end, "title": f["slide_title"], "summary": f["description"], "on_screen_text": f["on_screen_text"]})
    doc, use = analyze.stage2(brt, analyze.build_prompt(row, info, total, segs, captions), model=MODEL, min_claims=max(5, round(total / 60 * 0.6)), extra_fields=EXTRA)
    analyze.verify_quotes(doc, segs, captions)
    (ROOT / "data/kimi_test" / f"{VID}.{TAG}.json").write_text(json.dumps(doc, indent=1))
    print(json.dumps({"model": MODEL, "claims": len(doc["claims"]), "verified": sum(c["quote_verified"] for c in doc["claims"]), "entities": len(doc["entities"]), "usage": use}, indent=1))


if __name__ == "__main__":
    main()
