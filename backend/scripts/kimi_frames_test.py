"""One-video test: (A) Kimi sees frame images + captions directly vs (B) Nova's per-frame text + captions -> Kimi.
    uv run python scripts/kimi_frames_test.py uaQa-9xmfhs"""
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import analyze, config, db, frames  # noqa: E402

ONLY_B = "--only-b" in sys.argv
VID = next((a for a in sys.argv[1:] if not a.startswith("--")), "uaQa-9xmfhs")
OUT = ROOT / "data" / "kimi_test"
FRAME_DIR = OUT / VID
SYSTEM_A = analyze.SYSTEM.replace(
    "timestamped segment descriptions produced by a video model (it can misread or invent details)",
    "frames sampled from the video at slide changes, each with its timestamp (read the slide text yourself, exactly)").replace(
    "copied exactly from the transcript or a segment's on_screen_text", "copied exactly from the transcript or from text visible on a frame")


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FRAME_DIR.mkdir(exist_ok=True)
    s3, brt, _ = analyze.clients()
    with db.conn() as c:
        row = dict(c.execute("SELECT * FROM videos WHERE id = ?", (VID,)).fetchone())
    info = analyze.load_info(s3, VID)
    captions = json.loads((ROOT / "data" / "poc20" / f"{VID}.captions.json").read_text())
    fr = json.loads((ROOT / "data" / "frames20" / f"{VID}.frames.json").read_text())
    total = fr["duration_sec"]

    if not ONLY_B and not list(FRAME_DIR.glob("f_*.jpg")):  # frames were not kept last time; re-extract (same ffmpeg settings)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "v.mp4"
            s3.download_file(config.S3_BUCKET, f"{config.S3_PREFIX}/{VID}/{VID}.mp4", str(src))
            for t, path in frames.extract_frames(src, Path(tmp)):
                path.rename(FRAME_DIR / f"f_{int(t):05d}.jpg")
    files = sorted(FRAME_DIR.glob("f_*.jpg"))
    times = [int(f.stem.split("_")[1]) for f in files]
    print(f"frames on disk: {len(files)}; nova frames: {len(fr['frames'])}", flush=True)

    # ---- B: Nova per-frame text -> Kimi (text only) ----
    segs = []
    for i, f in enumerate(fr["frames"]):
        end = fr["frames"][i + 1]["t"] if i + 1 < len(fr["frames"]) else total
        segs.append({"idx": i, "start_sec": f["t"], "end_sec": end, "title": f["slide_title"], "summary": f["description"], "on_screen_text": f["on_screen_text"]})
    min_claims = max(5, round(total / 60 * 0.6))
    doc_b, use_b = analyze.stage2(brt, analyze.build_prompt(row, info, total, segs, captions), min_claims=min_claims)

    if ONLY_B:
        (OUT / f"{VID}.B2.json").write_text(json.dumps(doc_b, indent=1))
        print(json.dumps({"claims": len(doc_b["claims"]), "usage": use_b}, indent=1))
        return

    # ---- A: Kimi sees the images + captions directly ----
    head = analyze.build_prompt(row, info, total, [], captions).split("Segments (video model):")[0]
    cap_txt = "\n".join(f"[{c['start_sec']:.0f}s] {c['text']}" for c in captions)
    blocks = [{"text": head + f"Caption transcript:\n{cap_txt}\n\nFrames follow, numbered from 0; each is the slide on screen from its timestamp until the next frame. "
                                "Use the frame number as the segment idx; segments[] must list every frame."}]
    ends = times[1:] + [total]
    for i, (f, t) in enumerate(zip(files, times, strict=True)):
        blocks.append({"text": f"Frame {i} ({t}s-{ends[i]}s):"})
        blocks.append({"image": {"format": "jpeg", "source": {"bytes": f.read_bytes()}}})
    blocks.append({"text": "Now call the tool."})
    doc_a, use_a = analyze.stage2(brt, "", min_claims=min_claims, blocks=blocks, system=SYSTEM_A)

    # ---- compare ----
    ref_caps = norm(" ".join(c["text"] for c in captions))
    ref_nova = norm(" ".join(s["on_screen_text"] + " " + s["title"] for s in segs))
    pf = ROOT / "data" / "poc20" / f"{VID}.record.json"
    peg = json.loads(pf.read_text()) if pf.exists() else None

    def vr(doc, ref):
        qs = [norm(c.get("quote", "")) for c in doc["claims"]]
        return sum(1 for q in qs if q and q in ref), sum(1 for q in qs if q)

    def ents(doc):
        return {e["name"].lower() for e in doc["entities"]}

    res = {}
    for name, doc, use in (("A pure Kimi (images)", doc_a, use_a), ("B Nova text -> Kimi", doc_b, use_b)):
        q_caps = vr(doc, ref_caps)
        q_any = vr(doc, ref_caps + " " + ref_nova)
        types = {t: sum(c["claim_type"] == t for c in doc["claims"]) for t in ("fact", "quantitative", "forecast", "opinion", "recommendation")}
        res[name] = {"claims": len(doc["claims"]), "types": types, "quotes_with_text": q_any[1], "quotes_in_captions": q_caps[0],
                     "quotes_in_captions_or_nova_text": q_any[0], "entities": len(doc["entities"]), "speakers": [s["name"] for s in doc["speakers"]],
                     "topics": len(doc["topics"]), "sources": len(doc["source_documents"]), "flags": len(doc["quality_flags"]),
                     "usage": {k: use[k] for k in ("input_tokens", "output_tokens", "attempts", "secs", "usd_est")}}
    a_e, b_e = ents(doc_a), ents(doc_b)
    res["entity_overlap"] = {"A": len(a_e), "B": len(b_e), "both": len(a_e & b_e), "only_A": sorted(a_e - b_e), "only_B": sorted(b_e - a_e),
                             "pegasus_record": sorted(ents(peg["document"])) if peg else None}
    (OUT / f"{VID}.A.json").write_text(json.dumps(doc_a, indent=1))
    (OUT / f"{VID}.B.json").write_text(json.dumps(doc_b, indent=1))
    (OUT / f"{VID}.result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
