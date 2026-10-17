"""Compare Pegasus stage-1 output with the frame-based reading, per video.   uv run python scripts/compare_frames.py"""
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POC, FR = ROOT / "data" / "poc20", ROOT / "data" / "frames20"
STOP = set("the and for with that this from are was were have has not you your our they their will can all any but its into over than then also more most such each per via out one two use used using about which when what where while these those there here been being would could should may might very just only both other some many much".split())


def tokens(text: str) -> tuple[set[str], set[str]]:
    """(all distinctive tokens, technical tokens: contain a digit or are ALL-CAPS acronyms)"""
    raw = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\.]*[A-Za-z0-9]|[A-Za-z0-9]", text)
    allt = {t.lower().strip(".-") for t in raw if len(t) >= 3 and t.lower() not in STOP}
    tech = {t.lower().strip(".-") for t in raw if len(t) >= 2 and (any(c.isdigit() for c in t) or (t.isupper() and len(t) >= 2))}
    return allt, tech


def recall(a: set, b: set) -> float | None:
    return len(a & b) / len(a) if a else None


def main():
    rows = []
    for sf in sorted(POC.glob("*.stage1.json")):
        vid = sf.name.split(".")[0]
        ff = FR / f"{vid}.frames.json"
        if not ff.exists():
            continue
        s1, fr = json.loads(sf.read_text()), json.loads(ff.read_text())
        pt = " ".join(s["on_screen_text"] + " " + s["title"] for s in s1["segments"])
        ft = " ".join(f["on_screen_text"] + " " + f["slide_title"] for f in fr["frames"])
        p_all, p_tech = tokens(pt)
        f_all, f_tech = tokens(ft)
        starts = [s["start_sec"] for s in s1["segments"]][1:]
        ftimes = [f["t"] for f in fr["frames"]]
        near = sum(any(abs(t - s) <= 10 for t in ftimes) for s in starts)
        slide_cov = sum(1 for f in fr["frames"] if f["is_slide"]) / max(1, len(fr["frames"]))
        rows.append({
            "id": vid, "dur_min": round(fr["duration_sec"] / 60), "pegasus_segments": len(s1["segments"]), "frames": len(fr["frames"]),
            "frames_cover_pegasus_text": recall(p_all, f_all), "pegasus_covers_frame_text": recall(f_all, p_all),
            "frames_cover_pegasus_tech": recall(p_tech, f_tech), "pegasus_covers_frame_tech": recall(f_tech, p_tech),
            "pegasus_boundaries_near_a_frame": near / len(starts) if starts else None,
            "frame_text_words": len(ft.split()), "pegasus_text_words": len(pt.split()),
            "usd_pegasus": s1["meta"]["usd"], "usd_frames": fr["meta"]["usd"], "frame_errors": fr["meta"]["errors"], "slide_frames_share": slide_cov,
        })

    def pct(v):
        return "  -  " if v is None else f"{v * 100:4.0f}%"

    print(f"{'video':12s}{'min':>4}{'peg seg':>8}{'frames':>7} | text: frm⊇peg peg⊇frm | tech: frm⊇peg peg⊇frm | peg cuts≈frame | words peg/frm | $peg  $frm")
    for r in rows:
        print(f"{r['id']:12s}{r['dur_min']:>4}{r['pegasus_segments']:>8}{r['frames']:>7} |       {pct(r['frames_cover_pegasus_text'])}   {pct(r['pegasus_covers_frame_text'])} |       {pct(r['frames_cover_pegasus_tech'])}   {pct(r['pegasus_covers_frame_tech'])} |      {pct(r['pegasus_boundaries_near_a_frame'])}      | {r['pegasus_text_words']:>5}/{r['frame_text_words']:<5} | {r['usd_pegasus']:.2f}  {r['usd_frames']:.3f}")
    if rows:
        def mean(k):
            v = [r[k] for r in rows if r[k] is not None]
            return statistics.mean(v) if v else None
        print("\nMEAN over", len(rows), "videos:", {k: (round(mean(k), 3) if mean(k) is not None else None) for k in
              ("frames_cover_pegasus_text", "pegasus_covers_frame_text", "frames_cover_pegasus_tech", "pegasus_covers_frame_tech", "pegasus_boundaries_near_a_frame", "slide_frames_share")})
        print(f"cost: Pegasus ${sum(r['usd_pegasus'] for r in rows):.2f} vs frames ${sum(r['usd_frames'] for r in rows):.2f}  (stage 1 only)")
    (FR / "comparison.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    sys.exit(main())
