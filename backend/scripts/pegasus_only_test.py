"""Pure Pegasus: one general-mode call on the whole video returns the full structured analysis.   uv run python scripts/pegasus_only_test.py <video_id>"""
import copy
import json
import sys
import time
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import analyze, config, db  # noqa: E402

VID = sys.argv[1]
OUT = ROOT / "data" / "kimi_test"


def schema():
    s = copy.deepcopy(analyze.DOC_SCHEMA)
    s["properties"].pop("segments")  # no per-segment input in this mode
    s["required"] = [r for r in s["required"] if r != "segments"]
    return s


def main():
    s3, brt, account = analyze.clients()
    with db.conn() as c:
        row = dict(c.execute("SELECT * FROM videos WHERE id = ?", (VID,)).fetchone())
    info = analyze.load_info(s3, VID)
    sch = schema()
    chapters = "\n".join(f"{c['start_sec']:.0f}s {c['title']}" for c in info.get("chapters", []))
    prompt = (
        "You are analyzing a conference talk from the Open Compute Project for a searchable knowledge base. Watch and listen to the whole video, then fill the schema.\n"
        f"YouTube title: {row['title']}\nDescription: {(info.get('description') or '')[:1500]}\n{('Chapters:' + chr(10) + chapters) if chapters else ''}\n\n"
        "Rules: use only what is said or shown; never invent speaker names (empty list if none is given). "
        "Claims must be atomic (one assertion each), typed fact/quantitative/forecast/opinion/recommendation, covering the whole talk at roughly 1-2 claims per minute (at least 8, at most 40). "
        "Each claim's quote must be copied exactly from what is said or from text on screen (empty if you cannot); start_sec/end_sec are seconds from the start of the video; segment_idxs may be empty []. "
        "Do not list the Open Compute Project, the OCP Global Summit or YouTube as entities. Entity kinds: organization, product, architecture, specification, project, hardware, other; use canonical spellings. "
        "Put any white paper, report or URL shown or mentioned in source_documents. Add quality_flags for suspicious on-screen text or conflicts between what is said and shown.")
    body = {"analysis_mode": "general", "inputPrompt": prompt, "temperature": 0, "maxOutputTokens": 16000,
            "responseFormat": {"type": "json_schema", "jsonSchema": sch},
            "mediaSource": {"s3Location": {"uri": f"s3://{config.S3_BUCKET}/{config.S3_PREFIX}/{VID}/{VID}.mp4", "bucketOwner": account}}}
    t0 = time.monotonic()
    r = json.loads(brt.invoke_model(modelId=analyze.PEGASUS, body=json.dumps(body))["body"].read())
    secs = time.monotonic() - t0
    (OUT / f"{VID}.PP.raw.json").write_text(json.dumps(r, indent=1))
    print(f"call took {secs:.0f}s | stopReason={r.get('stopReason')} | keys={list(r)}", flush=True)
    doc = json.loads(r["message"])
    try:
        jsonschema.validate(doc, sch)
        print("schema: valid")
    except jsonschema.ValidationError as e:
        print("schema: INVALID ->", e.message[:200])
    (OUT / f"{VID}.PP.json").write_text(json.dumps(doc, indent=1))
    print(f"claims={len(doc.get('claims', []))} entities={len(doc.get('entities', []))} speakers={[s['name'] for s in doc.get('speakers', [])]} topics={len(doc.get('topics', []))}")
    dur = json.loads((ROOT / "data/frames20" / f"{VID}.frames.json").read_text())["duration_sec"]
    print(f"video {dur}s; billed video ~${dur * analyze.PEGASUS_USD_PER_SEC:.2f} + output {r.get('output_tokens', '?')} tokens")


if __name__ == "__main__":
    main()
