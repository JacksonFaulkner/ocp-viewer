"""Load analysis records (data/<dir>/<id>.record.json) into Postgres and, optionally, S3."""
import json
from pathlib import Path

import boto3

from app import config, pg


def _clean(x):
    """Postgres text cannot hold NUL bytes; model output occasionally contains them."""
    if isinstance(x, str):
        return x.replace("\x00", "")
    if isinstance(x, list):
        return [_clean(v) for v in x]
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    return x


def load_record(c, rec: dict, captions: list[dict], s3_uri: str | None) -> dict:
    rec, captions = _clean(rec), _clean(captions)
    v, d, a = rec["video"], rec["document"], rec["analysis"]
    vid = v["id"]
    c.execute("DELETE FROM videos WHERE id = %s", (vid,))  # cascades; keeps reloads idempotent
    c.execute(
        """INSERT INTO videos (id, url, title, duration_sec, upload_date, view_count, channel, s3_uri, summary, talk_type,
                               session, topics, schema_version, analysis, record_s3_uri)
           VALUES (%s,%s,%s,%s,to_date(%s,'YYYYMMDD'),%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        (vid, v["url"], v["title"], v["duration_sec"], v.get("upload_date"), v.get("view_count"), v.get("channel"), v["s3_uri"],
         d["summary"], d["talk_type"], json.dumps(d.get("session")), d["topics"], rec["schema_version"], json.dumps(a), s3_uri),
    )
    for s in d["speakers"]:
        c.execute("INSERT INTO speakers (video_id,name,affiliation,role,evidence) VALUES (%s,%s,%s,%s,%s)",
                  (vid, s["name"], s.get("affiliation"), s.get("role"), s.get("evidence")))
    for e in d["entities"]:
        row = c.execute(
            """INSERT INTO entities (name, kind) VALUES (%s,%s)
               ON CONFLICT (lower(name)) DO UPDATE SET name = entities.name RETURNING id""", (e["name"], e["kind"])).fetchone()
        c.execute("""INSERT INTO video_entities (video_id, entity_id, raw_name, raw_kind, segment_idxs) VALUES (%s,%s,%s,%s,%s)
                     ON CONFLICT DO NOTHING""", (vid, row[0], e["name"], e["kind"], e["segment_idxs"]))
    for s in d["source_documents"]:
        c.execute("INSERT INTO source_documents (video_id,title,url,evidence) VALUES (%s,%s,%s,%s)", (vid, s["title"], s.get("url"), s.get("evidence")))
    for f in d["quality_flags"]:
        c.execute("INSERT INTO quality_flags (video_id,location,issue) VALUES (%s,%s,%s)", (vid, f["where"], f["issue"]))

    chunk_rows = [("video", None, 0, v["duration_sec"],
                   " ".join(x for x in (d["summary"], "Topics: " + "; ".join(d["topics"])) if x), {"talk_type": d["talk_type"]})]
    for s in rec["segments"]:
        sid = c.execute("""INSERT INTO segments (video_id,idx,start_sec,end_sec,title,summary,on_screen_text,keep,merged_idxs,note)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                        (vid, s["idx"], s["start_sec"], s["end_sec"], s["title"], s["summary"], s["on_screen_text"], s["keep"], s["merged_idxs"], s.get("note"))).fetchone()[0]
        if s["keep"]:
            text = " ".join(x for x in (s["title"] + ".", s["summary"], ("On screen: " + s["on_screen_text"]) if s["on_screen_text"] else "") if x)
            chunk_rows.append(("segment", sid, s["start_sec"], s["end_sec"], text, {"idx": s["idx"]}))
    for cl in d["claims"]:
        cid = c.execute("""INSERT INTO claims (video_id,claim,claim_type,quote,start_sec,end_sec,segment_idxs,source_cited_on_slide,confidence,quote_verified)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                        (vid, cl["claim"], cl["claim_type"], cl.get("quote"), cl["start_sec"], cl["end_sec"], cl["segment_idxs"],
                         cl.get("source_cited_on_slide"), cl["confidence"], cl.get("quote_verified"))).fetchone()[0]
        chunk_rows.append(("claim", cid, cl["start_sec"], cl["end_sec"], cl["claim"],
                           {"claim_type": cl["claim_type"], "confidence": cl["confidence"], "quote_verified": cl.get("quote_verified")}))
    for i, cap in enumerate(captions):
        end = captions[i + 1]["start_sec"] if i + 1 < len(captions) else cap["start_sec"] + 15
        tid = c.execute("INSERT INTO transcript_chunks (video_id,start_sec,text) VALUES (%s,%s,%s) RETURNING id", (vid, cap["start_sec"], cap["text"])).fetchone()[0]
        chunk_rows.append(("transcript", tid, cap["start_sec"], end, cap["text"], {}))
    c.cursor().executemany("INSERT INTO chunks (video_id,kind,ref_id,start_sec,end_sec,context,text,meta) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                  [(vid, k, ref, st, en, v["title"], t, json.dumps(m)) for k, ref, st, en, t, m in chunk_rows])
    return {"segments": len(rec["segments"]), "claims": len(d["claims"]), "transcript": len(captions), "chunks": len(chunk_rows)}


def load_dir(directory: Path, push_s3: bool = True) -> list[tuple[str, dict]]:
    s3 = boto3.client("s3", region_name=config.AWS_REGION) if push_s3 else None
    results = []
    with pg.connect() as c:
        for f in sorted(directory.glob("*.record.json")):
            rec = json.loads(f.read_text())
            vid = rec["video"]["id"]
            cap_f = directory / f"{vid}.captions.json"
            captions = json.loads(cap_f.read_text()) if cap_f.exists() else []
            s3_uri = None
            if s3:
                prefix = f"{config.S3_PREFIX.rsplit('/', 1)[0]}/analysis/{vid}"
                s3.upload_file(str(f), config.S3_BUCKET, f"{prefix}/record.json")
                if captions:
                    s3.upload_file(str(cap_f), config.S3_BUCKET, f"{prefix}/captions.json")
                s3_uri = f"s3://{config.S3_BUCKET}/{prefix}/record.json"
            results.append((vid, load_record(c, rec, captions, s3_uri)))
    return results


def load_s3(ids: list[str] | None = None) -> list[tuple[str, dict]]:
    """Rebuild the database from the S3 analysis store. Needs only boto3 credentials, no local files."""
    from app import store

    results, failed = [], []
    for vid in ids or store.list_ids():
        rec = store.read_json(vid, "record")
        if rec is None:
            continue
        captions = store.read_json(vid, "captions") or []
        s3_uri = f"s3://{config.S3_BUCKET}/{store.prefix()}/{vid}/record.json"
        try:
            with pg.connect() as c:  # one transaction per video: a bad record rolls back alone
                results.append((vid, load_record(c, rec, captions, s3_uri)))
        except Exception as e:
            failed.append((vid, str(e)[:160]))
    for vid, err in failed:
        print(f"FAILED {vid}: {err}", flush=True)
    return results
