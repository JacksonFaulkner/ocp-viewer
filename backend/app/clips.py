"""Helpers for turning search hits into playable video clips."""
import re
from functools import lru_cache
from urllib.parse import quote

import boto3

from app import config, pg

MAX_CLIP_SEC = 120
MERGE_GAP_SEC = 30
PAD_SEC = 5


@lru_cache(maxsize=1)
def _s3():
    return boto3.client("s3", region_name=config.AWS_REGION)


def _split_s3(uri: str) -> tuple[str, str]:
    m = re.match(r"s3://([^/]+)/(.+)", uri)
    if not m:
        raise ValueError(f"bad s3 uri: {uri}")
    return m.group(1), m.group(2)


def media_urls(video: dict, start: float, end: float, expires: int = 3600) -> dict:
    """Presigned S3 URL for the real mp4 with a media-fragment range, plus YouTube fallbacks."""
    bucket, key = _split_s3(video["s3_uri"])
    signed = _s3().generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key, "ResponseContentType": "video/mp4"}, ExpiresIn=expires)
    vid = video["id"]
    return {
        "play_url": f"{signed}#t={start:g},{end:g}",
        "youtube_url": f"https://www.youtube.com/watch?v={quote(vid)}&t={int(start)}s",
        "youtube_embed_url": f"https://www.youtube.com/embed/{quote(vid)}?start={int(start)}&end={int(end) + 1}",
        "expires_in": expires,
    }


def transcript_for(c, video_id: str, start: float, end: float) -> list[dict]:
    rows = c.execute("SELECT start_sec, text FROM transcript_chunks WHERE video_id = %s AND start_sec >= %s AND start_sec < %s ORDER BY start_sec",
                     (video_id, start - 15, end)).fetchall()
    return [{"start_sec": r[0], "text": r[1]} for r in rows]


def slides_for(c, video_id: str, start: float, end: float) -> list[dict]:
    rows = c.execute("SELECT idx, start_sec, end_sec, title, summary, on_screen_text FROM segments WHERE video_id = %s AND keep AND end_sec > %s AND start_sec < %s ORDER BY start_sec",
                     (video_id, start, end)).fetchall()
    return [dict(zip(("idx", "start_sec", "end_sec", "title", "summary", "on_screen_text"), r)) for r in rows]


def build_candidates(hits: list[dict], durations: dict[str, float], max_candidates: int = 8, per_video: int = 3) -> list[dict]:
    """Cluster nearby hits per video into clips of at most MAX_CLIP_SEC, ranked by summed fusion score."""
    by_video: dict[str, list[dict]] = {}
    for h in hits:
        if h["kind"] != "video" and h["start_sec"] is not None:
            by_video.setdefault(h["video_id"], []).append(h)
    clusters = []
    for vid, hs in by_video.items():
        hs.sort(key=lambda h: h["start_sec"])
        cur = None
        for h in hs:
            end = max(h["end_sec"] or h["start_sec"], h["start_sec"] + 10)
            if cur and h["start_sec"] - cur["end"] <= MERGE_GAP_SEC and max(cur["end"], end) - cur["start"] <= MAX_CLIP_SEC:
                cur["end"] = max(cur["end"], end)
                cur["score"] += h["score"]
                cur["hits"].append(h)
            else:
                cur = {"video_id": vid, "title": h["title"], "url": h["url"], "start": h["start_sec"], "end": end, "score": h["score"], "hits": [h]}
                clusters.append(cur)
    clusters.sort(key=lambda c: -c["score"])
    seen: dict[str, int] = {}
    out = []
    for cl in clusters:
        if seen.get(cl["video_id"], 0) >= per_video:
            continue
        seen[cl["video_id"]] = seen.get(cl["video_id"], 0) + 1
        dur = durations.get(cl["video_id"]) or cl["end"] + PAD_SEC
        cl["start"] = max(0.0, cl["start"] - PAD_SEC)
        cl["end"] = min(float(dur), cl["end"] + PAD_SEC)
        out.append(cl)
        if len(out) >= max_candidates:
            break
    return out


def video_rows(video_ids: list[str]) -> dict[str, dict]:
    if not video_ids:
        return {}
    with pg.connect() as c:
        rows = c.execute("SELECT id, title, url, s3_uri, duration_sec FROM videos WHERE id = ANY(%s)", (video_ids,)).fetchall()
    return {r[0]: dict(zip(("id", "title", "url", "s3_uri", "duration_sec"), r)) for r in rows}
