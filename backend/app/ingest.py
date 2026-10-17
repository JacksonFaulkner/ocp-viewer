"""Download videos with yt-dlp and upload them to S3."""
import logging
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import yt_dlp

from app import config, db
from app.sheet import recent_videos

log = logging.getLogger(__name__)


def _ydl_opts(outdir: Path) -> dict:
    h = config.MAX_HEIGHT
    return {
        "format": f"bv*[height<={h}]+ba/b[height<={h}]",
        "merge_output_format": "mp4",
        "outtmpl": str(outdir / "%(id)s.%(ext)s"),
        "writeinfojson": True,
        "quiet": True,
        "noprogress": True,
        "retries": 5,
    }


def download_and_upload(url: str, vid: str, s3) -> tuple[str, int]:
    with tempfile.TemporaryDirectory() as tmp:
        outdir = Path(tmp)
        with yt_dlp.YoutubeDL(_ydl_opts(outdir)) as ydl:
            ydl.download([url])
        video = outdir / f"{vid}.mp4"
        key = f"{config.S3_PREFIX}/{vid}/{vid}.mp4"
        s3.upload_file(str(video), config.S3_BUCKET, key)
        info = outdir / f"{vid}.info.json"
        if info.exists():
            s3.upload_file(str(info), config.S3_BUCKET, f"{config.S3_PREFIX}/{vid}/{vid}.info.json")
        return key, video.stat().st_size


def sync_sheet(limit: int | None = None) -> int:
    videos = recent_videos(config.SHEET_PATH, config.MAX_AGE_DAYS)
    videos = videos[:limit] if limit else videos
    with db.conn() as c:
        db.upsert_pending(c, videos)
    return len(videos)


def _work(row, s3) -> tuple[str, int, float]:
    t0 = time.monotonic()
    key, size = download_and_upload(row["url"], row["id"], s3)
    return key, size, time.monotonic() - t0


def run(limit: int | None = None, retry_failed: bool = False, workers: int = 8) -> dict[str, int]:
    if not config.S3_BUCKET:
        raise RuntimeError("S3_BUCKET not set")
    sync_sheet(limit)
    s3 = boto3.client("s3", region_name=config.AWS_REGION)
    statuses = ("pending", "failed") if retry_failed else ("pending",)
    counts = {"done": 0, "failed": 0}
    t_start = time.monotonic()
    with db.conn() as c:
        todo = c.execute(
            f"SELECT id, url FROM videos WHERE status IN ({','.join('?' * len(statuses))})", statuses
        ).fetchall()
        # Workers only download/upload; the main thread owns the sqlite connection.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_work, row, s3): row["id"] for row in todo}
            for fut in as_completed(futs):
                vid = futs[fut]
                try:
                    key, size, secs = fut.result()
                    db.set_status(c, vid, "done", s3_key=key, size_bytes=size, error=None)
                    counts["done"] += 1
                    log.info("done %s %.1f MiB in %.1fs [%d/%d]", vid, size / 2**20, secs, sum(counts.values()), len(todo))
                except Exception as e:
                    db.set_status(c, vid, "failed", error=str(e)[:500])
                    counts["failed"] += 1
                    log.error("failed %s: %s", vid, str(e)[:200])
    log.info("finished %s in %.0fs", counts, time.monotonic() - t_start)
    return counts
