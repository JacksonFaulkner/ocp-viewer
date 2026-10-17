# ocp-video-ingest

Pulls recent Open Compute Project YouTube videos (listed in `data/channel.xlsx`, an Apify channel-scraper export) with yt-dlp and uploads them to S3. Status tracked in SQLite.

```bash
cp .env.example .env   # set S3_BUCKET
uv sync
uv run python -m app.cli plan            # count videos within MAX_AGE_DAYS (default 14)
uv run python -m app.cli ingest --limit 2  # try a couple first
uv run python -m app.cli ingest
uv run uvicorn main:app --reload         # GET /videos /stats, POST /ingest
```

Needs `ffmpeg` on PATH. S3 layout: `<S3_PREFIX>/<video_id>/<video_id>.mp4` + `.info.json`.
