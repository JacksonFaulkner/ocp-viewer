import argparse
import logging

from app import config, ingest


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m app.cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan", help="show how many videos match the age filter; no downloads")
    r = sub.add_parser("ingest", help="download pending videos and upload to S3")
    r.add_argument("--limit", type=int, help="only consider the first N matching videos")
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--retry-failed", action="store_true")
    sub.add_parser("db-init", help="apply sql/001_init.sql to DATABASE_URL")
    ld = sub.add_parser("load", help="load analysis records into Postgres (and S3)")
    ld.add_argument("directory", nargs="?", default="data/poc20")
    ld.add_argument("--no-s3", action="store_true")
    ld.add_argument("--from-s3", action="store_true", help="rebuild from the S3 analysis store instead of a local directory")
    up = sub.add_parser("sync-up", help="push local analysis outputs to the S3 analysis store")
    up.add_argument("directory")
    em = sub.add_parser("embed", help="embed chunks that have no embedding yet")
    em.add_argument("--limit", type=int)
    sub.add_parser("stats", help="row counts")
    se = sub.add_parser("search", help="hybrid search")
    se.add_argument("query")
    se.add_argument("-k", type=int, default=8)
    se.add_argument("--kind", action="append", choices=["video", "segment", "claim", "transcript"])
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.cmd == "db-init":
        from app import pg

        pg.init_schema()
        print("schema applied")
    elif args.cmd == "load":
        from pathlib import Path

        from app import load

        if args.from_s3:
            for vid, n in load.load_s3():
                print(vid, n)
            return
        for vid, n in load.load_dir(config.ROOT / args.directory if not Path(args.directory).is_absolute() else Path(args.directory), not args.no_s3):
            print(vid, n)
    elif args.cmd == "sync-up":
        from pathlib import Path

        from app import store

        d = Path(args.directory)
        print(store.upload_dir(d if d.is_absolute() else config.ROOT / d))
    elif args.cmd == "embed":
        from app import embed

        print("embedded", embed.embed_pending(limit=args.limit))
    elif args.cmd == "stats":
        from app import pg

        with pg.connect() as c:
            for t in ("videos", "speakers", "entities", "video_entities", "segments", "claims", "transcript_chunks", "chunks"):
                print(f"{t:18s}", c.execute(f"SELECT count(*) FROM {t}").fetchone()[0])
            print("embedded chunks   ", c.execute("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL").fetchone()[0])
    elif args.cmd == "search":
        from app import search

        for r in search.search(args.query, args.k, args.kind):
            print(f"{r['score']:.4f} [{r['kind']}] sem={r['sem_rank']} lex={r['lex_rank']}  {r['title'][:60]}\n    {r['text'][:220]}\n    {r['link']}")
    elif args.cmd == "plan":
        from app.sheet import recent_videos

        vs = recent_videos(config.SHEET_PATH, config.MAX_AGE_DAYS)
        print(f"{len(vs)} videos <= {config.MAX_AGE_DAYS} days old")
    else:
        print(ingest.run(args.limit, args.retry_failed, args.workers))


if __name__ == "__main__":
    main()
