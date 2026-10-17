import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


_load_env()

SHEET_PATH = Path(os.environ.get("SHEET_PATH", ROOT / "data" / "channel.xlsx"))
DB_PATH = Path(os.environ.get("DB_PATH", ROOT / "data" / "videos.db"))
S3_BUCKET = os.environ.get("S3_BUCKET", "")
S3_PREFIX = os.environ.get("S3_PREFIX", "ocp/videos").strip("/")
MAX_AGE_DAYS = int(os.environ.get("MAX_AGE_DAYS", "14"))
MAX_HEIGHT = int(os.environ.get("MAX_HEIGHT", "1080"))
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://ocp:ocp@127.0.0.1:5433/ocp")
