"""Read the Apify channel-scraper export and filter to recent videos."""
import re
from dataclasses import dataclass
from pathlib import Path

import openpyxl

_AGE = re.compile(r"(\d+)\s*(h|d|w|mo|y)\b")
_DAYS = {"h": 1 / 24, "d": 1, "w": 7, "mo": 30, "y": 365}


@dataclass(frozen=True)
class SheetVideo:
    id: str
    url: str
    title: str
    age_days: float
    duration: str | None
    view_count: int | None
    position: int


def parse_age_days(text: str) -> float | None:
    """'10h ago' -> 0.4, '2w ago' -> 14, '2mo ago' -> 60. Streamed/premiere labels return None."""
    m = _AGE.search(text or "")
    return int(m.group(1)) * _DAYS[m.group(2)] if m else None


def read_videos(path: Path) -> list[SheetVideo]:
    ws = openpyxl.load_workbook(path, read_only=True).active
    rows = ws.iter_rows(values_only=True)
    ix = {name: i for i, name in enumerate(next(rows))}
    out = []
    for pos, r in enumerate(rows):
        if r[ix["type"]] != "video":
            continue
        age = parse_age_days(r[ix["date"]])
        if age is None:
            continue
        out.append(
            SheetVideo(
                id=r[ix["id"]],
                url=r[ix["url"]],
                title=r[ix["title"]],
                age_days=age,
                duration=r[ix["duration"]],
                view_count=r[ix["viewCount"]],
                position=pos,
            )
        )
    return out


def recent_videos(path: Path, max_age_days: int) -> list[SheetVideo]:
    return [v for v in read_videos(path) if v.age_days <= max_age_days]
