"""S3 is the source of truth for analysis outputs: ocp/analysis/<video_id>/{record,captions,frames}.json."""
import json
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from app import config

_KINDS = {"record": "{vid}.record.json", "captions": "{vid}.captions.json", "frames": "{vid}.frames.json"}


def _s3():
    return boto3.client("s3", region_name=config.AWS_REGION)


def prefix() -> str:
    return f"{config.S3_PREFIX.rsplit('/', 1)[0]}/analysis"


def upload_dir(directory: Path) -> dict[str, int]:
    """Push every <id>.record.json (plus its captions/frames if present) found in `directory`."""
    s3, counts = _s3(), {"records": 0, "captions": 0, "frames": 0}
    for rec_file in sorted(directory.glob("*.record.json")):
        vid = rec_file.name.removesuffix(".record.json")
        for kind, pattern in _KINDS.items():
            f = directory / pattern.format(vid=vid)
            if f.exists():
                s3.upload_file(str(f), config.S3_BUCKET, f"{prefix()}/{vid}/{kind}.json", ExtraArgs={"ContentType": "application/json"})
                counts["records" if kind == "record" else kind] += 1
    return counts


def list_ids() -> list[str]:
    s3, ids = _s3(), set()
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=config.S3_BUCKET, Prefix=prefix() + "/", Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            ids.add(cp["Prefix"][len(prefix()) + 1:].strip("/"))
    return sorted(ids)


def read_json(vid: str, kind: str):
    try:
        obj = _s3().get_object(Bucket=config.S3_BUCKET, Key=f"{prefix()}/{vid}/{kind}.json")
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise
    return json.loads(obj["Body"].read())
