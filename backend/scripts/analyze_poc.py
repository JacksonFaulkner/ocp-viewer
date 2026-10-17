"""POC: analyze one S3 video with Pegasus 1.5 and write RAG-ready JSON.

    uv run python scripts/analyze_poc.py N_4_1orETrc
"""
import json
import sys
import time
from pathlib import Path

import boto3
from botocore.config import Config

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import config  # noqa: E402

MODEL = "us.twelvelabs.pegasus-1-5-v1:0"

DOC_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "3-5 sentence summary of the talk"},
        "talk_type": {"type": "string", "description": "e.g. keynote, technical talk, panel, demo"},
        "speakers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "As stated or shown on screen; empty if unknown"},
                    "affiliation": {"type": "string"},
                },
            },
        },
        "topics": {"type": "array", "items": {"type": "string"}, "description": "5-10 specific technical topics"},
        "entities": {
            "type": "array",
            "description": "Companies, products, standards, projects, hardware mentioned",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "kind": {"type": "string", "enum": ["company", "product", "standard", "project", "hardware", "other"]},
                },
            },
        },
        "key_claims": {
            "type": "array",
            "description": "Concrete factual claims, numbers, or results stated in the talk",
            "items": {"type": "string"},
        },
    },
    "required": ["summary", "topics"],
}

SEGMENT_DEFS = [
    {
        "id": "section",
        "description": "A distinct section of the talk covering one idea; typically 30s-3min",
        "fields": [
            {"name": "title", "type": "string", "description": "Short heading for the section"},
            {"name": "summary", "type": "string", "description": "2-3 sentences, self-contained so it makes sense out of context"},
            {"name": "on_screen_text", "type": "string", "description": "Key text, labels and numbers visible on slides; empty if none"},
            {"name": "keywords", "type": "string", "description": "Comma-separated technical terms, products and standards discussed"},
        ],
    }
]


def invoke(client, uri: str, body: dict) -> tuple[dict, float]:
    body["mediaSource"] = {"s3Location": {"uri": uri, "bucketOwner": ACCOUNT}}
    t0 = time.monotonic()
    r = client.invoke_model(modelId=MODEL, body=json.dumps(body))
    return json.loads(r["body"].read()), time.monotonic() - t0


if __name__ == "__main__":
    vid = sys.argv[1]
    sess = boto3.Session(region_name=config.AWS_REGION)
    ACCOUNT = sess.client("sts").get_caller_identity()["Account"]
    client = sess.client("bedrock-runtime", config=Config(read_timeout=600, retries={"max_attempts": 1}))
    uri = f"s3://{config.S3_BUCKET}/{config.S3_PREFIX}/{vid}/{vid}.mp4"
    out = ROOT / "data" / "poc"

    doc, t1 = invoke(client, uri, {
        "analysis_mode": "general",
        "inputPrompt": "Analyze this conference talk video for a searchable knowledge base. Fill the schema accurately; use only what is said or shown.",
        "responseFormat": {"type": "json_schema", "jsonSchema": DOC_SCHEMA},
        "maxOutputTokens": 4096,
    })
    (out / f"{vid}.document.raw.json").write_text(json.dumps(doc, indent=2))
    print(f"document mode: {t1:.1f}s", list(doc))

    seg, t2 = invoke(client, uri, {
        "analysis_mode": "time_based_metadata",
        "segment_definitions": SEGMENT_DEFS,
        "maxOutputTokens": 8192, "temperature": 0,
    })
    (out / f"{vid}.segments.raw.json").write_text(json.dumps(seg, indent=2))
    print(f"segment mode: {t2:.1f}s", list(seg), seg.get("input_tokens"), seg.get("output_tokens"))
