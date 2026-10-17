"""Embed chunks with Bedrock Titan Text Embeddings v2 (1024-d, normalized)."""
import json
import time
from concurrent.futures import ThreadPoolExecutor

import boto3

from app import config, pg

MODEL = "amazon.titan-embed-text-v2:0"
DIMS = 1024


def _client():
    return boto3.client("bedrock-runtime", region_name=config.AWS_REGION)


def embed_text(client, text: str) -> list[float]:
    body = json.dumps({"inputText": text[:20000], "dimensions": DIMS, "normalize": True})
    for attempt in range(8):
        try:
            return json.loads(client.invoke_model(modelId=MODEL, body=body)["body"].read())["embedding"]
        except client.exceptions.ThrottlingException:
            time.sleep(min(2 ** attempt, 60))  # Titan embed is rate-limited per account; back off and retry
    raise RuntimeError("embedding throttled after 8 attempts")


def vec_literal(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def embed_pending(batch: int = 200, workers: int = 4, limit: int | None = None) -> int:
    client = _client()
    done = 0
    with pg.connect() as c:
        while True:
            rows = c.execute("SELECT id, context, text FROM chunks WHERE embedding IS NULL ORDER BY id LIMIT %s", (batch,)).fetchall()
            if not rows or (limit and done >= limit):
                break
            with ThreadPoolExecutor(workers) as pool:
                vecs = list(pool.map(lambda r: embed_text(client, f"{r[1]}: {r[2]}"), rows))
            c.cursor().executemany("UPDATE chunks SET embedding = %s::halfvec, embed_model = %s WHERE id = %s",
                          [(vec_literal(v), MODEL, r[0]) for r, v in zip(rows, vecs)])
            c.commit()
            done += len(rows)
            print(f"embedded {done}", flush=True)
    return done
