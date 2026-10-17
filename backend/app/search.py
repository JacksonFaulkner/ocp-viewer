"""Hybrid search over chunks: Postgres full-text + pgvector similarity, merged with reciprocal-rank fusion."""
import re

from app import pg
from app.embed import _client, embed_text, vec_literal

_SQL = """
WITH q AS (SELECT to_tsquery('english', %(tsq)s) AS tsq),
sem AS (
    SELECT id, row_number() OVER (ORDER BY embedding <=> %(vec)s::halfvec) AS r
    FROM chunks WHERE embedding IS NOT NULL {filt}
    ORDER BY embedding <=> %(vec)s::halfvec LIMIT %(pool)s),
lex AS (
    SELECT c.id, row_number() OVER (ORDER BY ts_rank_cd(c.tsv, q.tsq) DESC) AS r
    FROM chunks c, q WHERE c.tsv @@ q.tsq {filt_c}
    ORDER BY ts_rank_cd(c.tsv, q.tsq) DESC LIMIT %(pool)s)
SELECT c.id, c.video_id, c.kind, c.start_sec, c.end_sec, c.text, c.meta, v.title, v.url,
       coalesce(1.0/(60+sem.r), 0) + coalesce(1.0/(60+lex.r), 0) AS score, sem.r AS sem_rank, lex.r AS lex_rank
FROM chunks c JOIN videos v ON v.id = c.video_id
LEFT JOIN sem ON sem.id = c.id LEFT JOIN lex ON lex.id = c.id
WHERE sem.id IS NOT NULL OR lex.id IS NOT NULL
ORDER BY score DESC LIMIT %(k)s
"""


def _or_query(text: str) -> str:
    """'how do I add SFF-8024' -> 'how | do | i | add | sff | 8024' (stopwords are dropped by to_tsquery)."""
    terms = dict.fromkeys(t.lower() for t in re.findall(r"[A-Za-z0-9]+", text) if len(t) > 1)
    return " | ".join(terms) or "x"


def search(query: str, k: int = 10, kinds: list[str] | None = None, pool: int = 60) -> list[dict]:
    vec = vec_literal(embed_text(_client(), query))
    filt = "AND kind = ANY(%(kinds)s)" if kinds else ""
    filt_c = "AND c.kind = ANY(%(kinds)s)" if kinds else ""
    params = {"tsq": _or_query(query), "vec": vec, "pool": pool, "k": k, "kinds": kinds}
    with pg.connect() as c:
        cur = c.execute(_SQL.format(filt=filt, filt_c=filt_c), params)
        cols = [d.name for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    for r in rows:
        r["link"] = f"{r['url']}&t={int(r['start_sec'] or 0)}s"
    return rows
