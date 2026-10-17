CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS videos (
    id              text PRIMARY KEY,
    url             text NOT NULL,
    title           text NOT NULL,
    duration_sec    integer,
    upload_date     date,
    view_count      integer,
    channel         text,
    s3_uri          text,
    summary         text,
    talk_type       text,
    session         jsonb,
    topics          text[],
    schema_version  integer,
    analysis        jsonb,
    record_s3_uri   text,
    loaded_at       timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS speakers (
    id          serial PRIMARY KEY,
    video_id    text NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    name        text NOT NULL,
    affiliation text,
    role        text,
    evidence    text
);
CREATE INDEX IF NOT EXISTS speakers_name_trgm ON speakers USING gin (name gin_trgm_ops);

-- Corpus-level entities. Names are deduped case-insensitively for now; real alias resolution comes later.
CREATE TABLE IF NOT EXISTS entities (
    id      serial PRIMARY KEY,
    name    text NOT NULL,
    kind    text,
    UNIQUE (name)
);
CREATE UNIQUE INDEX IF NOT EXISTS entities_lower_name ON entities (lower(name));
CREATE INDEX IF NOT EXISTS entities_name_trgm ON entities USING gin (name gin_trgm_ops);

CREATE TABLE IF NOT EXISTS entity_aliases (
    alias       text PRIMARY KEY,
    entity_id   integer NOT NULL REFERENCES entities(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS video_entities (
    video_id      text NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    entity_id     integer NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    raw_name      text NOT NULL,
    raw_kind      text,
    segment_idxs  integer[],
    PRIMARY KEY (video_id, entity_id)
);
CREATE INDEX IF NOT EXISTS video_entities_entity ON video_entities (entity_id);

CREATE TABLE IF NOT EXISTS segments (
    id             serial PRIMARY KEY,
    video_id       text NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    idx            integer NOT NULL,
    start_sec      real NOT NULL,
    end_sec        real NOT NULL,
    title          text,
    summary        text,
    on_screen_text text,
    keep           boolean NOT NULL DEFAULT true,
    merged_idxs    integer[],
    note           text,
    UNIQUE (video_id, idx)
);

CREATE TABLE IF NOT EXISTS claims (
    id                    serial PRIMARY KEY,
    video_id              text NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    claim                 text NOT NULL,
    claim_type            text,
    quote                 text,
    start_sec             real,
    end_sec               real,
    segment_idxs          integer[],
    source_cited_on_slide text,
    confidence            text,
    quote_verified        boolean
);
CREATE INDEX IF NOT EXISTS claims_video ON claims (video_id);
CREATE INDEX IF NOT EXISTS claims_type ON claims (claim_type);

CREATE TABLE IF NOT EXISTS transcript_chunks (
    id         serial PRIMARY KEY,
    video_id   text NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    start_sec  real NOT NULL,
    text       text NOT NULL
);
CREATE INDEX IF NOT EXISTS transcript_video ON transcript_chunks (video_id, start_sec);

CREATE TABLE IF NOT EXISTS source_documents (
    id        serial PRIMARY KEY,
    video_id  text NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    title     text,
    url       text,
    evidence  text
);

CREATE TABLE IF NOT EXISTS quality_flags (
    id        serial PRIMARY KEY,
    video_id  text NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    location  text,
    issue     text
);

-- One searchable row per video / segment / claim / transcript window.
-- `context` is the video title, included in both the full-text index and the embedding input.
CREATE TABLE IF NOT EXISTS chunks (
    id           bigserial PRIMARY KEY,
    video_id     text NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    kind         text NOT NULL CHECK (kind IN ('video', 'segment', 'claim', 'transcript')),
    ref_id       integer,
    start_sec    real,
    end_sec      real,
    context      text NOT NULL,
    text         text NOT NULL,
    meta         jsonb NOT NULL DEFAULT '{}',
    tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', context || ' ' || text)) STORED,
    embedding    halfvec(1024),
    embed_model  text
);
CREATE INDEX IF NOT EXISTS chunks_tsv ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding ON chunks USING hnsw (embedding halfvec_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_video_kind ON chunks (video_id, kind);
