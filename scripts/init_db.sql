-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Audio files metadata
CREATE TABLE IF NOT EXISTS audio_files (
    id          SERIAL PRIMARY KEY,
    filename    TEXT NOT NULL UNIQUE,
    source_url  TEXT,
    duration_s  FLOAT,
    speaker_a   TEXT,
    speaker_b   TEXT,
    topic       TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Speaker turn chunks (the searchable unit)
CREATE TABLE IF NOT EXISTS chunks (
    id              SERIAL PRIMARY KEY,
    audio_file_id   INT REFERENCES audio_files(id) ON DELETE CASCADE,
    speaker         TEXT NOT NULL,
    text            TEXT NOT NULL,
    start_time      FLOAT NOT NULL,
    end_time        FLOAT NOT NULL,
    embedding       vector(384),
    text_search     tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for hybrid search
CREATE INDEX IF NOT EXISTS idx_chunks_text_search ON chunks USING gin (text_search);
CREATE INDEX IF NOT EXISTS idx_chunks_audio_file ON chunks (audio_file_id);

-- Note: IVFFlat index on embeddings is created after data is loaded
-- (it needs rows to build the index properly).
-- Run: CREATE INDEX idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);
