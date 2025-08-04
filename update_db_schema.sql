-- 기존 documents 테이블에 표 관련 컬럼 추가

-- chunk_type 컬럼 추가 (text, table, table_section 등)
ALTER TABLE documents ADD COLUMN IF NOT EXISTS chunk_type VARCHAR(50) DEFAULT 'text';

-- metadata 컬럼 추가 (JSON 형태로 표 메타데이터 저장)
ALTER TABLE documents ADD COLUMN IF NOT EXISTS metadata JSONB;

-- 표 타입별 인덱스 추가
CREATE INDEX IF NOT EXISTS idx_documents_chunk_type ON documents(chunk_type);
CREATE INDEX IF NOT EXISTS idx_documents_metadata ON documents USING GIN(metadata);

-- 기존 데이터의 chunk_type을 'text'로 설정
UPDATE documents SET chunk_type = 'text' WHERE chunk_type IS NULL;