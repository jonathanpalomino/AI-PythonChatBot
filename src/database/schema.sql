-- =============================================================================
-- Database Schema for RAG Chatbot
-- PostgreSQL 14+
-- =============================================================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- ENUMS
-- =============================================================================

CREATE TYPE message_role AS ENUM ('user', 'assistant', 'system');
CREATE TYPE processing_status AS ENUM ('pending', 'processing', 'completed', 'error');
CREATE TYPE visibility_type AS ENUM ('public', 'private', 'shared');
CREATE TYPE hallucination_mode AS ENUM ('strict', 'balanced', 'creative');
CREATE TYPE tool_mode AS ENUM ('agent', 'manual');
CREATE TYPE tooltype AS ENUM ('http_request', 'sql_query', 'rag_search', 'custom');

-- =============================================================================
-- TABLE: prompt_templates
-- =============================================================================

CREATE TABLE prompt_templates (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(100) NOT NULL,
    visibility visibility_type DEFAULT 'public',
    system_prompt TEXT NOT NULL,
    user_prompt_template TEXT,
    variables JSONB DEFAULT '[]'::jsonb,
    settings JSONB DEFAULT '{}'::jsonb,
    version INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT true,
    created_by VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_prompt_templates_category ON prompt_templates(category);
CREATE INDEX idx_prompt_templates_visibility ON prompt_templates(visibility);
CREATE INDEX idx_prompt_templates_active ON prompt_templates(is_active) WHERE is_active = true;

-- =============================================================================
-- TABLE: qdrant_collections
-- =============================================================================

CREATE TABLE qdrant_collections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    display_name VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(100),
    vector_count INTEGER DEFAULT 0,
    last_synced TIMESTAMP,
    is_active BOOLEAN DEFAULT true,
    visibility visibility_type DEFAULT 'public',
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_qdrant_collections_active ON qdrant_collections(is_active) WHERE is_active = true;

-- =============================================================================
-- TABLE: projects
-- =============================================================================

CREATE TABLE projects (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_projects_active ON projects(is_active) WHERE is_active = true;

-- =============================================================================
-- TABLE: conversations
-- =============================================================================

CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(500) NOT NULL,
    project_id UUID REFERENCES projects(id) ON DELETE SET NULL,
    prompt_template_id UUID REFERENCES prompt_templates(id) ON DELETE SET NULL,
    settings JSONB DEFAULT '{}'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_conversations_updated ON conversations(updated_at DESC);
CREATE INDEX idx_conversations_project ON conversations(project_id);
CREATE INDEX idx_conversations_template ON conversations(prompt_template_id);
CREATE INDEX idx_conversations_settings ON conversations USING gin(settings);

-- =============================================================================
-- TABLE: messages
-- =============================================================================

CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role message_role NOT NULL,
    content TEXT NOT NULL,
    thinking_content TEXT DEFAULT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    attachments JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_messages_conversation_created ON messages(conversation_id, created_at);
CREATE INDEX idx_messages_metadata ON messages USING gin(metadata);

-- =============================================================================
-- TABLE: files
-- =============================================================================

CREATE TABLE files (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    conversation_id UUID REFERENCES conversations(id) ON DELETE SET NULL,
    file_name VARCHAR(500) NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    file_size BIGINT NOT NULL,
    storage_path TEXT NOT NULL,
    mime_type VARCHAR(255),
    processed BOOLEAN DEFAULT false,
    processing_status processing_status DEFAULT 'pending',
    metadata JSONB DEFAULT '{}'::jsonb,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_files_conversation ON files(conversation_id);
CREATE INDEX idx_files_project ON files(project_id);
CREATE INDEX idx_files_type ON files(file_type);
CREATE INDEX idx_files_processed ON files(processed) WHERE processed = true;

-- =============================================================================
-- TABLE: tool_configurations
-- =============================================================================

CREATE TABLE tool_configurations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    tool_name VARCHAR(100) NOT NULL,
    config JSONB DEFAULT '{}'::jsonb,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(conversation_id, tool_name)
);

CREATE INDEX idx_tool_configs_conversation ON tool_configurations(conversation_id);
CREATE INDEX idx_tool_configs_active ON tool_configurations(is_active) WHERE is_active = true;

-- =============================================================================
-- TABLE: custom_tools
-- =============================================================================

CREATE TABLE custom_tools (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    tool_type tooltype NOT NULL DEFAULT 'http_request',
    configuration JSONB DEFAULT '{}'::jsonb,
    visibility visibility_type DEFAULT 'public',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_custom_tools_active ON custom_tools(is_active) WHERE is_active = true;
CREATE INDEX idx_custom_tools_visibility ON custom_tools(visibility);
CREATE INDEX idx_custom_tools_type ON custom_tools(tool_type);

CREATE TRIGGER update_custom_tools_updated_at BEFORE UPDATE ON custom_tools
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- TABLE: conversation_memory
-- =============================================================================

CREATE TABLE conversation_memory (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    summary TEXT NOT NULL,
    key_points JSONB DEFAULT '[]'::jsonb,
    token_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_conversation_memory_conv ON conversation_memory(conversation_id);

-- =============================================================================
-- TABLE: llm_models
-- =============================================================================

CREATE TABLE llm_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    provider VARCHAR(50) NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    model_type VARCHAR(50) DEFAULT 'chat',
    context_window INTEGER DEFAULT 4096,
    supports_streaming BOOLEAN DEFAULT true,
    supports_function_calling BOOLEAN DEFAULT false,
    is_active BOOLEAN DEFAULT true,
    is_custom BOOLEAN DEFAULT false,
    supports_thinking BOOLEAN DEFAULT false,
    last_seen TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    cpu_supported BOOLEAN DEFAULT true,
    gpu_required BOOLEAN DEFAULT false,
    parent_retrieval_supported BOOLEAN DEFAULT true,
    UNIQUE(provider, model_name)
);

CREATE INDEX idx_llm_models_provider ON llm_models(provider);
CREATE INDEX idx_llm_models_active ON llm_models(is_active) WHERE is_active = true;

CREATE TRIGGER update_llm_models_updated_at BEFORE UPDATE ON llm_models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- TRIGGERS for updated_at
-- =============================================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_prompt_templates_updated_at BEFORE UPDATE ON prompt_templates
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_projects_updated_at BEFORE UPDATE ON projects
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_conversations_updated_at BEFORE UPDATE ON conversations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_qdrant_collections_updated_at BEFORE UPDATE ON qdrant_collections
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_tool_configurations_updated_at BEFORE UPDATE ON tool_configurations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- SEED DATA: Default Prompt Templates
-- =============================================================================

INSERT INTO prompt_templates (name, description, category, visibility, system_prompt, variables, settings) VALUES
(
    'Asistente de Código',
    'Experto en análisis y revisión de código con múltiples lenguajes',
    'code',
    'public',
    'Eres un experto desarrollador de software con amplia experiencia en {language}. Tu tarea es analizar código, encontrar bugs, sugerir mejoras y explicar conceptos de forma clara. Siempre proporciona ejemplos concretos y justifica tus sugerencias.',
    '[
        {"name": "language", "type": "select", "options": ["JavaScript", "TypeScript", "Python", "Java", "PL/SQL"], "default": "JavaScript", "required": true},
        {"name": "focus", "type": "select", "options": ["Bugs", "Performance", "Security", "Best Practices", "All"], "default": "All", "required": false}
    ]'::jsonb,
    '{
        "recommended_provider": "claude",
        "recommended_model": "claude-sonnet-3.5",
        "temperature": 0.3,
        "default_tools": [],
        "hallucination_mode": "balanced",
        "_comment": "code_analyzer removed - now automatic on file upload"
    }'::jsonb
),
(
    'Analista de Documentos',
    'Especialista en analizar y extraer información de documentos administrativos',
    'docs',
    'public',
    'Eres un analista experto en documentos administrativos y legales. Tu tarea es extraer información precisa, resumir contenido y responder preguntas específicas sobre documentos. SIEMPRE cita la sección o página exacta de donde obtuviste la información. Si no encuentras la información en el documento, indícalo claramente.',
    '[
        {"name": "doc_type", "type": "select", "options": ["Contrato", "Reporte", "Manual", "Procedimiento", "General"], "default": "General", "required": false}
    ]'::jsonb,
    '{
        "recommended_provider": "openai",
        "recommended_model": "gpt-4",
        "temperature": 0.1,
        "default_tools": ["rag_search"],
        "hallucination_mode": "strict",
        "_comment": "document_processor removed - now automatic on file upload"
    }'::jsonb
),
(
    'Consultor RAG',
    'Experto que busca en bases de conocimiento específicas para responder',
    'docs',
    'public',
    'Eres un consultor técnico experto. Tu tarea es buscar información en la documentación disponible y proporcionar respuestas precisas basadas ÚNICAMENTE en esa información. Siempre cita las fuentes exactas. Si la información no está en la documentación, indícalo claramente y NO inventes respuestas.',
    '[]'::jsonb,
    '{
        "recommended_provider": "local",
        "recommended_model": "mistral",
        "temperature": 0.2,
        "default_tools": ["rag_search"],
        "hallucination_mode": "strict"
    }'::jsonb
),
(
    'Asistente General',
    'Asistente conversacional para consultas generales',
    'general',
    'public',
    'Eres un asistente útil y versátil. Puedes ayudar con una amplia variedad de tareas: responder preguntas, generar ideas, explicar conceptos y más. Sé claro, conciso y útil. Si no estás seguro de algo, admítelo.',
    '[]'::jsonb,
    '{
        "recommended_provider": "openai",
        "recommended_model": "gpt-4",
        "temperature": 0.7,
        "default_tools": [],
        "hallucination_mode": "balanced"
    }'::jsonb
);

-- =============================================================================
-- SEED DATA: Sample Qdrant Collections
-- =============================================================================

INSERT INTO qdrant_collections (name, display_name, description, category, visibility, metadata) VALUES
(
    'api-documentation',
    'Documentación API REST',
    'Documentación completa de los endpoints de la API',
    'docs',
    'public',
    '{
        "source_path": "./docs/api",
        "embedding_model": "mxbai-embed-large",
        "indexed_file_count": 0
    }'::jsonb
),
(
    'plsql-manual',
    'Manual de PL/SQL',
    'Guías y referencias de procedimientos almacenados',
    'code',
    'public',
    '{
        "source_path": "./docs/plsql",
        "embedding_model": "mxbai-embed-large",
        "indexed_file_count": 0
    }'::jsonb
),
(
    'admin-procedures',
    'Procedimientos Administrativos',
    'Manuales y procedimientos del área administrativa',
    'admin',
    'public',
    '{
        "source_path": "./docs/admin",
        "embedding_model": "mxbai-embed-large",
        "indexed_file_count": 0
    }'::jsonb
);

-- =============================================================================
-- VIEWS (Optional - for analytics)
-- =============================================================================

CREATE VIEW conversation_stats AS
SELECT 
    c.id,
    c.title,
    c.created_at,
    COUNT(m.id) as message_count,
    COUNT(DISTINCT f.id) as file_count,
    MAX(m.created_at) as last_message_at
FROM conversations c
LEFT JOIN messages m ON c.id = m.conversation_id
LEFT JOIN files f ON c.id = f.conversation_id
GROUP BY c.id, c.title, c.created_at;

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE prompt_templates IS 'Plantillas de prompts predefinidas y personalizadas';
COMMENT ON TABLE qdrant_collections IS 'Registro de colecciones de vectores en Qdrant';
COMMENT ON TABLE conversations IS 'Conversaciones del chatbot';
COMMENT ON TABLE messages IS 'Mensajes individuales dentro de conversaciones';
COMMENT ON TABLE files IS 'Archivos adjuntos a conversaciones';
COMMENT ON TABLE tool_configurations IS 'Configuración de tools por conversación';
COMMENT ON TABLE conversation_memory IS 'Memoria/resumen de conversaciones para contexto';