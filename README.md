# 🤖 RAG Chatbot - Sistema Inteligente Multi-Proveedor con Enfoque Empresarial

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](LICENSE)

> **Sistema de chatbot con RAG (Retrieval-Augmented Generation) diseñado con arquitectura escalable de nivel empresarial, soporte multi-proveedor de LLMs, procesamiento avanzado de documentos y sistema extensible basado en herramientas.**

---

## ⚠️ Disclaimer

Este es un **proyecto personal de código abierto** desarrollado para demostrar competencias técnicas en arquitectura de software, integración de IA y desarrollo backend. El sistema implementa patrones y prácticas de nivel empresarial aplicables a entornos de producción.

---

## 📋 Tabla de Contenidos

- [Descripción General](#-descripción-general)
- [Stack Tecnológico](#-stack-tecnológico)
- [Arquitectura del Sistema](#-arquitectura-del-sistema)
- [Características Principales](#-características-principales)
- [Competencias Técnicas del Proyecto](#-competencias-técnicas-del-proyecto)
- [Instalación y Configuración](#-instalación-y-configuración)
- [Documentación de API](#-documentación-de-api)
- [Guía de Uso](#-guía-de-uso)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Proveedores de LLM](#-proveedores-de-llm)
- [Performance y Optimización](#-performance-y-optimización)
- [Seguridad](#-seguridad)
- [Testing](#-testing)
- [Licencia](#-licencia)

---

## 🎯 Descripción General

Sistema de chatbot con arquitectura empresarial que implementa **Retrieval-Augmented Generation (RAG)** con soporte para múltiples proveedores de Large Language Models (LLMs). El sistema permite conversaciones contextuales enriquecidas con información de documentos, ejecuta herramientas personalizadas, y proporciona control granular sobre el comportamiento y calidad de las respuestas.

### Casos de Uso

Este sistema está diseñado para escenarios de producción tales como:

- **Knowledge Base Chatbot**: Respuestas basadas en documentación técnica, manuales y bases de conocimiento
- **Análisis de Documentos**: Procesamiento y consulta de documentos en múltiples formatos
- **Asistente de Código**: Análisis y consulta de repositorios de código fuente
- **Integración con Sistemas**: Conexión con bases de datos, APIs externas y servicios
- **Búsqueda Semántica Avanzada**: Recuperación de información con embeddings y reranking

---

## 🛠 Stack Tecnológico

### Backend Core
- **Python 3.11+** - Lenguaje principal
- **FastAPI 0.115+** - Framework web async/await de alto rendimiento
- **SQLAlchemy 2.0** - ORM con soporte async
- **Pydantic V2** - Validación de datos y serialización

### Bases de Datos y Storage
- **PostgreSQL** - Base de datos relacional principal
- **Qdrant** - Base de datos vectorial para embeddings y búsqueda semántica
- **Redis** - Cache, gestión de sesiones y cancelación de streams

### LLM Providers
- **Ollama** - Modelos locales (Mistral, Llama 3, Qwen, DeepSeek R1, Gemma)
- **OpenAI** - GPT-4 Turbo, GPT-3.5 Turbo
- **Anthropic** - Claude 3.5 Sonnet, Claude 3 Opus
- **OpenRouter** - Acceso unificado a múltiples modelos
- **Groq** - Inferencia ultra-rápida
- **Google** - Gemini (preparado para integración)

### Document Processing
- **PyPDF2** - Procesamiento de PDFs
- **python-docx** - Documentos Word
- **openpyxl** - Excel y hojas de cálculo
- **python-pptx** - Presentaciones PowerPoint
- **BeautifulSoup4** - Parsing de HTML
- **Markdown** - Procesamiento avanzado de Markdown con detección de Obsidian

### Infrastructure
- **Docker & Docker Compose** - Containerización
- **Uvicorn** - Servidor ASGI de alto rendimiento
- **Nginx** - Reverse proxy (producción)

---

## 🏗 Arquitectura del Sistema

### Patrón de Diseño

El sistema implementa una **arquitectura de capas** con separación clara de responsabilidades:

```
┌─────────────────────────────────────────────────────┐
│              API Layer (FastAPI)                    │
│   - REST Endpoints                                  │
│   - Request/Response Validation (Pydantic)          │
│   - SSE Streaming                                   │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│           Service Layer (Business Logic)            │
│   - ChatOrchestrator (Flujo de conversación)       │
│   - FileProcessor (Procesamiento de documentos)    │
│   - EmbeddingService (Generación de embeddings)    │
│   - ConversationMemory (Memoria semántica)         │
│   - StreamCancelManager (Gestión de cancelación)   │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│          Provider Layer (LLM Abstraction)           │
│   - ProviderManager (Factory)                      │
│   - BaseProvider (Interface)                        │
│   - LocalProvider, OpenAIProvider, etc.            │
│   - CancellableStreamMixin                         │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│            Tool System (Extensible)                 │
│   - BaseTool (Abstract)                            │
│   - ToolRegistry (Singleton)                        │
│   - RAGTool, HTTPTool, SQLTool                     │
│   - CustomToolExecutor                              │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│        Data Layer (Persistence)                     │
│   - PostgreSQL (Metadata, conversaciones)          │
│   - Qdrant (Vectores, búsqueda semántica)         │
│   - Redis (Cache, stream management)               │
│   - File System / S3 (Almacenamiento de archivos)  │
└─────────────────────────────────────────────────────┘
```

### Diagrama de Arquitectura Detallado

```mermaid
graph TB
    subgraph "Frontend Layer"
        UI[Angular Frontend]
    end

    subgraph "API Layer - FastAPI"
        API[REST API Endpoints]
        SSE[SSE Streaming]
        CORS[CORS Middleware]
        VALID[Pydantic Validation]
    end

    subgraph "Service Layer"
        ORCH[ChatOrchestrator]
        FILEPROC[FileProcessor]
        EMBED[EmbeddingService]
        MEM[ConversationMemory]
        CANCEL[StreamCancelManager]
        HYBRID[HybridSearch]
        RERANK[Reranker]
        PDF[PDFService]
    end

    subgraph "Provider Layer"
        PMGR[ProviderManager]
        LOCAL[LocalProvider - Ollama]
        OPENAI[OpenAIProvider]
        CLAUDE[ClaudeProvider]
        GROQ[GroqProvider]
        ROUTER[OpenRouterProvider]
    end

    subgraph "Tool System"
        TREG[ToolRegistry]
        RAGT[RAGTool]
        HTTP[HTTPTool]
        SQL[SQLTool]
        CUSTOM[CustomToolExecutor]
    end

    subgraph "Document Processing"
        PDF_LOAD[PDF Loader]
        DOCX_LOAD[Word Loader]
        EXCEL_LOAD[Excel Loader]
        MD_LOAD[Markdown Loader]
        CODE_LOAD[Code Loader]
        OBS_DETECT[Obsidian Detector]
    end

    subgraph "Data Layer"
        PG[(PostgreSQL)]
        QD[(Qdrant Vector DB)]
        RD[(Redis Cache)]
        FS[File System]
    end

    subgraph "External Services"
        OLLAMA_SVC[Ollama Service]
        OPENAI_SVC[OpenAI API]
        CLAUDE_SVC[Anthropic API]
        GROQ_SVC[Groq API]
        OR_SVC[OpenRouter API]
    end

    %% Frontend to API
    UI -->|HTTP/SSE| API
    
    %% API Layer
    API --> CORS
    API --> VALID
    API --> SSE
    
    %% API to Services
    API --> ORCH
    API --> FILEPROC
    API --> PDF
    
    %% Orchestrator Flow
    ORCH --> PMGR
    ORCH --> TREG
    ORCH --> MEM
    ORCH --> CANCEL
    
    %% File Processing
    FILEPROC --> PDF_LOAD
    FILEPROC --> DOCX_LOAD
    FILEPROC --> EXCEL_LOAD
    FILEPROC --> MD_LOAD
    FILEPROC --> CODE_LOAD
    FILEPROC --> OBS_DETECT
    FILEPROC --> EMBED
    
    %% Embedding and Storage
    EMBED --> QD
    
    %% Memory System
    MEM --> HYBRID
    MEM --> RERANK
    HYBRID --> QD
    
    %% Provider Management
    PMGR --> LOCAL
    PMGR --> OPENAI
    PMGR --> CLAUDE
    PMGR --> GROQ
    PMGR --> ROUTER
    
    %% Providers to External Services
    LOCAL --> OLLAMA_SVC
    OPENAI --> OPENAI_SVC
    CLAUDE --> CLAUDE_SVC
    GROQ --> GROQ_SVC
    ROUTER --> OR_SVC
    
    %% Tool System
    TREG --> RAGT
    TREG --> HTTP
    TREG --> SQL
    TREG --> CUSTOM
    RAGT --> QD
    
    %% Stream Cancellation
    CANCEL --> RD
    
    %% Data Persistence
    ORCH --> PG
    FILEPROC --> PG
    FILEPROC --> FS
    MEM --> PG
    
    %% Styling
    classDef frontend fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    classDef api fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef service fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef provider fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
    classDef tool fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    classDef data fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef external fill:#e0f2f1,stroke:#004d40,stroke-width:2px
    classDef document fill:#f1f8e9,stroke:#33691e,stroke-width:2px
    
    class UI frontend
    class API,SSE,CORS,VALID api
    class ORCH,FILEPROC,EMBED,MEM,CANCEL,HYBRID,RERANK,PDF service
    class PMGR,LOCAL,OPENAI,CLAUDE,GROQ,ROUTER provider
    class TREG,RAGT,HTTP,SQL,CUSTOM tool
    class PG,QD,RD,FS data
    class OLLAMA_SVC,OPENAI_SVC,CLAUDE_SVC,GROQ_SVC,OR_SVC external
    class PDF_LOAD,DOCX_LOAD,EXCEL_LOAD,MD_LOAD,CODE_LOAD,OBS_DETECT document
```

### Flujo de Datos Principal

```mermaid
sequenceDiagram
    participant U as Usuario
    participant F as Frontend
    participant A as API Layer
    participant O as ChatOrchestrator
    participant P as ProviderManager
    participant T as ToolRegistry
    participant Q as Qdrant
    participant PG as PostgreSQL
    participant LLM as LLM Provider

    U->>F: Envía mensaje
    F->>A: POST /conversations/{id}/stream
    A->>O: process_message_stream()
    
    O->>PG: Guardar mensaje usuario
    O->>T: Seleccionar herramientas
    
    alt Modo Agent
        O->>P: Generar con function calling
        P->>LLM: Stream con tools
        LLM-->>P: Tool call requerido
        P-->>O: Tool call decision
        O->>T: Ejecutar herramienta
        T->>Q: Búsqueda RAG
        Q-->>T: Resultados
        T-->>O: Resultados tool
        O->>P: Continuar generación
    else Modo Manual
        O->>T: Ejecutar RAG (heurística)
        T->>Q: Búsqueda semántica
        Q-->>T: Contexto relevante
        T-->>O: Contexto
        O->>P: Generar con contexto
    end
    
    P->>LLM: Stream respuesta
    LLM-->>P: Chunks
    P-->>O: Chunks
    O-->>A: SSE Events
    A-->>F: Server-Sent Events
    F-->>U: Streaming UI
    
    O->>PG: Guardar mensaje asistente
    O->>PG: Actualizar metadata (tokens, cost)
```

### Componentes Clave

#### 1. **ChatOrchestrator**
Orquestador central que coordina:
- Gestión del flujo de conversación
- Selección y ejecución de herramientas (modo Agent vs Manual)
- Control de alucinaciones (Strict, Balanced, Creative)
- Memoria semántica y recuperación de contexto
- Streaming con soporte de cancelación

#### 2. **Provider Manager**
Sistema de abstracción para múltiples LLMs:
- Interface unificada para todos los proveedores
- Auto-sincronización de modelos disponibles
- Cálculo automático de costos por token
- Soporte para function calling (OpenAI, Claude)
- Streaming cancelable con Redis

#### 3. **Tool System**
Arquitectura extensible para herramientas:
- **Tool Registry**: Registro dinámico de herramientas
- **Tool Discovery**: Detección automática de herramientas
- **Custom Tools**: Sistema de herramientas definidas por usuario
- **Tool Configurations**: Configuraciones por conversación

#### 4. **Document Processing Pipeline**
Pipeline robusto para procesamiento de documentos:
- Detección automática de formato
- Chunking inteligente con overlap configurable
- Generación de embeddings
- Indexación en Qdrant
- Metadata extraction

---

## ✨ Características Principales

### 🎭 Modos de Operación

#### Modo Agent (Function Calling)
```json
{
  "settings": {
    "tool_mode": "agent",
    "available_tools": ["rag_search", "http_request", "sql_query"],
    "allow_tool_chaining": true
  }
}
```
- La IA decide autónomamente qué herramientas usar
- Encadenamiento de herramientas múltiples
- Requiere modelos con function calling (OpenAI, Claude)

#### Modo Manual (Heurístico)
```json
{
  "settings": {
    "tool_mode": "manual",
    "enabled_tools": ["rag_search"],
    "memory_config": {
      "semantic_enabled": true,
      "search_k": 5
    }
  }
}
```
- El sistema decide basado en heurísticas y configuración
- Compatible con todos los modelos (incluyendo locales)
- Control predictible del comportamiento

### 🧠 Control de Alucinaciones

Sistema de tres niveles para controlar la calidad de respuestas:

| Modo | Temperature | Características | Uso |
|------|-------------|-----------------|-----|
| **Strict** | 0.0 - 0.2 | Requiere fuentes verificables, alta precisión | Documentación técnica, compliance |
| **Balanced** | 0.4 - 0.6 | Balance entre precisión y creatividad | Uso general, asistencia |
| **Creative** | 0.7 - 1.0 | Permite inferencias, brainstorming | Ideación, contenido creativo |

### 🔍 RAG Avanzado

#### Búsqueda Híbrida
- **Búsqueda Semántica**: Embeddings con `mxbai-embed-large`
- **BM25**: Búsqueda por palabras clave
- **Reranking**: Re-ordenamiento de resultados por relevancia
- **Filtros Contextuales**: Por proyecto, fecha, tipo de documento

#### Procesamiento de Documentos
Soporta **15+ formatos** de archivo:
- 📄 **Texto**: PDF, DOCX, TXT, Markdown, HTML
- 📊 **Datos**: Excel, CSV, JSON
- 🖼️ **Presentaciones**: PPTX
- 💻 **Código**: Python, JavaScript, Java, C++, SQL, PL/SQL
- 📝 **Obsidian**: Detección automática de vaults, wikilinks, tags

#### Características de Procesamiento
- **Chunking Inteligente**: Respeta límites de párrafos y secciones
- **Metadata Extraction**: Título, autor, fecha, idioma
- **Obsidian Integration**: 
  - Detección automática de vaults
  - Resolución de wikilinks
  - Extracción de frontmatter
  - Navegación de gráfico de notas

### 🛠️ Sistema de Herramientas

#### Herramientas Integradas

**1. RAG Search**
```python
{
  "tool_name": "rag_search",
  "config": {
    "collections": ["project-docs", "api-reference"],
    "k": 5,
    "score_threshold": 0.5,
    "hybrid_search": true,
    "rerank": true
  }
}
```

**2. HTTP Tool**
```python
{
  "tool_name": "http_request",
  "config": {
    "url": "https://api.example.com/data",
    "method": "GET",
    "headers": {"Authorization": "Bearer ${token}"},
    "timeout": 30
  }
}
```

**3. SQL Tool**
```python
{
  "tool_name": "sql_query",
  "config": {
    "database_type": "postgresql",
    "connection_string": "postgresql://...",
    "query_template": "SELECT * FROM users WHERE name LIKE '%{name}%'",
    "read_only": true
  }
}
```

**4. Custom Tools**
- Sistema de templates para crear herramientas personalizadas
- Validación de configuración con JSON Schema
- Ejecución aislada y segura

### 💬 Streaming y Cancelación

#### Server-Sent Events (SSE)
```python
# Streaming de respuestas en tiempo real
async for chunk in stream_chat(...):
    yield f"data: {json.dumps(chunk)}\n\n"
```

#### Cancelación de Streams
- Cancelación inmediata vía Redis PubSub
- Limpieza de recursos
- Estado consistente post-cancelación
- Timeout configurable

### 📊 Gestión de Proyectos

Organización jerárquica de conversaciones y archivos:
```
Proyecto
├── Conversaciones
│   ├── Conversación 1
│   │   ├── Mensajes
│   │   └── Configuración
│   └── Conversación 2
└── Archivos
    ├── Archivo 1 (embeddings en Qdrant)
    └── Archivo 2
```

### 🔄 Auto-Sincronización de Modelos

El sistema sincroniza automáticamente modelos de proveedores:
- **Detección de características**: Function calling, streaming, vision
- **Cálculo de costos**: Tokens de entrada/salida
- **Metadata**: Context window, tipo de modelo, requisitos de hardware
- **Actualización periódica**: En startup y bajo demanda

---

## 💼 Competencias Técnicas del Proyecto

Este proyecto personal demuestra la aplicación de principios y patrones de desarrollo de nivel empresarial:

### 1. **Arquitectura de Software**
- ✅ Diseño de sistemas escalables con arquitectura de capas
- ✅ Principios SOLID y patrones de diseño (Factory, Strategy, Singleton, Observer)
- ✅ Separación de responsabilidades (API, Service, Provider, Data layers)
- ✅ Diseño orientado a interfaces y abstracción

### 2. **Python Moderno y Async Programming**
- ✅ Python 3.11+ con type hints completos
- ✅ Async/await con asyncio
- ✅ Context managers asincrónicos
- ✅ Generators y async generators
- ✅ Dataclasses y Pydantic models

### 3. **APIs RESTful y Web Development**
- ✅ FastAPI con validación automática (Pydantic)
- ✅ OpenAPI/Swagger documentation
- ✅ Server-Sent Events (SSE) para streaming
- ✅ Manejo de errores estructurado
- ✅ CORS y middleware de seguridad

### 4. **Bases de Datos**
- ✅ SQLAlchemy 2.0 async ORM
- ✅ Migraciones y gestión de esquemas
- ✅ Optimización de queries
- ✅ Bases de datos vectoriales (Qdrant)
- ✅ Redis para cache y PubSub

### 5. **Inteligencia Artificial y LLMs**
- ✅ Integración con múltiples providers (OpenAI, Anthropic, Ollama, etc.)
- ✅ Retrieval-Augmented Generation (RAG)
- ✅ Embeddings y búsqueda semántica
- ✅ Function calling / Tool use
- ✅ Prompt engineering y control de alucinaciones
- ✅ Streaming de respuestas

### 6. **Procesamiento de Documentos**
- ✅ Parser para 15+ formatos de archivo
- ✅ Chunking y segmentación inteligente
- ✅ Extracción de metadata
- ✅ Detección de encoding y lenguaje
- ✅ Procesamiento de Markdown avanzado (Obsidian)

### 7. **DevOps y Deployment**
- ✅ Docker y Docker Compose
- ✅ Configuración por entorno (.env)
- ✅ Logging estructurado
- ✅ Health checks y monitoreo
- ✅ Manejo de procesos (Qdrant, Redis startup)

### 8. **Patrones de Diseño Avanzados**
- ✅ **Factory Pattern**: ProviderManager
- ✅ **Strategy Pattern**: Tool system
- ✅ **Singleton Pattern**: ToolRegistry
- ✅ **Observer Pattern**: Stream cancellation
- ✅ **Template Method**: Document loaders
- ✅ **Dependency Injection**: Database sessions

### 9. **Testing y Quality Assurance**
- ✅ Unit tests con pytest
- ✅ Validación de datos con Pydantic
- ✅ Error handling robusto
- ✅ Logging para debugging

### 10. **Seguridad**
- ✅ Validación de inputs
- ✅ SQL injection prevention (parametrized queries)
- ✅ API key management
- ✅ CORS configuration
- ✅ Read-only mode para SQL queries

---

## 🚀 Instalación y Configuración

### Requisitos Previos

- **Python 3.11+**
- **Docker & Docker Compose**
- **Git**
- **(Opcional) Ollama** para modelos locales

### 1. Clonar el Repositorio

```bash
git clone <repository-url>
cd PythonChatBot
```

### 2. Crear Entorno Virtual

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar Variables de Entorno

```bash
cp .env.example .env
```

Editar `.env` con tus configuraciones:

```env
# General
ENVIRONMENT=development
DEBUG=True
APP_NAME=RAG Chatbot
APP_VERSION=1.0.0

# Database
DATABASE_URL=postgresql://chatbot_ia:chatbot_ia@localhost:5432/chatbot_ia_db

# Qdrant
QDRANT_URL=http://localhost:6333
VECTOR_SIZE=1024

# Redis
REDIS_URL=redis://localhost:6379/0

# Ollama (Local)
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=mxbai-embed-large
LLM_MODEL=mistral

# API Keys (opcionales)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-...
GROQ_API_KEY=gsk_...

# File Storage
UPLOAD_DIR=./data/uploads
MAX_UPLOAD_SIZE=104857600
```

### 5. Levantar Infraestructura

```bash
docker-compose up -d
```

Verificar que los servicios estén corriendo:

```bash
docker-compose ps
```

Deberías ver:
- ✅ `chatbot-postgres` (Puerto 5432)
- ✅ `chatbot-qdrant` (Puertos 6333, 6334)
- ✅ `chatbot-redis` (Puerto 6379)

### 6. Inicializar Base de Datos

La base de datos se inicializa automáticamente en el primer arranque. Si necesitas reinicializarla:

```bash
python -c "from src.database.connection import init_db; init_db()"
```

### 7. (Opcional) Instalar y Configurar Ollama

Para usar modelos locales:

```bash
# Instalar Ollama
# Windows: Descargar desde https://ollama.com
# Linux: curl -fsSL https://ollama.com/install.sh | sh

# Descargar modelos
ollama pull mistral
ollama pull mxbai-embed-large
ollama pull llama3.2
ollama pull qwen2.5
ollama pull deepseek-r1
```

### 8. Ejecutar la Aplicación

#### Modo Desarrollo (con auto-reload)

```bash
python main.py
```

#### Modo Producción

```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --workers 4
```

La API estará disponible en: **http://localhost:8001**

### 9. Verificar Instalación

#### Health Check

```bash
curl http://localhost:8001/health
```

Respuesta esperada:
```json
{
  "status": "healthy",
  "environment": "development",
  "database": "connected"
}
```

#### Documentación Interactiva

Abre en tu navegador:
- **Swagger UI**: http://localhost:8001/api/v1/docs
- **ReDoc**: http://localhost:8001/api/v1/redoc

---

## 📚 Documentación de API

### Endpoints Principales

#### 🗨️ Conversaciones

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `POST` | `/api/v1/conversations` | Crear conversación |
| `GET` | `/api/v1/conversations` | Listar conversaciones |
| `GET` | `/api/v1/conversations/{id}` | Obtener conversación |
| `PATCH` | `/api/v1/conversations/{id}` | Actualizar configuración |
| `DELETE` | `/api/v1/conversations/{id}` | Eliminar conversación |
| `POST` | `/api/v1/conversations/{id}/chat` | Enviar mensaje |
| `POST` | `/api/v1/conversations/{id}/stream` | Chat con streaming (SSE) |
| `POST` | `/api/v1/conversations/{id}/cancel` | Cancelar stream activo |
| `GET` | `/api/v1/conversations/{id}/export/pdf` | Exportar conversación a PDF |

#### 💬 Mensajes

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/api/v1/messages` | Listar mensajes de conversación |
| `GET` | `/api/v1/messages/{id}` | Obtener mensaje específico |
| `DELETE` | `/api/v1/messages/{id}` | Eliminar mensaje |

#### 📁 Archivos

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `POST` | `/api/v1/files/upload` | Subir archivo |
| `GET` | `/api/v1/files` | Listar archivos |
| `GET` | `/api/v1/files/{id}` | Obtener información de archivo |
| `DELETE` | `/api/v1/files/{id}` | Eliminar archivo (incluye vectores) |
| `GET` | `/api/v1/files/{id}/download` | Descargar archivo |

#### 🗂️ Proyectos

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `POST` | `/api/v1/projects` | Crear proyecto |
| `GET` | `/api/v1/projects` | Listar proyectos |
| `GET` | `/api/v1/projects/{id}` | Obtener proyecto |
| `PATCH` | `/api/v1/projects/{id}` | Actualizar proyecto |
| `DELETE` | `/api/v1/projects/{id}` | Eliminar proyecto (cascada) |
| `GET` | `/api/v1/projects/{id}/files` | Listar archivos del proyecto |
| `POST` | `/api/v1/projects/{id}/enable-rag` | Habilitar RAG para proyecto |

#### 🛠️ Herramientas

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/api/v1/tools` | Listar herramientas disponibles |
| `POST` | `/api/v1/tools` | Crear herramienta personalizada |
| `GET` | `/api/v1/tools/{id}` | Obtener herramienta |
| `PUT` | `/api/v1/tools/{id}` | Actualizar herramienta |
| `DELETE` | `/api/v1/tools/{id}` | Eliminar herramienta |
| `GET` | `/api/v1/tools/templates` | Listar templates disponibles |

#### 🗄️ Colecciones (Qdrant)

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/api/v1/collections` | Listar colecciones |
| `GET` | `/api/v1/collections/{name}` | Información de colección |
| `DELETE` | `/api/v1/collections/{name}` | Eliminar colección |
| `POST` | `/api/v1/collections/{name}/search` | Búsqueda semántica |

#### 🤖 Proveedores

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/api/v1/providers` | Listar proveedores y modelos |

#### 📝 Prompts

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `POST` | `/api/v1/prompts` | Crear plantilla de prompt |
| `GET` | `/api/v1/prompts` | Listar plantillas |
| `GET` | `/api/v1/prompts/{id}` | Obtener plantilla |
| `PUT` | `/api/v1/prompts/{id}` | Actualizar plantilla |
| `DELETE` | `/api/v1/prompts/{id}` | Eliminar plantilla |

---

## 📖 Guía de Uso

### Ejemplo 1: Crear Conversación Simple

```bash
curl -X POST http://localhost:8001/api/v1/conversations \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Conversación de Prueba",
    "settings": {
      "provider": "local",
      "model": "mistral",
      "temperature": 0.7,
      "tool_mode": "manual"
    }
  }'
```

### Ejemplo 2: Chat con RAG

```bash
# 1. Subir documento
curl -X POST http://localhost:8001/api/v1/files/upload \
  -F "file=@documentation.pdf" \
  -F "conversation_id=<conversation_id>"

# 2. Habilitar RAG en la conversación
curl -X PATCH http://localhost:8001/api/v1/conversations/<conversation_id> \
  -H "Content-Type: application/json" \
  -d '{
    "settings": {
      "enabled_tools": ["rag_search"],
      "tool_configurations": [{
        "tool_name": "rag_search",
        "config": {
          "k": 5,
          "score_threshold": 0.6
        }
      }]
    }
  }'

# 3. Hacer pregunta
curl -X POST http://localhost:8001/api/v1/conversations/<conversation_id>/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "¿Qué dice el documento sobre autenticación?",
    "stream": false
  }'
```

### Ejemplo 3: Streaming con Cancelación

```python
import requests
import json

# Iniciar stream
response = requests.post(
    "http://localhost:8001/api/v1/conversations/<id>/stream",
    json={"message": "Explica arquitectura de microservicios"},
    stream=True
)

# Leer chunks
for line in response.iter_lines():
    if line:
        data = json.loads(line.decode('utf-8').replace('data: ', ''))
        print(data.get('content', ''), end='', flush=True)

# Cancelar (desde otro cliente)
requests.post(f"http://localhost:8001/api/v1/conversations/<id>/cancel")
```

### Ejemplo 4: Modo Agent con Herramientas

```bash
curl -X POST http://localhost:8001/api/v1/conversations \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Asistente con Tools",
    "settings": {
      "provider": "openai",
      "model": "gpt-4-turbo-preview",
      "tool_mode": "agent",
      "available_tools": ["rag_search", "http_request"],
      "allow_tool_chaining": true
    }
  }'
```

---

## 📁 Estructura del Proyecto

```
PythonChatBot/
├── main.py                           # Entry point FastAPI
├── requirements.txt                  # Dependencias Python
├── docker-compose.yml                # Infraestructura local
├── .env.example                      # Template de configuración
├── Makefile                          # Comandos de desarrollo
│
├── src/                              # Código fuente
│   ├── api/v1/                       # Endpoints REST
│   │   ├── conversations.py          # CRUD conversaciones + chat
│   │   ├── messages.py               # Gestión de mensajes
│   │   ├── files.py                  # Upload/download de archivos
│   │   ├── projects.py               # Gestión de proyectos
│   │   ├── tools.py                  # CRUD herramientas
│   │   ├── collections.py            # Gestión colecciones Qdrant
│   │   └── prompts.py                # Templates de prompts
│   │
│   ├── services/                     # Lógica de negocio
│   │   ├── chat_orchestrator.py      # ⭐ Orquestador principal de chat
│   │   ├── file_processor.py         # Procesamiento de archivos
│   │   ├── embedding_service.py      # Generación de embeddings
│   │   ├── conversation_memory.py    # Memoria semántica
│   │   ├── stream_cancel_manager.py  # Gestión de cancelación
│   │   ├── hybrid_search.py          # Búsqueda híbrida (semantic + BM25)
│   │   ├── reranker.py               # Re-ranking de resultados
│   │   ├── pdf_service.py            # Generación de PDFs
│   │   └── ...
│   │
│   ├── providers/                    # Abstracción de LLM providers
│   │   ├── manager.py                # ⭐ ProviderManager (Factory)
│   │   └── cancellable_stream.py     # Mixin para cancelación
│   │
│   ├── tools/                        # Sistema de herramientas
│   │   ├── base_tool.py              # Clase abstracta base
│   │   ├── rag_tool.py               # Búsqueda RAG
│   │   ├── http_tool.py              # Requests HTTP
│   │   ├── sql_tool.py               # Queries SQL
│   │   ├── custom_tool.py            # Herramientas personalizadas
│   │   └── tool_discovery.py         # Auto-descubrimiento
│   │
│   ├── document_loaders/             # Procesadores de documentos
│   │   ├── pdf_loader.py             # PDFs
│   │   ├── word_loader.py            # DOCX (con tablas)
│   │   ├── excel_loader.py           # Excel/XLSX
│   │   ├── markdown_loader.py        # Markdown + Obsidian
│   │   ├── code_loader.py            # Código fuente
│   │   ├── html_loader.py            # HTML
│   │   ├── json_loader.py            # JSON
│   │   ├── csv_loader.py             # CSV
│   │   ├── pptx_loader.py            # PowerPoint
│   │   ├── sql_plsql_loader.py       # SQL/PL-SQL
│   │   ├── obsidian_detector.py      # Detección de Obsidian vaults
│   │   └── ...
│   │
│   ├── models/                       # SQLAlchemy models
│   │   ├── models.py                 # Modelos principales
│   │   └── llm_models.py             # Modelos LLM (sincronizados)
│   │
│   ├── schemas/                      # Pydantic schemas
│   │   └── schemas.py                # Request/Response schemas
│   │
│   ├── database/                     # Gestión de BD
│   │   ├── connection.py             # Configuración SQLAlchemy async
│   │   └── schema.sql                # Esquema SQL
│   │
│   ├── config/                       # Configuración
│   │   └── settings.py               # Settings con Pydantic
│   │
│   └── utils/                        # Utilidades
│       └── logger.py                 # Logger estructurado
│
├── scripts/                          # Scripts de utilidad
│   ├── init_db.py                    # Inicializar BD
│   ├── sync_tool_templates.py        # Sincronizar templates
│   └── ...
│
├── tests/                            # Tests
│   ├── test_api.py
│   ├── test_tools.py
│   └── ...
│
├── data/                             # Datos locales
│   ├── uploads/                      # Archivos subidos
│   └── vault/                        # Vault de documentos
│
├── qdrant/                           # Ejecutable de Qdrant (Windows)
│   └── qdrant.exe
│
└── redis/                            # Redis portable (Windows)
    └── start.bat
```

---

## 🤖 Proveedores de LLM

### Configuración por Proveedor

#### 1. Ollama (Local)

```bash
# Instalar Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Descargar modelos
ollama pull mistral
ollama pull mxbai-embed-large
```

**Configuración**:
```json
{
  "provider": "local",
  "model": "mistral",
  "temperature": 0.7
}
```

**Ventajas**:
- ✅ Sin costos
- ✅ Privacidad total
- ✅ Sin límites de rate
- ✅ Offline

**Modelos Recomendados**:
- `mistral` - Balance calidad/velocidad
- `llama3.2` - Muy capaz, más lento
- `qwen2.5` - Soporte para thinking
- `deepseek-r1` - Razonamiento avanzado
- `gemma2` - Eficiente, rápido

#### 2. OpenAI

```env
OPENAI_API_KEY=sk-...
```

```json
{
  "provider": "openai",
  "model": "gpt-4-turbo-preview",
  "temperature": 0.7
}
```

**Modelos Disponibles**:
- `gpt-4-turbo-preview` - Más capaz
- `gpt-4` - Versión estable
- `gpt-3.5-turbo` - Rápido y económico

#### 3. Anthropic (Claude)

```env
ANTHROPIC_API_KEY=sk-ant-...
```

```json
{
  "provider": "anthropic",
  "model": "claude-3-5-sonnet-20241022",
  "temperature": 0.7
}
```

**Modelos Disponibles**:
- `claude-3-5-sonnet-20241022` - Último, muy capaz
- `claude-3-opus-20240229` - Máxima capacidad
- `claude-3-sonnet-20240229` - Balance
- `claude-3-haiku-20240307` - Rápido

#### 4. OpenRouter

```env
OPENROUTER_API_KEY=sk-or-...
```

Acceso a 100+ modelos de diferentes proveedores:
- Meta Llama
- Google Gemini
- Mistral AI
- Y muchos más

#### 5. Groq

```env
GROQ_API_KEY=gsk_...
```

Inferencia ultra-rápida (hasta 500 tokens/seg)

---

## ⚡ Performance y Optimización

### Recomendaciones de Performance

#### 1. Modelos Locales
```bash
# Usar GPU para mejor performance
CUDA_VISIBLE_DEVICES=0 ollama serve

# Configurar cantidad de GPUs
{
  "num_gpu": 1,
  "num_thread": 8
}
```

#### 2. Limitar Contexto
```json
{
  "max_context_messages": 20,  // Limitar historial
  "max_tokens": 2000            // Limitar respuesta
}
```

#### 3. Optimizar RAG
```json
{
  "rag_config": {
    "k": 3,                      // Menos chunks
    "score_threshold": 0.7,      // Mayor threshold
    "use_reranking": false       // Desactivar si no es crítico
  }
}
```

#### 4. Cache de Redis
```python
# Habilitar cache de embeddings
CACHE_EMBEDDINGS=true
CACHE_TTL=3600  # 1 hora
```

#### 5. Connection Pooling
```env
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20
```

### Benchmarks Aproximados

| Provider | Modelo | Tokens/seg | Latencia Primer Token |
|----------|--------|------------|-----------------------|
| Groq | llama-3.1-70b | 500 | ~200ms |
| OpenAI | gpt-4-turbo | 100 | ~500ms |
| Ollama (GPU) | mistral | 50-80 | ~100ms |
| Ollama (CPU) | mistral | 10-20 | ~200ms |

---

## 🔐 Seguridad

### Buenas Prácticas Implementadas

#### 1. Validación de Inputs
```python
# Pydantic valida automáticamente todos los inputs
class ConversationCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    settings: ConversationSettings
```

#### 2. SQL Injection Prevention
```python
# Siempre usar queries parametrizadas
stmt = select(Conversation).where(Conversation.id == conversation_id)
```

#### 3. Gestión de API Keys
```env
# NUNCA commitear .env
# Usar variables de entorno en producción
OPENAI_API_KEY=${OPENAI_API_KEY}
```

#### 4. CORS Configurado
```python
# Solo permitir orígenes específicos en producción
CORS_ORIGINS=["https://myapp.com"]
```

#### 5. Read-Only SQL
```json
{
  "sql_tool_config": {
    "read_only": true  // Solo SELECT, no INSERT/UPDATE/DELETE
  }
}
```

### Recomendaciones para Producción

1. **Usar HTTPS**: Siempre en producción
2. **Implementar autenticación**: JWT, OAuth2
3. **Rate limiting**: Limitar requests por IP
4. **Secrets management**: AWS Secrets Manager, HashiCorp Vault
5. **Auditoría**: Logging de todas las acciones sensibles

---

## 🧪 Testing

```bash
# Instalar dependencias de testing
pip install pytest pytest-asyncio httpx

# Ejecutar todos los tests
pytest

# Tests específicos
pytest tests/test_api.py
pytest tests/test_tools.py -v

# Con coverage
pytest --cov=src --cov-report=html
```

### Estructura de Tests

```python
# tests/test_api.py
import pytest
from httpx import AsyncClient
from main import app

@pytest.mark.asyncio
async def test_create_conversation():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/api/v1/conversations", json={
            "title": "Test Conversation",
            "settings": {"provider": "local", "model": "mistral"}
        })
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Test Conversation"
```

---

## 📄 Licencia

Este proyecto está licenciado bajo la **GNU General Public License v3.0**. Ver el archivo [LICENSE](LICENSE) para más detalles.

### ¿Qué significa esto?

- ✅ **Uso libre**: Puedes usar este software para cualquier propósito
- ✅ **Modificación**: Puedes modificar el código según tus necesidades
- ✅ **Distribución**: Puedes distribuir copias originales o modificadas
- ✅ **Uso comercial**: Puedes usar el software comercialmente
- ⚠️ **Copyleft**: Trabajos derivados deben usar GPL v3
- ⚠️ **Sin garantías**: El software se proporciona "tal cual" sin garantías

**Protección Legal**: GPL v3 incluye cláusulas de limitación de responsabilidad (secciones 15 y 16) que protegen al autor de cualquier responsabilidad derivada del uso del software. El software se proporciona "TAL CUAL" (AS IS), sin garantías de ningún tipo.

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una branch para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la branch (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## 📞 Contacto y Soporte

Para preguntas, issues o sugerencias:

1. Revisa el [FAQ](FAQ.md)
2. Abre un issue en GitHub
3. Consulta la documentación interactiva: `/api/v1/docs`

---

## 📚 Recursos Adicionales

- **[QUICKSTART.md](QUICKSTART.md)** - Guía rápida de inicio
- **[FAQ.md](FAQ.md)** - Preguntas frecuentes
- **[API Docs](http://localhost:8001/api/v1/docs)** - Documentación interactiva
- **[Qdrant Documentation](https://qdrant.tech/documentation/)** - Base de datos vectorial
- **[FastAPI Documentation](https://fastapi.tiangolo.com/)** - Framework web

---

## 🎓 Sobre Este Proyecto

Este proyecto personal demuestra:

### Arquitectura y Diseño
- **Backend Development**: Diseño de APIs RESTful escalables con FastAPI
- **Async Programming**: Programación asíncrona de alto rendimiento
- **Clean Architecture**: Separación de responsabilidades y patrones empresariales
- **AI/ML Integration**: Integración de múltiples proveedores de LLM
- **Vector Databases**: Implementación de RAG con Qdrant
- **Document Processing**: Sistema extensible para 15+ formatos
- **Real-time Streaming**: Server-Sent Events con cancelación
- **DevOps**: Docker, containerización, configuración por entorno

### Principios Aplicados
- **Escalabilidad**: Arquitectura preparada para cargas de producción
- **Mantenibilidad**: Código organizado, documentado y testeable
- **Extensibilidad**: Sistema de herramientas y proveedores pluggables
- **Best Practices**: SOLID, patrones de diseño, testing

### Highlights
- ✅ Sistema RAG con arquitectura empresarial y múltiples proveedores de LLM
- ✅ Sistema extensible de herramientas dinámico (RAG, HTTP, SQL, Custom)
- ✅ Procesamiento de 15+ formatos de documentos con chunking inteligente
- ✅ API REST con streaming (SSE) y cancelación en tiempo real
- ✅ Integración de bases de datos vectoriales (Qdrant) para búsqueda semántica
- ✅ Auto-sincronización de modelos de IA con detección de capacidades
- ✅ Control de alucinaciones con tres niveles de precisión
- ✅ Arquitectura async completa con SQLAlchemy 2.0, FastAPI y asyncio

---

**Proyecto personal desarrollado con ❤️ usando Python, FastAPI y tecnologías de IA de última generación.**

*Este es un proyecto de código abierto liberado bajo GPL v3. El código puede ser usado, modificado y distribuido libremente bajo los términos de esta licencia.*
