# Quick Start Guide - RAG Chatbot

Guía rápida para poner en marcha el sistema en 10 minutos.

---

## 📋 Pre-requisitos

✅ Python 3.11+  
✅ Docker & Docker Compose  
✅ Git  
✅ 8GB RAM mínimo (para Ollama)

---

## 🚀 Setup en 5 Pasos

### **Paso 1: Clonar e Instalar**

```bash
# Clonar repositorio
git clone <repo-url>
cd rag-chatbot

# Crear virtual environment
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### **Paso 2: Configurar Variables de Entorno**

```bash
# Copiar template
cp .env.example .env

# Editar .env (mínimo necesario para local):
# DATABASE_URL=postgresql://chatbot:chatbot@localhost:5432/chatbot
# QDRANT_URL=http://localhost:6333
# OLLAMA_BASE_URL=http://localhost:11434
```

### **Paso 3: Levantar Infraestructura**

```bash
# Iniciar servicios (PostgreSQL, Qdrant, Redis)
docker-compose up -d

# Verificar que estén corriendo
docker-compose ps

# Deberías ver 3 servicios: postgres, qdrant, redis
```

### **Paso 4: Instalar Ollama (LLM Local)**

```bash
# En Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh

# Iniciar Ollama
ollama serve  # En otra terminal

# Descargar modelos
ollama pull mistral
ollama pull mxbai-embed-large
```

### **Paso 5: Inicializar Base de Datos**

```bash
# Crear tablas y seed data
python scripts/init_db.py

# Deberías ver:
# ✅ Database initialized
# ✅ 4 prompt templates created
# ✅ 3 collections created
```

---

## ✅ Verificar Instalación

### **Test 1: Health Check**

```bash
# Iniciar API
python main.py

# En otra terminal:
curl http://localhost:8000/health

# Respuesta esperada:
# {"status":"healthy","environment":"local","database":"connected"}
```

### **Test 2: Swagger UI**

Abre en tu navegador:

```
http://localhost:8000/api/v1/docs
```

Deberías ver la documentación interactiva de la API.

### **Test 3: API Tests**

```bash
python scripts/test_api.py

# Deberías ver:
# ✅ Health check passed
# ✅ Providers endpoint working
# ✅ Found 4 prompt templates
# ✅ Found 3 collections
# ✅ Found 3 tools
# ✅ Conversation created
# ✅ Chat response received
```

---

## 💬 Primera Conversación

### **Usando cURL:**

```bash
# 1. Crear conversación
curl -X POST http://localhost:8000/api/v1/conversations \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Mi primera conversación",
    "settings": {
      "provider": "local",
      "model": "mistral",
      "temperature": 0.7,
      "tool_mode": "manual",
      "enabled_tools": []
    }
  }'

# Copia el "id" de la respuesta

# 2. Enviar mensaje
curl -X POST "http://localhost:8000/api/v1/conversations/{ID}/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hola, ¿cómo funciona este sistema?"
  }'
```

### **Usando Python:**

```python
import requests

API_BASE = "http://localhost:8000/api/v1"

# Crear conversación
response = requests.post(f"{API_BASE}/conversations", json={
    "title": "Test con Python",
    "settings": {
        "provider": "local",
        "model": "mistral",
        "temperature": 0.7,
        "tool_mode": "manual",
        "enabled_tools": []
    }
})

conv_id = response.json()["id"]
print(f"Conversación creada: {conv_id}")

# Enviar mensaje
response = requests.post(
    f"{API_BASE}/conversations/{conv_id}/chat",
    json={"message": "Hola!"}
)

print(response.json()["message"]["content"])
```

---

## 🛠️ Habilitar Tools

### **RAG Search:**

```bash
# 1. Sincronizar documentos (usando tu código existente)
python -c "
from sync.syncer import QdrantSyncer
syncer = QdrantSyncer(
    vault_path='./data/vault',
    collection_name='api-documentation'
)
syncer.sync()
"

# 2. Crear conversación con RAG habilitado
curl -X POST http://localhost:8000/api/v1/conversations \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Chat con RAG",
    "settings": {
      "provider": "local",
      "model": "mistral",
      "tool_mode": "manual",
      "enabled_tools": ["rag_search"]
    }
  }'

# 3. Configurar RAG tool para la conversación
curl -X POST http://localhost:8000/api/v1/tools/configurations \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "{CONV_ID}",
    "tool_name": "rag_search",
    "config": {
      "collections": ["api-documentation"],
      "k": 5,
      "score_threshold": 0.5
    }
  }'
```

---

## 🐛 Troubleshooting

### **Error: No puede conectar a PostgreSQL**

```bash
# Verificar que Docker esté corriendo
docker-compose ps

# Reiniciar servicios
docker-compose restart postgres
```

### **Error: Ollama no responde**

```bash
# Verificar que Ollama esté corriendo
curl http://localhost:11434/api/tags

# Si no responde, iniciar Ollama:
ollama serve
```

### **Error: Qdrant connection failed**

```bash
# Verificar Qdrant
curl http://localhost:6333/collections

# Reiniciar Qdrant
docker-compose restart qdrant
```

### **Error: Tool 'X' not found**

```bash
# Verificar tools registrados
curl http://localhost:8000/api/v1/tools/available

# Deberías ver: rag_search, code_analyzer, document_processor
```

---

## 📚 Próximos Pasos

1. **Usar Prompts Predefinidos**: Explora `/api/v1/prompts`
2. **Subir Documentos**: Usa `/api/v1/files/upload`
3. **Modo Agent**: Configura `tool_mode: "agent"` para que la IA decida qué tools usar
4. **Control de Alucinaciones**: Experimenta con `hallucination_mode: "strict"`

---

## 🎓 Comandos Útiles (Makefile)

```bash
make help          # Ver todos los comandos
make setup         # Setup completo automático
make run           # Iniciar API
make test-api      # Probar endpoints
make logs          # Ver logs de Docker
make clean         # Limpiar cache
```

---

## 📖 Documentación Completa

- **API Docs**: http://localhost:8000/api/v1/docs
- **README.md**: Documentación completa
- **STRUCTURE.md**: Estructura del proyecto

---

¡Listo! Ahora tienes un sistema RAG completo funcionando. 🎉
