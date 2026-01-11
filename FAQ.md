# FAQ - Preguntas Frecuentes

---

## 🤔 General

### **¿Qué es este proyecto?**

Un sistema de chatbot con RAG (Retrieval Augmented Generation) que permite:

- Chat con múltiples proveedores de LLM (local/remoto)
- Búsqueda semántica en documentos (RAG)
- Análisis de código
- Control de alucinaciones
- Herramientas extensibles

### **¿Qué significa "Modo Agent" vs "Modo Manual"?**

- **Modo Agent**: La IA decide automáticamente qué herramientas usar (requiere OpenAI o Claude)
- **Modo Manual**: Tú configuras qué herramientas se ejecutan (funciona con cualquier LLM, incluso local)

---

## 🛠️ Instalación

### **¿Qué necesito instalar?**

**Mínimo (local):**

- Python 3.11+
- Docker (para PostgreSQL, Qdrant, Redis)
- Ollama (para LLM local)

**Opcional (para providers remotos):**

- API keys de OpenAI, Claude, Gemini, etc.

### **¿Puedo usar sin Docker?**

Sí, pero tendrías que instalar PostgreSQL, Qdrant y Redis manualmente. Docker simplifica todo.

### **¿Funciona en Windows?**

Sí, pero necesitas:

- WSL2 (Windows Subsystem for Linux)
- Docker Desktop
- Python instalado en Windows o WSL

---

## 🤖 LLM Providers

### **¿Qué modelos puedo usar?**

**Local (gratis):**

- Mistral 7B
- Llama 2/3
- Phi-3
- Cualquier modelo de Ollama

**Remotos (requieren API key):**

- OpenAI: GPT-4, GPT-3.5
- Anthropic: Claude 3.5 Sonnet, Opus
- Google: Gemini Pro
- OpenRouter: Acceso a múltiples modelos

### **¿Cuál es el mejor modelo para cada tarea?**

| Tarea            | Modelo Recomendado | Por qué              |
|------------------|--------------------|----------------------|
| Análisis de docs | Claude 3.5 Sonnet  | Mejor contexto largo |
| Code review      | GPT-4              | Excelente con código |
| RAG búsquedas    | Mistral (local)    | Rápido y económico   |
| Creatividad      | GPT-4, Claude Opus | Más imaginativos     |

### **¿Cómo cambio de modelo?**

En la creación de conversación:

```json
{
  "settings": {
    "provider": "openai",  // o "local", "anthropic"
    "model": "gpt-4"       // o "mistral", "claude-3-5-sonnet"
  }
}
```

---

## 📚 RAG (Búsqueda en Documentos)

### **¿Cómo funciona RAG?**

1. Subes documentos → Se procesan y guardan en Qdrant (base de datos vectorial)
2. Haces una pregunta → El sistema busca partes relevantes del documento
3. La IA responde basándose en esos fragmentos

### **¿Qué tipos de documentos soporta?**

- ✅ Markdown (.md)
- ✅ Word (.docx)
- ✅ PDF (.pdf)
- ✅ Código fuente (.js, .ts, .py, .java, .sql)
- 🔜 PowerPoint (.pptx) - próximamente

### **¿Cómo subo documentos?**

**Opción 1: API**

```bash
curl -X POST http://localhost:8000/api/v1/files/upload \
  -F "file=@documento.pdf"
```

**Opción 2: Sync completo (tu código existente)**

```python
from sync.syncer import QdrantSyncer
syncer = QdrantSyncer(vault_path='./data/vault')
syncer.sync()
```

### **¿Qué es una "colección" en Qdrant?**

Una colección es como una "carpeta" de documentos relacionados. Por ejemplo:

- `api-documentation` → Docs de tu API
- `plsql-procedures` → Procedimientos almacenados
- `admin-manuals` → Manuales administrativos

Puedes buscar en una o varias colecciones a la vez.

---

## 🎭 Prompts

### **¿Qué es un prompt template?**

Una "receta" predefinida que le dice a la IA cómo comportarse. Por ejemplo:

- "Asistente de Código" → Experto en programación
- "Analista de Documentos" → Extrae info precisa de docs

### **¿Puedo crear mis propios prompts?**

¡Sí! Desde la API:

```bash
POST /api/v1/prompts
```

O duplicar uno existente y modificarlo.

### **¿Qué son las "variables" en un prompt?**

Campos dinámicos que el usuario completa. Por ejemplo:

```
Prompt: "Eres experto en {language} con {años} de experiencia"
Variables:
  - language: "Python"
  - años: 10
Resultado: "Eres experto en Python con 10 años de experiencia"
```

---

## 🛡️ Control de Alucinaciones

### **¿Qué significa "modo estricto"?**

La IA **solo** responde con información verificable de los documentos. Si no tiene la info, lo dice claramente.

### **¿Cuándo usar cada modo?**

| Modo         | Cuándo Usar                           | Temperature |
|--------------|---------------------------------------|-------------|
| **Strict**   | Contratos, docs legales, info crítica | 0.0 - 0.2   |
| **Balanced** | Uso general, análisis técnico         | 0.3 - 0.5   |
| **Creative** | Brainstorming, ideas, exploración     | 0.7 - 1.0   |

### **¿Cómo funciona la validación?**

En modo estricto, el sistema:

1. Verifica que cada respuesta tenga fuentes
2. Detecta frases especulativas ("probablemente", "podría ser")
3. Asigna un "confidence score"
4. Puede rechazar responder si no hay fuentes

---

## 🔧 Tools (Herramientas)

### **¿Qué tools están disponibles?**

1. **RAG Search** - Busca en documentos

### **¿Cómo agrego un nuevo tool?**

Crea una clase que herede de `BaseTool`:

```python
from tools.base_tool import BaseTool, ToolCategory, ToolResult


class MyCustomTool(BaseTool):
    @property
    def name(self) -> str:
        return "my_tool"

    async def execute(self, **kwargs) -> ToolResult:
        # Tu lógica aquí
        return ToolResult(success=True, data={"result": "..."})


# Registrar
tool_registry.register(MyCustomTool())
```

### **¿Cuándo usar modo Agent vs Manual?**

| Modo       | Ventajas                          | Desventajas                        |
|------------|-----------------------------------|------------------------------------|
| **Agent**  | Automático, inteligente           | Requiere OpenAI/Claude, más caro   |
| **Manual** | Control total, funciona con local | Menos flexible, hay que configurar |

---

## 💾 Base de Datos

### **¿Por qué PostgreSQL?**

- Soporte de JSONB (flexible para metadata)
- Transacciones ACID
- Escalable
- Gratuito

### **¿Qué guarda en la base de datos?**

- Conversaciones y mensajes
- Prompt templates
- Configuración de tools
- Metadata de archivos
- **NO** guarda los vectores (esos van en Qdrant)

### **¿Puedo usar otra base de datos?**

Sí, pero tendrías que adaptar los modelos SQLAlchemy. PostgreSQL es la opción más probada.

---

## 🚀 Deployment

### **¿Cómo lo despliego en producción?**

El mismo código funciona en local y AWS:

1. **Cambiar .env:**
   ```bash
   DATABASE_URL=postgresql://user@rds-endpoint:5432/chatbot
   QDRANT_URL=https://your-qdrant-cloud.io
   ```

2. **Deploy options:**
    - AWS ECS/Fargate
    - EC2 con Docker
    - Kubernetes (si necesitas escalado masivo)

### **¿Cuánto cuesta?**

**Local:** Gratis (solo hardware)

**AWS (ejemplo):**

- RDS (PostgreSQL): ~$50/mes
- Qdrant Cloud: ~$50/mes (o self-hosted en EC2)
- ElastiCache (Redis): ~$15/mes
- ECS Fargate: ~$30/mes
- **Total:** ~$145/mes + costos de LLM APIs

**Costos de LLM:**

- Local (Ollama): $0
- GPT-4: ~$0.03/1K tokens
- Claude 3.5 Sonnet: ~$0.003/1K tokens

---

## 🐛 Problemas Comunes

### **"Tool 'rag_search' not found"**

El tool no está registrado. Verifica en `main.py`:

```python
tool_registry.register(RAGTool())
```

### **"Conversation not found"**

Estás usando un UUID incorrecto. Verifica con:

```bash
GET /api/v1/conversations
```

### **Ollama no responde**

```bash
# Verificar que esté corriendo
ollama list

# Si no, iniciar
ollama serve
```

### **PostgreSQL connection failed**

```bash
# Verificar Docker
docker-compose ps

# Reiniciar
docker-compose restart postgres
```

---

## 📊 Performance

### **¿Qué tan rápido es?**

Depende del provider:

- **Local (Ollama)**: 5-20 tokens/seg (depende de tu GPU)
- **OpenAI API**: 30-50 tokens/seg
- **Claude API**: 40-60 tokens/seg

### **¿Puedo usar GPU?**

Sí, Ollama detecta automáticamente la GPU. Para NVIDIA:

```bash
# Verificar
nvidia-smi

# Ollama usará GPU automáticamente
```

### **¿Cuántos usuarios soporta?**

Depende de tu infraestructura:

- **Local**: 1-5 usuarios concurrentes
- **AWS (small)**: 10-50 usuarios
- **AWS (scaled)**: 100+ usuarios (con load balancer)

---

## 🔐 Seguridad

### **¿Cómo protejo las API keys?**

- **Nunca** comitees `.env` a git
- Usa variables de entorno en producción
- Rota keys periódicamente

### **¿Hay autenticación?**

No está implementada aún. Para producción, agrega:

- JWT tokens
- OAuth2
- API keys por usuario

### **¿Los datos están encriptados?**

- En tránsito: Sí (HTTPS en producción)
- En reposo: Depende de tu DB (RDS soporta encryption)

---

¿Más preguntas? Abre un issue en GitHub o contacta al equipo.
