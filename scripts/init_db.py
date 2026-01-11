#!/usr/bin/env python
# =============================================================================
# scripts/init_db.py
# Initialize database with seed data
# =============================================================================
"""
Script para inicializar la base de datos con datos seed
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.connection import init_db, get_db_session
from src.models.models import PromptTemplate, QdrantCollection, VisibilityType, HallucinationMode
from src.config.settings import settings


def seed_prompt_templates():
    """Create default prompt templates"""
    print("📝 Creating default prompt templates...")

    templates = [
        {
            "name": "Asistente de Código",
            "description": "Experto en análisis y revisión de código",
            "category": "code",
            "visibility": VisibilityType.PUBLIC,
            "system_prompt": """Eres un experto desarrollador de software con amplia experiencia en {language}.
Tu tarea es analizar código, encontrar bugs, sugerir mejoras y explicar conceptos de forma clara.
Siempre proporciona ejemplos concretos y justifica tus sugerencias.""",
            "variables": [
                {
                    "name": "language",
                    "type": "select",
                    "options": ["JavaScript", "TypeScript", "Python", "Java", "PL/SQL"],
                    "default": "JavaScript",
                    "required": True
                }
            ],
            "settings": {
                "recommended_provider": "claude",
                "recommended_model": "claude-sonnet-3.5",
                "temperature": 0.3,
                "default_tools": ["code_analyzer"],
                "hallucination_mode": HallucinationMode.BALANCED.value
            }
        },
        {
            "name": "Analista de Documentos",
            "description": "Especialista en analizar documentos administrativos",
            "category": "docs",
            "visibility": VisibilityType.PUBLIC,
            "system_prompt": """Eres un analista experto en documentos administrativos y legales.
Tu tarea es extraer información precisa, resumir contenido y responder preguntas específicas.
SIEMPRE cita la sección o página exacta. Si no encuentras la información, indícalo claramente.""",
            "variables": [],
            "settings": {
                "recommended_provider": "openai",
                "recommended_model": "gpt-4",
                "temperature": 0.1,
                "default_tools": ["rag_search", "document_processor"],
                "hallucination_mode": HallucinationMode.STRICT.value
            }
        },
        {
            "name": "Consultor RAG",
            "description": "Experto en buscar información en documentación",
            "category": "docs",
            "visibility": VisibilityType.PUBLIC,
            "system_prompt": """Eres un consultor técnico experto.
Busca información en la documentación disponible y proporciona respuestas precisas ÚNICAMENTE basadas en esa información.
Siempre cita las fuentes. Si no está en la documentación, indícalo claramente.""",
            "variables": [],
            "settings": {
                "recommended_provider": "local",
                "recommended_model": "mistral",
                "temperature": 0.2,
                "default_tools": ["rag_search"],
                "hallucination_mode": HallucinationMode.STRICT.value
            }
        },
        {
            "name": "Asistente General",
            "description": "Asistente conversacional versátil",
            "category": "general",
            "visibility": VisibilityType.PUBLIC,
            "system_prompt": """Eres un asistente útil y versátil.
Puedes ayudar con una amplia variedad de tareas: responder preguntas, generar ideas, explicar conceptos.
Sé claro, conciso y útil. Si no estás seguro, admítelo.""",
            "variables": [],
            "settings": {
                "recommended_provider": "openai",
                "recommended_model": "gpt-4",
                "temperature": 0.7,
                "default_tools": [],
                "hallucination_mode": HallucinationMode.BALANCED.value
            }
        }
    ]

    with get_db_session() as db:
        for template_data in templates:
            # Check if exists
            existing = db.query(PromptTemplate).filter(
                PromptTemplate.name == template_data["name"]
            ).first()

            if existing:
                print(f"  ⏭️  Template '{template_data['name']}' already exists")
                continue

            template = PromptTemplate(**template_data)
            db.add(template)
            print(f"  ✅ Created template: {template_data['name']}")

        db.commit()


def seed_collections():
    """Create default Qdrant collections registry"""
    print("\n📦 Creating default Qdrant collections...")

    collections = [
        {
            "name": "api-documentation",
            "display_name": "Documentación API REST",
            "description": "Documentación completa de los endpoints de la API",
            "category": "docs",
            "visibility": VisibilityType.PUBLIC,
            "metadata": {
                "source_path": "./docs/api",
                "embedding_model": "mxbai-embed-large"
            }
        },
        {
            "name": "plsql-manual",
            "display_name": "Manual de PL/SQL",
            "description": "Guías y referencias de procedimientos almacenados",
            "category": "code",
            "visibility": VisibilityType.PUBLIC,
            "metadata": {
                "source_path": "./docs/plsql",
                "embedding_model": "mxbai-embed-large"
            }
        },
        {
            "name": "admin-procedures",
            "display_name": "Procedimientos Administrativos",
            "description": "Manuales y procedimientos del área administrativa",
            "category": "admin",
            "visibility": VisibilityType.PUBLIC,
            "metadata": {
                "source_path": "./docs/admin",
                "embedding_model": "mxbai-embed-large"
            }
        }
    ]

    with get_db_session() as db:
        for collection_data in collections:
            # Check if exists
            existing = db.query(QdrantCollection).filter(
                QdrantCollection.name == collection_data["name"]
            ).first()

            if existing:
                print(f"  ⏭️  Collection '{collection_data['name']}' already exists")
                continue

            collection = QdrantCollection(**collection_data)
            db.add(collection)
            print(f"  ✅ Created collection: {collection_data['display_name']}")

        db.commit()


def main():
    """Main initialization"""
    print("=" * 60)
    print("🚀 RAG Chatbot - Database Initialization")
    print("=" * 60)
    print()

    # Validate configuration
    print("🔍 Validating configuration...")
    settings.validate()
    print()

    # Initialize database schema
    print("🗄️  Creating database schema...")
    init_db()
    print()

    # Seed data
    seed_prompt_templates()
    seed_collections()

    print()
    print("=" * 60)
    print("✅ Database initialized successfully!")
    print("=" * 60)
    print()
    print("📊 Summary:")
    print("   - 4 Prompt Templates created")
    print("   - 3 Qdrant Collections registered")
    print("   - 3 Tools available (RAG)")
    print()
    print("Next steps:")
    print("1. Start the API: python main.py")
    print("2. Access docs: http://localhost:8000/api/v1/docs")
    print("3. Test API: python scripts/test_api.py")
    print("4. Sync documents to Qdrant (optional)")
    print()


if __name__ == "__main__":
    main()
