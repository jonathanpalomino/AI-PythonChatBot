#!/usr/bin/env python3
# =============================================================================
# scripts/test_obsidian_vault.py
# Script para analizar vaults de Obsidian y indexar en Qdrant
# =============================================================================
"""
Script CLI para analizar vaults de Obsidian con soporte completo para:
- Vault completo
- Carpeta específica (recursivo)
- Archivo individual

Con opciones configurables:
- Modelo de IA para generación de contexto
- Modelo de embeddings
- Colección de Qdrant (se crea si no existe)
"""
import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional, List

from src.services.embedding.embedding_service import EmbeddingService

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.settings import get_qdrant_config
from src.database.connection import AsyncSessionLocal
from src.document_loaders.obsidian_note_adapter import ObsidianNoteAdapter
from src.document_loaders.obsidian_detector import ObsidianDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ObsidianVaultIndexer:
    """Indexador de vaults de Obsidian a Qdrant con configuración flexible"""

    def __init__(
        self,
        ai_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        collection_name: str = "obsidian_test"
    ):
        """
        Args:
            ai_model: Modelo de IA para generación de contexto (opcional)
            embedding_model: Modelo de embeddings (opcional, usa default del sistema)
            collection_name: Nombre de la colección Qdrant
        """
        self.ai_model = ai_model
        self.embedding_model = embedding_model
        self.collection_name = collection_name

        # Inicializar servicios
        qdrant_config = get_qdrant_config()
        self.qdrant = AsyncQdrantClient(**qdrant_config)
        self.adapter = ObsidianNoteAdapter()
        self.detector = ObsidianDetector()

        logger.info(
            f"Initialized indexer",
            extra={
                "ai_model": ai_model or "default",
                "embedding_model": embedding_model or "default",
                "collection": collection_name
            }
        )

    async def ensure_collection_exists(self, db: AsyncSession) -> None:
        """
        Asegura que la colección existe, créala si no.

        Args:
            db: Sesión de base de datos para obtener dimensión de embeddings
        """
        try:
            # Verificar si la colección existe
            collections = await self.qdrant.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name in collection_names:
                logger.info(f"Collection '{self.collection_name}' already exists")
                print(f"✅ Collection '{self.collection_name}' already exists")
                return

            # Crear colección nueva
            logger.info(f"Creating collection '{self.collection_name}'")
            print(f"🔄 Creating collection '{self.collection_name}'...")

            # Obtener dimensión de embeddings generando un embedding de prueba
            embedding_service = EmbeddingService()
            test_embedding = await embedding_service.generate_embedding(
                "test",
                model=self.embedding_model
            )
            vector_size = len(test_embedding)

            # Crear colección
            await self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )

            logger.info(
                f"Collection created",
                extra={
                    "collection": self.collection_name,
                    "vector_size": vector_size
                }
            )
            print(f"✅ Collection '{self.collection_name}' created (dimension: {vector_size})")

        except Exception as e:
            error_msg = str(e)

            # Detectar si Qdrant no está corriendo
            if "connection" in error_msg.lower() or "connect" in error_msg.lower():
                logger.error("Qdrant is not running or not accessible")
                print("\n" + "=" * 80)
                print("❌ ERROR: Cannot connect to Qdrant")
                print("=" * 80)
                print("\n🔴 Qdrant is not running or not accessible on the configured host/port.")
                print("\n📋 To fix this, start Qdrant:")
                print("\n  Option 1 (Local executable):")
                print("    .\\qdrant\\qdrant.exe")
                print("\n  Option 2 (Docker):")
                print("    docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant")
                print("\n  Option 3 (Check if already running):")
                print("    Visit http://localhost:6333/dashboard")
                print("\n" + "=" * 80)
                raise SystemExit(1)

            logger.error(f"Error ensuring collection exists: {e}", exc_info=True)
            raise

    async def analyze_and_index_vault(
        self,
        vault_path: Path,
        db: AsyncSession,
        note_names: Optional[List[str]] = None
    ) -> dict:
        """
        Analiza vault completo o subset y indexa en Qdrant.

        Args:
            vault_path: Ruta al vault
            db: Sesión de base de datos
            note_names: Lista opcional de nombres de notas específicas

        Returns:
            Dict con estadísticas del proceso
        """
        # Detectar contexto
        context = self.detector.detect(vault_path)

        if not context.is_obsidian:
            logger.warning(f"Path {vault_path} is not an Obsidian vault")
            print(f"⚠️  Path is not an Obsidian vault: {vault_path}")
            return {"success": False, "error": "Not an Obsidian vault"}

        print(f"\n📁 Vault detected: {context.vault_root}")
        print(f"📊 Files found: {len(context.files)}")

        # Cargar notas
        print(f"\n🔄 Loading notes...")
        notes = await self.adapter.load_vault(
            vault_path=vault_path,
            note_names=note_names,
            include_graph=True
        )

        if not notes:
            print("⚠️  No notes loaded")
            return {"success": False, "error": "No notes loaded"}

        print(f"✅ Loaded {len(notes)} notes")

        # Convertir a chunks
        print(f"\n🔄 Converting to RAG chunks...")
        all_chunks = []
        for note in notes:
            chunks = note.to_qdrant_chunks()
            all_chunks.extend(chunks)

        print(f"✅ Generated {len(all_chunks)} chunks")

        # Generar embeddings e indexar
        print(f"\n🔄 Generating embeddings and indexing...")
        embedding_service = EmbeddingService()
        indexed_count = 0

        points = []
        for idx, chunk in enumerate(all_chunks):
            try:
                # Generar embedding
                embedding = await embedding_service.generate_embedding(
                    chunk["content"],
                    model=self.embedding_model
                )

                # Crear point para Qdrant
                point = PointStruct(
                    id=indexed_count,
                    vector=embedding,
                    payload={
                        **chunk["metadata"],
                        "content": chunk["content"]
                    }
                )
                points.append(point)
                indexed_count += 1

                # Mostrar progreso cada 10 chunks
                if indexed_count % 10 == 0:
                    print(f"  Progress: {indexed_count}/{len(all_chunks)} chunks")

            except Exception as e:
                logger.error(f"Error processing chunk {idx}: {e}")
                print(f"⚠️  Error processing chunk {idx}: {e}")

        # Indexar en Qdrant en batch
        if points:
            try:
                await self.qdrant.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                print(f"\n✅ Indexed {len(points)} chunks in Qdrant")
            except Exception as e:
                logger.error(f"Error indexing in Qdrant: {e}", exc_info=True)
                print(f"\n❌ Error indexing: {e}")
                return {"success": False, "error": str(e)}

        # Estadísticas
        stats = {
            "success": True,
            "notes_processed": len(notes),
            "chunks_generated": len(all_chunks),
            "chunks_indexed": indexed_count,
            "collection": self.collection_name,
            "vault_path": str(vault_path)
        }

        # Estadísticas de grafo
        hubs = [n for n in notes if n.graph_node and n.graph_node.is_hub]
        indexes = [n for n in notes if n.graph_node and n.graph_node.is_index]

        stats["graph_stats"] = {
            "total_edges": sum(len(n.graph_edges) for n in notes),
            "hubs": [h.note_id for h in hubs],
            "indexes": [i.note_id for i in indexes]
        }

        return stats

    async def analyze_and_index_folder(
        self,
        folder_path: Path,
        vault_root: Path,
        db: AsyncSession
    ) -> dict:
        """
        Analiza una carpeta específica dentro de un vault (recursivo).

        Args:
            folder_path: Ruta a la carpeta
            vault_root: Raíz del vault
            db: Sesión de base de datos

        Returns:
            Dict con estadísticas
        """
        if not folder_path.is_dir():
            print(f"❌ Path is not a directory: {folder_path}")
            return {"success": False, "error": "Not a directory"}

        # Buscar archivos .md recursivamente en la carpeta
        md_files = list(folder_path.rglob("*.md"))

        if not md_files:
            print(f"⚠️  No .md files found in {folder_path}")
            return {"success": False, "error": "No markdown files found"}

        print(f"\n📁 Folder: {folder_path.relative_to(vault_root)}")
        print(f"📊 Found {len(md_files)} markdown files")

        # Extraer nombres de notas (sin extensión)
        note_names = [f.stem for f in md_files]

        # Usar analyze_and_index_vault con note_names específicos
        return await self.analyze_and_index_vault(
            vault_path=vault_root,
            db=db,
            note_names=note_names
        )

    async def analyze_and_index_file(
        self,
        file_path: Path,
        db: AsyncSession
    ) -> dict:
        """
        Analiza un archivo individual.

        Args:
            file_path: Ruta al archivo .md
            db: Sesión de base de datos

        Returns:
            Dict con estadísticas
        """
        if not file_path.is_file() or file_path.suffix != ".md":
            print(f"❌ Path is not a .md file: {file_path}")
            return {"success": False, "error": "Not a markdown file"}

        print(f"\n📄 File: {file_path.name}")

        # Detectar vault padre
        context = self.detector.detect(file_path)

        # Cargar nota individual
        print(f"🔄 Loading note...")
        note = await self.adapter.load_note(
            note_path=file_path,
            include_graph=context.is_obsidian,
            resolve_transclusions=True,
            vault_context=context if context.is_obsidian else None
        )

        print(f"✅ Loaded note: {note.note_id}")
        print(f"  - Wikilinks: {len(note.wikilinks)}")
        print(f"  - Transclusions: {len(note.transclusions)}")
        print(f"  - Callouts: {len(note.callouts)}")

        # Convertir a chunks
        chunks = note.to_qdrant_chunks()
        print(f"  - Chunks: {len(chunks)}")

        # Generar embeddings e indexar
        print(f"\n🔄 Generating embeddings...")
        embedding_service = EmbeddingService()

        points = []
        for idx, chunk in enumerate(chunks):
            embedding = await embedding_service.generate_embedding(
                chunk["content"],
                model=self.embedding_model
            )

            point = PointStruct(
                id=idx,
                vector=embedding,
                payload={
                    **chunk["metadata"],
                    "content": chunk["content"]
                }
            )
            points.append(point)

        # Indexar
        if points:
            await self.qdrant.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"✅ Indexed {len(points)} chunks")

        return {
            "success": True,
            "note_id": note.note_id,
            "chunks_indexed": len(points),
            "collection": self.collection_name,
            "wikilinks": len(note.wikilinks),
            "is_obsidian": context.is_obsidian
        }

    async def close(self):
        """Cierra conexiones"""
        await self.qdrant.close()


async def main():
    """Entrada principal del script"""
    parser = argparse.ArgumentParser(
        description="Analyze Obsidian vaults and index in Qdrant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze entire vault
  python test_obsidian_vault.py --vault D:/MyVault --collection my_vault

  # Analyze specific folder (recursive)
  python test_obsidian_vault.py --folder D:/MyVault/Projects --vault D:/MyVault

  # Analyze single file
  python test_obsidian_vault.py --file D:/MyVault/Note.md

  # With custom models
  python test_obsidian_vault.py --vault D:/MyVault --ai-model llama3.1:8b --embedding-model nomic-embed-text
        """
    )

    # Opciones de entrada
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--vault",
        type=Path,
        help="Path to Obsidian vault (analyzes entire vault)"
    )
    input_group.add_argument(
        "--folder",
        type=Path,
        help="Path to folder within vault (recursive analysis)"
    )
    input_group.add_argument(
        "--file",
        type=Path,
        help="Path to single markdown file"
    )

    # Opciones de configuración
    parser.add_argument(
        "--vault-root",
        type=Path,
        help="Vault root (required when using --folder)"
    )
    parser.add_argument(
        "--ai-model",
        type=str,
        help="AI model for context generation (e.g., 'llama3.1:8b')"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        help="Embedding model (e.g., 'nomic-embed-text')"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="obsidian_test",
        help="Qdrant collection name (default: obsidian_test)"
    )
    parser.add_argument(
        "--notes",
        type=str,
        nargs="+",
        help="Specific note names to process (without .md extension)"
    )

    args = parser.parse_args()

    print(args)

    # Validaciones
    if args.folder and not args.vault_root:
        parser.error("--folder requires --vault-root")

    # Banner
    print("=" * 80)
    print("🔮 OBSIDIAN VAULT INDEXER")
    print("=" * 80)

    # Crear indexer
    indexer = ObsidianVaultIndexer(
        ai_model=args.ai_model,
        embedding_model=args.embedding_model,
        collection_name=args.collection
    )

    try:
        # Obtener sesión de base de datos
        async with AsyncSessionLocal() as db:
            # Asegurar que la colección existe
            await indexer.ensure_collection_exists(db)

            # Ejecutar análisis según modo
            if args.vault:
                stats = await indexer.analyze_and_index_vault(
                    vault_path=args.vault,
                    db=db,
                    note_names=args.notes
                )
            elif args.folder:
                stats = await indexer.analyze_and_index_folder(
                    folder_path=args.folder,
                    vault_root=args.vault_root,
                    db=db
                )
            else:  # args.file
                stats = await indexer.analyze_and_index_file(
                    file_path=args.file,
                    db=db
                )

            # Mostrar resultados
            print("\n" + "=" * 80)
            print("📊 RESULTS")
            print("=" * 80)

            if stats.get("success"):
                print("✅ Status: SUCCESS")
                for key, value in stats.items():
                    if key != "success" and not isinstance(value, dict):
                        print(f"  {key}: {value}")

                if "graph_stats" in stats:
                    print("\n🔗 Graph Statistics:")
                    graph = stats["graph_stats"]
                    print(f"  Total edges: {graph['total_edges']}")
                    if graph["hubs"]:
                        print(f"  Hub notes: {', '.join(graph['hubs'][:5])}")
                    if graph["indexes"]:
                        print(f"  Index notes: {', '.join(graph['indexes'][:5])}")
            else:
                print(f"❌ Status: FAILED")
                print(f"  Error: {stats.get('error')}")

            print("=" * 80)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {e}")
        return 1
    finally:
        await indexer.close()

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
