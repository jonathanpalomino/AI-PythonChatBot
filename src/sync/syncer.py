# src/sync/syncer.py
"""
Sistema de sincronizaciÃ³n incremental con Qdrant
"""
import hashlib
from pathlib import Path
from typing import List, Optional

from ollama import Client as OllamaClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm

from src.config.settings import settings
from src.services.embedding_service import EmbeddingService
from .tracker import SyncTracker
from ..document_loaders import DocumentLoaderFactory


class QdrantSyncer:
    """Sincroniza documentos con Qdrant de forma incremental"""

    def __init__(
        self,
        vault_path: Optional[Path] = None,
        collection_name: Optional[str] = None,
        recreate: bool = False,
        keep_alive: str = "30m"  # Mantener modelo durante sincronizaciÃ³n
    ):
        self.vault_path = vault_path or settings.get_vault_path()
        self.collection_name = collection_name or settings.COLLECTION_NAME
        self.recreate = recreate
        self.keep_alive = keep_alive

        # Clientes
        self.qdrant = QdrantClient(url=settings.QDRANT_URL)
        self.ollama = OllamaClient(host=settings.OLLAMA_BASE_URL)

        # Tracker para sincronizaciÃ³n incremental
        self.tracker = SyncTracker(settings.TRACKING_FILE)

        # EstadÃ­sticas
        self.stats = {
            'processed': 0,
            'added': 0,
            'updated': 0,
            'deleted': 0,
            'skipped': 0,
            'errors': 0
        }

        # Pre-cargar modelo de embeddings
        self._preload_embedding_model()

    def _preload_embedding_model(self):
        """Pre-carga el modelo de embeddings al iniciar"""
        print("â³ Pre-cargando modelo de embeddings...")
        try:
            self.ollama.embeddings(
                model=settings.EMBEDDING_MODEL,
                prompt="warmup",
                keep_alive=self.keep_alive
            )
            print(f"âœ… Modelo {settings.EMBEDDING_MODEL} cargado en memoria\n")
        except Exception as e:
            print(f"âš ï¸  Error pre-cargando modelo: {e}\n")

    def initialize(self):
        """Inicializa el sistema de sincronizaciÃ³n"""
        print("ðŸš€ Inicializando sincronizaciÃ³n...\n")

        # Validar configuraciÃ³n
        settings.validate()

        # Cargar tracking (a menos que sea recreate)
        if not self.recreate:
            self.tracker.load()
        else:
            print("âš ï¸  Modo RECREATE: se eliminarÃ¡ la colecciÃ³n existente\n")

        # Configurar colecciÃ³n
        self._setup_collection()

    def _setup_collection(self):
        """Crea o recrea la colecciÃ³n en Qdrant"""
        try:
            # Verificar si existe
            self.qdrant.get_collection(self.collection_name)

            if self.recreate:
                print(f"ðŸ—‘ï¸  Eliminando colecciÃ³n existente '{self.collection_name}'...")
                self.qdrant.delete_collection(self.collection_name)
                self._create_collection()
                self.tracker.metadata = {}  # Limpiar tracking
            else:
                print(f"âœ… ColecciÃ³n '{self.collection_name}' existe")

        except Exception:
            # No existe, crear
            self._create_collection()

    def _create_collection(self):
        """Crea la colecciÃ³n en Qdrant"""
        print(f"ðŸ“¦ Creando colecciÃ³n '{self.collection_name}'...")
        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=settings.VECTOR_SIZE,  # mxbai-embed-large = 1024
                distance=Distance.COSINE
            )
        )
        print("âœ… ColecciÃ³n creada")

    def generate_embedding(self, text: str) -> List[float]:
        """Genera embedding usando Ollama con manejo de textos largos"""
        embedding_service = EmbeddingService()
        return embedding_service.generate_embedding_sync(text)

    def find_documents(self) -> List[Path]:
        """Encuentra todos los documentos soportados en el vault"""
        documents = []
        supported_exts = DocumentLoaderFactory.get_supported_extensions()

        for ext in supported_exts:
            documents.extend(self.vault_path.rglob(f'*{ext}'))

        # Filtrar directorios ocultos (.obsidian, etc)
        documents = [
            doc for doc in documents
            if not any(part.startswith('.') for part in doc.parts)
        ]

        return sorted(documents)

    def _generate_point_id(self, file_path: str, section_index: int) -> str:
        """Genera ID Ãºnico y determinÃ­stico para un punto"""
        hash_input = f"{file_path}:{section_index}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def _split_long_section(self, section, max_chars: int = 2000) -> List[dict]:
        """
        Divide una secciÃ³n larga en chunks mÃ¡s pequeÃ±os

        Args:
            section: DocumentSection a dividir
            max_chars: TamaÃ±o mÃ¡ximo de caracteres por chunk

        Returns:
            Lista de diccionarios con title y content
        """
        content = section.content

        if len(content) <= max_chars:
            return [{'title': section.title, 'content': content, 'metadata': section.metadata}]

        chunks = []
        current_pos = 0
        chunk_num = 1

        while current_pos < len(content):
            # Extraer chunk
            chunk_end = current_pos + max_chars
            chunk_text = content[current_pos:chunk_end]

            # Intentar cortar en un punto natural (pÃ¡rrafo, frase)
            if chunk_end < len(content):
                # Buscar Ãºltimo salto de lÃ­nea
                last_newline = chunk_text.rfind('\n\n')
                if last_newline > max_chars * 0.5:  # Al menos 50% del chunk
                    chunk_text = chunk_text[:last_newline]
                    current_pos += last_newline + 2
                else:
                    # Buscar Ãºltimo punto
                    last_period = chunk_text.rfind('. ')
                    if last_period > max_chars * 0.5:
                        chunk_text = chunk_text[:last_period + 1]
                        current_pos += last_period + 2
                    else:
                        # Ãšltimo espacio
                        last_space = chunk_text.rfind(' ')
                        if last_space > max_chars * 0.5:
                            chunk_text = chunk_text[:last_space]
                            current_pos += last_space + 1
                        else:
                            current_pos = chunk_end
            else:
                current_pos = len(content)

            chunks.append({
                'title': f"{section.title} (parte {chunk_num})" if len(
                    chunks) > 0 or current_pos < len(
                    content) else section.title,
                'content': chunk_text.strip(),
                'metadata': section.metadata
            })
            chunk_num += 1

        return chunks

    def sync_document(self, doc_path: Path):
        """Sincroniza un documento individual"""
        try:
            # Obtener loader apropiado
            loader = DocumentLoaderFactory.get_loader(doc_path)
            if not loader:
                print(f"âš ï¸  No hay loader para: {doc_path.name}")
                return

            # Cargar documento
            doc = loader.load(doc_path)
            # Usar ruta relativa al vault en lugar de cwd
            rel_path = str(doc_path.relative_to(self.vault_path))

            # Verificar cambios
            changed, reason = self.tracker.has_changed(rel_path, doc.content)

            if not changed:
                self.stats['skipped'] += 1
                return

            print(f"ðŸ”„ {doc.file_name} ({reason})")

            # Si es modificaciÃ³n, eliminar vectores antiguos
            if reason == "modificado":
                old_ids = self.tracker.get_point_ids(rel_path)
                if old_ids:
                    self.qdrant.delete(
                        collection_name=self.collection_name,
                        points_selector=old_ids
                    )
                    print(f"   ðŸ—‘ï¸  Eliminados {len(old_ids)} vectores antiguos")

            # Generar puntos para cada secciÃ³n (con chunking si es necesario)
            points = []
            point_ids = []
            chunk_index = 0

            for section in doc.sections:
                # Dividir secciÃ³n si es muy larga
                chunks = self._split_long_section(section, max_chars=2000)

                for chunk in chunks:
                    # Texto completo para embedding
                    text_to_embed = f"{chunk['title']}\n\n{chunk['content']}"

                    # Generar embedding
                    try:
                        embedding = self.generate_embedding(text_to_embed)
                    except Exception as e:
                        print(f"   âŒ Error generando embedding: {e}")
                        self.stats['errors'] += 1
                        continue

                    # ID Ãºnico
                    point_id = self._generate_point_id(rel_path, chunk_index)
                    point_ids.append(point_id)
                    chunk_index += 1

                    # Preparar payload
                    payload = {
                        'file': doc.file_name,
                        'filePath': rel_path,
                        'section': chunk['title'],
                        'sectionLevel': section.level,
                        'content': chunk['content'],
                        'fullText': text_to_embed,
                        **doc.metadata,
                        **chunk['metadata']
                    }

                    points.append(PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload
                    ))

            # Insertar en Qdrant
            if points:
                self.qdrant.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                print(f"   âœ… Indexadas {len(points)} secciones")

            # Actualizar tracking
            self.tracker.update_file(rel_path, doc.content, point_ids)

            # Actualizar estadÃ­sticas
            if reason == "nuevo":
                self.stats['added'] += 1
            else:
                self.stats['updated'] += 1

            self.stats['processed'] += 1

        except Exception as e:
            print(f"   âŒ Error: {e}")
            self.stats['errors'] += 1

    def clean_deleted_files(self, current_files: List[Path]):
        """Elimina vectores de archivos que ya no existen"""
        # Usar rutas relativas al vault
        current_paths = {str(f.relative_to(self.vault_path)) for f in current_files}
        tracked_paths = set(self.tracker.get_tracked_files())

        deleted = tracked_paths - current_paths

        if deleted:
            print(f"\nðŸ§¹ Limpiando {len(deleted)} archivo(s) eliminado(s)...")

            for file_path in deleted:
                print(f"ðŸ—‘ï¸  {Path(file_path).name}")

                # Eliminar vectores de Qdrant
                point_ids = self.tracker.get_point_ids(file_path)
                if point_ids:
                    self.qdrant.delete(
                        collection_name=self.collection_name,
                        points_selector=point_ids
                    )

                # Eliminar del tracking
                self.tracker.remove_file(file_path)
                self.stats['deleted'] += 1

    def sync(self, pattern: Optional[str] = None):
        """
        Ejecuta sincronizaciÃ³n completa

        Args:
            pattern: PatrÃ³n regex opcional para filtrar archivos
        """
        self.initialize()

        print(f"ðŸ“ Vault: {self.vault_path}")
        print(f"ðŸŽ¯ Collection: {self.collection_name}")
        print(f"ðŸ¤– Embedding: {settings.EMBEDDING_MODEL}")
        print(f"ðŸ“Š Vector size: {settings.VECTOR_SIZE}\n")

        # Buscar documentos
        print("ðŸ” Buscando documentos...")
        documents = self.find_documents()

        # Aplicar filtro si existe
        if pattern:
            import re
            regex = re.compile(pattern)
            documents = [doc for doc in documents if regex.search(str(doc))]
            print(f"   Filtro aplicado: {pattern}")

        print(f"   Encontrados: {len(documents)} documentos\n")

        if not documents:
            print("âš ï¸  No se encontraron documentos para procesar")
            return

        # Sincronizar cada documento
        for doc_path in tqdm(documents, desc="Sincronizando"):
            self.sync_document(doc_path)

        # Limpiar archivos eliminados
        self.clean_deleted_files(documents)

        # Guardar tracking
        self.tracker.save()

        # Mostrar estadÃ­sticas
        self._print_stats()

    def _print_stats(self):
        """Imprime estadÃ­sticas finales"""
        print("\n" + "=" * 60)
        print("ðŸ“Š EstadÃ­sticas de SincronizaciÃ³n:")
        print(f"   Total procesados: {self.stats['processed']}")
        print(f"   âœ… Nuevos: {self.stats['added']}")
        print(f"   ðŸ”„ Actualizados: {self.stats['updated']}")
        print(f"   â­ï¸  Sin cambios: {self.stats['skipped']}")
        print(f"   ðŸ—‘ï¸  Eliminados: {self.stats['deleted']}")
        print(f"   âŒ Errores: {self.stats['errors']}")

        # Info de la colecciÃ³n
        collection_info = self.qdrant.get_collection(self.collection_name)
        print(f"\nðŸ“¦ ColecciÃ³n '{self.collection_name}':")
        print(f"   Vectores totales: {collection_info.points_count}")

        # Stats del tracker
        tracker_stats = self.tracker.get_stats()
        print(f"\nðŸ“ˆ Tracking:")
        print(f"   Archivos rastreados: {tracker_stats['total_files']}")
        print(f"   Secciones totales: {tracker_stats['total_sections']}")
        print("=" * 60)
