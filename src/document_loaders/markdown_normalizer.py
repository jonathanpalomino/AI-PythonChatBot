# =============================================================================
# src/document_loaders/markdown_normalizer.py
# Generic Markdown normalization utilities
# =============================================================================
"""
Normalización y sanitización de contenido Markdown genérico.
Sin dependencias de Obsidian ni otras variantes específicas.
"""
import re
from typing import Dict, List, Tuple, Optional


class MarkdownNormalizer:
    """Normaliza sintaxis Markdown estándar (headings, listas, tablas, bloques de código)"""

    @staticmethod
    def normalize_headings(content: str) -> str:
        """
        Normaliza headings para asegurar formato consistente.
        
        - Convierte headings ATX-style (# Heading) a formato estándar
        - Asegura espacio después del #
        - Remueve espacios en blanco extras al final
        
        Args:
            content: Contenido markdown
            
        Returns:
            Contenido con headings normalizados
        """
        lines = []
        for line in content.split('\n'):
            # Detectar ATX-style heading
            match = re.match(r'^(\s*)(#{1,6})\s*(.+?)\s*#*\s*$', line)
            if match:
                indent, hashes, title = match.groups()
                # Normalizar: eliminar indent, asegurar un espacio, limpiar trailing #
                normalized = f"{hashes} {title.strip()}"
                lines.append(normalized)
            else:
                lines.append(line)
        
        return '\n'.join(lines)

    @staticmethod
    def extract_code_blocks(content: str) -> Tuple[str, List[Dict[str, str]]]:
        """
        Extrae bloques de código fenced (```) del contenido.
        
        Args:
            content: Contenido markdown
            
        Returns:
            Tupla de (contenido sin bloques de código, lista de bloques extraídos)
        """
        code_blocks = []
        placeholder_template = "<<<CODE_BLOCK_{}>>>"
        
        def replacer(match):
            lang = match.group(1) or ""
            code = match.group(2)
            index = len(code_blocks)
            code_blocks.append({
                "language": lang.strip(),
                "code": code,
                "index": index
            })
            return placeholder_template.format(index)
        
        # Pattern para fenced code blocks
        pattern = r'```(\w+)?\n(.*?)```'
        content_without_code = re.sub(pattern, replacer, content, flags=re.DOTALL)
        
        return content_without_code, code_blocks

    @staticmethod
    def restore_code_blocks(content: str, code_blocks: List[Dict[str, str]]) -> str:
        """
        Restaura bloques de código previamente extraídos.
        
        Args:
            content: Contenido con placeholders
            code_blocks: Lista de bloques de código
            
        Returns:
            Contenido con bloques de código restaurados
        """
        for block in code_blocks:
            placeholder = f"<<<CODE_BLOCK_{block['index']}>>>"
            lang = block['language']
            code = block['code']
            code_block = f"```{lang}\n{code}```"
            content = content.replace(placeholder, code_block)
        
        return content

    @staticmethod
    def normalize_tables(content: str) -> str:
        """
        Normaliza tablas markdown para formato consistente.
        
        Args:
            content: Contenido markdown
            
        Returns:
            Contenido con tablas normalizadas
        """
        # Asegurar que separadores de tabla tengan formato correcto
        # Pattern: línea con |---|---| para separador de header
        lines = []
        for line in content.split('\n'):
            # Detectar separador de tabla
            if re.match(r'^\s*\|?\s*[-:]+\s*(\|\s*[-:]+\s*)+\|?\s*$', line):
                # Normalizar separador
                parts = [p.strip() for p in line.split('|') if p.strip()]
                normalized = '| ' + ' | '.join(['---'] * len(parts)) + ' |'
                lines.append(normalized)
            else:
                lines.append(line)
        
        return '\n'.join(lines)

    @staticmethod
    def clean_frontmatter(content: str) -> Tuple[str, Optional[str]]:
        """
        Extrae y limpia frontmatter YAML del contenido.
        
        Args:
            content: Contenido markdown con posible frontmatter
            
        Returns:
            Tupla de (contenido sin frontmatter, frontmatter YAML o None)
        """
        match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
        if match:
            frontmatter = match.group(1)
            content_without = content[match.end():]
            return content_without, frontmatter
        
        return content, None

    @staticmethod
    def normalize_lists(content: str) -> str:
        """
        Normaliza listas para formato consistente (espacios, bullets).
        
        Args:
            content: Contenido markdown
            
        Returns:
            Contenido con listas normalizadas
        """
        lines = []
        for line in content.split('\n'):
            # Normalizar listas desordenadas (*, -, +)
            match = re.match(r'^(\s*)([*\-+])\s+(.+)$', line)
            if match:
                indent, bullet, text = match.groups()
                # Normalizar a usar - como bullet
                normalized = f"{indent}- {text}"
                lines.append(normalized)
            # Normalizar listas ordenadas
            elif re.match(r'^(\s*)(\d+)\.\s+(.+)$', line):
                match = re.match(r'^(\s*)(\d+)\.\s+(.+)$', line)
                indent, num, text = match.groups()
                normalized = f"{indent}{num}. {text}"
                lines.append(normalized)
            else:
                lines.append(line)
        
        return '\n'.join(lines)

    @staticmethod
    def strip_html_comments(content: str) -> str:
        """
        Remueve comentarios HTML del contenido.
        
        Args:
            content: Contenido markdown
            
        Returns:
            Contenido sin comentarios HTML
        """
        # Remover comentarios HTML: <!-- comentario -->
        pattern = r'<!--.*?-->'
        return re.sub(pattern, '', content, flags=re.DOTALL)

    @staticmethod
    def normalize_all(content: str) -> str:
        """
        Aplica todas las normalizaciones en orden seguro.
        
        Args:
            content: Contenido markdown crudo
            
        Returns:
            Contenido markdown normalizado
        """
        # 1. Remover comentarios HTML
        content = MarkdownNormalizer.strip_html_comments(content)
        
        # 2. Extraer código (para no modificarlo)
        content_without_code, code_blocks = MarkdownNormalizer.extract_code_blocks(content)
        
        # 3. Normalizar headings
        content_without_code = MarkdownNormalizer.normalize_headings(content_without_code)
        
        # 4. Normalizar listas
        content_without_code = MarkdownNormalizer.normalize_lists(content_without_code)
        
        # 5. Normalizar tablas
        content_without_code = MarkdownNormalizer.normalize_tables(content_without_code)
        
        # 6. Restaurar código
        content = MarkdownNormalizer.restore_code_blocks(content_without_code, code_blocks)
        
        return content

    @staticmethod
    def split_by_sentences(text: str, max_length: int = 500) -> List[str]:
        """
        Divide texto en oraciones para chunking más granular.
        
        Args:
            text: Texto a dividir
            max_length: Longitud máxima aproximada de cada chunk
            
        Returns:
            Lista de chunks de texto
        """
        # Dividir por puntos seguidos de espacio y mayúscula
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > max_length and current_chunk:
                # Guardar chunk actual
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Agregar último chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
