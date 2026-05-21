# =============================================================================
# src/services/intent/extractors.py
# Target Extraction Logic
# =============================================================================
"""
Extractor de targets (symbol names) desde queries de usuario.
Usa regex patterns definidos en config.py.
"""

import re
from typing import Optional, List, Tuple, Dict, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)

class TargetExtractor:
    """
    Extrae targets (symbol names, file names, etc.) desde queries usando regex.

    Examples:
         extractor = TargetExtractor()
         patterns = [r"método\\s+(\\w+)", r"function\\s+(\\w+)"]
         extractor.extract("muéstrame el método authenticate", patterns)
        'authenticate'
    """

    def extract(
        self,
        query: str,
        patterns: List[str]
    ) -> Optional[str]:
        """
        Extrae target usando lista de regex patterns en orden de prioridad.

        Args:
            query: Query del usuario
            patterns: Lista de regex patterns (se prueban en orden)

        Returns:
            Target extraído (string) o None si no se encontró
        """
        if not patterns:
            logger.debug("No patterns provided for extraction")
            return None

        query_clean = query.strip()

        for i, pattern in enumerate(patterns):
            try:
                match = re.search(pattern, query_clean, re.IGNORECASE)

                if match and len(match.groups()) > 0:
                    target = match.group(1).strip()

                    # Validación básica
                    if self._is_valid_target(target):
                        logger.debug(
                            f"Target extracted: '{target}' (pattern {i+1}/{len(patterns)})"
                        )
                        return target
                    else:
                        logger.debug(
                            f"Invalid target '{target}' (too short or invalid chars)"
                        )

            except re.error as e:
                logger.warning(f"Invalid regex pattern: {pattern} - {e}")
                continue

        logger.debug(f"No target extracted from: {query_clean[:60]}")
        return None

    def extract_multiple(
        self,
        query: str,
        patterns: List[str],
        max_results: int = 5
    ) -> List[str]:
        """
        Extrae múltiples targets de una query.
        Útil para queries como "métodos authenticate y validate".

        Args:
            query: Query del usuario
            patterns: Lista de regex patterns
            max_results: Máximo número de targets a retornar

        Returns:
            Lista de targets extraídos
        """
        if not patterns:
            return []

        targets = []
        query_clean = query.strip()

        for pattern in patterns:
            try:
                matches = re.finditer(pattern, query_clean, re.IGNORECASE)

                for match in matches:
                    if len(match.groups()) > 0:
                        target = match.group(1).strip()
                        if self._is_valid_target(target) and target not in targets:
                            targets.append(target)
                            if len(targets) >= max_results:
                                break

                if len(targets) >= max_results:
                    break

            except re.error as e:
                logger.warning(f"Invalid regex pattern: {pattern} - {e}")
                continue

        if targets:
            logger.debug(f"Extracted {len(targets)} targets: {targets}")

        return targets

    def _is_valid_target(self, target: str) -> bool:
        """
        Valida que un target sea razonable.

        Criterios:
        - Longitud >= 2 caracteres
        - No contiene solo números
        - No contiene caracteres especiales problemáticos

        Args:
            target: Target a validar

        Returns:
            True si es válido, False si no
        """
        if not target or len(target) < 2:
            return False

        # Evitar targets que son solo números
        if target.isdigit():
            return False

        # Evitar targets con caracteres problemáticos
        invalid_chars = [';', ':', '\\n', '\\t', '\\r']
        if any(char in target for char in invalid_chars):
            return False

        # Evitar palabras reservadas comunes
        stopwords = {'el', 'la', 'los', 'las', 'de', 'del', 'the', 'a', 'an'}
        if target.lower() in stopwords:
            return False

        return True

    def extract_with_context(
        self,
        query: str,
        patterns: List[str]
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Extrae target + contexto adicional (metadata).

        Args:
            query: Query del usuario
            patterns: Lista de regex patterns

        Returns:
            (target, context_dict) donde context incluye:
            - 'original_query': Query original
            - 'pattern_used': Pattern que funcionó
            - 'match_start': Posición inicio del match
            - 'match_end': Posición fin del match
        """
        if not patterns:
            return None, {}

        query_clean = query.strip()

        for i, pattern in enumerate(patterns):
            try:
                match = re.search(pattern, query_clean, re.IGNORECASE)

                if match and len(match.groups()) > 0:
                    target = match.group(1).strip()

                    if self._is_valid_target(target):
                        context = {
                            'original_query': query,
                            'pattern_used': pattern,
                            'pattern_index': i,
                            'match_start': match.start(),
                            'match_end': match.end(),
                            'full_match': match.group(0)
                        }
                        return target, context

            except re.error:
                continue

        return None, {}

    def extract_and_mask(
        self,
        query: str,
        patterns: List[str],
        mask_token: str = ""
    ) -> Tuple[str, Optional[str]]:
        """
        Extrae target de la query y retorna la query con el target enmascarado/removido.
        Ideal para normalizar inputs antes de calcular embeddings (Entity Masking).

        Ejemplo:
             extract_and_mask("dame el codigo de get_value", [r"codigo de (\\w+)"])
             ("dame el codigo de ", "get_value")

        Args:
            query: Query original del usuario
            patterns: Lista de regex patterns
            mask_token: String con el que reemplazar el target extraído

        Returns:
            Tuple[query_enmascarada, target_extraido]
        """
        if not patterns:
            return (query, None)

        target, context = self.extract_with_context(query, patterns)

        if not target or not context:
            return (query, None)

        # Usar la metadata del match para reemplazar
        original: str = context['original_query']
        start = context['match_start']
        end = context['match_end']
        full_match: str = context['full_match']

        # El pattern puede o no incluir el prefijo (ej: "codigo de ").
        # Reemplazamos SOLO el substring exacto que capturó el grupo (el target real)
        # dentro del match entero, para mantener los prefijos intactos.
        
        # Encontramos dónde está el target dentro del full_match
        target_idx = full_match.find(target)
        if target_idx != -1:
            masked_match = full_match[:target_idx] + mask_token + full_match[target_idx + len(target):]
            masked_query = original[:start] + masked_match + original[end:]
            # Limpiar espacios dobles que puedan haber quedado
            masked_query = re.sub(r'\s+', ' ', masked_query).strip()
            return (masked_query, target)
        
        # Fallback de seguridad
        return (query, target)

# Singleton instance (opcional, para facilitar uso)
_extractor = TargetExtractor()

def extract_target(query: str, patterns: List[str]) -> Optional[str]:
    """
    Helper function para extraer target rápidamente.

    Args:
        query: Query del usuario
        patterns: Lista de regex patterns

    Returns:
        Target extraído o None
    """
    return _extractor.extract(query, patterns)
