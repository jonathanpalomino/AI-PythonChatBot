# src/services/query_intent_analyzer.py
"""
Analiza la intención de una consulta para determinar estrategia de navegación
"""

from typing import Dict, Literal, Optional
from dataclasses import dataclass
import re
from src.utils.logger import get_logger

@dataclass
class NavigationIntent:
    """Intención de navegación determinada desde la query"""
    direction: Literal["up", "down", "bidirectional"]  # Dirección
    max_depth: int  # Profundidad máxima
    max_nodes: int  # Nodos máximos
    priority: Literal["general", "specific", "exhaustive"]  # Prioridad
    confidence: float  # Confianza en la decisión (0-1)
    reasoning: str  # Por qué se tomó esta decisión

class QueryIntentAnalyzer:
    """
    Analiza queries para determinar cómo navegar el grafo Obsidian
    
    Ejemplos:
    - "¿Qué es ServerGroups?" → UP (contexto general)
    - "¿Qué aplicaciones usan JBoss?" → DOWN (detalles específicos)
    - "Explica la arquitectura completa" → BIDIRECTIONAL (todo el contexto)
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Patrones para detectar intención ASCENDENTE (buscar contexto general)
        self.context_patterns = [
            r'\b(qu[eé] es|define|explica|contexto de)\b',
            r'\b(para qu[eé] sirve|cu[aá]l es el prop[oó]sito)\b',
            r'\b(arquitectura|overview|visi[oó]n general)\b',
            r'\b(relacionado con|parte de)\b',
            r'\b(qu[eé] contiene|qu[eé] incluye)\b',
        ]
        
        # Patrones para detectar intención DESCENDENTE (buscar detalles)
        self.detail_patterns = [
            r'\b(qu[eé] aplicaciones|qu[eé] servicios|qu[eé] programas)\b',
            r'\b(lista|enumerate|muestra)\b',
            r'\b(cu[aá]les son|cu[aá]ntos|nombres de)\b',
            r'\b(especificaciones|detalles|configuraci[oó]n de)\b',
            r'\b(usa|utiliza|depende de|consume)\b',
        ]
        
        # Patrones para detectar intención BIDIRECCIONAL (exhaustiva)
        self.exhaustive_patterns = [
            r'\b(toda la|completo|exhaustivo|todo sobre)\b',
            r'\b(diagrama|mapa|estructura completa)\b',
            r'\b(relaciones|dependencias|conexiones)\b',
            r'\b(flujo completo|end-to-end)\b',
        ]
        
        # Indicadores de especificidad (bajo = general, alto = específico)
        self.specificity_indicators = {
            'high': [r'\b(aplicaci[oó]n|servicio|programa|archivo|clase)\s+[A-Z]'],
            'low': [r'\b(todo|general|overview|arquitectura|contexto)\b']
        }
    
    def analyze(
        self,
        query: str,
        current_note: Optional[str] = None,
        note_metadata: Optional[Dict] = None
    ) -> NavigationIntent:
        """
        Analiza la query y determina la estrategia de navegación
        
        Args:
            query: Pregunta del usuario
            current_note: Nota desde donde se parte (opcional)
            note_metadata: Metadata de la nota actual (is_hub, is_index, etc)
        
        Returns:
            NavigationIntent con estrategia determinada
        """
        query_lower = query.lower()
        
        # Score de cada dirección
        scores = {
            'up': 0,
            'down': 0,
            'bidirectional': 0
        }
        
        # 1. ANÁLISIS DE PATRONES EN LA QUERY
        
        # Buscar contexto (ascendente)
        for pattern in self.context_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                scores['up'] += 2
                self.logger.debug(f"Context pattern matched: {pattern}")
        
        # Buscar detalles (descendente)
        for pattern in self.detail_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                scores['down'] += 2
                self.logger.debug(f"Detail pattern matched: {pattern}")
        
        # Buscar exhaustividad (bidireccional)
        for pattern in self.exhaustive_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                scores['bidirectional'] += 3
                self.logger.debug(f"Exhaustive pattern matched: {pattern}")
        
        # 2. ANÁLISIS DE ESPECIFICIDAD
        specificity = self._analyze_specificity(query_lower)
        
        if specificity == 'high':
            scores['down'] += 1  # Específico → buscar detalles
        elif specificity == 'low':
            scores['up'] += 1  # General → buscar contexto
        
        # 3. ANÁLISIS DE LA NOTA ACTUAL (si se proporciona)
        if note_metadata:
            is_hub = note_metadata.get('is_hub', False)
            is_index = note_metadata.get('is_index', False)
            is_atomic = note_metadata.get('note_type') == 'atomic'
            
            if is_hub:
                # Si estamos en un hub, probablemente queremos bajar
                scores['down'] += 1
                self.logger.debug("Current note is hub → favor DOWN")
            
            elif is_atomic:
                # Si estamos en nota atómica, probablemente queremos subir
                scores['up'] += 1
                self.logger.debug("Current note is atomic → favor UP")
            
            elif is_index:
                # Si es índice, bidireccional es buena opción
                scores['bidirectional'] += 1
                self.logger.debug("Current note is index → favor BIDIRECTIONAL")
        
        # 4. DETERMINAR DIRECCIÓN GANADORA
        max_score = max(scores.values())
        
        # Si no hay score claro, default a bidireccional limitado
        if max_score == 0:
            direction = "bidirectional"
            confidence = 0.3
            reasoning = "No clear pattern detected - using safe bidirectional search"
        else:
            direction = max(scores, key=scores.get)
            confidence = min(max_score / 5.0, 1.0)  # Normalizar a 0-1
            reasoning = self._build_reasoning(scores, query_lower, note_metadata)
        
        # 5. DETERMINAR PROFUNDIDAD Y LÍMITES SEGÚN DIRECCIÓN
        depth_config = self._get_depth_config(
            direction=direction,
            confidence=confidence,
            specificity=specificity
        )
        
        intent = NavigationIntent(
            direction=direction,
            max_depth=depth_config['max_depth'],
            max_nodes=depth_config['max_nodes'],
            priority=depth_config['priority'],
            confidence=confidence,
            reasoning=reasoning
        )
        
        self.logger.info(
            f"Query intent analyzed",
            extra={
                "query": query[:60],
                "direction": direction,
                "depth": depth_config['max_depth'],
                "nodes": depth_config['max_nodes'],
                "confidence": f"{confidence:.2f}"
            }
        )
        
        return intent
    
    def _analyze_specificity(self, query: str) -> Literal["high", "low", "medium"]:
        """Determina si la query es específica o general"""
        high_count = sum(
            1 for pattern in self.specificity_indicators['high']
            if re.search(pattern, query, re.IGNORECASE)
        )
        low_count = sum(
            1 for pattern in self.specificity_indicators['low']
            if re.search(pattern, query, re.IGNORECASE)
        )
        
        if high_count > low_count:
            return "high"
        elif low_count > high_count:
            return "low"
        else:
            return "medium"
    
    def _build_reasoning(
        self,
        scores: Dict[str, int],
        query: str,
        note_metadata: Optional[Dict]
    ) -> str:
        """Construye explicación de por qué se eligió esta dirección"""
        parts = []
        
        winner = max(scores, key=scores.get)
        
        if scores[winner] >= 2:
            if winner == 'up':
                parts.append("Query seeks contextual/general information")
            elif winner == 'down':
                parts.append("Query seeks specific details/instances")
            else:
                parts.append("Query seeks comprehensive/exhaustive information")
        
        if note_metadata:
            note_type = note_metadata.get('note_type', 'unknown')
            parts.append(f"Starting from {note_type} note")
        
        return "; ".join(parts) if parts else "Default heuristic applied"
    
    def _get_depth_config(
        self,
        direction: str,
        confidence: float,
        specificity: str
    ) -> Dict:
        """
        Determina configuración de profundidad según dirección y confianza
        
        REGLAS:
        - UP (ascendente): menos profundidad, más nodos (buscar contexto amplio)
        - DOWN (descendente): más profundidad, menos nodos (buscar detalles)
        - BIDIRECTIONAL: profundidad media, nodos medios (exploración balanceada)
        """
        if direction == "up":
            # ASCENDENTE: Subir hasta encontrar contexto general
            return {
                'max_depth': 2 if confidence > 0.7 else 3,
                'max_nodes': 8,
                'priority': 'general'
            }
        
        elif direction == "down":
            # DESCENDENTE: Bajar para encontrar detalles específicos
            if specificity == 'high':
                # Muy específico → bajar profundo pero limitado
                return {
                    'max_depth': 3,
                    'max_nodes': 10,
                    'priority': 'specific'
                }
            else:
                # General → bajar moderado con más amplitud
                return {
                    'max_depth': 2,
                    'max_nodes': 12,
                    'priority': 'specific'
                }
        
        else:  # bidirectional
            # BIDIRECCIONAL: Balance entre amplitud y profundidad
            return {
                'max_depth': 2,
                'max_nodes': 15,
                'priority': 'exhaustive'
            }
