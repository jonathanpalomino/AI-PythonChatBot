# =============================================================================
# src/document_loaders/obsidian_tree_navigator.py
# Navegador inteligente de grafos Obsidian - REFACTORED
# =============================================================================
"""
Navegador inteligente de grafos Obsidian con soporte para ciclos.
Optimizado para documentación técnica con relaciones recursivas.

✅ REFACTORIZADO: NavigationIntent movido aquí (era de query_intent_analyzer)
"""

import datetime
from src.utils.date_utils import get_current_utc
from typing import Dict, List, Set, Optional, Tuple, Literal
from dataclasses import dataclass
from collections import OrderedDict, deque
import time
from src.utils.logger import get_logger


# =============================================================================
# NavigationIntent - Específico para navegación Obsidian
# =============================================================================

@dataclass
class NavigationIntent:
    """
    Intención de navegación para Obsidian.

    Determina cómo navegar por el grafo de notas:
    - up: Contexto general (incoming links, hacia conceptos más generales)
    - down: Detalles específicos (outgoing links, hacia implementaciones)
    - bidirectional: Contexto completo (explora ambas direcciones)
    """
    direction: Literal["up", "down", "bidirectional"]
    max_depth: int = 2
    max_nodes: int = 15
    confidence: float = 0.8
    reasoning: str = ""


@dataclass
class NavigationResult:
    """Resultado de navegación con contexto expandido"""
    root_note: str
    visited_notes: List[str]
    context_layers: Dict[int, List[str]]  # {depth: [notas]}
    total_links: int
    cycles_detected: List[Tuple[str, str]]  # (origen, destino)
    execution_time_ms: float
    strategy_used: str


# =============================================================================
# ObsidianTreeNavigator - Navegador con cache LRU + TTL
# =============================================================================

class ObsidianTreeNavigator:
    """
    Navegador eficiente de grafos Obsidian con:
    - Detección de ciclos
    - Expansión BFS por capas
    - Priorización por tipo de nota (hubs primero)
    - Cache de rutas frecuentes (LRU + TTL)
    """

    MAX_CACHE_SIZE = 100
    CACHE_TTL_SECONDS = 3600  # 1 hora

    def __init__(self, graph: Dict[str, Dict], cache_enabled: bool = True):
        self.graph = graph
        self.logger = get_logger(__name__)
        self.cache_enabled = cache_enabled

        # Cache con TTL
        self._navigation_cache: OrderedDict[
            str, Tuple[NavigationResult, datetime.datetime]] = OrderedDict()
        self._cycle_cache: Set[Tuple[str, str]] = set()

        # Pre-clasificar notas por importancia
        self.hubs = self._identify_hubs()
        self.indexes = self._identify_indexes()

    def _identify_hubs(self) -> List[str]:
        """Identifica notas hub (muchos backlinks)"""
        return sorted(
            [note for note, data in self.graph.items()
             if len(data.get('in', [])) > 5],
            key=lambda n: len(self.graph[n].get('in', [])),
            reverse=True
        )

    def _identify_indexes(self) -> List[str]:
        """Identifica notas índice (muchos outgoing links)"""
        return sorted(
            [note for note, data in self.graph.items()
             if len(data.get('out', [])) > 10],
            key=lambda n: len(self.graph[n].get('out', [])),
            reverse=True
        )

    def navigate_from_note(
        self,
        start_note: str,
        max_depth: int = 2,
        max_nodes: int = 15,
        strategy: str = "smart"
    ) -> NavigationResult:
        """
        Navega desde una nota expandiendo contexto inteligentemente.

        Args:
            start_note: Nota inicial
            max_depth: Profundidad máxima
            max_nodes: Límite de nodos a visitar
            strategy: "smart", "bfs", o "hub-first"

        Returns:
            NavigationResult con notas visitadas y capas de contexto
        """
        start_time = time.perf_counter()
        cache_key = f"{start_note}:{max_depth}:{max_nodes}:{strategy}"

        # Check cache con TTL
        if self.cache_enabled and cache_key in self._navigation_cache:
            result, timestamp = self._navigation_cache[cache_key]
            age = (get_current_utc() - timestamp).total_seconds()

            if age < self.CACHE_TTL_SECONDS:
                self.logger.debug(f"Cache HIT for {start_note} (age: {age:.1f}s)")
                return result
            else:
                del self._navigation_cache[cache_key]
                self.logger.debug(f"Cache EXPIRED for {start_note}")

        # Validar que la nota existe
        if start_note not in self.graph:
            self.logger.warning(f"Note {start_note} not found in graph")
            return NavigationResult(
                root_note=start_note,
                visited_notes=[],
                context_layers={},
                total_links=0,
                cycles_detected=[],
                execution_time_ms=0,
                strategy_used=strategy
            )

        # Ejecutar navegación según estrategia
        if strategy == "smart":
            result = self._navigate_smart(start_note, max_depth, max_nodes)
        elif strategy == "hub-first":
            result = self._navigate_hub_first(start_note, max_depth, max_nodes)
        else:
            result = self._navigate_bfs(start_note, max_depth, max_nodes)

        result.execution_time_ms = (time.perf_counter() - start_time) * 1000

        # Guardar en cache (LRU)
        if self.cache_enabled:
            self._navigation_cache[cache_key] = (result, get_current_utc())
            if len(self._navigation_cache) > self.MAX_CACHE_SIZE:
                self._navigation_cache.popitem(last=False)

        self.logger.info(
            "Navigation completed",
            extra={
                "root": start_note,
                "visited": len(result.visited_notes),
                "layers": len(result.context_layers),
                "cycles": len(result.cycles_detected),
                "time_ms": f"{result.execution_time_ms:.2f}",
                "strategy": strategy
            }
        )

        return result

    def navigate_with_intent(
        self,
        start_note: str,
        intent: NavigationIntent
    ) -> NavigationResult:
        """
        Navega según la intención detectada.

        Args:
            start_note: Nota de inicio
            intent: Intención con dirección y parámetros

        Returns:
            NavigationResult con el contexto apropiado
        """
        self.logger.info(
            "Navigating with intent",
            extra={
                "note": start_note,
                "direction": intent.direction,
                "depth": intent.max_depth,
                "reasoning": intent.reasoning
            }
        )

        if intent.direction == "up":
            return self._navigate_upward(start_note, intent.max_depth, intent.max_nodes)
        elif intent.direction == "down":
            return self._navigate_downward(start_note, intent.max_depth, intent.max_nodes)
        else:  # bidirectional
            return self._navigate_bidirectional(start_note, intent.max_depth, intent.max_nodes)

    # =========================================================================
    # Métodos de navegación (sin cambios, solo limpieza de imports)
    # =========================================================================

    def _navigate_smart(self, start_note: str, max_depth: int, max_nodes: int) -> NavigationResult:
        """Navegación inteligente: prioriza hubs"""
        visited: Set[str] = set()
        context_layers: Dict[int, List[str]] = {}
        cycles_detected: List[Tuple[str, str]] = []

        # Cola de prioridad: (profundidad, nota, padre)
        queue = deque([(0, start_note, None)])
        visited.add(start_note)
        context_layers[0] = [start_note]

        while queue and len(visited) < max_nodes:
            depth, current, parent = queue.popleft()
            if depth >= max_depth:
                continue

            # Obtener vecinos (outgoing + incoming para documentación)
            neighbors = self._get_relevant_neighbors(current, visited)

            # Ordenar vecinos por prioridad
            neighbors = self._prioritize_neighbors(neighbors, current)

            for neighbor in neighbors:
                if len(visited) >= max_nodes:
                    break

                # Detectar ciclo
                if neighbor in visited:
                    cycles_detected.append((current, neighbor))
                    self.logger.debug(f"Cycle detected: {current} → {neighbor}")
                    continue

                visited.add(neighbor)
                queue.append((depth + 1, neighbor, current))

                if depth + 1 not in context_layers:
                    context_layers[depth + 1] = []
                context_layers[depth + 1].append(neighbor)

        return NavigationResult(
            root_note=start_note,
            visited_notes=list(visited),
            context_layers=context_layers,
            total_links=len(visited) - 1,
            cycles_detected=cycles_detected,
            execution_time_ms=0,
            strategy_used="smart"
        )

    def _navigate_upward(self, start_note: str, max_depth: int, max_nodes: int) -> NavigationResult:
        """Navegación ASCENDENTE: hacia contexto general (incoming links)"""
        visited: Set[str] = set()
        context_layers: Dict[int, List[str]] = {}
        cycles_detected: List[Tuple[str, str]] = []

        queue = deque([(0, start_note, None)])
        visited.add(start_note)
        context_layers[0] = [start_note]

        while queue and len(visited) < max_nodes:
            depth, current, parent = queue.popleft()
            if depth >= max_depth:
                continue

            if current not in self.graph:
                continue

            incoming = self.graph[current].get('in', [])
            incoming_sorted = sorted(
                [n for n in incoming if n not in visited],
                key=lambda n: len(self.graph.get(n, {}).get('in', [])),
                reverse=True
            )

            for parent_note in incoming_sorted:
                if len(visited) >= max_nodes:
                    break
                if parent_note in visited:
                    cycles_detected.append((current, parent_note))
                    continue

                visited.add(parent_note)
                queue.append((depth + 1, parent_note, current))

                if depth + 1 not in context_layers:
                    context_layers[depth + 1] = []
                context_layers[depth + 1].append(parent_note)

                if parent_note in self.hubs[:3]:
                    break

        return NavigationResult(
            root_note=start_note,
            visited_notes=list(visited),
            context_layers=context_layers,
            total_links=len(visited) - 1,
            cycles_detected=cycles_detected,
            execution_time_ms=0,
            strategy_used="upward"
        )

    def _navigate_downward(self, start_note: str, max_depth: int,
                           max_nodes: int) -> NavigationResult:
        """Navegación DESCENDENTE: hacia detalles específicos (outgoing links)"""
        visited: Set[str] = set()
        context_layers: Dict[int, List[str]] = {}
        cycles_detected: List[Tuple[str, str]] = []

        queue = deque([(0, start_note, None)])
        visited.add(start_note)
        context_layers[0] = [start_note]

        while queue and len(visited) < max_nodes:
            depth, current, parent = queue.popleft()
            if depth >= max_depth:
                continue

            if current not in self.graph:
                continue

            outgoing = self.graph[current].get('out', [])
            outgoing_sorted = sorted(
                [n for n in outgoing if n not in visited],
                key=lambda n: self._get_note_priority_score(n, priority='atomic'),
                reverse=True
            )

            for child_note in outgoing_sorted:
                if len(visited) >= max_nodes:
                    break
                if child_note in visited:
                    cycles_detected.append((current, child_note))
                    continue

                visited.add(child_note)
                queue.append((depth + 1, child_note, current))

                if depth + 1 not in context_layers:
                    context_layers[depth + 1] = []
                context_layers[depth + 1].append(child_note)

        return NavigationResult(
            root_note=start_note,
            visited_notes=list(visited),
            context_layers=context_layers,
            total_links=len(visited) - 1,
            cycles_detected=cycles_detected,
            execution_time_ms=0,
            strategy_used="downward"
        )

    def _navigate_bidirectional(self, start_note: str, max_depth: int,
                                max_nodes: int) -> NavigationResult:
        """Navegación BIDIRECCIONAL: contexto completo"""
        return self._navigate_smart(start_note, max_depth, max_nodes)

    def _navigate_hub_first(self, start_note: str, max_depth: int,
                            max_nodes: int) -> NavigationResult:
        """Estrategia hub-first: sube al hub más cercano, luego expande"""
        visited: Set[str] = {start_note}
        context_layers: Dict[int, List[str]] = {0: [start_note]}
        cycles_detected: List[Tuple[str, str]] = []

        nearest_hub = self._find_nearest_hub(start_note, max_depth=3)
        if nearest_hub and nearest_hub != start_note:
            visited.add(nearest_hub)
            context_layers[1] = [nearest_hub]

        if nearest_hub:
            hub_expansion = self._navigate_smart(nearest_hub, max_depth - 1, max_nodes)
            visited.update(hub_expansion.visited_notes)

            for depth, notes in hub_expansion.context_layers.items():
                layer_depth = depth + 1
                if layer_depth not in context_layers:
                    context_layers[layer_depth] = []
                context_layers[layer_depth].extend(notes)

            cycles_detected.extend(hub_expansion.cycles_detected)

        return NavigationResult(
            root_note=start_note,
            visited_notes=list(visited),
            context_layers=context_layers,
            total_links=len(visited) - 1,
            cycles_detected=cycles_detected,
            execution_time_ms=0,
            strategy_used="hub-first"
        )

    def _navigate_bfs(self, start_note: str, max_depth: int, max_nodes: int) -> NavigationResult:
        """BFS estándar sin priorización"""
        visited: Set[str] = {start_note}
        context_layers: Dict[int, List[str]] = {0: [start_note]}
        cycles_detected: List[Tuple[str, str]] = []
        queue = deque([(0, start_note)])

        while queue and len(visited) < max_nodes:
            depth, current = queue.popleft()
            if depth >= max_depth:
                continue

            neighbors = self._get_relevant_neighbors(current, visited)
            for neighbor in neighbors:
                if len(visited) >= max_nodes:
                    break
                if neighbor in visited:
                    cycles_detected.append((current, neighbor))
                    continue

                visited.add(neighbor)
                queue.append((depth + 1, neighbor))

                if depth + 1 not in context_layers:
                    context_layers[depth + 1] = []
                context_layers[depth + 1].append(neighbor)

        return NavigationResult(
            root_note=start_note,
            visited_notes=list(visited),
            context_layers=context_layers,
            total_links=len(visited) - 1,
            cycles_detected=cycles_detected,
            execution_time_ms=0,
            strategy_used="bfs"
        )

    # =========================================================================
    # Métodos auxiliares
    # =========================================================================

    def _get_relevant_neighbors(self, note: str, visited: Set[str]) -> List[str]:
        """Obtiene vecinos relevantes (bidireccional)"""
        if note not in self.graph:
            return []

        outgoing = self.graph[note].get('out', [])
        incoming = self.graph[note].get('in', [])
        all_neighbors = set(outgoing + incoming)

        return [n for n in all_neighbors if n not in visited]

    def _prioritize_neighbors(self, neighbors: List[str], current_note: str) -> List[str]:
        """Prioriza vecinos: hubs primero, luego indexes, luego alfabético"""

        def priority_score(note: str) -> Tuple[int, int, str]:
            is_hub = 0 if note in self.hubs else 1
            is_index = 0 if note in self.indexes else 1
            return (is_hub, is_index, note)

        return sorted(neighbors, key=priority_score)

    def _find_nearest_hub(self, start_note: str, max_depth: int = 3) -> Optional[str]:
        """Encuentra el hub más cercano subiendo por el grafo"""
        visited = set()
        queue = deque([(start_note, 0)])

        while queue:
            current, depth = queue.popleft()
            if depth > max_depth or current in visited:
                continue

            visited.add(current)

            if current in self.hubs:
                return current

            if current in self.graph:
                for parent in self.graph[current].get('in', []):
                    if parent not in visited:
                        queue.append((parent, depth + 1))

        return self.hubs[0] if self.hubs else None

    def _get_note_priority_score(self, note: str, priority: str = 'general') -> float:
        """Calcula score de prioridad de una nota"""
        if note not in self.graph:
            return 0.0

        incoming_count = len(self.graph[note].get('in', []))
        outgoing_count = len(self.graph[note].get('out', []))
        total_links = incoming_count + outgoing_count

        if priority == 'general':
            return incoming_count + (outgoing_count * 0.3)
        elif priority == 'atomic':
            return 10.0 / (total_links + 1)
        else:  # balanced
            ideal_links = 5
            deviation = abs(total_links - ideal_links)
            return 10.0 / (deviation + 1)

    def get_context_summary(self, result: NavigationResult) -> str:
        """Genera un resumen textual del contexto navegado"""
        summary_parts = [
            f"📍 Contexto desde: **{result.root_note}**",
            f"🔗 Notas relacionadas: {len(result.visited_notes)}",
        ]

        for depth in sorted(result.context_layers.keys()):
            notes = result.context_layers[depth]
            if depth == 0:
                summary_parts.append(f"\\n**Nivel {depth} (Origen):**")
            else:
                summary_parts.append(f"\\n**Nivel {depth}:**")

            summary_parts.append(f" - {', '.join(notes[:10])}")
            if len(notes) > 10:
                summary_parts.append(f" - ... y {len(notes) - 10} más")

        if result.cycles_detected:
            summary_parts.append(f"\\n⚠️ Ciclos detectados: {len(result.cycles_detected)}")
            for origin, dest in result.cycles_detected[:3]:
                summary_parts.append(f" - {origin} ↔ {dest}")

        return "\\n".join(summary_parts)
