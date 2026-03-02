import logging
import pickle
from pathlib import Path
from typing import List, Dict

import networkx as nx

from src.services.analysis.codebase_analyzer import CodebaseAnalyzer, LANGUAGE_BY_EXTENSION, \
    CLASS_REGEX, FUNCTION_REGEX

logger = logging.getLogger(__name__)

class CodeGraphBuilder:
    """
    Builds and manages a global dependency graph of the codebase.
    """

    GRAPH_FILE = "code_dependency_graph.pkl"

    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.graph = nx.DiGraph()
        self.analyzer = CodebaseAnalyzer()
        self.storage_path = self.root_dir / "data" / self.GRAPH_FILE
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def load_graph(self) -> bool:
        """Loads the graph from disk if it exists."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "rb") as f:
                    self.graph = pickle.load(f)
                logger.info(f"Loaded dependency graph with {self.graph.number_of_nodes()} nodes.")
                return True
            except Exception as e:
                logger.error(f"Failed to load graph: {e}")
        return False

    def save_graph(self):
        """Saves the current graph to disk."""
        try:
            with open(self.storage_path, "wb") as f:
                pickle.dump(self.graph, f)
            logger.info(f"Saved dependency graph with {self.graph.number_of_nodes()} nodes.")
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")

    def build_graph(self):
        """Scans the codebase and rebuilds the graph from scratch."""
        logger.info("Starting full graph build...")
        self.graph = nx.DiGraph()

        # Extensions to scan
        extensions = [k for k in LANGUAGE_BY_EXTENSION.keys()] + ['.sql']

        # 1. First Pass: Create Nodes (Files, Classes, Functions)
        for ext in extensions:
            for file_path in self.root_dir.rglob(f"*{ext}"):
                if any(x in str(file_path) for x in [".venv", "__pycache__", "node_modules", ".git"]):
                    continue

                self._process_file_structure(file_path)

        # 2. Second Pass: Link Dependencies (Imports, Calls)
        # Note: True call graph is hard without full LSP. We use name-matching heuristics.
        self._link_dependencies()

        self.save_graph()
        logger.info("Graph build completed.")

    def _process_file_structure(self, file_path: Path):
        """Extracts structure from a file and adds nodes."""
        rel_path = str(file_path.relative_to(self.root_dir))
        lang = LANGUAGE_BY_EXTENSION.get(file_path.suffix, 'unknown')
        if file_path.suffix == '.sql': lang = 'sql'

        # Add File Node
        self.graph.add_node(rel_path, type="file", language=lang)

        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')

            # Extract Symbols
            if lang == 'python':
                self._extract_python_structure(rel_path, content)
            else:
                self._extract_regex_structure(rel_path, content, lang)

        except Exception as e:
            logger.warning(f"Failed to process {rel_path} for graph: {e}")

    def _extract_python_structure(self, file_node: str, content: str):
        """Uses AST analyzer to add rich nodes."""
        analysis = self.analyzer.analyze_file(content, file_node)
        if "error" in analysis:
            return

        symbols = analysis.get("symbols", [])
        imports = analysis.get("imports", [])

        # Store raw imports on file node for linking later
        self.graph.nodes[file_node]["imports"] = imports

        for sym in symbols:
            # Node ID: file_path::SymbolName
            # (Handling nested classes could be file_path::Parent.Child)
            node_id = f"{file_node}::{sym.name}"
            if sym.parent:
                node_id = f"{file_node}::{sym.parent}.{sym.name}"

            self.graph.add_node(node_id, type=sym.type, name=sym.name, doc=sym.docstring)
            self.graph.add_edge(file_node, node_id, params={}, type="contains")

            # Store dependencies (calls) to link later
            self.graph.nodes[node_id]["pending_deps"] = sym.dependencies

            # Link to parent class/function if applicable
            # (Simple hierarchy for now: File -> Symbol. If Parent exists, maybe File -> Parent -> Child?)
            # For simplicity, we just keep File -> Symbol for now, but if we parsed parent correctly:
            if sym.parent:
                parent_id = f"{file_node}::{sym.parent}"
                if self.graph.has_node(parent_id):
                     self.graph.add_edge(parent_id, node_id, type="contains")

    def _extract_regex_structure(self, file_node: str, content: str, lang: str):
        """Uses Regex fallback."""
        classes = []
        functions = []

        if lang in CLASS_REGEX:
            for match in CLASS_REGEX[lang].finditer(content):
                name = match.group(1)
                node_id = f"{file_node}::{name}"
                self.graph.add_node(node_id, type="class", name=name)
                self.graph.add_edge(file_node, node_id, type="contains")
                classes.append(name)

        if lang in FUNCTION_REGEX:
             for match in FUNCTION_REGEX[lang].finditer(content):
                name = match.group(1)
                node_id = f"{file_node}::{name}"
                self.graph.add_node(node_id, type="function", name=name)
                self.graph.add_edge(file_node, node_id, type="contains")
                functions.append(name)

    def _link_dependencies(self):
        """
        Attempts to link nodes based on imports and name usage.
        Heuristic: If Node A uses name "User", and there is a Node "path/to/User", link them.
        """
        # Create a lookup map: SymbolName -> [List of NodeIDs defining it]
        symbol_map = {}
        for node, data in self.graph.nodes(data=True):
            if data.get("type") in ["class", "function", "method"]:
                name = data.get("name")
                if name:
                    if name not in symbol_map: symbol_map[name] = []
                    symbol_map[name].append(node)

        # Link Python Imports (File Level)
        for node, data in self.graph.nodes(data=True):
            if data.get("type") == "file" and "imports" in data:
                for imp in data["imports"]:
                    # Imprt could be "src.services.file_processor"
                    # We try to match file nodes
                    # Naive match: convert dots to slashes
                    possible_path = imp.replace(".", "/") + ".py"
                    for target_node in self.graph.nodes():
                        if target_node.endswith(possible_path):
                            self.graph.add_edge(node, target_node, type="imports")

        # Link Symbol Dependencies (Calls)
        for node, data in self.graph.nodes(data=True):
            pending_deps = data.get("pending_deps", [])
            for dep_name in pending_deps:
                # If dependencies are explicit imports (e.g. from x import Y), we might have them
                # But typically we just have the name "Y".
                # We check our symbol map.
                if dep_name in symbol_map:
                    targets = symbol_map[dep_name]
                    # Heuristic: If multiple targets, we might link all or try to be smart matching imports.
                    # For now, link all (Caller Graph will be "noisy" but inclusive).
                    for target in targets:
                        # Don't link start node to itself or siblings in same file (optional)
                        if target.split("::")[0] != node.split("::")[0]:
                             self.graph.add_edge(node, target, type="calls")

    # === QUERY METHODS ===

    def get_callers(self, symbol_name: str) -> List[Dict]:
        """Finds who calls/depends on the given symbol."""
        # Find nodes matching the name
        targets = [n for n, d in self.graph.nodes(data=True) if d.get("name") == symbol_name]
        results = []
        for target in targets:
            # Incoming edges
            for src, dst, data in self.graph.in_edges(target, data=True):
                # We want edges that are NOT "contains" (which is just file structure)
                if data.get("type") in ["calls", "imports", "inherits"]:
                    results.append({
                        "source": src,
                        "relationship": data.get("type"),
                        "target": target
                    })
        return results

    def get_dependencies(self, symbol_name: str) -> List[Dict]:
        """Finds what the symbol depends on."""
        sources = [n for n, d in self.graph.nodes(data=True) if d.get("name") == symbol_name]
        results = []
        for source in sources:
            # Outgoing edges
            for src, dst, data in self.graph.out_edges(source, data=True):
                if data.get("type") in ["calls", "imports", "inherits"]:
                    results.append({
                        "source": source,
                        "relationship": data.get("type"),
                        "target": dst
                    })
        return results

    def get_context_subgraph(self, symbol_name: str, depth: int = 1) -> Dict:
        """Returns a subgraph around the symbol for context."""
        centers = [n for n, d in self.graph.nodes(data=True) if d.get("name") == symbol_name]
        if not centers:
            return {"error": "Symbol not found in graph"}

        nodes = set(centers)
        # Traverse simple breadth-first
        current_layer = set(centers)
        for _ in range(depth):
            next_layer = set()
            for node in current_layer:
                # Add neighbors (both directions to see callers and callees)
                for neighbor in self.graph.successors(node):
                    next_layer.add(neighbor)
                for neighbor in self.graph.predecessors(node):
                    next_layer.add(neighbor)
            nodes.update(next_layer)
            current_layer = next_layer

        subgraph = self.graph.subgraph(list(nodes))
        return nx.node_link_data(subgraph)

