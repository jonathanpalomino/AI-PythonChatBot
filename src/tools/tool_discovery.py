# =============================================================================
# src/tools/tool_discovery.py
# Tool Discovery System
# =============================================================================
"""
Sistema para descubrir automáticamente herramientas físicas disponibles
"""

import importlib
import inspect
import pkgutil
from typing import Dict, List, Type, Any, Optional
from src.tools.base_tool import BaseTool, ToolCategory
from src.utils.logger import get_logger


logger = get_logger(__name__)


class ToolDiscovery:
    """Sistema de descubrimiento automático de herramientas físicas"""
    
    def __init__(self):
        self._discovered_tools: Dict[str, Type[BaseTool]] = {}
        self._tool_metadata: Dict[str, Dict[str, Any]] = {}
    
    def discover_tools(self, package_path: str = "src.tools") -> Dict[str, Type[BaseTool]]:
        """
        Descubrir automáticamente todas las herramientas físicas disponibles
        
        Args:
            package_path: Ruta del paquete donde buscar herramientas
            
        Returns:
            Diccionario con herramientas descubiertas (nombre -> clase)
        """
        try:
            # Importar el paquete
            package = importlib.import_module(package_path)
            
            # Descubrir todos los módulos en el paquete
            discovered_tools = {}
            
            for importer, modname, ispkg in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
                if ispkg:
                    continue  # Saltar subpaquetes
                
                try:
                    # Importar el módulo
                    module = importlib.import_module(modname)
                    
                    # Buscar clases que hereden de BaseTool
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (obj != BaseTool and 
                            issubclass(obj, BaseTool) and 
                            obj.__module__ == module.__name__):
                            
                            # Validar que tenga los atributos necesarios
                            if self._is_valid_tool_class(obj):
                                tool_name = obj.name if hasattr(obj, 'name') else name
                                discovered_tools[tool_name] = obj
            
                                # Crear instancia para obtener metadata
                                instance = obj()
            
                                # Guardar metadata
                                self._tool_metadata[tool_name] = {
                                    'class_name': name,
                                    'module': modname,
                                    'description': getattr(instance, 'description', ''),
                                    'category': instance.category.value,
                                    'enabled_by_default': getattr(instance, 'enabled_by_default', False)
                                }
            
                                logger.info(
                                    f"Tool discovered: {tool_name} ({name})",
                                    extra={
                                        'class_name': name,
                                        'tool_module': modname,
                                        'category': self._tool_metadata[tool_name]['category']
                                    }
                                )
                                
                except Exception as e:
                    logger.warning(
                        f"Failed to import module {modname}: {e}",
                        extra={'tool_module': modname}
                    )
                    continue
            
            self._discovered_tools = discovered_tools
            logger.info(
                f"Tool discovery completed: {len(discovered_tools)} tools found",
                extra={'tools': list(discovered_tools.keys())}
            )
            
            return discovered_tools
            
        except Exception as e:
            logger.error(f"Tool discovery failed: {e}", exc_info=True)
            return {}
    
    def get_discovered_tools(self) -> Dict[str, Type[BaseTool]]:
        """Obtener herramientas descubiertas"""
        return self._discovered_tools.copy()
    
    def get_tool_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Obtener metadata de herramientas descubiertas"""
        return self._tool_metadata.copy()
    
    def get_tool_class(self, tool_name: str) -> Optional[Type[BaseTool]]:
        """Obtener clase de herramienta por nombre"""
        return self._discovered_tools.get(tool_name)
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Obtener información detallada de una herramienta"""
        if tool_name in self._tool_metadata:
            metadata = self._tool_metadata[tool_name].copy()
            tool_class = self._discovered_tools.get(tool_name)
            if tool_class:
                metadata['parameters'] = [
                    {
                        'name': p.name,
                        'type': p.type,
                        'description': p.description,
                        'required': p.required,
                        'default': p.default,
                        'enum': p.enum
                    }
                    for p in tool_class().get_parameters()
                ]
            return metadata
        return None
    
    def _is_valid_tool_class(self, cls: Type[BaseTool]) -> bool:
        """Validar que una clase sea una herramienta válida"""
        try:
            # Verificar que tenga los atributos requeridos
            required_attrs = ['name', 'description', 'category']
            for attr in required_attrs:
                if not hasattr(cls, attr):
                    logger.warning(f"Tool class {cls.__name__} missing required attribute: {attr}")
                    return False
            
            # Verificar que tenga método get_parameters
            if not hasattr(cls, 'get_parameters'):
                logger.warning(f"Tool class {cls.__name__} missing get_parameters method")
                return False
            
            # Intentar instanciar la herramienta para validar que no tenga errores
            instance = cls()
            if not isinstance(instance, BaseTool):
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Tool class {cls.__name__} validation failed: {e}")
            return False
    
    def print_discovered_tools(self):
        """Imprimir información de herramientas descubiertas"""
        if not self._discovered_tools:
            print("No tools discovered")
            return
        
        print(f"\n=== Discovered Tools ({len(self._discovered_tools)}) ===")
        for tool_name, tool_class in self._discovered_tools.items():
            metadata = self._tool_metadata.get(tool_name, {})
            print(f"\n• {tool_name}")
            print(f"  Class: {metadata.get('class_name', 'Unknown')}")
            print(f"  Module: {metadata.get('module', 'Unknown')}")
            print(f"  Category: {metadata.get('category', 'Unknown')}")
            print(f"  Enabled by default: {metadata.get('enabled_by_default', False)}")
            print(f"  Description: {metadata.get('description', 'No description')}")


# Instancia global del sistema de descubrimiento
tool_discovery = ToolDiscovery()


def discover_all_tools() -> Dict[str, Type[BaseTool]]:
    """Función de conveniencia para descubrir todas las herramientas"""
    return tool_discovery.discover_tools()


def get_tool_class(tool_name: str) -> Optional[Type[BaseTool]]:
    """Función de conveniencia para obtener una clase de herramienta"""
    return tool_discovery.get_tool_class(tool_name)


def get_all_tool_info() -> Dict[str, Dict[str, Any]]:
    """Función de conveniencia para obtener información de todas las herramientas"""
    info = {}
    for tool_name in tool_discovery.get_discovered_tools().keys():
        tool_info = tool_discovery.get_tool_info(tool_name)
        if tool_info:
            info[tool_name] = tool_info
    return info