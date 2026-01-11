# =============================================================================
# src/tools/custom_tool.py
# Custom Tool Implementation
# =============================================================================
"""
Tool para ejecutar herramientas personalizadas definidas por el usuario
"""

from typing import List, Dict, Any, Optional, get_type_hints
import inspect
import re
from uuid import UUID

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.settings import settings
from src.models.models import CustomTool
from src.tools.base_tool import BaseTool, ToolCategory, ToolParameter, ToolResult
from src.utils.logger import get_logger


class  CustomToolExecutor(BaseTool):
    """Tool para ejecutar herramientas personalizadas"""

    def __init__(self, custom_tool_id: UUID, db: Optional[AsyncSession] = None):
        self.custom_tool_id = custom_tool_id
        self.db = db
        self.logger = get_logger(__name__)
        self._custom_tool_config = None
        
        # Importar sistema de descubrimiento de herramientas
        from src.tools.tool_discovery import tool_discovery
        
        # Asegurarse de que las herramientas estén descubiertas
        if not tool_discovery.get_discovered_tools():
            tool_discovery.discover_tools()
            
        super().__init__()

    async def _load_custom_tool_config(self) -> CustomTool:
        """Cargar la configuración de la herramienta personalizada desde la base de datos"""
        if self._custom_tool_config:
            return self._custom_tool_config

        if not self.db:
            raise ValueError("Database session is required to load custom tool configuration")

        from sqlalchemy import select
        result = await self.db.execute(
            select(CustomTool).filter(CustomTool.id == self.custom_tool_id)
        )
        custom_tool = result.scalars().first()

        if not custom_tool:
            raise ValueError(f"Custom tool with id {self.custom_tool_id} not found")

        self._custom_tool_config = custom_tool
        return custom_tool

    # =============================================================================
    # Tool Definition
    # =============================================================================

    @property
    def name(self) -> str:
        # Use overridden name if available, otherwise use ID-based name
        if hasattr(self, '_name') and self._name:
            return self._name
        return f"custom_tool_{self.custom_tool_id}"

    @property
    def tool_type(self) -> str:
        """Obtener el tipo de herramienta física subyacente"""
        if self._custom_tool_config:
            return self._custom_tool_config.tool_type
        return "unknown"

    @property
    def description(self) -> str:
        config = self._custom_tool_config
        if config:
            return config.description or f"Custom tool for {config.name}"
        return "Custom tool"

    @property
    def category(self) -> ToolCategory:
        # Default category
        default_cat = ToolCategory.UTILITY
        
        # Try to determine from configuration if loaded
        if self._custom_tool_config:
            tool_type = self._custom_tool_config.tool_type
            if tool_type == "rag_search":
                return ToolCategory.RAG
            elif tool_type == "http_request":
                return ToolCategory.WEB
            elif tool_type == "sql_query":
                return ToolCategory.UTILITY
        
        return default_cat

    def get_parameters(self) -> List[ToolParameter]:
        """Obtener los parámetros definidos para la herramienta personalizada"""
        if not self._custom_tool_config:
            return []

        parameters = []
        # Use configuration field instead of parameters
        config = self._custom_tool_config.configuration or {}
        
        # Extract parameters from configuration if available
        if "parameters" in config:
            for param in config["parameters"]:
                parameters.append(
                    ToolParameter(
                        name=param.get("name", ""),
                        type=param.get("type", "string"),
                        description=param.get("description", ""),
                        required=param.get("required", True),
                        default=param.get("default"),
                        enum=param.get("enum")
                    )
                )
        
        
        # 2. Add automatically discovered parameters from templates if they aren't explicitly defined
        explicit_names = {p.name for p in parameters}
        discovered_names = self._find_template_tags(config)
        
        for name in discovered_names:
            if name not in explicit_names:
                parameters.append(
                    ToolParameter(
                        name=name,
                        type="string",
                        description=f"Parameter detected in tool configuration: {name}",
                        required=True
                    )
                )
        
        return parameters

    def _find_template_tags(self, obj: Any) -> set:
        """Recursively find all unique {{tags}} in a configuration object"""
        tags = set()
        if isinstance(obj, str):
            matches = re.findall(r'\{\{([^}]+)\}\}', obj)
            for m in matches:
                tags.add(m.strip())
        elif isinstance(obj, list):
            for item in obj:
                tags.update(self._find_template_tags(item))
        elif isinstance(obj, dict):
            for v in obj.values():
                tags.update(self._find_template_tags(v))
        return tags

    # =============================================================================
    # Execution
    # =============================================================================

    async def execute(self, **kwargs) -> ToolResult:
        """Ejecutar la herramienta personalizada delegando a la herramienta física correspondiente"""
        try:
            # Cargar configuración
            config = await self._load_custom_tool_config()

            # Resolve parameter aliases before validation
            # (e.g., if LLM extracts 'nombre_ciudad' but template needs 'nom_ciudad')
            self._resolve_parameter_aliases(config, kwargs)

            # Validar entrada
            await self.validate_input(**kwargs)

            # Delegar a la herramienta física según el tipo
            return await self._execute_physical_tool(config, **kwargs)

        except Exception as e:
            self.logger.error(
                f"Custom tool execution error: {e}",
                exc_info=True,
                extra={"tool_id": str(self.custom_tool_id)}
            )
            return ToolResult(
                success=False,
                data=None,
                error=str(e)
            )

    async def _execute_physical_tool(self, config: CustomTool, **kwargs) -> ToolResult:
        """Ejecutar la herramienta física correspondiente al tipo de herramienta"""
        try:
            # Obtener la herramienta física desde el registro de herramientas o descubrimiento automático
            physical_tool = self._get_physical_tool_from_registry(config.tool_type)
            
            if physical_tool is None:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Unsupported tool type: {config.tool_type}"
                )
            
            # Crear una nueva instancia para evitar problemas de estado compartido
            physical_tool_class = physical_tool.__class__
            physical_tool = physical_tool_class()
            class_name = physical_tool_class.__name__
            
            # Obtener la configuración de la herramienta
            tool_config = config.configuration or {}
            
            # Interpolate variables in the configuration using kwargs (LLM provided params)
            interpolated_config = self._interpolate_value(tool_config, kwargs)
            
            # Combinar configuración interpolada con parámetros de entrada
            # La configuración interpolada tiene PRECEDENCIA sobre kwargs para evitar que
            # alias o parámetros accidentales sobreescriban la URL u otros campos críticos del template.
            execution_params = {**kwargs, **interpolated_config}
            
            # Validar que la herramienta física tenga el método execute
            if not hasattr(physical_tool, 'execute'):
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Physical tool {class_name} does not have execute method"
                )
            
            # Filtrar parámetros para que solo se pasen los que la herramienta física acepta
            filtered_params = self._filter_valid_parameters(
                physical_tool.execute,
                execution_params
            )
            
            # Ejecutar la herramienta física con los parámetros filtrados
            self.logger.debug(f"Ejecutando {class_name} con parámetros: {list(filtered_params.keys())}")
            if "collections" in filtered_params:
                self.logger.info(f"Custom tool '{config.name}' usando colecciones: {filtered_params['collections']}")
                
            result = await physical_tool.execute(**filtered_params)
            
            # Registrar resultado
            if result.success:
                self.logger.info(
                    f"Physical tool executed successfully via custom tool",
                    extra={
                        "tool_id": str(self.custom_tool_id),
                        "tool_name": config.name,
                        "physical_tool": class_name,
                        "tool_type": config.tool_type
                    }
                )
            else:
                self.logger.error(
                    f"Physical tool execution failed via custom tool",
                    extra={
                        "tool_id": str(self.custom_tool_id),
                        "tool_name": config.name,
                        "physical_tool": class_name,
                        "tool_type": config.tool_type,
                        "error": result.error
                    }
                )
            
            return result
            
        except Exception as e:
            self.logger.error(
                f"Physical tool execution error: {e}",
                exc_info=True,
                extra={
                    "tool_id": str(self.custom_tool_id),
                    "tool_type": config.tool_type
                }
            )
            return ToolResult(
                success=False,
                data=None,
                error=str(e)
            )

    def _resolve_parameter_aliases(self, config: CustomTool, kwargs: Dict[str, Any]) -> None:
        """
        Attempt to resolve missing template variables from other provided parameters.
        支持双向映射 (Bidirectional Mapping):
        1. Si se provee la llave (k) pero falta el tag ({{tag}}): tag = k
        2. Si se provee el tag ({{tag}}) pero falta la llave (k): k = tag
        """
        tool_config = config.configuration or {}
        self.logger.debug(f"Attempting bidirectional parameter resolution for {self.name}...")
        
        # Scan recursively for mappings in the config
        def find_mappings(obj):
            mappings = {}
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, str):
                        # Find tags in the value like {{tag}}
                        matches = re.findall(r'\{\{([^}]+)\}\}', v)
                        # Solo aplicamos mapeo automático si hay exactamente un tag en el valor
                        # para evitar ambigüedades en strings complejos como "/api/{{v1}}/{{v2}}"
                        if len(matches) == 1 and v.strip() == f"{{{{{matches[0].strip()}}}}}":
                            tag = matches[0].strip()
                            
                            # Logica Bidireccional:
                            # 1. Caso 'nombre_ciudad' -> '{{nom_ciudad}}'
                            # LLM extrajo el nombre de la llave técnica, llenamos la variable
                            if k in kwargs and tag not in kwargs:
                                mappings[tag] = kwargs[k]
                                
                            # 2. Caso 'll_param_idx' <- '{{nom_ciudad}}' (Stress Test)
                            # LLM extrajo el tag semántico, llenamos la llave técnica esperada por el endpoint
                            elif tag in kwargs and k not in kwargs:
                                mappings[k] = kwargs[tag]
                                
                    # Recurse into nested structures
                    mappings.update(find_mappings(v))
            elif isinstance(obj, list):
                for item in obj:
                    mappings.update(find_mappings(item))
            return mappings

        new_mappings = find_mappings(tool_config)
        if new_mappings:
            self.logger.info(f"Bidirectional aliasing for {self.name}: {new_mappings}")
            kwargs.update(new_mappings)

    def _interpolate_value(self, value: Any, variables: Dict[str, Any]) -> Any:
        """
        Recursively interpolate variables in strings within lists or dicts.
        Uses {{variable_name}} syntax.
        """
        if isinstance(value, str):
            # perform interpolation
            result = value
            for var_name, var_value in variables.items():
                placeholder = f"{{{{{var_name}}}}}"
                if placeholder in result:
                    # Convert non-string values to string for replacement
                    str_value = str(var_value) if not isinstance(var_value, str) else var_value
                    result = result.replace(placeholder, str_value)
            return result
        elif isinstance(value, list):
            return [self._interpolate_value(item, variables) for item in value]
        elif isinstance(value, dict):
            return {k: self._interpolate_value(v, variables) for k, v in value.items()}
        return value

    def _get_physical_tool_from_registry(self, tool_type: str):
        """
        Obtener la herramienta física desde el registro de herramientas.
        Busca herramientas cuyo nombre coincida con el tool_type.
        """
        from src.tools.base_tool import tool_registry
        from src.tools.tool_discovery import tool_discovery
        
        # Primero intentar desde el registro de herramientas
        physical_tool = tool_registry.get(tool_type)
        if physical_tool:
            return physical_tool
        
        # Si no está en el registro, intentar desde el sistema de descubrimiento
        tool_class = tool_discovery.get_tool_class(tool_type)
        if tool_class:
            # Crear una instancia temporal para obtener una herramienta registrable
            try:
                instance = tool_class()
                return instance
            except Exception as e:
                self.logger.warning(f"Failed to instantiate discovered tool {tool_type}: {e}")
        
        # Si no se encuentra, intentar mapeo de legacy (para compatibilidad)
        legacy_mapping = {
            "http_request": "http_request",
            "sql_query": "sql_query",
            "rag_search": "rag_search"
        }
        
        if tool_type in legacy_mapping:
            mapped_name = legacy_mapping[tool_type]
            physical_tool = tool_registry.get(mapped_name)
            if physical_tool:
                return physical_tool
        
        return None

    def _filter_valid_parameters(self, method, params: Dict[str, Any]) -> Dict[str, Any]:
        """Filter parameters to only include those accepted by the method"""
        try:
            # Get the signature of the method
            sig = inspect.signature(method)
            type_hints = get_type_hints(method)
            
            # Get all parameter names from the method signature
            valid_params = {}
            for param_name, param_value in params.items():
                if param_name in sig.parameters:
                    valid_params[param_name] = param_value
            
            return valid_params
            
        except Exception as e:
            # If there's an error in filtering, log it but return all params
            self.logger.warning(
                f"Error filtering parameters: {e}. Passing all parameters.",
                extra={"error": str(e)}
            )
            return params

    def _is_valid_url(self, url: str) -> bool:
        """Validar que la URL sea segura y bien formada"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            
            # Validar que tenga esquema y dominio
            if not parsed.scheme or not parsed.netloc:
                return False
            
            # Validar que el esquema sea HTTP o HTTPS
            if parsed.scheme not in ["http", "https"]:
                return False
            
            # Validar que no sea una URL local o privada
            if parsed.netloc.startswith("localhost") or parsed.netloc.startswith("127.0.0.1"):
                return False
            
            return True
        except Exception:
            return False