# =============================================================================
# src/services/intent/parameter_extractor.py
# Generic Parameter Extraction for Custom Tools
# =============================================================================
"""
ParameterExtractor: Extracción genérica de parámetros para custom tools.

Este módulo complementa a IntentRouter:
- IntentRouter: Clasifica intents predefinidos en INTENT_REGISTRY
- ParameterExtractor: Extrae parámetros para custom tools (no registradas)

Usa los prompts de SystemPrompts para consistencia con el sistema.
"""

import asyncio
import hashlib
import json
import re
from typing import List, Dict, Any, Optional

from src.config.prompts import SystemPrompts
from src.providers.manager import provider_manager, ChatMessage
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ParameterExtractor:
    """
    Extractor genérico de parámetros usando LLM.
    
    Diseñado para custom tools que no están en INTENT_REGISTRY.
    Usa los prompts centralizados de SystemPrompts.
    """
    
    def __init__(self, cache_size: int = 500):
        self.logger = logger
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_size = cache_size
    
    def _cache_key(self, message: str, params: List[Dict]) -> str:
        """Genera clave de cache única."""
        param_sig = tuple((p['name'], p.get('type', 'string')) for p in params)
        combined = f"{message.lower().strip()}:{param_sig}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    async def extract(
        self,
        user_message: str,
        parameters: List[Dict[str, Any]],
        provider: str = "local",
        model: str = "qwen2.5:3b",
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """
        Extrae parámetros de un mensaje de usuario.
        
        Args:
            user_message: Mensaje del usuario
            parameters: Lista de definiciones de parámetros
            provider: Proveedor LLM
            model: Modelo LLM
            timeout: Timeout en segundos
            
        Returns:
            Dict con parámetros extraídos
        """
        if not parameters:
            return {}
        
        # Check cache
        cache_key = self._cache_key(user_message, parameters)
        if cache_key in self._cache:
            self.logger.debug(f"Parameter extraction cache HIT")
            return self._cache[cache_key]
        
        # Build prompts using SystemPrompts
        system_prompt = self._build_system_prompt(parameters)
        user_prompt = SystemPrompts.EXTRACTION_USER_MESSAGE_TEMPLATE.format(
            user_message=user_message,
            expected_keys=", ".join([p.get('name', '') for p in parameters])
        )
        
        self.logger.info(f"Extraction prompt built for {len(parameters)} parameters")
        
        try:
            # Call LLM with timeout
            p = provider_manager.get_provider(provider)
            response = await asyncio.wait_for(
                p.chat(
                    messages=[
                        ChatMessage(role="system", content=system_prompt),
                        ChatMessage(role="user", content=user_prompt)
                    ],
                    model=model,
                    temperature=0.05
                ),
                timeout=timeout
            )
            
            # Log raw response for debugging
            self.logger.info(f"LLM raw response: {response.content[:300]}...")
            
            # Parse response
            extracted = self._parse_json_response(response.content)
            
            # Validate and clean
            validated = self._validate_params(extracted, parameters)
            
            # Cache result
            if validated:
                self._cache[cache_key] = validated
                # LRU eviction
                if len(self._cache) > self._cache_size:
                    self._cache.pop(next(iter(self._cache)))
            
            self.logger.info(f"Extracted parameters: {validated}")
            return validated
            
        except asyncio.TimeoutError:
            self.logger.error(f"Parameter extraction timeout after {timeout}s")
            return {}
        except Exception as e:
            self.logger.error(f"Parameter extraction failed: {e}", exc_info=True)
            return {}
    
    def _build_system_prompt(self, parameters: List[Dict[str, Any]]) -> str:
        """
        Construye el prompt del sistema usando SystemPrompts.
        
        Usa EXTRACTION_GENERIC_TEMPLATE para custom tools (sin mapeo de acciones).
        """
        # Build parameters description
        params_desc = ""
        for p in parameters:
            name = p.get('name', 'unknown')
            ptype = p.get('type', 'string')
            pdesc = p.get('description', SystemPrompts.EXTRACTION_NO_DESCRIPTION)
            
            desc = f"- '{name}' ({ptype}): {pdesc}"
            if p.get('required'):
                desc += SystemPrompts.EXTRACTION_REQUIRED_LABEL
            if 'enum' in p and p['enum']:
                desc += SystemPrompts.EXTRACTION_ALLOWED_VALUES_LABEL.format(enum=p['enum'])
            params_desc += desc + "\n"
        
        # Use the generic template (no action mapping needed for custom tools)
        prompt = SystemPrompts.EXTRACTION_GENERIC_TEMPLATE.format(
            params_desc=params_desc
        )
        
        return prompt
    
    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Robust JSON parsing with regex for LLM outputs."""
        self.logger.debug(f"Raw LLM response for extraction: {content}")
        
        # Clean up markdown code block markers
        cleaned_content = content.strip()
        
        # Remove ```json or ``` markers
        cleaned_content = re.sub(r'^```json?\n', '', cleaned_content, flags=re.MULTILINE)
        cleaned_content = re.sub(r'\n```$', '', cleaned_content)
        
        # Try direct JSON parse first
        try:
            parsed = json.loads(cleaned_content)
            self.logger.debug(f"Direct JSON parse succeeded: {parsed}")
            return parsed
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON object in the content
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned_content)
        if match:
            try:
                parsed = json.loads(match.group(0))
                self.logger.debug(f"Regex JSON parse succeeded: {parsed}")
                return parsed
            except json.JSONDecodeError:
                pass
        
        # Try to extract key-value pairs as last resort
        extracted = {}
        kv_pattern = r'"(\w+)"\s*:\s*"([^"]*)"'
        matches = re.findall(kv_pattern, cleaned_content)
        for key, value in matches:
            extracted[key] = value
        
        if extracted:
            self.logger.info(f"Regex fallback extracted: {extracted}")
            return extracted
        
        self.logger.warning(f"Failed to parse JSON from LLM: {content[:100]}...")
        return {}
    
    def _validate_params(
        self,
        extracted: Dict[str, Any],
        parameters: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Valida y filtra parámetros extraídos."""
        
        param_names = {p['name'] for p in parameters}
        validated = {}
        
        for name, value in extracted.items():
            if name not in param_names:
                self.logger.debug(f"Skipping unknown parameter: {name}")
                continue
            
            # Skip null/empty values (as per SystemPrompts rules)
            if value is None or str(value).lower() in ("none", "null", "undefined", ""):
                self.logger.debug(f"Skipping null/empty value for: {name}")
                continue
            
            validated[name] = value
        
        return validated


# Singleton instance
_extractor: Optional[ParameterExtractor] = None
_extractor_lock = asyncio.Lock()


async def get_parameter_extractor() -> ParameterExtractor:
    """
    Obtiene instancia singleton de ParameterExtractor.
    
    Returns:
        ParameterExtractor instance
    """
    global _extractor
    
    if _extractor is not None:
        return _extractor
    
    async with _extractor_lock:
        if _extractor is None:
            _extractor = ParameterExtractor()
            logger.info("ParameterExtractor singleton initialized")
    
    return _extractor
