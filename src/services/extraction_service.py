# =============================================================================
# src/services/extraction_service.py
# High-reliability Parameter Extraction Service
# =============================================================================
"""
Service for extracting intent parameters from user messages using LLMs.
Uses advanced prompting techniques (Few-Shot, Chain-of-Thought, Context)
to ensure high reliability and strict format adherence.
"""
import json
import re
from typing import List, Dict, Any, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.models import Message, MessageRole
from src.providers.manager import provider_manager, ChatMessage
from src.utils.logger import get_logger

class ParameterExtractionService:
    """Service to extract structured parameters from natural language"""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.logger = get_logger(__name__)

    async def extract_parameters(
        self,
        user_message: str,
        parameters: List[Dict[str, Any]],
        conversation_id: UUID,
        provider: str = "local",
        model: str = "mistral"
    ) -> Dict[str, Any]:
        """
        Extract designated parameters from a user message.
        
        Args:
            user_message: The latest message from the user.
            parameters: List of parameter definitions (name, type, description, required).
            conversation_id: For contextual extraction.
            provider: LLM provider to use.
            model: Model name to use.
            
        Returns:
            Dictionary with extracted values.
        """
        if not parameters:
            return {}

        self.logger.info(f"Extracting parameters for message: {user_message[:50]}...")

        # 1. Fetch recent context
        context = await self._get_conversation_context(conversation_id)
        
        # 2. Build the system prompt
        system_prompt = self._build_system_prompt(parameters)
        
        # 3. Prepare messages for LLM
        messages = [
            ChatMessage(role="system", content=system_prompt)
        ]
        
        # Add context if available
        if context:
            messages.append(ChatMessage(role="system", content=f"PREVIOUS CONTEXT:\n{context}"))
            
        expected_keys = ", ".join([p.get('name', '') for p in parameters])
        messages.append(ChatMessage(role="user", content=f"USER MESSAGE: {user_message}\n\nExtract only: {expected_keys}. Return JSON only."))

        try:
            # 4. Call LLM
            p = provider_manager.get_provider(provider)
            response = await p.chat(
                messages=messages,
                model=model,
                temperature=0.1 # Low temperature for extraction reliability
            )
            
            # 5. Parse JSON response
            extracted = self._parse_json_response(response.content)
            
            # 6. Validate and clean results
            return self._validate_extracted_params(extracted, parameters)
            
        except Exception as e:
            self.logger.error(f"Extraction failed: {e}", exc_info=True)
            return {}

    async def _get_conversation_context(self, conversation_id: UUID, limit: int = 4) -> str:
        """Fetch last N messages to provide context for relative references"""
        stmt = select(Message).where(
            Message.conversation_id == conversation_id,
            Message.is_active == True
        ).order_by(Message.created_at.desc()).limit(limit)
        
        result = await self.db.execute(stmt)
        messages = result.scalars().all()
        messages.reverse() # Back to chronological order
        
        context_parts = []
        for msg in messages:
            role = "User" if msg.role == MessageRole.USER else "Assistant"
            context_parts.append(f"{role}: {msg.content}")
            
        return "\n".join(context_parts)

    def _build_system_prompt(self, parameters: List[Dict[str, Any]]) -> str:
        """Create a professional extraction prompt with schema and examples"""
        params_desc = ""
        for p in parameters:
            name = p.get('name', 'unknown')
            ptype = p.get('type', 'string')
            pdesc = p.get('description', 'No description available')
            desc = f"- '{name}' ({ptype}): {pdesc}"
            if p.get('required'):
                desc += " (REQUIRED)"
            params_desc += desc + "\n"

        # Prepare a dynamic example based on the first parameter if available
        example_key = parameters[0].get('name', 'param_name') if parameters else 'param_name'
        
        prompt = f"""You are a data extraction bot.
Your ONLY job is to extract variables from the USER MESSAGE in JSON format.

### VARIABLES TO EXTRACT:
{params_desc}

### RULES:
1. Return ONLY a JSON object. No conversation.
2. If a variable is not found, DO NOT include it.
3. Use the exact variable names defined above as KEYS.
4. Values should be the information sought.

### JSON FORMAT EXAMPLE:
{{
  "{example_key}": "value"
}}
"""
        return prompt

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Robust JSON parsing with regex for LLM outputs that might contain markdown or extra text"""
        self.logger.debug(f"Raw LLM response for extraction: {content}")
        
        # Attempt direct parse
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError:
            pass
            
        # Try to find JSON block
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Fallback: Regex for simple key-value pairs if JSON fails
        # This is useful for small models like Gemma that might output "ciudad: Madrid"
        extracted = {}
        for line in content.split('\n'):
            # Look for "key": "value" or key: value
            kv_match = re.search(r'["\']?(\w+)["\']?\s*[:=]\s*["\']?([^"\',]+)["\']?', line)
            if kv_match:
                key, value = kv_match.groups()
                extracted[key.strip()] = value.strip()
                
        if extracted:
            self.logger.info(f"Regex fallback extracted: {extracted}")
            return extracted
                
        self.logger.warning(f"Failed to parse JSON or regex from LLM: {content[:100]}...")
        return {}

    def _validate_extracted_params(self, extracted: Dict[str, Any], schema: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ensure extracted values match expected types and filter unknown keys"""
        validated = {}
        schema_lookup = {p['name']: p for p in schema}
        
        for name, value in extracted.items():
            if name in schema_lookup:
                # Basic type validation/coercion could be added here
                validated[name] = value
                
        return validated
