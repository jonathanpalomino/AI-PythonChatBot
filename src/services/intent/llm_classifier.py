# src/services/intent/llm_classifier.py

import json
from typing import Optional, Dict, Any

import httpx

from src.services.intent.config import INTENT_REGISTRY
from src.utils.logger import get_logger

logger = get_logger(__name__)

_llm_classifier_instance: Optional["LLMClassifier"] = None


def build_classification_prompt(user_message: str) -> str:
    intents_description = "\n".join([
        f"- {name}: {intent.description}"
        for name, intent in INTENT_REGISTRY.items()
    ])
    return f"""Analiza el mensaje y responde con JSON estricto.

Intents disponibles:
{intents_description}

Mensaje: "{user_message}"

Responde SOLO con JSON:
{{"intent": "<nombre>", "target": "<símbolo_o_null>", "confidence": <0.0-1.0>}}"""


class LLMClassifier:
    """Clasificador de intents basado en LLM. Fallback del IntentRouter por embeddings."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.logger = get_logger(__name__)

    async def classify(
        self,
        user_message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Clasifica el intent usando el LLM configurado.

        Returns:
            Dict con keys: intent, target, confidence
        """
        effective_model = model or "qwen2.5:3b"
        prompt = build_classification_prompt(user_message)

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": effective_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "options": {"temperature": 0.0},
                        "format": "json",
                        "stream": False
                    }
                )
                response.raise_for_status()
                content = response.json()["message"]["content"]
                result = json.loads(content)

                self.logger.info(
                    f"LLMClassifier result: intent={result.get('intent')}, "
                    f"target={result.get('target')}, "
                    f"confidence={result.get('confidence')}"
                )
                return result

        except Exception as e:
            self.logger.error(f"LLMClassifier failed: {e}")
            return {"intent": "rag_search", "target": None, "confidence": 0.0}


async def get_llm_classifier() -> LLMClassifier:
    """Singleton lazy loader."""
    global _llm_classifier_instance
    if _llm_classifier_instance is None:
        _llm_classifier_instance = LLMClassifier()
    return _llm_classifier_instance
