import json
import re
import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, AsyncGenerator, Iterator, Union

from pydantic import BaseModel
from litellm import acompletion
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from ollama import Client as OllamaClient

from src.config.settings import settings
from src.providers.cancellable_stream import CancellableProviderMixin
from src.utils.date_utils import get_current_utc
from src.utils.logger import get_logger, get_payload_request_logger, get_payload_response_logger
from src.models.llm_models import LLMModel

logger = get_logger(__name__)
payload_request_logger = get_payload_request_logger()
payload_response_logger = get_payload_response_logger()

if settings.OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
if settings.ANTHROPIC_API_KEY:
    os.environ["ANTHROPIC_API_KEY"] = settings.ANTHROPIC_API_KEY
if settings.OPENROUTER_API_KEY:
    os.environ["OPENROUTER_API_KEY"] = settings.OPENROUTER_API_KEY
if settings.GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY

class ProviderType(str, Enum):
    """LLM provider types"""
    LOCAL = "local"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OPENROUTER = "openrouter"
    GROQ = "groq"
    WEB_EMULATOR = "web_emulator"
    CUSTOM = "custom"


class ModelType(str, Enum):
    """Model capability types"""
    CHAT = "chat"  # Standard chat/completion models
    EMBEDDING = "embedding"  # Embedding-only models
    MULTIMODAL = "multimodal"  # Models that support images/video
    VISION = "vision"  # Vision-specific models
    REASONING = "reasoning"  # Reasoning/Thinking models (e.g. o1, r1)
    CODE = "code"  # Code-specific models
    GENERAL = "general"  # General models (e.g. o1, r1)
    OTHER = "other"  # Other specialized models


@dataclass
class ModelInfo:
    """Model information"""
    name: str
    provider: ProviderType
    context_window: int
    supports_function_calling: bool
    supports_streaming: bool
    model_type: ModelType = ModelType.CHAT  # Default to chat
    # Database-sourced attributes for frontend filtering
    supports_thinking: bool = False  # Can emit <think> tags (gemma3, qwen3, deepseek-r1)
    is_active: bool = True  # Model is available for use
    is_custom: bool = False  # Manually added by user (not from provider)
    cost_per_1k_input: Optional[float] = None
    cost_per_1k_output: Optional[float] = None
    # Pricing details (optional, for display/logic)
    is_free: bool = False
    pricing: Optional[Dict[str, Any]] = None  # Raw pricing object if available
    # Hardware requirements and capabilities
    cpu_supported: bool = True  # Can run on CPU
    gpu_required: bool = False  # Requires GPU
    parent_retrieval_supported: bool = True  # Supports parent document retrieval
    # False = provider gestiona historial externamente (ej. pestaña de browser)
    supports_message_history: bool = True


@dataclass
class ChatMessage:
    """Standard chat message format"""
    role: str  # "user", "assistant", "system"
    content: Union[str, List[Dict[str, Any]]]  # Text or multimodal content blocks


@dataclass
class ChatResponse:
    """Standard chat response"""
    content: str
    model: str
    provider: str
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    tool_calls: Optional[List[Dict]] = None
    finish_reason: Optional[str] = None
    thinking_content: Optional[str] = None  # Content from thinking field/tags
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# =============================================================================
# Base Provider
# =============================================================================

class BaseProvider(ABC):
    """Base class for all LLM providers"""

    def __init__(self):
        # self.provider_type: ProviderType = None
        if getattr(self, "provider_type", None) is None:
            raise ValueError("provider_type must be set by subclass before BaseProvider.__init__")
        self._validate_credentials()

    @abstractmethod
    def _validate_credentials(self):
        """Validate provider credentials"""
        pass

    @abstractmethod
    async def chat(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> ChatResponse:
        """Send chat request"""
        pass

    @abstractmethod
    async def stream_chat(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream chat response"""
        pass

    @abstractmethod
    def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models"""
        pass

    def calculate_cost(self, tokens_input: int, tokens_output: int, model_info: ModelInfo) -> float:
        """Calculate cost based on token usage"""
        if model_info.cost_per_1k_input is None:
            return 0.0

        cost_input = (tokens_input / 1000) * model_info.cost_per_1k_input
        cost_output = (tokens_output / 1000) * model_info.cost_per_1k_output
        return cost_input + cost_output


class CustomProviderConfig(BaseModel):
    name: str
    base_url: str
    model_prefix: str
    api_key_env: Optional[str] = None


class LiteLLMProviderBase(BaseProvider, CancellableProviderMixin):
    """
    Clase base que utiliza LiteLLM para manejar la lógica de chat y stream_chat.
    Las subclases solo necesitan definir el prefijo del proveedor ("ollama/", "groq/", etc.)
    y cómo obtener la lista de modelos.
    """

    def __init__(self, provider_type: ProviderType, litellm_prefix: str = ""):
        self.provider_type = provider_type
        self.litellm_prefix = litellm_prefix
        super().__init__()

    def _validate_credentials(self):
        """La validación se delega a LiteLLM o a la configuración de entorno previa"""
        pass

    def _format_messages(self, messages: List[ChatMessage]) -> List[Dict]:
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    def _get_request_params(
        self,
        model: str,
        messages: List[Dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict:
        """Build request parameters based on provider type."""
        litellm_model = f"{self.litellm_prefix}{model}" if self.litellm_prefix else model

        request_params = {
            "model": litellm_model,
            "messages": messages,
        }

        if stream:
            request_params["stream"] = True

        if temperature is not None:
            request_params["temperature"] = temperature
        if max_tokens is not None:
            request_params["max_tokens"] = max_tokens
        if tools is not None:
            request_params["tools"] = tools
            request_params["tool_choice"] = kwargs.get("tool_choice", "auto")

        # Pasamos parámetros adicionales que LiteLLM soporta
        if self.provider_type == ProviderType.LOCAL:
            if "num_ctx" in kwargs:
                request_params["num_ctx"] = kwargs["num_ctx"]
            request_params["api_base"] = settings.OLLAMA_BASE_URL
        elif self.provider_type == ProviderType.OPENROUTER:
            request_params["extra_headers"] = {
                "HTTP-Referer": "https://github.com/your-repo",
                "X-Title": settings.APP_NAME,
            }
        elif self.provider_type == ProviderType.CUSTOM:
            if hasattr(self, "config"):
                request_params["api_base"] = self.config.base_url

        return request_params

    async def chat(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> ChatResponse:

        formatted_messages = self._format_messages(messages)
        request_params = self._get_request_params(
            model=model,
            messages=formatted_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            stream=False,
            **kwargs
        )

        payload_request_logger.info(f"LiteLLM Request ({self.provider_type.value}): {json.dumps(request_params, indent=2, default=str)}")

        response = await acompletion(**request_params)

        try:
            res_dict = response.model_dump()
            payload_response_logger.info(f"LiteLLM Response ({self.provider_type.value}): {json.dumps(res_dict, indent=2, default=str)}")
        except Exception as e:
            payload_response_logger.info(f"LiteLLM Response (non-serializable): {str(e)}")

        message = response.choices[0].message
        raw_content = message.content or ""

        thinking_content = None
        final_content = raw_content

        # Procesamiento de think tags: formato chino (o estándar XML) y formato OpenAI (json field)
        think_pattern = re.compile(r'<(?:think|思考)>(.*?)</(?:think|思考)>', re.DOTALL | re.IGNORECASE)
        # Algunos modelos pueden retornar "思考\n..." en vez de XML.
        chinese_pattern = re.compile(r'(?:思考|think)\n(.*?)(?:```|$)', re.DOTALL | re.IGNORECASE)

        think_matches = think_pattern.findall(raw_content)
        if think_matches:
            thinking_content = "\n".join(think_matches)
            final_content = think_pattern.sub('', raw_content).strip()
        else:
            chinese_matches = chinese_pattern.findall(raw_content)
            if chinese_matches and ("```" in raw_content or "\n" in raw_content):
                # Extraemos el thinking de formato alternativo
                thinking_content = "\n".join(chinese_matches)
                final_content = chinese_pattern.sub('', raw_content).strip()
                # Remove left over "思考" or "think" si no fue capturado bien
                final_content = re.sub(r'^(思考|think)\n', '', final_content, flags=re.IGNORECASE).strip()

        # Handle Format 2: OpenAI thinking field o JSON de litellm
        if not thinking_content and raw_content.startswith("["):
            try:
                parsed_content = json.loads(raw_content)
                if isinstance(parsed_content, list):
                    think_parts = []
                    text_parts = []
                    for item in parsed_content:
                        if item.get("type") == "thinking" and "thinking" in item:
                            think_parts.append(item["thinking"])
                        elif item.get("type") == "text" and "content" in item:
                            text_parts.append(item["content"])
                    if think_parts:
                        thinking_content = "\n".join(think_parts)
                        final_content = "".join(text_parts).strip()
            except json.JSONDecodeError:
                pass

        if hasattr(message, "thinking") and message.thinking:
            if thinking_content:
                thinking_content += "\n" + message.thinking
            else:
                thinking_content = message.thinking

        # Extraer tools
        tool_calls = None
        if hasattr(message, "tool_calls") and message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
                for tc in message.tool_calls
            ]

        usage = response.usage

        return ChatResponse(
            content=final_content,
            model=model,
            provider=self.provider_type.value,
            tokens_used=usage.total_tokens if usage else 0,
            tool_calls=tool_calls,
            finish_reason=response.choices[0].finish_reason if response.choices else None,
            thinking_content=thinking_content,
            metadata={
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "thinking_content": thinking_content
            }
        )

    def _process_streaming_thinking(self, delta_content: str, in_think_block: bool) -> Iterator[Dict]:
        """Process streaming content and detect thinking tags."""

        if not in_think_block:
            if "<think>" in delta_content:
                parts = delta_content.split("<think>")
                if parts[0].strip():
                    yield {"type": "content", "chunk": parts[0]}
                delta_content = parts[1] if len(parts) > 1 else ""
                in_think_block = True
            elif "思考\n" in delta_content:
                parts = delta_content.split("思考\n", 1)
                if parts[0].strip():
                    yield {"type": "content", "chunk": parts[0]}
                delta_content = parts[1] if len(parts) > 1 else ""
                in_think_block = True

        if in_think_block:
            if "</think>" in delta_content:
                parts = delta_content.split("</think>")
                yield {"type": "thinking", "content": parts[0]}
                in_think_block = False
                if len(parts) > 1 and parts[1]:
                    yield {"type": "content", "chunk": parts[1]}
            elif "```" in delta_content and "思考" not in delta_content:
                parts = delta_content.split("```", 1)
                yield {"type": "thinking", "content": parts[0]}
                in_think_block = False
                yield {"type": "content", "chunk": f"```{parts[1]}" if len(parts) > 1 else "```"}
            else:
                if delta_content:
                    yield {"type": "thinking", "content": delta_content}
        else:
            if delta_content:
                yield {"type": "content", "chunk": delta_content}

        yield {"in_think_block": in_think_block}

    async def stream_chat(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:

        formatted_messages = self._format_messages(messages)
        request_params = self._get_request_params(
            model=model,
            messages=formatted_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
        )

        payload_request_logger.info(f"LiteLLM Stream Request ({self.provider_type.value}): {json.dumps(request_params, indent=2, default=str)}")

        stream = await acompletion(**request_params)

        in_think_block = False

        async for chunk in stream:
            if not chunk.choices or not chunk.choices[0].delta:
                continue

            delta_content = chunk.choices[0].delta.content or ""

            delta_thinking = getattr(chunk.choices[0].delta, "thinking", None)
            if getattr(chunk.choices[0].delta, "reasoning", None):
                delta_thinking = chunk.choices[0].delta.reasoning

            if delta_thinking:
                yield json.dumps({"type": "thinking", "content": delta_thinking})
                if delta_content:
                    yield json.dumps({"type": "content", "chunk": delta_content})
                continue

            if self.provider_type in [ProviderType.LOCAL, ProviderType.CUSTOM] and delta_content:
                for event in self._process_streaming_thinking(delta_content, in_think_block):
                    if "in_think_block" in event:
                        in_think_block = event["in_think_block"]
                    else:
                        yield json.dumps(event)
                continue

            if delta_content:
                yield delta_content


# =============================================================================
# Proveedores Modificados con LiteLLM
# =============================================================================

class LiteOllamaProvider(LiteLLMProviderBase):
    def __init__(self):
        self.client = OllamaClient(host=settings.OLLAMA_BASE_URL)
        super().__init__(ProviderType.LOCAL, litellm_prefix="ollama/")
        self._gpu_count = None
        self._cpu_threads = None

    def _detect_gpu_count(self) -> int:
        """Detect number of available GPUs (Linux and Windows compatible)"""
        if self._gpu_count is not None:
            return self._gpu_count

        import platform

        try:
            # Try nvidia-smi first (Linux/Windows)
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=count', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                gpu_count = int(result.stdout.strip().split('\n')[-1]) + 1  # nvidia-smi counts from 0
                self._gpu_count = gpu_count
                logger.info(f"Detected {gpu_count} GPU(s) via nvidia-smi")
                return gpu_count
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError, FileNotFoundError):
            pass

        try:
            # Try rocm-smi for AMD GPUs (Linux)
            result = subprocess.run(
                ['rocm-smi', '--showproductname'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                # Count lines that contain GPU info
                gpu_count = len([line for line in result.stdout.split('\n') if 'GPU' in line and 'ID' in line])
                self._gpu_count = gpu_count
                logger.info(f"Detected {gpu_count} GPU(s) via rocm-smi")
                return gpu_count
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            pass

        # Windows-specific detection using wmic
        if platform.system() == 'Windows':
            try:
                # Try wmic for Windows GPU detection
                result = subprocess.run(
                    ['wmic', 'path', 'win32_VideoController', 'get', 'name'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    # Count non-empty lines (excluding header)
                    lines = [line.strip() for line in result.stdout.split('\n') if line.strip()]
                    if len(lines) > 1:  # More than just header
                        gpu_count = len(lines) - 1  # Subtract header
                        self._gpu_count = gpu_count
                        logger.info(f"Detected {gpu_count} GPU(s) via wmic on Windows")
                        return gpu_count
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                pass

        # Fallback to CPU mode if no GPUs detected
        self._gpu_count = 0
        logger.info("No GPUs detected, using CPU mode")
        return 0

    def _detect_cpu_threads(self) -> int:
        """Detect number of available CPU threads (Linux and Windows compatible)"""
        import platform

        try:
            # Try to get CPU count from /proc/cpuinfo (Linux)
            if platform.system() == 'Linux':
                with open('/proc/cpuinfo', 'r') as f:
                    content = f.read()
                    # Count processor entries
                    thread_count = content.count('processor\t:')
                    if thread_count > 0:
                        logger.info(f"Detected {thread_count} CPU thread(s) via /proc/cpuinfo")
                        return thread_count
        except (FileNotFoundError, PermissionError):
            pass

        try:
            # Try lscpu command (Linux)
            if platform.system() == 'Linux':
                result = subprocess.run(
                    ['lscpu', '--parse=CPU'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    # Count non-comment lines
                    thread_count = len([line for line in result.stdout.split('\n') if line and not line.startswith('#')])
                    if thread_count > 0:
                        logger.info(f"Detected {thread_count} CPU thread(s) via lscpu")
                        return thread_count
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            pass

        try:
            # Try nproc command (Linux)
            if platform.system() == 'Linux':
                result = subprocess.run(
                    ['nproc'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    thread_count = int(result.stdout.strip())
                    logger.info(f"Detected {thread_count} CPU thread(s) via nproc")
                    return thread_count
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError, FileNotFoundError):
            pass

        # Windows-specific detection using wmic
        if platform.system() == 'Windows':
            try:
                # Try wmic for Windows CPU detection
                result = subprocess.run(
                    ['wmic', 'cpu', 'get', 'NumberOfLogicalProcessors'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    # Parse output to get thread count
                    lines = [line.strip() for line in result.stdout.split('\n') if line.strip()]
                    if len(lines) >= 2:  # Header + data
                        thread_count = int(lines[1])  # First data line
                        logger.info(f"Detected {thread_count} CPU thread(s) via wmic on Windows")
                        return thread_count
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError, FileNotFoundError):
                pass

        # Fallback to Python's os.cpu_count() (works on both Linux and Windows)
        import os
        thread_count = os.cpu_count()
        if thread_count is not None:
            logger.info(f"Detected {thread_count} CPU thread(s) via os.cpu_count()")
            return thread_count

        # Final fallback
        logger.warning("Could not detect CPU threads, using default value of 4")
        return 4

    def _validate_credentials(self):
        """Check if Ollama is accessible"""
        try:
            # Simple ping
            self.client.list()
        except Exception as e:
            raise ConnectionError(f"Cannot connect to Ollama: {e}")

    def _prepare_request_body(self, model: str, messages: List[Dict], options: Dict, stream: bool = False) -> Dict:
        """Prepare and log the request body for Ollama API calls"""
        request_body = {
            "model": model,
            "messages": messages,
            "options": options,
            "keep_alive": "5m"
        }

        if stream:
            request_body["stream"] = True

        # Log the request body
        ollama_url = f"{settings.OLLAMA_BASE_URL}/api/chat"
        payload_request_logger.info(f"HTTP Request: POST {ollama_url} - Request body: {json.dumps(request_body, indent=2)}")

        return request_body

    async def chat(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> ChatResponse:
        """Send chat request to Ollama"""
        import re
        import asyncio

        # Convert messages
        ollama_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens:
            options["num_predict"] = max_tokens


        # Solo incluir parámetros si se proporcionan en kwargs
        if "num_ctx" in kwargs:
            options["num_ctx"] = kwargs["num_ctx"]
        if "num_gpu" in kwargs:
            options["num_gpu"] = kwargs["num_gpu"]
        if "num_thread" in kwargs:
            options["num_thread"] = kwargs["num_thread"]
        if "num_batch" in kwargs:
            options["num_batch"] = kwargs["num_batch"]

        # Prepare and log request body
        self._prepare_request_body(model, ollama_messages, options, stream=False)

        # Ollama doesn't support function calling natively
        # We would need to implement prompt-based tool calling

        # Run blocking Ollama call in thread pool to avoid blocking the event loop
        response = await asyncio.to_thread(
            self.client.chat,
            model=model,
            messages=ollama_messages,
            options=options,
            keep_alive="5m"  # Keep model in memory for 5 minutes to avoid reload overhead
        )

        # Log the response (only serializable parts)
        def make_serializable(obj):
            """Recursively convert objects to serializable format"""
            if isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            else:
                # For non-serializable objects, return their string representation
                return str(obj)

        try:
            serializable_response = make_serializable(response)
            ollama_url = f"{settings.OLLAMA_BASE_URL}/api/chat"
            payload_request_logger.info(f"HTTP Response: POST {ollama_url} - Response: {json.dumps(serializable_response, indent=2)}")
        except Exception as e:
            ollama_url = f"{settings.OLLAMA_BASE_URL}/api/chat"
            payload_request_logger.info(f"HTTP Response: POST {ollama_url} - Response: <non-serializable response: {str(e)}>")

        # Extract content - handle Qwen3 "thinking" mode responses
        raw_content = response["message"].get("content", "")
        thinking_content = None
        final_content = raw_content

        # Check for <think>...</think> tags (Qwen3 thinking mode)
        # Pattern: content can be wrapped in <think>thinking</think> + actual response
        think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE)
        think_matches = think_pattern.findall(raw_content)

        if think_matches:
            # Store thinking content in metadata
            thinking_content = "\n".join(think_matches)
            # Remove thinking blocks from content to get actual response
            final_content = think_pattern.sub('', raw_content).strip()

        # Check for separate 'thinking' field (DeepSeek, etc.)
        # Some models return thinking content in a separate field
        separate_thinking = response.get("thinking")
        if separate_thinking:
            if thinking_content:
                # Combine with existing thinking content
                thinking_content += "\n" + separate_thinking
            else:
                thinking_content = separate_thinking

        # If content is still empty after removing think tags, check for 'thinking' field
        # Some Ollama versions may put thinking content in a separate field
        if not final_content and response.get("message", {}).get("thinking"):
            # The thinking was separate, but we still have no content
            # This means the model only generated thinking without final output
            # Use the raw content as-is (which may be empty)
            final_content = raw_content

        # Final fallback: if content is truly empty after all processing,
        # log a warning and return empty string (handled by caller)

        return ChatResponse(
            content=final_content,
            model=model,
            provider=self.provider_type.value,
            tokens_used=response.get("eval_count", 0),
            finish_reason=response.get("done_reason"),
            thinking_content=thinking_content,  # Direct access to thinking content
            metadata={
                "response_time": response.get("total_duration"),
                "thinking_content": thinking_content  # Also store in metadata for compatibility
            }
        )

    async def stream_chat(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream chat response"""
        import re
        import asyncio

        ollama_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens:
            options["num_predict"] = max_tokens


        # Solo incluir parámetros si se proporcionan en kwargs
        if "num_ctx" in kwargs:
            options["num_ctx"] = kwargs["num_ctx"]
        if "num_gpu" in kwargs:
            options["num_gpu"] = kwargs["num_gpu"]
        if "num_thread" in kwargs:
            options["num_thread"] = kwargs["num_thread"]
        if "num_batch" in kwargs:
            options["num_batch"] = kwargs["num_batch"]


        # Prepare and log request body
        self._prepare_request_body(model, ollama_messages, options, stream=True)

        # Run blocking Ollama stream initialization in thread pool to avoid blocking the event loop
        stream = await asyncio.to_thread(
            self.client.chat,
            model=model,
            messages=ollama_messages,
            options=options,
            stream=True,
            keep_alive="5m"  # Keep model in memory for 5 minutes to avoid reload overhead
        )

        # Log the stream initialization (note: we can't log the full response for streaming)
        ollama_url = f"{settings.OLLAMA_BASE_URL}/api/chat"
        logger.info(f"HTTP Stream Initialized: POST {ollama_url} - Model: {model}, Messages: {len(ollama_messages)}")

        # For streaming, we need to handle Qwen3 thinking tags
        # NEW: Emit thinking content to frontend, THEN filter for final response
        buffer = ""
        in_think_block = False
        accumulated_thinking = []  # Collect thinking chunks

        # NEW: Handle both thinking and response fields for DeepSeek models
        for chunk in stream:
            # DEBUG: Log raw chunk structure to understand the format
            chunk_type = type(chunk).__name__
            chunk_dir = [attr for attr in dir(chunk) if not attr.startswith('_')]
            logger.debug(f"Raw chunk type: {chunk_type}, attributes: {chunk_dir}")

            # Extract content from chunk - handle both dict and object formats
            # Ollama Python client returns objects with .message.content attribute
            if isinstance(chunk, dict):
                message_obj = chunk.get("message", {})
                if isinstance(message_obj, dict):
                    response_content = message_obj.get("content", "")
                    thinking_content = message_obj.get("thinking", "")
                else:
                    response_content = getattr(message_obj, 'content', '')
                    thinking_content = getattr(message_obj, 'thinking', '')
            else:
                # Ollama Python client returns objects
                message_obj = getattr(chunk, 'message', None)
                if message_obj:
                    response_content = getattr(message_obj, 'content', '')
                    thinking_content = getattr(message_obj, 'thinking', '')
                else:
                    # Fallback: some versions return content directly
                    response_content = getattr(chunk, 'content', '')
                    thinking_content = getattr(chunk, 'thinking', '')

            # Skip empty chunks (Ollama heartbeats)
            if not response_content and not thinking_content:
                continue

            # Log the chunk only if it has content
            try:
                chunk_dict = {
                    "message": {
                        "content": response_content,
                        "thinking": thinking_content
                    }
                }
                payload_response_logger.info(f"Full chunk: {json.dumps(chunk_dict, indent=2)}")
            except Exception as e:
                payload_response_logger.info(f"Full chunk (error): {str(e)}")

            # Log for debugging
            #logger.info(f"Chunk content - response: '{response_content}', thinking: '{thinking_content}'")

            # NEW: Also check for "Thinking..." prefix in response (DeepSeek format)
            if not thinking_content and response_content and response_content.startswith("Thinking..."):
                # Extract thinking content from response
                thinking_content = response_content
                response_content = ""  # Clear response since it's actually thinking
                #logger.info(f"Extracted thinking from response: '{thinking_content}'")

            # Emit thinking content if present
            if thinking_content:
                # Clean up thinking content (remove "Thinking..." prefix and "...done thinking." suffix)
                clean_thinking = thinking_content.replace("Thinking...", "").replace("...done thinking.", "")
                if clean_thinking:  # Ensure there is content to emit
                    #logger.info(f"Emitting thinking content: '{clean_thinking}'")
                    yield json.dumps({"type": "thinking", "content": clean_thinking})

            # Emit response content if present
            if response_content:
                yield json.dumps({"type": "content", "chunk": response_content})

        # Flush remaining buffer if not in think block
        if buffer and not in_think_block:
            # Final cleanup: remove any remaining think tags
            clean_content = re.sub(r'<think>.*?</think>', '', buffer,
                                   flags=re.DOTALL | re.IGNORECASE)
            if clean_content.strip():
                yield clean_content

    def get_available_models(self) -> List[ModelInfo]:
        """Get available Ollama models"""
        models = self.client.list()

        model_list = []
        for model in models.get("models", []):
            #model_name = model["name"].lower()
            # Handle both dictionary and object access for compatibility
            if isinstance(model, dict):
                model_name = model.get("model", "").lower()
                model_val = model.get("model")
            else:
                model_name = model.model.lower()
                model_val = model.model

            # Determine model type based on name patterns
            if any(embed_term in model_name for embed_term in [
                "embed", "embedding", "bge", "e5", "gte", "instructor", "mxbai"
            ]):
                model_type = ModelType.EMBEDDING
                supports_function_calling = False
            elif any(vision_term in model_name for vision_term in [
                "vision", "llava", "bakllava", "minicpm-v"
            ]):
                model_type = ModelType.VISION
                supports_function_calling = False
            elif any(mm_term in model_name for mm_term in [
                "multimodal", "llava", "cogvlm", "fuyu"
            ]):
                model_type = ModelType.MULTIMODAL
                supports_function_calling = False
            else:
                # Default to chat for standard LLMs
                model_type = ModelType.CHAT
                # Check if model supports function calling (tools)
                # Generally, newer and larger models support this
                supports_function_calling = any(tool_term in model_name for tool_term in [
                    "llama3", "mistral", "mixtral", "gemma2", "qwen", "command"
                ]) and "embed" not in model_name

            model_list.append(
                ModelInfo(
                    name=model_val,
                    provider=ProviderType.LOCAL,
                    context_window=8192,  # Default, could be higher
                    supports_function_calling=supports_function_calling,
                    supports_streaming=True,
                    model_type=model_type,
                    cost_per_1k_input=0.0,  # Free
                    cost_per_1k_output=0.0,
                    # Hardware requirements - most Ollama models can run on CPU
                    cpu_supported=True,
                    gpu_required=False,
                    parent_retrieval_supported=True
                )
            )

        return model_list


class LiteOpenAIProvider(LiteLLMProviderBase):
    def __init__(self):
        super().__init__(ProviderType.OPENAI, litellm_prefix="openai/")

    def get_available_models(self) -> List[ModelInfo]:
        """Get available OpenAI models"""
        return [
            ModelInfo(
                name="gpt-4-turbo-preview",
                provider=ProviderType.OPENAI,
                context_window=128000,
                supports_function_calling=True,
                supports_streaming=True,
                model_type=ModelType.CHAT,
                cost_per_1k_input=0.01,
                cost_per_1k_output=0.03,
                # OpenAI models run on cloud infrastructure
                cpu_supported=False,
                gpu_required=True,
                parent_retrieval_supported=True
            ),
            ModelInfo(
                name="gpt-4",
                provider=ProviderType.OPENAI,
                context_window=8192,
                supports_function_calling=True,
                supports_streaming=True,
                model_type=ModelType.CHAT,
                cost_per_1k_input=0.03,
                cost_per_1k_output=0.06,
                # OpenAI models run on cloud infrastructure
                cpu_supported=False,
                gpu_required=True,
                parent_retrieval_supported=True
            ),
            ModelInfo(
                name="gpt-3.5-turbo",
                provider=ProviderType.OPENAI,
                context_window=16385,
                supports_function_calling=True,
                supports_streaming=True,
                model_type=ModelType.CHAT,
                cost_per_1k_input=0.0005,
                cost_per_1k_output=0.0015,
                # OpenAI models run on cloud infrastructure
                cpu_supported=False,
                gpu_required=True,
                parent_retrieval_supported=True
            )
        ]


class LiteAnthropicProvider(LiteLLMProviderBase):
    def __init__(self):
        super().__init__(ProviderType.ANTHROPIC, litellm_prefix="anthropic/")



class LiteOpenRouterProvider(LiteLLMProviderBase):
    def __init__(self):
        super().__init__(ProviderType.OPENROUTER, litellm_prefix="openrouter/")

    def get_available_models(self) -> List[ModelInfo]:
        """Get available OpenRouter models"""
        # Note: OpenRouter has MANY models. We should probably fetch them dynamically
        # or cache them. For now, we'll implement a dynamic fetch if possible,
        # otherwise return a curated list of top models.

        # Ideally, we should fetch from https://openrouter.ai/api/v1/models
        # But this is a sync method, so we'll use requests or valid hardcoded popular ones

        # Since this method is synchronous in the interface, we'll try to fetch using requests
        # or fallback to a popular list.

        try:
            import requests
            response = requests.get("https://openrouter.ai/api/v1/models")
            if response.status_code == 200:
                data = response.json()
                models = []
                for m in data.get("data", []):
                    # Basic filtering for too many models?
                    # For now, include all or popular ones.
                    # OpenRouter returns A LOT of models.

                    # Logic to determine capabilities
                    model_id = m.get("id")
                    name = m.get("name", model_id)
                    context = m.get("context_length", 8192)

                    # Cost (per 1M tokens usually, convert to 1K)
                    pricing_obj = m.get("pricing", {})
                    prompt_price_str = pricing_obj.get("prompt", "0")
                    completion_price_str = pricing_obj.get("completion", "0")

                    try:
                        prompt_price = float(prompt_price_str) * 1000
                        completion_price = float(completion_price_str) * 1000
                    except (ValueError, TypeError):
                        prompt_price = 0.0
                        completion_price = 0.0

                    # Determine if free
                    is_free = (prompt_price == 0.0 and completion_price == 0.0)

                    # Infer type
                    model_type = ModelType.CHAT
                    if "vision" in model_id.lower():
                        model_type = ModelType.VISION

                    models.append(ModelInfo(
                        name=model_id, # Use ID as the value to send
                        provider=ProviderType.OPENROUTER,
                        context_window=context,
                        supports_function_calling=False, # Hard to know dynamically without more metadata
                        supports_streaming=True,
                        model_type=model_type,
                        cost_per_1k_input=prompt_price,
                        cost_per_1k_output=completion_price,
                        is_free=is_free,
                        pricing=pricing_obj,
                        cpu_supported=False,
                        gpu_required=True,
                        parent_retrieval_supported=True
                    ))
                return models
        except Exception as e:
            print(f"⚠️  Failed to fetch OpenRouter models dynamically: {e}")
            pass

        # Fallback list
        return [
            ModelInfo(
                name="openai/gpt-4o",
                provider=ProviderType.OPENROUTER,
                context_window=128000,
                supports_function_calling=True,
                supports_streaming=True,
                model_type=ModelType.CHAT,
                cost_per_1k_input=0.005,
                cost_per_1k_output=0.015,
                cpu_supported=False,
                gpu_required=True,
                parent_retrieval_supported=True
            ),
            ModelInfo(
                name="anthropic/claude-3.5-sonnet",
                provider=ProviderType.OPENROUTER,
                context_window=200000,
                supports_function_calling=True,
                supports_streaming=True,
                model_type=ModelType.CHAT,
                cost_per_1k_input=0.003,
                cost_per_1k_output=0.015,
                cpu_supported=False,
                gpu_required=True,
                parent_retrieval_supported=True
            ),
            ModelInfo(
                name="google/gemini-pro-1.5",
                provider=ProviderType.OPENROUTER,
                context_window=1000000,
                supports_function_calling=True,
                supports_streaming=True,
                model_type=ModelType.CHAT,
                cost_per_1k_input=0.0035,
                cost_per_1k_output=0.0105,
                cpu_supported=False,
                gpu_required=True,
                parent_retrieval_supported=True
            )
        ]


class LiteGroqProvider(LiteLLMProviderBase):
    def __init__(self):
        super().__init__(ProviderType.GROQ, litellm_prefix="groq/")

    def get_available_models(self) -> List[ModelInfo]:
        """Get available Groq models"""
        # Groq doesn't check credentials on list models by default in all lib versions,
        # but better to provide a static list or fetch if possible.
        # Groq supports: Llama 3 8b/70b, Mixtral 8x7b, Gemma 7b

        return [
            ModelInfo(
                name="llama3-70b-8192",
                provider=ProviderType.GROQ,
                context_window=8192,
                supports_function_calling=True,
                supports_streaming=True,
                model_type=ModelType.CHAT,
                cost_per_1k_input=0.0, # Groq is often free/low cost preview. Update as needed.
                cost_per_1k_output=0.0,
                cpu_supported=False,
                gpu_required=True,
                parent_retrieval_supported=True
            ),
             ModelInfo(
                name="llama3-8b-8192",
                provider=ProviderType.GROQ,
                context_window=8192,
                supports_function_calling=True,
                supports_streaming=True,
                model_type=ModelType.CHAT,
                 cost_per_1k_input=0.0,
                cost_per_1k_output=0.0,
                cpu_supported=False,
                gpu_required=True,
                parent_retrieval_supported=True
            ),
            ModelInfo(
                name="mixtral-8x7b-32768",
                provider=ProviderType.GROQ,
                context_window=32768,
                supports_function_calling=True,
                supports_streaming=True,
                model_type=ModelType.CHAT,
                 cost_per_1k_input=0.0,
                cost_per_1k_output=0.0,
                cpu_supported=False,
                gpu_required=True,
                parent_retrieval_supported=True
            ),
             ModelInfo(
                name="gemma-7b-it",
                provider=ProviderType.GROQ,
                context_window=8192,
                supports_function_calling=True, # Gemma 7b on Groq implementation status check
                supports_streaming=True,
                model_type=ModelType.CHAT,
                 cost_per_1k_input=0.0,
                cost_per_1k_output=0.0,
                cpu_supported=False,
                gpu_required=True,
                parent_retrieval_supported=True
            )
        ]


class CustomLiteLLMProvider(LiteLLMProviderBase):
    """LiteLLM provider for custom providers"""
    def __init__(self, config: CustomProviderConfig):
        super().__init__(
            provider_type=ProviderType.CUSTOM,
            litellm_prefix=f"{config.model_prefix}/" if config.model_prefix else ""
        )
        self.config = config
        if config.api_key_env:
            api_key = os.getenv(config.api_key_env)
            if api_key:
                os.environ[f"{config.name.upper()}_API_KEY"] = api_key

    def get_available_models(self) -> List[ModelInfo]:
        return []

class ProviderManager:
    """Manages all LLM providers powered by LiteLLM"""

    def __init__(self):
        self.providers: Dict[str, BaseProvider] = {}
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize available providers using LiteLLM classes"""
        try:
            self.providers[ProviderType.LOCAL.value] = LiteOllamaProvider()
            logger.info("✅ LiteLLM Local provider (Ollama) initialized")
        except Exception as e:
            logger.warning(f"⚠️  Local provider unavailable: {e}")

        if settings.OPENAI_API_KEY:
            try:
                self.providers[ProviderType.OPENAI.value] = LiteOpenAIProvider()
                logger.info("✅ LiteLLM OpenAI provider initialized")
            except Exception as e:
                pass

        if settings.ANTHROPIC_API_KEY:
            try:
                self.providers[ProviderType.ANTHROPIC.value] = LiteAnthropicProvider()
                logger.info("✅ LiteLLM Anthropic provider initialized")
            except Exception as e:
                pass

        if settings.OPENROUTER_API_KEY:
            try:
                self.providers[ProviderType.OPENROUTER.value] = LiteOpenRouterProvider()
                logger.info("✅ LiteLLM OpenRouter provider initialized")
            except Exception as e:
                pass

        if settings.GROQ_API_KEY:
            try:
                self.providers[ProviderType.GROQ.value] = LiteGroqProvider()
                logger.info("✅ LiteLLM Groq provider initialized")
            except Exception as e:
                pass

    def get_provider(self, provider_type: str) -> BaseProvider:
        if provider_type in self.providers:
            return self.providers[provider_type]
        raise ValueError(f"Provider {provider_type} not available in Manager V2")

    def register_custom_provider(
        self,
        provider_name: str,
        base_url: str,
        model_prefix: str,
        api_key_env: Optional[str] = None
    ):
        """Register a custom provider dynamically"""
        config = CustomProviderConfig(
            name=provider_name,
            base_url=base_url,
            model_prefix=model_prefix,
            api_key_env=api_key_env
        )
        provider = CustomLiteLLMProvider(config)
        self.providers[provider_name] = provider
        logger.info(f"✅ Custom provider {provider_name} registered")

    def get_available_providers(self) -> List[str]:
        return list(self.providers.keys())

    async def sync_available_models(self, db_session: AsyncSession) -> List[str]:
        """
        Synchronize available models from all providers to database.
        """
        logger.info("🔄 Syncing models from providers (V2)...")
        uncertain_models = []

        all_provider_models: List[ModelInfo] = []
        for provider_name, provider in self.providers.items():
            try:
                models = provider.get_available_models()
                all_provider_models.extend(models)
            except Exception as e:
                logger.warning(f"⚠️ Failed to fetch models from {provider_name}: {e}")

        for model_info in all_provider_models:
            provider_val = model_info.provider.value if isinstance(model_info.provider, ProviderType) else str(model_info.provider)
            stmt = select(LLMModel).where(
                LLMModel.provider == provider_val,
                LLMModel.model_name == model_info.name
            )
            result = await db_session.execute(stmt)
            existing_model = result.scalar_one_or_none()

            inferred_type = model_info.model_type
            supports_thinking = model_info.supports_thinking

            if existing_model:
                if not existing_model.is_custom:
                    existing_model.model_type = inferred_type
                    existing_model.context_window = model_info.context_window
                    existing_model.supports_function_calling = model_info.supports_function_calling
                    existing_model.supports_streaming = model_info.supports_streaming
                    existing_model.supports_thinking = supports_thinking
                    existing_model.is_active = model_info.is_active
                    existing_model.last_seen = get_current_utc()
            else:
                new_model = LLMModel(
                    provider=provider_val,
                    model_name=model_info.name,
                    model_type=inferred_type,
                    context_window=model_info.context_window,
                    supports_streaming=model_info.supports_streaming,
                    supports_function_calling=model_info.supports_function_calling,
                    supports_thinking=supports_thinking,
                    is_active=model_info.is_active,
                    cpu_supported=model_info.cpu_supported,
                    gpu_required=model_info.gpu_required,
                    parent_retrieval_supported=model_info.parent_retrieval_supported
                )
                db_session.add(new_model)

        await db_session.commit()
        return uncertain_models







    async def get_available_models(self, db_session: AsyncSession = None) -> List[ModelInfo]:
        """
        Get all available models, preferring DB source if session provided.
        Fallback to live fetch if no DB session.
        """
        if not db_session:
            # Fallback to current behavior (live fetch from providers)
            all_models = []
            for p in self.providers.values():
                all_models.extend(p.get_available_models())

            # Sort alphabetically by name
            all_models.sort(key=lambda x: x.name)
            return all_models

        from sqlalchemy import select
        from src.models.llm_models import LLMModel

        # Fetch active models from DB
        stmt = select(LLMModel).where(LLMModel.is_active == True).order_by(LLMModel.model_name)
        result = await db_session.execute(stmt)
        db_models = result.scalars().all()

        # Convert to ModelInfo using database values
        return [
            ModelInfo(
                name=m.model_name,
                provider=ProviderType(m.provider),
                context_window=m.context_window,
                supports_function_calling=m.supports_function_calling,
                supports_streaming=m.supports_streaming,
                model_type=ModelType(m.model_type),
                # Database fields for frontend filtering
                supports_thinking=m.supports_thinking,
                is_active=m.is_active,
                is_custom=m.is_custom,
                cost_per_1k_input=m.cost_per_1k_input if hasattr(m, 'cost_per_1k_input') else 0.0,
                cost_per_1k_output=m.cost_per_1k_output if hasattr(m, 'cost_per_1k_output') else 0.0,
                # Hardware requirements and capabilities from database
                cpu_supported=m.cpu_supported,
                gpu_required=m.gpu_required,
                parent_retrieval_supported=m.parent_retrieval_supported,
                # Pricing
                is_free=m.is_free if hasattr(m, 'is_free') else False
            )
            for m in db_models
        ]


provider_manager = ProviderManager()
