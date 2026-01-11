# =============================================================================
# src/providers/manager.py
# Unified LLM Provider Manager
# =============================================================================
"""
Gestor unificado de providers de LLM (Local, OpenAI, Claude, Gemini, OpenRouter)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, AsyncGenerator
import json
import subprocess
import re

import anthropic
from ollama import Client as OllamaClient
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.settings import settings
from src.providers.cancellable_stream import CancellableProviderMixin
from src.utils.logger import get_logger

# Logger para el provider local
logger = get_logger(__name__)


class ProviderType(str, Enum):
    """LLM provider types"""
    LOCAL = "local"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OPENROUTER = "openrouter"
    GROQ = "groq"


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


@dataclass
class ChatMessage:
    """Standard chat message format"""
    role: str  # "user", "assistant", "system"
    content: str


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
        temperature: float = 0.7,
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
        temperature: float = 0.7,
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


# =============================================================================
# Local Provider (Ollama)
# =============================================================================

class LocalProvider(BaseProvider, CancellableProviderMixin):
    """Ollama local LLM provider with cancellation support"""

    def __init__(self):
        self.provider_type = ProviderType.LOCAL
        self.client = OllamaClient(host=settings.OLLAMA_BASE_URL)
        self._gpu_count = None  # Cache GPU count
        self._cpu_threads = None  # Cache CPU thread count
        super().__init__()

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
        logger.info(f"HTTP Request: POST http://localhost:11434/api/chat - Request body: {json.dumps(request_body, indent=2)}")

        return request_body

    async def chat(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> ChatResponse:
        """Send chat request to Ollama"""
        import re

        # Convert messages
        ollama_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        # Calculate dynamic context size based on message history
        total_chars = sum(len(msg.get("content", "")) for msg in ollama_messages)
        estimated_tokens = total_chars // 3  # Rough approximation: 3 chars per token
        # Use between 512 and 4096 tokens, defaulting to 2x estimated need
        #dynamic_ctx = min(max(512, estimated_tokens * 2), 4096)
        dynamic_ctx = min(max(512, estimated_tokens * 2), 8192)

        #options = {
        #    "temperature": temperature,
        #    "num_predict": max_tokens or 2000,
        #    #"num_ctx": kwargs.get("num_ctx", min(1024, dynamic_ctx)),
        #    "num_ctx": kwargs.get("num_ctx", dynamic_ctx),
        #    # Reduced from 2048, with dynamic sizing
        #    "num_gpu": kwargs.get("num_gpu", self._detect_gpu_count()),  # Auto-detect GPU count
        #    #"num_thread": kwargs.get("num_thread", self._detect_cpu_threads()),  # Auto-detect CPU #threads
        #    "num_batch": kwargs.get("num_batch", 512),  # Batch processing
        #}
        options = {
            "temperature": temperature,
            "num_predict": max_tokens or 2000,
        }

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

        response = self.client.chat(
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
            logger.info(f"HTTP Response: POST http://localhost:11434/api/chat - Response: {json.dumps(serializable_response, indent=2)}")
        except Exception as e:
            logger.info(f"HTTP Response: POST http://localhost:11434/api/chat - Response: <non-serializable response: {str(e)}>")

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
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream chat response"""
        import re

        ollama_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        # Calculate dynamic context size based on message history
        total_chars = sum(len(msg.get("content", "")) for msg in ollama_messages)
        estimated_tokens = total_chars // 3  # Rough approximation: 3 chars per token
        # Use between 512 and 4096 tokens, defaulting to 2x estimated need
        #dynamic_ctx = min(max(512, estimated_tokens * 2), 4096)
        dynamic_ctx = min(max(512, estimated_tokens * 2), 8192)

        #options = {
        #    "temperature": temperature,
        #    "num_predict": max_tokens or 2000,
        #    "num_ctx": kwargs.get("num_ctx", dynamic_ctx),
        #    "num_gpu": kwargs.get("num_gpu", self._detect_gpu_count()),  # Auto-detect GPU count
        #    "num_thread": kwargs.get("num_thread", self._detect_cpu_threads()),  # Auto-detect CPU #threads
        #    "num_batch": kwargs.get("num_batch", 512),  # Batch processing
        #}
        options = {
            "temperature": temperature,
            "num_predict": max_tokens or 2000,
        }

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

        stream = self.client.chat(
            model=model,
            messages=ollama_messages,
            options=options,
            stream=True,
            keep_alive="5m"  # Keep model in memory for 5 minutes to avoid reload overhead
        )

        # Log the stream initialization (note: we can't log the full response for streaming)
        logger.info(f"HTTP Stream Initialized: POST http://localhost:11434/api/chat - Model: {model}, Messages: {len(ollama_messages)}")

        # For streaming, we need to handle Qwen3 thinking tags
        # NEW: Emit thinking content to frontend, THEN filter for final response
        buffer = ""
        in_think_block = False
        accumulated_thinking = []  # Collect thinking chunks

        # NEW: Handle both thinking and response fields for DeepSeek models
        for chunk in stream:
            # Log the entire chunk for debugging
            # Convert chunk to dict for JSON serialization
            try:
                chunk_dict = chunk if isinstance(chunk, dict) else {
                    "message": {
                        "content": getattr(chunk, 'content', ''),
                        "thinking": getattr(chunk, 'thinking', '')
                    }
                }
                logger.info(f"Full chunk: {json.dumps(chunk_dict, indent=2)}")
            except Exception as e:
                logger.info(f"Full chunk (non-serializable): {str(chunk)[:200]}...")

            # Extract message object from chunk
            message_obj = chunk.get("message", {})

            # Handle both response and thinking fields (DeepSeek format)
            response_content = message_obj.get("content", "")
            thinking_content = message_obj.get("thinking", "")

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
                #logger.info(f"Emitting response content: '{response_content}'")
                yield json.dumps({"type": "content", "chunk": response_content})

            # Skip chunks with no content
            if not response_content and not thinking_content:
                #logger.info("Skipping empty chunk")
                continue

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


# =============================================================================
# OpenAI Provider
# =============================================================================

class OpenAIProvider(BaseProvider, CancellableProviderMixin):
    """OpenAI provider (GPT-4, GPT-3.5) with cancellation support"""

    def __init__(self):
        self.provider_type = ProviderType.OPENAI
        super().__init__()
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    def _validate_credentials(self):
        """Validate OpenAI API key"""
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not configured")

    async def chat(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> ChatResponse:
        """Send chat request to OpenAI"""
        openai_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        request_params = {
            "model": model,
            "messages": openai_messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        if tools:
            request_params["tools"] = tools
            request_params["tool_choice"] = kwargs.get("tool_choice", "auto")

        response = await self.client.chat.completions.create(**request_params)

        message = response.choices[0].message

        # Extract tool calls if present
        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
                for tc in message.tool_calls
            ]

        return ChatResponse(
            content=message.content or "",
            model=response.model,
            provider=self.provider_type.value,
            tokens_used=response.usage.total_tokens,
            tool_calls=tool_calls,
            finish_reason=response.choices[0].finish_reason,
            metadata={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            }
        )

    async def stream_chat(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream chat response"""
        openai_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        stream = await self.client.chat.completions.create(
            model=model,
            messages=openai_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

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


# =============================================================================
# Anthropic Provider (Claude)
# =============================================================================

class AnthropicProvider(BaseProvider, CancellableProviderMixin):
    """Anthropic Claude provider with cancellation support"""

    def __init__(self):
        self.provider_type = ProviderType.ANTHROPIC
        super().__init__()
        self.client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

    def _validate_credentials(self):
        """Validate Anthropic API key"""
        if not settings.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not configured")

    async def chat(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> ChatResponse:
        """Send chat request to Claude"""
        # Separate system message
        system_messages = []
        claude_messages = []

        for msg in messages:
            if msg.role == "system":
                system_messages.append(msg.content)
            else:
                claude_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        request_params = {
            "model": model,
            "messages": claude_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096
        }

        # Concatenate all system messages
        system_message = "".join(system_messages) if system_messages else None

        if system_message:
            request_params["system"] = system_message

        if tools:
            request_params["tools"] = tools

        response = await self.client.messages.create(**request_params)

        # Extract content and tool calls
        content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input
                })

        return ChatResponse(
            content=content,
            model=response.model,
            provider=self.provider_type.value,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=response.stop_reason,
            metadata={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }
        )

    async def stream_chat(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream chat response"""
        system_message = None
        claude_messages = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                claude_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        request_params = {
            "model": model,
            "messages": claude_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
            "stream": True
        }

        if system_message:
            request_params["system"] = system_message

        async with self.client.messages.stream(**request_params) as stream:
            async for text in stream.text_stream:
                yield text

    def get_available_models(self) -> List[ModelInfo]:
        """Get available Claude models"""
        return [
            ModelInfo(
                name="claude-3-5-sonnet-20241022",
                provider=ProviderType.ANTHROPIC,
                context_window=200000,
                supports_function_calling=True,
                supports_streaming=True,
                model_type=ModelType.CHAT,
                cost_per_1k_input=0.003,
                cost_per_1k_output=0.015,
                # Anthropic models run on cloud infrastructure
                cpu_supported=False,
                gpu_required=True,
                parent_retrieval_supported=True
            ),
            ModelInfo(
                name="claude-3-opus-20240229",
                provider=ProviderType.ANTHROPIC,
                context_window=200000,
                supports_function_calling=True,
                supports_streaming=True,
                model_type=ModelType.CHAT,
                cost_per_1k_input=0.015,
                cost_per_1k_output=0.075,
                # Anthropic models run on cloud infrastructure
                cpu_supported=False,
                gpu_required=True,
                parent_retrieval_supported=True
            )
        ]


# =============================================================================
# OpenRouter Provider
# =============================================================================

class OpenRouterProvider(BaseProvider, CancellableProviderMixin):
    """OpenRouter provider (OpenAI-compatible) with cancellation support"""

    def __init__(self):
        self.provider_type = ProviderType.OPENROUTER
        super().__init__()
        self.client = AsyncOpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )

    def _validate_credentials(self):
        """Validate OpenRouter API key"""
        if not settings.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not configured")

    async def chat(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> ChatResponse:
        """Send chat request to OpenRouter"""
        openrouter_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        request_params = {
            "model": model,
            "messages": openrouter_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_headers": {
                "HTTP-Referer": "https://github.com/your-repo",  # Optional, for OpenRouter rankings
                "X-Title": settings.APP_NAME,  # Optional
            }
        }

        if tools:
            request_params["tools"] = tools
            request_params["tool_choice"] = kwargs.get("tool_choice", "auto")

        response = await self.client.chat.completions.create(**request_params)

        message = response.choices[0].message

        # Extract tool calls if present
        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
                for tc in message.tool_calls
            ]

        return ChatResponse(
            content=message.content or "",
            model=response.model,
            provider=self.provider_type.value,
            tokens_used=response.usage.total_tokens if response.usage else 0,
            tool_calls=tool_calls,
            finish_reason=response.choices[0].finish_reason,
            metadata={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0
            }
        )

    async def stream_chat(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream chat response"""
        openrouter_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        stream = await self.client.chat.completions.create(
            model=model,
            messages=openrouter_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            extra_headers={
                "HTTP-Referer": "https://github.com/your-repo",
                "X-Title": settings.APP_NAME,
            }
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

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


# =============================================================================
# Groq Provider
# =============================================================================

class GroqProvider(BaseProvider, CancellableProviderMixin):
    """Groq provider (OpenAI-compatible) with cancellation support"""

    def __init__(self):
        self.provider_type = ProviderType.GROQ
        super().__init__()
        self.client = AsyncOpenAI(
            api_key=settings.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )

    def _validate_credentials(self):
        """Validate Groq API key"""
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not configured")

    async def chat(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> ChatResponse:
        """Send chat request to Groq"""
        groq_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        request_params = {
            "model": model,
            "messages": groq_messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        if tools:
            request_params["tools"] = tools
            request_params["tool_choice"] = kwargs.get("tool_choice", "auto")

        response = await self.client.chat.completions.create(**request_params)

        message = response.choices[0].message

        # Extract tool calls if present
        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
                for tc in message.tool_calls
            ]

        return ChatResponse(
            content=message.content or "",
            model=response.model,
            provider=self.provider_type.value,
            tokens_used=response.usage.total_tokens if response.usage else 0,
            tool_calls=tool_calls,
            finish_reason=response.choices[0].finish_reason,
            metadata={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0
            }
        )

    async def stream_chat(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream chat response"""
        groq_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        stream = await self.client.chat.completions.create(
            model=model,
            messages=groq_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

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


# =============================================================================
# Provider Manager
# =============================================================================

class ProviderManager:
    """Manages all LLM providers"""

    def __init__(self):
        self.providers: Dict[ProviderType, BaseProvider] = {}
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize available providers"""
        # Local (always available if Ollama is running)
        try:
            self.providers[ProviderType.LOCAL] = LocalProvider()
            print("✅ Local provider (Ollama) initialized")
        except Exception as e:
            print(f"⚠️  Local provider unavailable: {e}")

        # OpenAI
        if settings.OPENAI_API_KEY:
            try:
                self.providers[ProviderType.OPENAI] = OpenAIProvider()
                print("✅ OpenAI provider initialized")
            except Exception as e:
                print(f"⚠️  OpenAI provider failed: {e}")

        # Anthropic
        if settings.ANTHROPIC_API_KEY:
            try:
                self.providers[ProviderType.ANTHROPIC] = AnthropicProvider()
                print("✅ Anthropic provider initialized")
            except Exception as e:
                print(f"⚠️  Anthropic provider failed: {e}")

        # OpenRouter
        if settings.OPENROUTER_API_KEY:
            try:
                self.providers[ProviderType.OPENROUTER] = OpenRouterProvider()
                print("✅ OpenRouter provider initialized")
            except Exception as e:
                print(f"⚠️  OpenRouter provider failed: {e}")

        # Groq
        if settings.GROQ_API_KEY:
            try:
                self.providers[ProviderType.GROQ] = GroqProvider()
                print("✅ Groq provider initialized")
            except Exception as e:
                print(f"⚠️  Groq provider failed: {e}")

    def get_provider(self, provider_type: str) -> BaseProvider:
        """Get provider by type"""
        provider_enum = ProviderType(provider_type)

        if provider_enum not in self.providers:
            # Try to lazily initialize local provider if it was unavailable at startup
            if provider_enum == ProviderType.LOCAL:
                try:
                    self.providers[ProviderType.LOCAL] = LocalProvider()
                    print("✅ Local provider (Ollama) initialized on-demand")
                except Exception as e:
                    raise ValueError(f"Provider {provider_type} not available: {e}")
            elif provider_enum == ProviderType.OPENROUTER and settings.OPENROUTER_API_KEY:
                 try:
                    self.providers[ProviderType.OPENROUTER] = OpenRouterProvider()
                    print("✅ OpenRouter provider initialized on-demand")
                 except Exception as e:
                    raise ValueError(f"Provider {provider_type} not available: {e}")
            elif provider_enum == ProviderType.GROQ and settings.GROQ_API_KEY:
                 try:
                    self.providers[ProviderType.GROQ] = GroqProvider()
                    print("✅ Groq provider initialized on-demand")
                 except Exception as e:
                    raise ValueError(f"Provider {provider_type} not available: {e}")
            else:
                raise ValueError(f"Provider {provider_type} not available")

        return self.providers[provider_enum]

    def get_available_providers(self) -> List[str]:
        """Get list of available provider names"""
        return [p.value for p in self.providers.keys()]

    async def sync_available_models(self, db_session: AsyncSession):
        """
        Sync available models from providers to database.
        Preserves existing manual configuration.
        Uses Ollama API metadata for better type detection.
        """
        from sqlalchemy import select
        from src.models.llm_models import LLMModel

        print("🔄 Syncing models from providers...")
        uncertain_models = []  # Track models needing manual classification

        # 1. Get all models from all providers
        all_provider_models: List[ModelInfo] = []
        for provider_name, provider in self.providers.items():
            try:
                models = provider.get_available_models()
                all_provider_models.extend(models)
            except Exception as e:
                print(f"⚠️  Failed to fetch models from {provider_name}: {e}")

        # 2. Sync with DB
        for model_info in all_provider_models:
            # Check if exists
            stmt = select(LLMModel).where(
                LLMModel.provider == model_info.provider.value,
                LLMModel.model_name == model_info.name
            )
            result = await db_session.execute(stmt)
            existing_model = result.scalar_one_or_none()

            # Infer capabilities
            inferred_type = model_info.model_type
            supports_thinking = False
            confidence = "high"  # high, medium, low

            # Enhanced detection for LOCAL (Ollama) models
            if model_info.provider == ProviderType.LOCAL and inferred_type == ModelType.CHAT:
                name_lower = model_info.name.lower()

                # Try to get detailed model info from Ollama
                try:
                    # Get the LocalProvider instance
                    local_provider = self.providers.get(ProviderType.LOCAL)
                    if local_provider and hasattr(local_provider, 'client'):
                        model_details = local_provider.client.show(model_info.name)

                        # Analyze template for reasoning indicators
                        template = model_details.get('template', '').lower()
                        system = model_details.get('system', '').lower()

                        # Strong indicators of TRUE reasoning models (deepseek-r1 style)
                        reasoning_indicators = [
                            'chain of thought', 'step by step reasoning',
                            'reasoning process', 'thought process'
                        ]

                        # Indicators of thinking capability (gemma3, qwen3 style)
                        thinking_indicators = [
                            '<think>', '<thinking>', '</think>',
                            '<reasoning>', '</reasoning>'
                        ]

                        has_thinking_tags = any(
                            indicator in template or indicator in system for indicator in
                            thinking_indicators)
                        has_reasoning_focus = any(
                            indicator in template or indicator in system for indicator in
                            reasoning_indicators)

                        if has_reasoning_focus:
                            # True reasoning model (deepseek-r1)
                            inferred_type = ModelType.REASONING
                            supports_thinking = True
                            confidence = "high"
                            print(f"  ✓ {model_info.name}: REASONING (detected via template)")
                        elif has_thinking_tags:
                            # Chat with thinking capability (gemma3, qwen3)
                            inferred_type = ModelType.CHAT
                            supports_thinking = True
                            confidence = "high"
                            print(
                                f"  ✓ {model_info.name}: CHAT with thinking (detected via template)")
                        else:
                            # Fallback to name heuristics
                            if any(k in name_lower for k in
                                   ["deepseek-r1", "r1", "reasoner", "cot", "qwq"]):
                                inferred_type = ModelType.REASONING
                                supports_thinking = True
                                confidence = "medium"
                                print(f"  ✓ {model_info.name}: REASONING (name heuristic)")
                            elif any(k in name_lower for k in
                                     ["gemma3", "qwen3", "qwen2.5", "thinking"]):
                                inferred_type = ModelType.CHAT
                                supports_thinking = True
                                confidence = "medium"
                                print(f"  ✓ {model_info.name}: CHAT with thinking (name heuristic)")
                            else:
                                # Cannot determine - mark as uncertain
                                confidence = "low"
                                uncertain_models.append(model_info.name)
                                print(
                                    f"  ? {model_info.name}: CHAT (uncertain - needs verification)")

                except Exception as e:
                    # Failed to get details, use name heuristics only
                    if any(
                        k in name_lower for k in ["deepseek-r1", "r1", "reasoner", "cot", "qwq"]):
                        inferred_type = ModelType.REASONING
                        supports_thinking = True
                        confidence = "medium"
                    elif any(k in name_lower for k in ["gemma3", "qwen3", "qwen2.5", "thinking"]):
                        inferred_type = ModelType.CHAT
                        supports_thinking = True
                        confidence = "medium"
                    else:
                        confidence = "low"
                        uncertain_models.append(model_info.name)

            if existing_model:
                # Update last_seen
                existing_model.last_seen = datetime.utcnow()
                # Only update capabilities if NOT custom (user manual override protection)
                if not existing_model.is_custom:
                    # Update context window if it changed
                    existing_model.context_window = model_info.context_window
                    # Don't overwrite type if it's already set to something specific,
                    # unless we are upgrading from generic CHAT to more specific REASONING
                    if existing_model.model_type == ModelType.CHAT.value and inferred_type == ModelType.REASONING:
                        existing_model.model_type = inferred_type.value
                    
                    # Update pricing info
                    if hasattr(existing_model, 'is_free'):
                         existing_model.is_free = model_info.is_free
                    if hasattr(existing_model, 'cost_per_1k_input'):
                        existing_model.cost_per_1k_input = model_info.cost_per_1k_input
                        existing_model.cost_per_1k_output = model_info.cost_per_1k_output
            else:
                # Create new
                new_model = LLMModel(
                    provider=model_info.provider.value,
                    model_name=model_info.name,
                    model_type=inferred_type.value,
                    context_window=model_info.context_window,
                    supports_streaming=model_info.supports_streaming,
                    supports_function_calling=model_info.supports_function_calling,
                    supports_thinking=supports_thinking,
                    is_active=True,
                    # Hardware requirements and capabilities
                    cpu_supported=model_info.cpu_supported,
                    gpu_required=model_info.gpu_required,
                    parent_retrieval_supported=model_info.parent_retrieval_supported,
                    # Pricing
                    is_free=model_info.is_free,
                    cost_per_1k_input=model_info.cost_per_1k_input,
                    cost_per_1k_output=model_info.cost_per_1k_output
                )
                db_session.add(new_model)

        await db_session.commit()
        print("✅ Model sync complete")

        # Report uncertain models
        if uncertain_models:
            print(f"\n⚠️  UNCERTAIN MODELS ({len(uncertain_models)}):")
            print("   The following models need manual classification:")
            for model in uncertain_models:
                print(f"   - {model}")
            print("   → Check Ollama library or run: ollama show <model_name>")

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

    def get_all_models(self) -> Dict[str, List[ModelInfo]]:
        """Deprecated: use get_available_models with DB session"""
        return self._get_all_models_legacy()

    def _get_all_models_legacy(self) -> Dict[str, List[ModelInfo]]:
        """Get all available models from all providers"""
        all_models = {}

        for provider_type, provider in self.providers.items():
            try:
                all_models[provider_type.value] = provider.get_available_models()
            except Exception as e:
                print(f"⚠️  Failed to get models from {provider_type}: {e}")
                all_models[provider_type.value] = []

        return all_models


# =============================================================================
# Global Manager Instance
# =============================================================================

provider_manager = ProviderManager()
