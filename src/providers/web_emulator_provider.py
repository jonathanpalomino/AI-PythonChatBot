"""
Web Emulator Provider — Multiple AI services via browser emulation (no official API).

Supported services (registered as "models"):
  - copilot-365  → https://m365.cloud.microsoft/chat   (Microsoft Copilot 365)
  - grok         → https://grok.com/                   (xAI Grok)
  - kimi         → https://www.kimi.com/               (Moonshot AI Kimi)

Design constraints:
  - supports_function_calling  = False  (no tools API)
  - supports_message_history   = False  (history managed by browser tab)
  - supports_streaming         = False  (full response read from DOM)

The provider expects ONE ChatMessage whose content is the fully-enriched
turn (system + tools + user question). For multimodal content (images), the content
field can contain a list of content blocks formatted for ChatGPT's input format.

Conversation history lives in the browser tab and must NOT be closed on API shutdown.
"""

import asyncio
import base64
import logging
import mimetypes
from collections import OrderedDict
from dataclasses import dataclass
from typing import AsyncGenerator, Optional, Union, List, Dict

from playwright.async_api import Browser, Page, Playwright, async_playwright
from markdownify import markdownify as md

try:
    from pathlib import Path
    import aiofiles
    
    async def _read_file_bytes(file_id: str) -> bytes:
        """Read file content as bytes for image processing."""
        from src.services.file_service import FileService
        from src.core.database import get_db_session
        
        async with get_db_session() as session:
            file_service = FileService(session)
            file_record = await file_service.get_by_id(file_id)
            if not file_record:
                raise FileNotFoundError(f"File not found: {file_id}")
            
            file_path = Path(file_record.file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not accessible: {file_path}")
            
            async with aiofiles.open(file_path, 'rb') as f:
                return await f.read()
    
    _HAS_IMAGE_SUPPORT = True
except ImportError:
    _HAS_IMAGE_SUPPORT = False
    logger.warning("Image support not available - missing aiofiles or Path")

from src.config.settings import settings
from src.providers.manager import (
    BaseProvider,
    CancellableProviderMixin,
    ChatMessage,
    ChatResponse,
    ModelInfo,
    ModelType,
    ProviderType,
)

logger = logging.getLogger(__name__)

TAB_TITLE_PREFIX = "WebEmulator::"


# =============================================================================
# Service Configuration
# =============================================================================

@dataclass
class WebServiceConfig:
    """Configuration for a specific web-emulated AI service."""
    name: str                         # Human-readable display name
    model_id: str                     # Model ID used in API (model field)
    url: str                          # Starting URL
    textarea_selectors: list          # Ordered CSS selectors for the input area
    send_button_selectors: list       # Ordered CSS selectors for the send button
    response_container_selector: str  # Selector for response containers (comma-separated)
    generating_indicators: list       # Selectors that appear while generating
    ui_error_indicators: list         # Selectors that indicate a UI error
    login_indicators: list            # Selectors that indicate login is required
    auth_url_patterns: list           # URL substrings confirming authenticated state
    input_delay_ms: int = 15          # Keystroke delay in ms


# ─── Copilot 365 ──────────────────────────────────────────────────────────────
COPILOT365_CONFIG = WebServiceConfig(
    name="Microsoft Copilot 365",
    model_id="copilot-365",
    url="https://m365.cloud.microsoft/chat",
    textarea_selectors=[
        "textarea[placeholder*='Ask']",
        "textarea[placeholder*='Message']",
        "textarea[placeholder*='Pregunta']",
        "textarea[placeholder*='Mensaje']",
        "textarea[aria-label*='Ask']",
        "textarea[aria-label*='Pregunta']",
        "textarea[data-testid*='input']",
        "textarea",
    ],
    send_button_selectors=[
        "button[aria-label*='Send']",
        "button[aria-label*='Enviar']",
        "button[aria-label*='Submit']",
        "button[type='submit']",
    ],
    response_container_selector=(
        "[data-testid='response'], [data-testid='response-content'], "
        ".markdown, div[role='region']"
    ),
    generating_indicators=[
        "[aria-label*='loading']",
        "[aria-label*='cargando']",
        "div[class*='typing']",
        "div[class*='generating']",
        "span:text('Generating')",
        "span:text('Generando')",
        "div:text('Copilot is thinking')",
        "div:text('Copilot está pensando')",
    ],
    ui_error_indicators=[
        "div:text('limit')",
        "div:text('límite')",
        "div:text('something went wrong')",
        "div:text('algo salió mal')",
        "[data-testid='error-message']",
    ],
    login_indicators=[
        "button:text('Sign in')",
        "button:text('Iniciar sesión')",
        "a[href*='login.microsoftonline']",
        "input[name='loginfmt']",
        "input[type='email']",
    ],
    auth_url_patterns=["m365.cloud.microsoft"],
)

# ─── Grok (xAI) ───────────────────────────────────────────────────────────────
# NOTE: Selectors verified against grok.com structure — update if xAI changes DOM
GROK_CONFIG = WebServiceConfig(
    name="Grok (xAI)",
    model_id="grok",
    url="https://grok.com/",
    textarea_selectors=[
        "div[contenteditable='true']",
        "[data-testid='chat-input']",
        "textarea[placeholder*='Ask']",
        "textarea[placeholder*='Message']",
        "textarea[placeholder*='Grok']",
        "textarea",
    ],
    send_button_selectors=[
        "button[aria-label*='Send']",
        "button[aria-label*='Enviar']",
        "[data-testid='send-button']",
        "button[type='submit']",
    ],
    response_container_selector=(
        "[data-testid='message'], .message-bubble, "
        ".response-content, .prose, [class*='message-content']"
    ),
    generating_indicators=[
        "button[aria-label*='Stop']",
        "[aria-label*='Generating']",
        "div[class*='typing']",
        "div[class*='loading']",
        "[data-testid='generating']",
    ],
    ui_error_indicators=[
        "[data-testid='error']",
        "div[class*='error-message']",
        "div:text('Something went wrong')",
        "div:text('Alta demanda')",
        "div:text('Grok está bajo un uso intensivo')",
        "div:text('limit')",
    ],
    login_indicators=[
        "button:text('Sign in')",
        "button:text('Log in')",
        "a[href*='/auth']",
        "a[href*='login']",
        "[data-testid='login']",
    ],
    auth_url_patterns=["grok.com/chat", "grok.com/"],
    input_delay_ms=10,
)

# ─── Kimi (Moonshot AI) ───────────────────────────────────────────────────────
# NOTE: Selectors verified against kimi.com structure — update if Moonshot changes DOM
KIMI_CONFIG = WebServiceConfig(
    name="Kimi (Moonshot AI)",
    model_id="kimi",
    url="https://www.kimi.com/",
    textarea_selectors=[
        "textarea[placeholder*='Ask']",
        "textarea[placeholder*='Message']",
        "textarea[placeholder*='Type']",
        "[data-testid='chat-input']",
        ".chat-input textarea",
        "textarea",
    ],
    send_button_selectors=[
        "button[aria-label*='Send']",
        "[data-testid='send-button']",
        ".send-btn",
        "button[type='submit']",
    ],
    response_container_selector=(
        ".chat-message, .message-content, .answer-content, "
        "[data-testid='message'], .markdown-body"
    ),
    generating_indicators=[
        "div[class*='loading']",
        "div[class*='generating']",
        "div[class*='typing']",
        "[data-testid='generating']",
        "[aria-label*='loading']",
    ],
    ui_error_indicators=[
        ".error-message",
        "[data-testid='error']",
        "div:text('Something went wrong')",
    ],
    login_indicators=[
        "button:text('Login')",
        "button:text('Sign in')",
        "a[href*='login']",
        "[data-testid='login-btn']",
        ".login-btn",
    ],
    auth_url_patterns=["kimi.com"],
    input_delay_ms=10,
)

# ─── ChatGPT (OpenAI) ──────────────────────────────────────────────────────────
CHATGPT_CONFIG = WebServiceConfig(
    name="ChatGPT (OpenAI)",
    model_id="chatgpt",
    url="https://chatgpt.com/",
    textarea_selectors=[
        "#prompt-textarea",
        "textarea[placeholder*='ChatGPT']",
        "textarea[placeholder*='Message']",
        "div[contenteditable='true']",
        "textarea",
    ],
    send_button_selectors=[
        "#composer-submit-button",
        "button[data-testid='send-button']",
        "button[aria-label*='Send']",
        "button[aria-label*='Enviar']",
    ],
    response_container_selector="div[data-message-author-role='assistant']",
    generating_indicators=[
        "button[aria-label*='Stop']",
        "button[aria-label*='Detener']",
        "div.streaming",
        "div.result-streaming",
    ],
    ui_error_indicators=[
        "div:text('Something went wrong')",
        "div:text('There was an error')",
        "div:text('algo salió mal')",
        "div:text('error generating')",
    ],
    login_indicators=[
        "a[href*='/auth/login']",
        "button:text('Log in')",
        "button:text('Iniciar sesión')",
        "div:text('Sign in')",
    ],
    auth_url_patterns=["chatgpt.com"],
)

# Registry: model_id → config
SERVICE_REGISTRY: dict = {
    cfg.model_id: cfg
    for cfg in [COPILOT365_CONFIG, GROK_CONFIG, KIMI_CONFIG, CHATGPT_CONFIG]
}


# =============================================================================
# Exceptions
# =============================================================================

class AuthenticationError(Exception):
    """Browser session not authenticated or session has expired."""


# =============================================================================
# Provider
# =============================================================================

class WebEmulatorProvider(BaseProvider, CancellableProviderMixin):
    """
    Unified web emulation provider.

    All web AI services share a single browser instance connected via CDP.
    Each service is exposed as a separate model. Tab lifecycle is per
    (model_id, conversation_id) pair, using an LRU cache.
    """

    provider_type = ProviderType.WEB_EMULATOR

    def __init__(self) -> None:
        self.provider_type = ProviderType.WEB_EMULATOR
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        # Cache key: "{model_id}::{conversation_id}"
        self._tab_cache: OrderedDict = OrderedDict()
        self.logger = logger
        super().__init__()

    def _validate_credentials(self):
        """Auth is managed by the browser session — no credentials needed."""
        pass

    # ─── Browser / CDP ────────────────────────────────────────────────────────

    async def _ensure_browser(self) -> None:
        if self._browser is not None:
            return

        self._playwright = await async_playwright().start()
        port = getattr(settings, "WEB_EMULATOR_REMOTE_DEBUGGING_PORT", 9222)
        endpoint = f"http://127.0.0.1:{port}"

        try:
            self._browser = await self._playwright.chromium.connect_over_cdp(endpoint)
            self.logger.info(f"Connected to browser via CDP on port {port}")
        except Exception as exc:
            await self._playwright.stop()
            self._playwright = None
            raise ConnectionError(
                f"Could not connect to browser on port {port}.\n"
                f"Start Edge with remote debugging:\n"
                f"  msedge.exe --remote-debugging-port={port} "
                f'--user-data-dir="%LOCALAPPDATA%\\Microsoft\\Edge\\User Data" '
                f"--start-minimized\n"
                f"(CDP endpoint: http://127.0.0.1:{port})"
            ) from exc

        await self._restore_tab_cache()

    async def _restore_tab_cache(self) -> None:
        """Re-associate open tabs with cache entries after an API restart."""
        if not self._browser:
            return
        context = self._browser.contexts[0]
        restored = 0
        for page in context.pages:
            try:
                title = await page.title()
                if title.startswith(TAB_TITLE_PREFIX):
                    # Format: "WebEmulator::{model_id}::{conversation_id}"
                    rest = title[len(TAB_TITLE_PREFIX):]
                    parts = rest.split("::", 1)
                    if len(parts) == 2:
                        cache_key = f"{parts[0]}::{parts[1]}"
                        self._tab_cache[cache_key] = page
                        restored += 1
            except Exception:
                continue
        if restored:
            self.logger.info(f"Restored {restored} web emulator tab(s) from previous session")

    # ─── Tab Management ───────────────────────────────────────────────────────

    async def _get_or_create_tab(self, model_id: str, conversation_id: str) -> Page:
        config = SERVICE_REGISTRY[model_id]
        cache_key = f"{model_id}::{conversation_id}"

        if cache_key in self._tab_cache:
            page = self._tab_cache[cache_key]
            if not page.is_closed():
                self._tab_cache.move_to_end(cache_key)
                return page
            del self._tab_cache[cache_key]
            self.logger.warning(f"Tab {cache_key} was closed externally. Creating new one.")

        context = self._browser.contexts[0]
        page = await context.new_page()
        await page.goto(config.url, wait_until="domcontentloaded")

        # Mark title for restoration after API restart
        tab_title = f"{TAB_TITLE_PREFIX}{model_id}::{conversation_id}"
        await page.evaluate(f"document.title = '{tab_title}'")

        # Confirm textarea is reachable before caching the tab
        textarea = await self._find_element(page, config.textarea_selectors, timeout=30000)
        if not textarea:
            await page.close()
            raise RuntimeError(
                f"Could not find input textarea on {config.url}. "
                f"Verify that the browser session is authenticated for {config.name}."
            )

        self._tab_cache[cache_key] = page
        self.logger.info(f"New tab created: {cache_key}")

        # LRU eviction
        max_tabs: int = getattr(settings, "WEB_EMULATOR_MAX_TABS", 20)
        if len(self._tab_cache) > max_tabs:
            oldest_key, _ = next(iter(self._tab_cache.items()))
            if oldest_key != cache_key:
                del self._tab_cache[oldest_key]
                self.logger.debug(f"LRU evicted tab: {oldest_key}")

        return page

    # ─── DOM Utilities ────────────────────────────────────────────────────────

    @staticmethod
    async def _find_element(page: Page, selectors: list, timeout: int = 5000, state: str = "attached"):
        for selector in selectors:
            try:
                element = await page.wait_for_selector(selector, timeout=timeout, state=state)
                if element:
                    return element
            except Exception:
                continue
        return None

    async def _wait_for_response_complete(self, page: Page, config: WebServiceConfig) -> None:
        timeout_ms = getattr(settings, "WEB_EMULATOR_TIMEOUT", 60000)
        total_timeout_s = timeout_ms / 1000
        deadline = asyncio.get_event_loop().time() + total_timeout_s

        # Phase 1: Wait for generation to start OR an error to appear
        generation_started = False
        while asyncio.get_event_loop().time() < (asyncio.get_event_loop().time() + 8): # 8s max to start
            # Check for errors first
            for err_sel in config.ui_error_indicators:
                try:
                    el = await page.query_selector(err_sel)
                    if el and await el.is_visible():
                        text = await el.inner_text()
                        raise RuntimeError(f"{config.name} error: {text}")
                except RuntimeError: raise
                except Exception: continue

            # Check if generation started
            for indicator in config.generating_indicators:
                try:
                    el = await page.query_selector(indicator)
                    if el and await el.is_visible():
                        generation_started = True
                        break
                except Exception: continue
            
            if generation_started:
                break
            await page.wait_for_timeout(500)

        if not generation_started:
            self.logger.warning(f"Generation did not start for {config.name} within 8s. Checking final state.")

        # Phase 2: Poll until all generating indicators disappear
        while asyncio.get_event_loop().time() < deadline:
            # Always check for errors during generation
            for err_sel in config.ui_error_indicators:
                try:
                    el = await page.query_selector(err_sel)
                    if el and await el.is_visible():
                        text = await el.inner_text()
                        raise RuntimeError(f"{config.name} error during generation: {text}")
                except RuntimeError: raise
                except Exception: continue

            any_generating = False
            for indicator in config.generating_indicators:
                try:
                    element = await page.query_selector(indicator)
                    if element and await element.is_visible():
                        any_generating = True
                        break
                except Exception: continue

            if not any_generating:
                await page.wait_for_timeout(1000)  # DOM stabilization
                return

            await page.wait_for_timeout(500)

        raise TimeoutError(
            f"{config.name} did not complete response within {timeout_ms}ms."
        )

    async def _extract_new_response(self, page: Page, config: WebServiceConfig, count_before: int) -> str:
        locator = page.locator(config.response_container_selector)
        total = await locator.count()

        if total > count_before:
            new_responses = []
            for i in range(count_before, total):
                try:
                    element = locator.nth(i)
                    html = await element.inner_html()
                    content = md(html, heading_style="ATX", bullets="-").strip()
                    if content:
                        new_responses.append(content)
                except Exception:
                    continue
            
            if new_responses:
                self.logger.info(f"Detected {len(new_responses)} alternative responses for {config.name}")
                combined = "\n\n---\n\n".join(new_responses)
                return combined
            else:
                raise RuntimeError(f"No valid new response element found for {config.name} after detecting {total - count_before} new elements.")
        
        elif total > 0:
            self.logger.warning(f"No new response element for {config.name}. Using last.")
            element = locator.last
            html = await element.inner_html()
            content = md(html, heading_style="ATX", bullets="-").strip()
            return content
        
        else:
            raise RuntimeError(f"No response container found in {config.name} DOM.")

    # ─── Public Interface ─────────────────────────────────────────────────────

    async def _prepare_message_content(self, message: ChatMessage) -> str:
        """
        Prepare message content for web emulator.
        Converts multimodal content blocks to text format supported by ChatGPT.
        For images, extracts base64 data and converts to description.
        """
        if isinstance(message.content, str):
            return message.content
        
        # Handle multimodal content blocks
        text_parts = []
        
        for block in message.content:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "image_url":
                # Extract image URL and convert to description
                image_url = block.get("image_url", {}).get("url", "")
                if image_url.startswith("data:image/"):
                    try:
                        # Extract base64 data
                        base64_data = image_url.split(";base64,")[-1]
                        image_bytes = base64.b64decode(base64_data)
                        
                        # Detect image type from data URL or fallback
                        image_type = image_url.split(":")[1].split(";")[0] if ":" in image_url else "image/jpeg"
                        
                        image_info = f"\n[IMAGE: {image_type} ({len(image_bytes)} bytes)]"
                        text_parts.append(image_info)
                        self.logger.debug(f"Processed image attachment: {image_type}")
                    except Exception as exc:
                        self.logger.warning(f"Failed to process image data: {exc}")
                        text_parts.append(f"\n[IMAGE ATTACHED - processing failed]")
                else:
                    text_parts.append(f"\n[IMAGE: {image_url}]")
        
        return "\n".join(text_parts)

    async def _attach_image_to_browser(self, page: Page, file_id: str) -> None:
        """
        Attempt to attach image to browser tab via CDP.
        This is a best-effort operation - if it fails, we'll append image info to text.
        """
        if not _HAS_IMAGE_SUPPORT:
            self.logger.info("Image support not available - attachments will be described in text")
            return
        
        try:
            # Try to find and interact with file upload input if exists
            upload_selectors = [
                "input[type='file']",
                "input[accept*='image']",
                "[data-testid*='upload']",
                ".upload-button",
                "[role='button'][aria-label*='Attach']",
                "[aria-label*='Upload']",
            ]
            
            for selector in upload_selectors:
                try:
                    file_input = await page.wait_for_selector(selector, timeout=2000, state="visible")
                    if file_input and await file_input.is_enabled():
                        try:
                            # Read actual file content
                            image_bytes = await _read_file_bytes(file_id)
                            
                            # Find file input and trigger upload
                            file_input_input = await page.evaluate_handle("""
                                async (element, bytes) => {
                                    return new Promise((resolve) => {
                                        element.setAttribute('files', '');
                                        
                                        const changeEvent = new Event('change', { bubbles: true });
                                        element.dispatchEvent(changeEvent);
                                        
                                        const fileInput = element.querySelector('input');
                                        if (fileInput) {
                                            const fileList = new DataTransfer();
                                            const blob = new Blob([bytes], { type: 'image/jpeg' });
                                            fileList.items.add(new File([blob], 'image.jpg'));
                                            fileInput.files = fileList.files;
                                            
                                            const inputEvent = new Event('input', { bubbles: true });
                                            fileInput.dispatchEvent(inputEvent);
                                        }
                                        
                                        resolve(true);
                                    });
                                }
                            """, [file_input, image_bytes])
                            
                            self.logger.info(f"Image attachment successful via {selector}")
                            return
                        except Exception as upload_exc:
                            self.logger.warning(f"Upload failed: {upload_exc} - will use text description")
                            return
                except Exception:
                    continue
            
            self.logger.info("No file upload UI found - attachments will be described in text")
        except Exception as exc:
            self.logger.warning(f"Image attachment attempt failed: {exc} - will use text description")

    async def chat(
        self,
        messages: list,
        model: str = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list | None = None,
        **kwargs,
    ) -> ChatResponse:
        config = SERVICE_REGISTRY.get(model)
        if not config:
            raise ValueError(
                f"Unknown web emulator model: '{model}'. "
                f"Available: {list(SERVICE_REGISTRY.keys())}"
            )

        await self._ensure_browser()

        conversation = kwargs.get("conversation")
        if not conversation:
            raise ValueError("WebEmulatorProvider requires 'conversation' in kwargs.")

        conversation_id = str(conversation.id)
        page = await self._get_or_create_tab(model, conversation_id)

        if not messages:
            raise ValueError("WebEmulatorProvider.chat() requires at least one message.")

        user_message = messages[-1]
        
        # Prepare content: convert multimodal to text with image descriptions
        user_message_text = await self._prepare_message_content(user_message)
        
        # Attach any images found in metadata
        if user_message.metadata and user_message.metadata.get("attachments"):
            for attachment in user_message.metadata["attachments"]:
                if attachment.get("file_type", "").startswith("image/"):
                    await self._attach_image_to_browser(page, attachment.get("file_id", ""))

        retry_attempts = getattr(settings, "WEB_EMULATOR_RETRY_ATTEMPTS", 3)
        last_error: Exception | None = None

        for attempt in range(retry_attempts + 1):
            try:
                count_before = await page.locator(config.response_container_selector).count()

                textarea = await self._find_element(page, config.textarea_selectors, timeout=5000)
                if not textarea:
                    raise RuntimeError(
                        f"Could not find textarea for {config.name}. "
                        f"Verify the tab is still on {config.url}."
                    )

                # 1. Clear and fill directly (much faster and avoids newline issues)
                await textarea.click()
                await textarea.fill("")
                await page.wait_for_timeout(100)
                await textarea.fill(user_message_text)
                await page.wait_for_timeout(500)

                # 2. Try to click send button
                sent = False
                for _ in range(3):
                    send_btn = await self._find_element(page, config.send_button_selectors, timeout=1000)
                    if send_btn and await send_btn.is_visible() and await send_btn.is_enabled():
                        await send_btn.click()
                        sent = True
                        break
                    await page.wait_for_timeout(500)

                # 3. Fallback to keyboard keys if button fails
                if not sent:
                    await page.keyboard.press("Control+Enter")
                    await page.wait_for_timeout(200)
                    await page.keyboard.press("Enter")
                    await page.keyboard.press("Enter")

                await self._wait_for_response_complete(page, config)
                content = await self._extract_new_response(page, config, count_before)

                if not content:
                    raise RuntimeError(f"{config.name} returned an empty response.")

                self.logger.info(
                    f"Response received from {config.name} for conv {conversation_id} "
                    f"({len(content)} chars)"
                )

                return ChatResponse(
                    content=content,
                    model=model,
                    provider=self.provider_type.value,
                    tokens_used=None,
                    finish_reason="stop",
                    metadata={
                        "service": config.name,
                        "url": config.url,
                        "browser_session": "cdp",
                        "conversation_tab_id": conversation_id,
                    },
                )

            except (AuthenticationError, ConnectionError):
                raise

            except Exception as exc:
                last_error = exc
                if attempt < retry_attempts:
                    self.logger.warning(
                        f"Attempt {attempt + 1}/{retry_attempts + 1} failed "
                        f"for {config.name}: {exc}. Retrying..."
                    )
                    await page.wait_for_timeout(2000)
                else:
                    self.logger.error(
                        f"All attempts failed for {config.name}: {exc}",
                        exc_info=True,
                    )
                    raise

        raise last_error

    async def stream_chat(
        self,
        messages: list,
        model: str = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list | None = None,
        **kwargs,
    ) -> AsyncGenerator:
        """Simulated streaming: delivers full response as a single chunk."""
        response = await self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            **kwargs,
        )
        yield response.content

    def get_available_models(self) -> list:
        return [
            ModelInfo(
                name="copilot-365",
                provider=ProviderType.WEB_EMULATOR,
                context_window=128000,
                supports_function_calling=False,
                supports_streaming=False,
                supports_message_history=False,
                model_type=ModelType.CHAT,
                is_free=True,
                cpu_supported=True,
                gpu_required=False,
                cost_per_1k_input=0.0,
                cost_per_1k_output=0.0,
            ),
            ModelInfo(
                name="grok",
                provider=ProviderType.WEB_EMULATOR,
                context_window=131072,
                supports_function_calling=False,
                supports_streaming=False,
                supports_message_history=False,
                model_type=ModelType.CHAT,
                is_free=True,
                cpu_supported=True,
                gpu_required=False,
                cost_per_1k_input=0.0,
                cost_per_1k_output=0.0,
            ),
            ModelInfo(
                name="kimi",
                provider=ProviderType.WEB_EMULATOR,
                context_window=128000,
                supports_function_calling=False,
                supports_streaming=False,
                supports_message_history=False,
                model_type=ModelType.CHAT,
                is_free=True,
                cpu_supported=True,
                gpu_required=False,
                cost_per_1k_input=0.0,
                cost_per_1k_output=0.0,
            ),
            ModelInfo(
                name="chatgpt",
                provider=ProviderType.WEB_EMULATOR,
                context_window=128000,
                supports_function_calling=False,
                supports_streaming=False,
                supports_message_history=False,
                model_type=ModelType.CHAT,
                is_free=True,
                cpu_supported=True,
                gpu_required=False,
                cost_per_1k_input=0.0,
                cost_per_1k_output=0.0,
            ),
        ]

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Release CDP references without closing the browser or its tabs."""
        self._tab_cache.clear()
        self._browser = None
        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception:
                pass
            self._playwright = None
        self.logger.info("WebEmulatorProvider: CDP references released (browser still active)")
