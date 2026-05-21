# =============================================================================
# main.py
# FastAPI application entry point
# =============================================================================
"""
RAG Chatbot API - Main application
"""
import sys
import asyncio

# ---------------------------------------------------------------------------
# Windows: Playwright (and any code using create_subprocess_exec) requires
# ProactorEventLoop. The default SelectorEventLoop in Python 3.8+ on Windows
# raises NotImplementedError for subprocess creation.
# Must be set BEFORE uvicorn / any asyncio code is imported.
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import atexit
import platform
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config.settings import settings
from src.database.connection import init_db, close_db_connections
from src.tools.base_tool import tool_registry
from src.tools.codebase_tool import CodebaseTool
from src.tools.http_tool import HTTPTool
from src.tools.obsidian_note_tool import ObsidianNoteTool  # New: Obsidian vault loader
from src.tools.rag_tool import RAGTool
from src.tools.sql_tool import SQLTool
from src.tools.git_tool import GitTool
from src.utils.health_checker import HealthChecker
from src.utils.logger import get_logger
from src.api.v1 import auth as auth_router

logger = get_logger(__name__)

# Import routers
from src.api.v1 import conversations, messages, prompts, files, collections, tools, projects, collections_ingest, oauth_router

# Global variable to hold Qdrant process
qdrant_process = None
# Global variable to hold Redis process
redis_process = None
# Global variable to hold Celery worker process
celery_worker_process = None
# Global variable to hold Celery flower process
celery_flower_process = None
# Flags to track if the app started the services (to avoid stopping external ones)
app_started_qdrant = False
app_started_redis = False
app_started_celery_worker = False
app_started_celery_flower = False


# =============================================================================
# Redis Process Management
# =============================================================================

def check_redis_health() -> bool:
    """
    Check if Redis is already running and accessible
    Returns True if Redis is healthy, False otherwise
    """
    return HealthChecker.check_socket('127.0.0.1', 6379, timeout=1)


def start_redis_windows():
    """
    Start Redis on Windows using the local start.bat
    Opens a separate console window for Redis
    """
    global redis_process

    # Check if Redis start.bat exists
    redis_dir = Path(__file__).parent / "redis"
    redis_start_bat = redis_dir / "start.bat"

    if not redis_start_bat.exists():
        logger.warning(f"Redis start script not found at: {redis_start_bat}")
        print(f"⚠️  Redis start script not found at: {redis_start_bat}")
        print(f"   Please start Redis manually")
        return False

    try:
        logger.info(f"Attempting to start Redis from: {redis_start_bat}")
        print(f"🚀 Starting Redis...")

        # Start Redis using cmd.exe explicitly to ensure window visibility
        # /c runs the command and then terminates the shell (but start.bat has pause at end)
        # We need the shell to stay alive if redis stays alive?
        # redis-server runs in foreground. cmd /c start.bat waits for it.
        # This allows capturing the PID of the cmd process.
        redis_process = subprocess.Popen(
            ["cmd.exe", "/c", str(redis_start_bat.name)], # Run by name, relying on cwd
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            cwd=str(redis_dir)
        )

        logger.info(f"Redis start script launched (PID: {redis_process.pid})")
        print(f"✅ Redis start script launched")

        # Register cleanup on exit
        atexit.register(stop_redis)

        # Wait for Redis to initialize with a loop
        print("   Waiting for Redis to initialize...", end="", flush=True)
        import time

        # Retry for up to 10 seconds
        for i in range(10):
            time.sleep(1)
            print(".", end="", flush=True)
            if check_redis_health():
                print(" Done!")
                print("✅ Redis is now accessible")
                return True

        print("\n⚠️  Redis start script launched but health check timed out")
        print("   Please check the Redis console window for errors.")
        return False

    except Exception as e:
        logger.error(f"Failed to start Redis: {e}", exc_info=True)
        print(f"\n❌ Failed to start Redis: {e}")
        return False


def ensure_redis_running():
    """
    Main function to ensure Redis is running
    1. First checks if Redis is already accessible
    2. If not, attempts to start it on Windows
    """
    logger.info("Checking Redis availability...")

    # First, check if Redis is already running
    if check_redis_health():
        return True

    # Redis is not running, try to start it
    logger.info("Redis is not running, attempting to start it...")
    print("🔄 Redis is not running, attempting to start it...")

    system = platform.system()

    if system == "Windows":
        return start_redis_windows()
    else:
        logger.info(f"Auto-start for Redis not implemented for {system}, please start manually if needed.")
        print(f"ℹ️  Redis auto-start only available on Windows. Please start manually if not running.")
        return False


def stop_redis():
    """
    Stop Redis server gracefully (only if we started the process object)
    """
    global redis_process

    if not app_started_redis:
        return

    if redis_process is None:
        return

    try:
        # Check if process is still running (this check is a bit loose for Popen with shell=True/bat)
        if redis_process.poll() is None:
            logger.info(f"Stopping Redis process (PID: {redis_process.pid})")
            print(f"🛑 Stopping Redis process...")

            # Terminate the process
            redis_process.terminate()

            # Since we launched a bat file, the actual redis-server might be a child.
            # Simple terminate might just kill the cmd wrapper.
            # But for simpledev usage this is often 'good enough' or we rely on the OS cleaning up console windows.
            # A more robust solution would kill the tree, but let's stick to simple consistency with Qdrant logic first.

            try:
                redis_process.wait(timeout=5)
                logger.info("Redis process stopped")
                print("✅ Redis process stopped")
            except subprocess.TimeoutExpired:
                redis_process.kill()
                logger.warning("Redis process force-killed after timeout")
                print("⚠️  Redis process force-killed after timeout")
        else:
            logger.info("Redis process already terminated")

    except Exception as e:
        logger.error(f"Error stopping Redis: {e}", exc_info=True)
        print(f"❌ Error stopping Redis: {e}")



# =============================================================================
# Celery Process Management
# =============================================================================

def start_celery_worker_windows():
    """
    Start Celery Worker on Windows using celery\\start_worker.bat
    Opens a separate console window
    """
    global celery_worker_process, app_started_celery_worker

    celery_dir = Path(__file__).parent / "celery"
    celery_worker_bat = celery_dir / "start_worker.bat"

    if not celery_worker_bat.exists():
        logger.warning(f"Celery worker start script not found at: {celery_worker_bat}")
        print(f"WARNING: Celery worker start script not found at: {celery_worker_bat}")
        print("Please start Celery Worker manually")
        return False

    try:
        logger.info(f"Attempting to start Celery Worker from: {celery_worker_bat}")
        print("Starting Celery Worker...")

        celery_worker_process = subprocess.Popen(
            ["cmd.exe", "/c", str(celery_worker_bat.name)],
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            cwd=str(celery_dir)
        )

        logger.info(f"Celery Worker start script launched (PID: {celery_worker_process.pid})")
        print("Celery Worker started")
        app_started_celery_worker = True

        # Register cleanup on exit (best effort)
        atexit.register(stop_celery_worker)

        return True

    except Exception as e:
        logger.error(f"Failed to start Celery Worker: {e}", exc_info=True)
        print(f"Failed to start Celery Worker: {e}")
        return False


def start_celery_flower_windows():
    """
    Start Flower Monitor on Windows using celery\\start_flower.bat
    Opens a separate console window
    """
    global celery_flower_process, app_started_celery_flower

    celery_dir = Path(__file__).parent / "celery"
    celery_flower_bat = celery_dir / "start_flower.bat"

    if not celery_flower_bat.exists():
        logger.warning(f"Flower start script not found at: {celery_flower_bat}")
        print(f"WARNING: Flower start script not found at: {celery_flower_bat}")
        print("Please start Flower manually")
        return False

    try:
        logger.info(f"Attempting to start Flower from: {celery_flower_bat}")
        print("Starting Flower Monitor...")

        celery_flower_process = subprocess.Popen(
            ["cmd.exe", "/c", str(celery_flower_bat.name)],
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            cwd=str(celery_dir)
        )

        logger.info(f"Flower start script launched (PID: {celery_flower_process.pid})")
        print("Flower Monitor started")
        app_started_celery_flower = True

        # Register cleanup on exit (best effort)
        atexit.register(stop_celery_flower)

        return True

    except Exception as e:
        logger.error(f"Failed to start Flower: {e}", exc_info=True)
        print(f"Failed to start Flower: {e}")
        return False


def ensure_celery_running():
    """
    Ensure Celery Worker and Flower are running
    Checks each service separately and starts only what's needed
    """
    logger.info("Checking Celery services availability...")

    worker_running = HealthChecker.check_celery_worker()
    flower_running = HealthChecker.check_celery_flower(port=5555)

    if worker_running and flower_running:
        logger.info("Both Celery Worker and Flower are already running")
        print("✅ Celery Worker and Flower are already running")
        return True

    system = platform.system()
    if system != "Windows":
        logger.info(f"Auto-start for Celery not implemented for {system}, please start manually.")
        print(f"INFO: Celery auto-start only available on Windows. Please start manually if not running.")
        return False

    # Start Worker if not running
    if not worker_running:
        logger.info("Celery Worker is not running, attempting to start it...")
        print("🔄 Celery Worker is not running, attempting to start it...")
        if not start_celery_worker_windows():
            return False
    else:
        logger.info("Celery Worker is already running")
        print("✅ Celery Worker is already running")

    # Start Flower if not running
    if not flower_running:
        logger.info("Flower is not running, attempting to start it...")
        print("🔄 Flower is not running, attempting to start it...")
        if not start_celery_flower_windows():
            return False
    else:
        logger.info("Flower is already running")
        print("✅ Flower is already running")

    return True


def stop_celery_worker():
    """
    Stop Celery Worker (best effort, only if we started it)
    """
    global celery_worker_process

    if not app_started_celery_worker:
        return

    try:
        logger.info("Stopping Celery Worker...")
        print("🛑 Stopping Celery Worker...")

        subprocess.run(
            ["taskkill", "/F", "/T", "/FI", "WINDOWTITLE eq Celery Worker"],
            capture_output=True
        )

        logger.info("Celery Worker stopped")
        print("✅ Celery Worker stopped")
    except Exception as e:
        logger.warning(f"Error stopping Celery Worker: {e}")


def stop_celery_flower():
    """
    Stop Flower Monitor (best effort, only if we started it)
    """
    global celery_flower_process

    if not app_started_celery_flower:
        return

    try:
        logger.info("Stopping Flower Monitor...")
        print("🛑 Stopping Flower Monitor...")

        subprocess.run(
            ["taskkill", "/F", "/T", "/FI", "WINDOWTITLE eq Flower Monitor"],
            capture_output=True
        )

        logger.info("Flower Monitor stopped")
        print("✅ Flower Monitor stopped")
    except Exception as e:
        logger.warning(f"Error stopping Flower: {e}")


def stop_celery():
    """
    Stop both Celery Worker and Flower (best effort, only if we started them)
    """
    stop_celery_worker()
    stop_celery_flower()



# =============================================================================
# Qdrant Process Management
# =============================================================================

def check_qdrant_health() -> bool:
    """
    Check if Qdrant is already running and accessible
    Returns True if Qdrant is healthy, False otherwise
    """
    if HealthChecker.check_http(f"{settings.QDRANT_URL}/healthz", timeout=2):
        logger.info("Qdrant is already running and healthy")
        print("✅ Qdrant is already running")
        return True
    return False


def start_qdrant_windows():
    """
    Start Qdrant on Windows using the local executable
    Opens a separate console window for Qdrant
    """
    global qdrant_process

    # Check if Qdrant executable exists
    qdrant_exe = Path(__file__).parent / "qdrant" / "qdrant.exe"

    if not qdrant_exe.exists():
        logger.warning(f"Qdrant executable not found at: {qdrant_exe}")
        print(f"⚠️  Qdrant executable not found at: {qdrant_exe}")
        print(f"   Please download Qdrant or start it manually")
        return False

    try:
        # Start Qdrant in a new console window
        # CREATE_NEW_CONSOLE flag opens a separate window (like clicking .exe manually)
        qdrant_process = subprocess.Popen(
            [str(qdrant_exe)],
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            cwd=str(qdrant_exe.parent)
        )

        logger.info(f"Qdrant started in separate window (PID: {qdrant_process.pid})")
        print(f"✅ Qdrant started in separate window (PID: {qdrant_process.pid})")

        # Register cleanup on exit
        atexit.register(stop_qdrant)

        # Wait a bit for Qdrant to initialize
        import time
        print("   Waiting for Qdrant to initialize...")
        time.sleep(3)

        # Verify it started successfully
        if check_qdrant_health():
            print("✅ Qdrant is now accessible")
            return True
        else:
            logger.warning("Qdrant process started but health check failed")
            print("⚠️  Qdrant started but may need more time to initialize")
            return True

    except Exception as e:
        logger.error(f"Failed to start Qdrant: {e}", exc_info=True)
        print(f"❌ Failed to start Qdrant: {e}")
        return False


def start_qdrant_linux():
    """
    Start Qdrant on Linux
    Attempts to use docker, otherwise provides instructions
    """
    global qdrant_process

    # Check if docker is available
    try:
        subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            check=True
        )
        docker_available = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        docker_available = False

    if docker_available:
        try:
            # Try to start Qdrant using docker
            logger.info("Starting Qdrant using Docker...")
            print("🐳 Starting Qdrant using Docker...")

            qdrant_process = subprocess.Popen(
                [
                    "docker", "run", "-d",
                    "--name", "qdrant",
                    "-p", "6333:6333",
                    "-p", "6334:6334",
                    "qdrant/qdrant"
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            qdrant_process.wait()

            if qdrant_process.returncode == 0:
                logger.info("Qdrant Docker container started")
                print("✅ Qdrant started in Docker container")

                # Wait for initialization
                import time
                time.sleep(3)

                if check_qdrant_health():
                    print("✅ Qdrant is now accessible")
                    return True

            else:
                stderr = qdrant_process.stderr.read().decode() if qdrant_process.stderr else ""
                logger.warning(f"Docker start failed: {stderr}")
                print(f"⚠️  Could not start Qdrant in Docker: {stderr}")

        except Exception as e:
            logger.error(f"Failed to start Qdrant with Docker: {e}", exc_info=True)
            print(f"❌ Failed to start Qdrant with Docker: {e}")

    # If docker not available or failed, provide instructions
    print("⚠️  Qdrant is not running. Please start it manually:")
    print("   Option 1 (Docker): docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant")
    print("   Option 2 (Binary): Download from https://github.com/qdrant/qdrant/releases")
    logger.warning("Qdrant not started - manual intervention required")

    return False


def ensure_qdrant_running():
    """
    Main function to ensure Qdrant is running
    1. First checks if Qdrant is already accessible
    2. If not, attempts to start it based on the OS
    """
    logger.info("Checking Qdrant availability...")

    # First, check if Qdrant is already running
    if check_qdrant_health():
        return True

    # Qdrant is not running, try to start it
    logger.info("Qdrant is not running, attempting to start it...")
    print("🔄 Qdrant is not running, attempting to start it...")

    system = platform.system()

    if system == "Windows":
        success = start_qdrant_windows()
        if success:
            app_started_qdrant = True
        return success
    elif system == "Linux":
        success = start_qdrant_linux()
        if success:
            app_started_qdrant = True
        return success
    elif system == "Darwin":  # macOS
        # Similar to Linux, could use docker or homebrew
        logger.info("macOS detected, attempting Docker start...")
        success = start_qdrant_linux()  # Reuse Linux logic for now
        if success:
            app_started_qdrant = True
        return success
    else:
        logger.warning(f"Unsupported OS: {system}")
        print(f"⚠️  Unsupported OS: {system}. Please start Qdrant manually.")
        return False


def stop_qdrant():
    """
    Stop Qdrant server gracefully (only if we started it)
    """
    global qdrant_process

    if qdrant_process is None:
        return

    try:
        # Check if it's still running
        if qdrant_process.poll() is None:
            logger.info(f"Stopping Qdrant (PID: {qdrant_process.pid})")
            print(f"🛑 Stopping Qdrant (PID: {qdrant_process.pid})")

            # Terminate the process
            qdrant_process.terminate()

            # Wait up to 5 seconds for graceful shutdown
            try:
                qdrant_process.wait(timeout=5)
                logger.info("Qdrant stopped gracefully")
                print("✅ Qdrant stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't stop
                qdrant_process.kill()
                logger.warning("Qdrant force-killed after timeout")
                print("⚠️  Qdrant force-killed after timeout")
        else:
            logger.info("Qdrant process already terminated")

    except Exception as e:
        logger.error(f"Error stopping Qdrant: {e}", exc_info=True)
        print(f"❌ Error stopping Qdrant: {e}")


async def sync_tool_templates(tools_to_sync: List):
    """
    Synchronize tool templates from Python code to database.
    This ensures that database templates match the current code implementation.

    Uses ToolService through service_factory (no direct database access).

    Args:
        tools_to_sync: List of tool instances to synchronize
    """
    from src.utils.service_factory import get_tool_service

    try:
        async with get_tool_service() as tool_service:
            created_count, updated_count = await tool_service.sync_tool_templates(tools_to_sync)
            print(f"✅ Tool templates synchronized: {created_count} created, {updated_count} updated")
    except Exception as e:
        logger.error(f"Template synchronization failed: {e}", exc_info=True)
        print(f"❌ Template synchronization failed: {e}")
        raise


async def initialize_tool_types():
    """
    Initialize tool types from the database at application startup.
    This ensures that custom tool types are loaded and available.

    Uses ToolService through service_factory (no direct database access).
    """
    from src.utils.service_factory import get_tool_service

    try:
        async with get_tool_service() as tool_service:
            # Load tool types through service (arquitectura en capas)
            tool_types = await tool_service.load_tool_types_for_initialization()
            logger.info(
                f"Tool types loaded from database: {list(tool_types.keys())}",
                extra={"tool_types": list(tool_types.keys())}
            )
            print(f"Tool types loaded: {len(tool_types)} types")
    except Exception as e:
        logger.error(f"Failed to initialize tool types: {e}", exc_info=True)
        print(f"Failed to initialize tool types: {e}")


# =============================================================================
# Lifespan Events
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - handles startup and shutdown"""
    logger.info(
        "Starting Chatbot API",
        extra={"version": settings.APP_VERSION, "environment": settings.ENVIRONMENT}
    )
    print("🚀 Starting Chatbot API...")

    # Start Qdrant
    ensure_qdrant_running()

    # Start Redis
    redis_ok = ensure_redis_running()

    # Start Celery (Worker + Flower)
    if redis_ok:
        ensure_celery_running()
    else:
        logger.warning("Redis unavailable; skipping Celery auto-start")
        print("⚠️  Redis unavailable; skipping Celery auto-start")

    # Initialize database
    init_db()
    logger.info("Database initialized")

    # ---------------------------------------------------------------------------
    # [Módulo C] Bootstrap Auth: roles y permisos del sistema (idempotente)
    # ---------------------------------------------------------------------------
    from src.utils.service_factory import get_permission_service
    try:
        async with get_permission_service() as perm_svc:
            auth_result = await perm_svc.bootstrap_system_roles()
            print(f"✅ Auth bootstrap: {auth_result['roles_created']} roles, "
                  f"{auth_result['permissions_created']} permissions created")
    except Exception as e:
        logger.error(f"Auth bootstrap failed: {e}", exc_info=True)
        print(f"❌ Auth bootstrap failed: {e}")
    # ---------------------------------------------------------------------------

    # Import service factory for tool operations
    from src.utils.service_factory import get_tool_service, get_session_for_provider_sync

    # Instantiate physical tools (no database needed)
    rag_tool = RAGTool()
    http_tool = HTTPTool()
    sql_tool = SQLTool()
    codebase_tool = CodebaseTool()
    obsidian_tool = ObsidianNoteTool()
    git_tool = GitTool()

    # Register physical tool instances (solo una vez)
    tool_registry.register(rag_tool)
    tool_registry.register(http_tool)
    tool_registry.register(sql_tool)
    tool_registry.register(codebase_tool)
    tool_registry.register(obsidian_tool)
    tool_registry.register(git_tool)

    logger.info(f"Physical tools registered: {len(tool_registry.list_names())}")
    print(f"✅ {len(tool_registry.list_names())} physical tools registered")

    # Sync tool templates to database using ToolService
    await sync_tool_templates(
        tools_to_sync=[rag_tool, http_tool, sql_tool, codebase_tool, obsidian_tool, git_tool]
    )

    # Initialize tool types from database
    await initialize_tool_types()

    # Load and register ONLY custom tool instances (NOT templates)
    try:
        async with get_tool_service() as tool_service:
            # ✅ IMPORTANTE: Obtener SOLO instancias personalizadas, NO templates
            # Los templates solo están en BD como referencia, no se registran
            custom_tools = await tool_service.get_custom_tools_for_startup()

            if custom_tools:
                logger.info(
                    f"Loading {len(custom_tools)} custom tool instances from database...")
                print(f"Loading {len(custom_tools)} custom tool instances from database...")

                # Get repositories from service for CustomToolExecutor
                custom_tool_repo = tool_service.custom_tool_repo
                file_repo = tool_service.file_repo

                for custom_tool in custom_tools:
                    try:
                        from src.tools.custom_tool import CustomToolExecutor
                        custom_tool_executor = CustomToolExecutor(
                            custom_tool_id=custom_tool.id,
                            file_repo=file_repo,
                            custom_tool_repo=custom_tool_repo
                        )
                        custom_tool_executor._name = custom_tool.name
                        tool_registry.register(custom_tool_executor)

                        # Registrar en IntentRouter dinámicamente
                        if custom_tool.intent_examples or (custom_tool.configuration and "intent_actions" in custom_tool.configuration):
                            from src.services.intent.router import get_intent_router
                            router = await get_intent_router()

                            # Verificar si tiene intent_actions configurados (sistema multi-action)
                            intent_actions = None
                            if custom_tool.configuration and "intent_actions" in custom_tool.configuration:
                                intent_actions = custom_tool.configuration["intent_actions"]

                            await router.register_custom_tool(
                                tool_name=custom_tool.name,
                                examples=custom_tool.intent_examples,  # Para compatibilidad
                                tool_type=str(custom_tool.tool_type),
                                intent_actions=intent_actions
                            )

                        logger.info(
                            f"Custom tool registered: {custom_tool.name} (ID: {custom_tool.id})")
                        print(f"✓ Custom tool registered: {custom_tool.name}")
                    except Exception as e:
                        logger.error(f"Failed to register custom tool {custom_tool.name}: {e}",
                                     exc_info=True)
                        print(f"✗ Failed to register custom tool {custom_tool.name}: {e}")
            else:
                logger.info("No custom tool instances found (only templates exist)")
                print("✓ No custom tool instances to load (templates are in DB for reference)")

    except Exception as e:
        logger.error(f"Failed to load custom tools: {e}", exc_info=True)
        print(f"Failed to load custom tools: {e}")

    # Sync LLM Models
    try:
        from src.providers.manager import provider_manager
        logger.info("Syncing LLM models...")
        print("Syncing LLM models from providers...")

        async with get_session_for_provider_sync() as session:
            await provider_manager.sync_available_models(session)
    except Exception as e:
        logger.error(f"Failed to sync models on startup: {e}")
        print(f"Failed to sync models: {e}")

    logger.info(
        f"Tools registered: {len(tool_registry.list_names())}",
        extra={"tools": tool_registry.list_names()}
    )
    print(f"🎉 {len(tool_registry.list_names())} tools registered")
    print(f"🚀 API ready on {settings.API_PREFIX}")

    yield

    # Shutdown
    try:
        from src.providers.manager import provider_manager, ProviderType
        web_emulator = provider_manager.providers.get(ProviderType.WEB_EMULATOR)
        if web_emulator and hasattr(web_emulator, 'close'):
            import asyncio
            if asyncio.iscoroutinefunction(web_emulator.close):
                await web_emulator.close()
            else:
                web_emulator.close()
            logger.info("Web Emulator provider closed (browser tabs preserved)")
    except Exception as e:
        logger.error(f"Error closing Web Emulator provider: {e}")

    close_db_connections()
    from src.database.connection import close_async_db_connections
    await close_async_db_connections()
    stop_qdrant()
    stop_redis()
    stop_celery()
    logger.info("Application shutdown complete")


# =============================================================================
# Create FastAPI App
# =============================================================================

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="RAG-powered chatbot with flexible tool system",
    docs_url=f"{settings.API_PREFIX}/docs",
    redoc_url=f"{settings.API_PREFIX}/redoc",
    openapi_url=f"{settings.API_PREFIX}/openapi.json",
    lifespan=lifespan
)

# =============================================================================
# Middleware
# =============================================================================

# CORS
if settings.DEBUG:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=settings.CORS_CREDENTIALS,
        allow_methods=settings.CORS_METHODS,
        allow_headers=settings.CORS_HEADERS,
    )


# =============================================================================
# Exception Handlers
# =============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    logger.warning(f"Resource not found: {request.url.path}", extra={"path": str(request.url.path)})
    return JSONResponse(
        status_code=404,
        content={
            "error": "Resource not found",
            "detail": str(exc),
            "error_code": "RESOURCE_NOT_FOUND"
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}", exc_info=True,
                 extra={"path": str(request.url.path)})
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "error_code": "INTERNAL_SERVER_ERROR"
        }
    )


# =============================================================================
# Routes
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": f"{settings.API_PREFIX}/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "environment": settings.ENVIRONMENT,
        "database": "connected"  # Could add actual DB check
    }


# Include routers
app.include_router(
    conversations.router,
    prefix=f"{settings.API_PREFIX}/conversations",
    tags=["Conversations"]
)

app.include_router(
    messages.router,
    prefix=f"{settings.API_PREFIX}/messages",
    tags=["Messages"]
)

app.include_router(
    prompts.router,
    prefix=f"{settings.API_PREFIX}/prompts",
    tags=["Prompt Templates"]
)

app.include_router(
    files.router,
    prefix=f"{settings.API_PREFIX}/files",
    tags=["Files"]
)

app.include_router(
    collections.router,
    prefix=f"{settings.API_PREFIX}/collections",
    tags=["Qdrant Collections"]
)

app.include_router(
    tools.router,
    prefix=f"{settings.API_PREFIX}/tools",
    tags=["Tools"]
)

app.include_router(
    projects.router,
    prefix=f"{settings.API_PREFIX}/projects",
    tags=["Projects"]
)

app.include_router(
    collections_ingest.router,
    prefix=f"{settings.API_PREFIX}/collections",
    tags=["Files Collections"]
)

# [Módulo C] Auth router
app.include_router(
    auth_router.router,
    prefix=f"{settings.API_PREFIX}/auth",
    tags=["Authentication"]
)

app.include_router(
    oauth_router.router,
    prefix=f"{settings.API_PREFIX}/oauth",
    tags=["OAuth"]
)

# =============================================================================
# Additional utility endpoints
# =============================================================================

@app.get(f"{settings.API_PREFIX}/providers")
async def list_providers():
    """List available LLM providers with enhanced model attributes from database"""
    from src.providers.manager import provider_manager
    from src.utils.service_factory import get_session_for_provider_sync

    providers = provider_manager.get_available_providers()

    # Get models from database using service factory
    async with get_session_for_provider_sync() as session:
        models_list = await provider_manager.get_available_models(session)

        # Group by provider
        models = {}
        for model in models_list:
            provider_key = model.provider.value
            if provider_key not in models:
                models[provider_key] = []

            models[provider_key].append({
                "name": model.name,
                "context_window": model.context_window,
                "supports_function_calling": model.supports_function_calling,
                "supports_streaming": model.supports_streaming,
                "model_type": model.model_type.value,
                # Database fields for frontend filtering
                "supports_thinking": model.supports_thinking,
                "is_active": model.is_active,
                "is_custom": model.is_custom,
                # Hardware requirements and capabilities
                "cpu_supported": model.cpu_supported,
                "gpu_required": model.gpu_required,
                "parent_retrieval_supported": model.parent_retrieval_supported,
                # Cost info
                "cost_per_1k_input": model.cost_per_1k_input,
                "cost_per_1k_output": model.cost_per_1k_output,
                "is_free": model.is_free if hasattr(model, 'is_free') else False
            })

    return {
        "providers": providers,
        "models": models
    }


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    if sys.platform == "win32":
        # On Windows, uvicorn with reload=True explicitly overrides the event loop
        # policy to WindowsSelectorEventLoopPolicy, which breaks Playwright's
        # subprocess creation (create_subprocess_exec → NotImplementedError).
        # Fix: run uvicorn.Server directly inside an explicit ProactorEventLoop.
        # Hot-reload is disabled on Windows; restart manually after code changes.
        config = uvicorn.Config(
            "main:app",
            host="0.0.0.0",
            port=8001,
            reload=False,          # reload=True forces SelectorEventLoop on Windows
            log_level=settings.LOG_LEVEL.lower(),
        )
        server = uvicorn.Server(config)
        loop = asyncio.ProactorEventLoop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve())
    else:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8001,
            reload=settings.DEBUG,
            log_level=settings.LOG_LEVEL.lower(),
        )
