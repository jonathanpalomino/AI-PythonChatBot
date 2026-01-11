# =============================================================================
# main.py
# FastAPI application entry point
# =============================================================================
"""
RAG Chatbot API - Main application
"""
import atexit
import platform
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config.settings import settings
from src.database.connection import init_db, close_db_connections
from src.tools.base_tool import tool_registry
from src.tools.rag_tool import RAGTool
from src.tools.http_tool import HTTPTool
from src.tools.sql_tool import SQLTool
from src.tools.custom_tool import CustomToolExecutor
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Import routers
from src.api.v1 import conversations, messages, prompts, files, collections, tools, projects

# Global variable to hold Qdrant process
qdrant_process = None
# Global variable to hold Redis process
redis_process = None


# =============================================================================
# Redis Process Management
# =============================================================================

def check_redis_health() -> bool:
    """
    Check if Redis is already running and accessible
    Returns True if Redis is healthy, False otherwise
    """
    import socket

    try:
        # Create a socket object
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # Set a timeout for the connection attempt
            s.settimeout(1)
            # Try to connect to localhost on port 6379 (default Redis port)
            result = s.connect_ex(('127.0.0.1', 6379))

            if result == 0:
                # logger.info("Redis is running (port 6379 is open)")
                return True

    except Exception as e:
        logger.debug(f"Redis health check failed: {e}")
        pass

    return False


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
        return True # Return True anyway clearly the user can see errors now

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
# Qdrant Process Management
# =============================================================================

def check_qdrant_health() -> bool:
    """
    Check if Qdrant is already running and accessible
    Returns True if Qdrant is healthy, False otherwise
    """
    import requests

    try:
        # Try to connect to Qdrant's health endpoint
        response = requests.get(
            f"{settings.QDRANT_URL}/healthz",
            timeout=2
        )

        if response.status_code == 200:
            logger.info("Qdrant is already running and healthy")
            print("✅ Qdrant is already running")
            return True

    except requests.exceptions.RequestException:
        # Qdrant is not accessible
        pass

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
        return start_qdrant_windows()
    elif system == "Linux":
        return start_qdrant_linux()
    elif system == "Darwin":  # macOS
        # Similar to Linux, could use docker or homebrew
        logger.info("macOS detected, attempting Docker start...")
        return start_qdrant_linux()  # Reuse Linux logic for now
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


async def sync_tool_templates():
    """
    Synchronize tool templates from Python code to database.
    This ensures that database templates match the current code implementation.
    """
    from src.database.connection import AsyncSessionLocal
    from sqlalchemy import select
    from src.models.models import CustomTool
    from src.tools.http_tool import HTTPTool
    from src.tools.sql_tool import SQLTool
    from src.tools.rag_tool import RAGTool

    try:
        async with AsyncSessionLocal() as db:
            # Get existing templates
            result = await db.execute(
                select(CustomTool).filter(CustomTool.is_template == True)
            )
            existing_templates = {tool.tool_type: tool for tool in result.scalars().all()}

            # Define tools to sync
            tools_to_sync = [
                HTTPTool(),
                SQLTool(),
                RAGTool()
            ]

            updated_count = 0
            created_count = 0

            for tool in tools_to_sync:
                tool_type = tool.name.lower()
                # Build config schema with example values
                properties = {}
                example_values = {}

                for param in tool.get_parameters():
                    properties[param.name] = {
                        "type": param.type,
                        "description": param.description,
                        "required": param.required,
                        "default": param.default,
                        "enum": param.enum,
                        "example": param.example  # Include example value from parameter definition
                    }
                    # Use parameter's example value, or default value, or a sensible default
                    if param.example is not None:
                        example_values[param.name] = param.example
                    elif param.default is not None:
                        example_values[param.name] = param.default
                    elif param.type == "string":
                        example_values[param.name] = f"example_{param.name}"
                    elif param.type == "integer":
                        example_values[param.name] = 100
                    elif param.type == "boolean":
                        example_values[param.name] = True
                    elif param.type == "object":
                        example_values[param.name] = {}
                    elif param.type == "array":
                        example_values[param.name] = []

                template_data = {
                    "name": tool.name,
                    "description": tool.description,
                    "tool_type": tool_type,  # Use tool name as type
                    "is_template": True,
                    "configuration": {},  # Default empty configuration for templates
                    "config_schema": {
                        "type": "object",
                        "properties": properties
                    },
                    "example": example_values  # Example values based on parameter definitions
                }

                if tool_type in existing_templates:
                    # Update existing template
                    existing = existing_templates[tool_type]
                    existing.name = template_data["name"]
                    existing.description = template_data["description"]
                    existing.configuration = template_data["configuration"]
                    existing.config_schema = template_data["config_schema"]
                    existing.example = template_data["example"]
                    updated_count += 1
                else:
                    # Create new template
                    new_template = CustomTool(
                        name=template_data["name"],
                        description=template_data["description"],
                        tool_type=tool_type,
                        is_template=True,
                        configuration=template_data["configuration"],
                        config_schema=template_data["config_schema"],
                        example=template_data["example"],
                        is_active=True
                    )
                    db.add(new_template)
                    created_count += 1

            await db.commit()

            logger.info(f"✅ Tool templates synchronized: {created_count} created, {updated_count} updated")
            print(f"✅ Tool templates synchronized: {created_count} created, {updated_count} updated")

    except Exception as e:
        logger.error(f"Template synchronization failed: {e}", exc_info=True)
        print(f"❌ Template synchronization failed: {e}")
        raise


async def initialize_tool_types():
    """
    Initialize tool types from the database at application startup.
    This ensures that custom tool types are loaded and available.
    """
    from src.database.connection import AsyncSessionLocal
    from src.api.v1.tools import load_tool_types_from_db
    from sqlalchemy import select
    from src.models.models import CustomTool

    try:
        async with AsyncSessionLocal() as db:
            # Load custom tools from database and register them in the tool registry
            # Exclude template tools (is_template=True) - only register actual custom tools
            custom_tools = await db.execute(
                select(CustomTool).filter(
                    CustomTool.is_active == True,
                    CustomTool.is_template == False
                )
            )
            custom_tools = custom_tools.scalars().all()

            if custom_tools:
                logger.info(f"Loading {len(custom_tools)} custom tools from database...")
                print(f"🔄 Loading {len(custom_tools)} custom tools from database...")

                for custom_tool in custom_tools:
                    try:
                        # Create executor for the custom tool
                        custom_tool_executor = CustomToolExecutor(custom_tool.id, db)

                        # Override the name to use the tool's actual name instead of ID-based name
                        custom_tool_executor._name = custom_tool.name

                        # Register in tool registry
                        tool_registry.register(custom_tool_executor)

                        logger.info(f"Custom tool registered: {custom_tool.name} (ID: {custom_tool.id})")
                        print(f"✅ Custom tool registered: {custom_tool.name}")

                    except Exception as e:
                        logger.error(f"Failed to register custom tool {custom_tool.name}: {e}", exc_info=True)
                        print(f"⚠️  Failed to register custom tool {custom_tool.name}: {e}")

            # Load tool types from database
            tool_types = await load_tool_types_from_db(db)
            logger.info(f"Tool types loaded from database: {list(tool_types.keys())}",
                       extra={"tool_types": list(tool_types.keys())})
            print(f"✅ Tool types loaded: {len(tool_types)} types")
    except Exception as e:
        logger.error(f"Failed to initialize tool types: {e}", exc_info=True)
        print(f"⚠️  Failed to initialize tool types: {e}")


# =============================================================================
# Lifespan Events
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events for startup and shutdown
    """
    # Startup
    logger.info("Starting RAG Chatbot API",
                extra={"version": settings.APP_VERSION, "environment": settings.ENVIRONMENT})
    print("🚀 Starting RAG Chatbot API...")

    # Ensure Qdrant is running (checks health first, then tries to start if needed)
    ensure_qdrant_running()

    # Ensure Redis is running
    ensure_redis_running()

    # Initialize database
    init_db()
    logger.info("Database initialized")

    # Sync tool templates from code to database
    await sync_tool_templates()

    # Initialize tool types from database
    await initialize_tool_types()

    # Register tools
    tool_registry.register(RAGTool())
    tool_registry.register(HTTPTool())
    tool_registry.register(SQLTool())
    # tool_registry.register(CodeAnalyzerTool())
    # tool_registry.register(DocumentProcessorTool())

    # Get DB session for tools that need it
    from src.database.connection import SessionLocal
    db = SessionLocal()
    # tool_registry.register(DeepThinkingTool(db))

    # Sync LLM Models
    try:
        from src.providers.manager import provider_manager
        from src.database.connection import AsyncSessionLocal

        logger.info("Syncing LLM models...")
        print("🔄 Syncing LLM models from providers...")

        async with AsyncSessionLocal() as session:
            await provider_manager.sync_available_models(session)

    except Exception as e:
        logger.error(f"Failed to sync models on startup: {e}")
        print(f"⚠️  Failed to sync models: {e}")
    finally:
        db.close()  # Close the sync session created earlier if any

    logger.info(f"Tools registered: {len(tool_registry.list_names())}",
                extra={"tools": tool_registry.list_names()})
    print(f"✅ {len(tool_registry.list_names())} tools registered")
    print(f"✅ API ready on {settings.API_PREFIX}")

    yield

    # Shutdown
    logger.info("Shutting down RAG Chatbot API")
    print("🛑 Shutting down...")
    close_db_connections()
    stop_qdrant()
    stop_redis()
    logger.info("Shutdown complete")
    print("✅ Goodbye!")


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
        content={"error": "Resource not found", "detail": str(exc)}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}", exc_info=True,
                 extra={"path": str(request.url.path)})
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
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


# =============================================================================
# Additional utility endpoints
# =============================================================================

@app.get(f"{settings.API_PREFIX}/providers")
async def list_providers():
    """List available LLM providers with enhanced model attributes from database"""
    from src.providers.manager import provider_manager
    from src.database.connection import AsyncSessionLocal

    providers = provider_manager.get_available_providers()

    # Get models from database (includes all llm_models table fields)
    async with AsyncSessionLocal() as session:
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

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
