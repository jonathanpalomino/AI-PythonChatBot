# =============================================================================
# src/tools/sql_tool.py
# SQL Query Tool
# =============================================================================
"""
Herramienta para ejecutar consultas SQL en bases de datos soportadas (Postgres, MySQL, Oracle).
"""

from typing import List, Dict, Any, Optional

from sqlalchemy import text
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from src.tools.base_tool import BaseTool, ToolCategory, ToolParameter, ToolResult
from src.utils.logger import get_logger


class SQLTool(BaseTool):
    """Herramienta para ejecutar consultas SQL en bases de datos."""

    def __init__(self):
        self.logger = get_logger(__name__)
        super().__init__()

    # Definición de la Herramienta

    @property
    def name(self) -> str:
        return "sql_query"

    @property
    def description(self) -> str:
        return "Ejecuta consultas SQL en bases de datos configuradas (PostgreSQL, MySQL, Oracle)."

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.UTILITY

    @property
    def enabled_by_default(self) -> bool:
        return False

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="database_type",
                type="string",
                description="Tipo de base de datos: 'postgresql', 'mysql', o 'oracle'.",
                required=True,
                enum=["postgresql", "mysql", "oracle"],
                example="postgresql"
            ),
            ToolParameter(
                name="host",
                type="string",
                description="Host de la base de datos o ruta del socket.",
                required=True,
                example="localhost"
            ),
            ToolParameter(
                name="port",
                type="integer",
                description="Puerto de la base de datos (por defecto: 5432 para PostgreSQL, 3306 para MySQL).",
                required=False,
                default=None,
                example=5432
            ),
            ToolParameter(
                name="database",
                type="string",
                description="Nombre de la base de datos.",
                required=True,
                example="mydatabase"
            ),
            ToolParameter(
                name="username",
                type="string",
                description="Nombre de usuario.",
                required=True,
                example="dbuser"
            ),
            ToolParameter(
                name="password",
                type="string",
                description="Contraseña de la base de datos.",
                required=True,
                example="dbpassword"
            ),
            ToolParameter(
                name="query",
                type="string",
                description="Consulta SQL a ejecutar.",
                required=True,
                example="SELECT * FROM customers WHERE active = True LIMIT 100"
            ),
            ToolParameter(
                name="parameters",
                type="object",
                description="Parámetros opcionales para la consulta.",
                required=False,
                default={},
                example={"status": "active", "limit": 100}
            ),
            ToolParameter(
                name="fetch_size",
                type="integer",
                description="Número máximo de filas a recuperar (por defecto: 100).",
                required=False,
                default=100,
                example=100
            )
        ]

    # Métodos Auxiliares

    def _get_default_port(self, database_type: str) -> int:
        """Obtiene el puerto por defecto según el tipo de base de datos."""
        port_map = {
            "postgresql": 5432,
            "mysql": 3306,
            "oracle": 1521
        }
        return port_map.get(database_type)

    def _create_database_url(
        self,
        database_type: str,
        username: str,
        password: str,
        host: str,
        port: Optional[int],
        database: str
    ) -> Optional[URL]:
        """Crea la URL de conexión según el tipo de base de datos."""
        if database_type == "postgresql":
            return URL.create(
                drivername="postgresql+asyncpg",
                username=username,
                password=password,
                host=host,
                port=port,
                database=database
            )
        elif database_type == "mysql":
            return URL.create(
                drivername="mysql+asyncmy",
                username=username,
                password=password,
                host=host,
                port=port,
                database=database
            )
        elif database_type == "oracle":
            # Oracle connection string format
            if ":" in host:
                # Host:port format
                return URL.create(
                    drivername="oracle+asyncoracledriver",
                    username=username,
                    password=password,
                    host=host,
                    port=port,
                    database=database
                )
            else:
                # SID or service name format
                return URL.create(
                    drivername="oracle+asyncoracledriver",
                    username=username,
                    password=password,
                    host=host,
                    port=port,
                    database=f"{host}:{port}/{database}"
                )
        return None

    # Ejecución

    async def execute(
        self,
        database_type: str,
        host: str,
        port: Optional[int],
        database: str,
        username: str,
        password: str,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        fetch_size: int = 100
    ) -> ToolResult:
        """Ejecuta la consulta SQL en la base de datos especificada."""
        try:
            # Validate inputs
            await self.validate_input(
                database_type=database_type,
                host=host,
                database=database,
                username=username,
                password=password,
                query=query,
                parameters=parameters or {},
                fetch_size=fetch_size
            )

            # Set default port if not provided
            if port is None:
                port = self._get_default_port(database_type)

            # Create database URL
            db_url = self._create_database_url(
                database_type=database_type,
                username=username,
                password=password,
                host=host,
                port=port,
                database=database
            )

            if db_url is None:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Unsupported database type: {database_type}"
                )

            # Create async engine and session
            engine = create_async_engine(
                str(db_url),
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=3600
            )

            AsyncSessionLocal = sessionmaker(
                bind=engine,
                class_=AsyncSession,
                expire_on_commit=False
            )

            async with AsyncSessionLocal() as session:
                try:
                    # Execute query
                    result = await session.execute(text(query), parameters or {})

                    # Fetch results
                    rows = result.fetchmany(fetch_size) if fetch_size > 0 else result.fetchall()

                    # Get column names
                    column_names = result.keys()

                    # Convert to list of dicts
                    results_data = []
                    for row in rows:
                        row_dict = {}
                        for i, col in enumerate(column_names):
                            row_dict[col] = row[i]
                        results_data.append(row_dict)

                    # Get row count
                    total_rows = len(results_data)

                    self.logger.info(
                        f"SQL query executed successfully",
                        extra={
                            "database_type": database_type,
                            "database": database,
                            "rows_fetched": total_rows,
                            "query_length": len(query)
                        }
                    )

                    return ToolResult(
                        success=True,
                        data={
                            "rows": results_data,
                            "count": total_rows,
                            "columns": column_names
                        },
                        metadata={
                            "database_type": database_type,
                            "database": database,
                            "query_length": len(query),
                            "rows_fetched": total_rows
                        }
                    )

                except Exception as e:
                    self.logger.error(
                        f"SQL query execution failed: {e}",
                        exc_info=True,
                        extra={
                            "database_type": database_type,
                            "database": database,
                            "query": query[:200]
                        }
                    )
                    return ToolResult(
                        success=False,
                        data=None,
                        error=f"SQL execution error: {str(e)}"
                    )

        except Exception as e:
            self.logger.error(
                f"SQL tool execution error: {e}",
                exc_info=True
            )
            return ToolResult(
                success=False,
                data=None,
                error=str(e)
            )

    # Métodos Auxiliares

    async def test_connection(
        self,
        database_type: str,
        host: str,
        port: Optional[int],
        database: str,
        username: str,
        password: str
    ) -> ToolResult:
        """Realiza una prueba de conexión a la base de datos."""
        try:
            # Set default port if not provided
            if port is None:
                port = self._get_default_port(database_type)

            # Create database URL
            db_url = self._create_database_url(
                database_type=database_type,
                username=username,
                password=password,
                host=host,
                port=port,
                database=database
            )

            if db_url is None:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Unsupported database type: {database_type}"
                )

            # Create async engine
            engine = create_async_engine(str(db_url))

            # Test connection
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))

            self.logger.info(
                f"Database connection test successful",
                extra={
                    "database_type": database_type,
                    "database": database,
                    "host": host
                }
            )

            return ToolResult(
                success=True,
                data={
                    "message": "Connection successful",
                    "database_type": database_type,
                    "database": database
                },
                metadata={
                    "database_type": database_type,
                    "database": database
                }
            )

        except Exception as e:
            self.logger.error(
                f"Database connection test failed: {e}",
                exc_info=True,
                extra={
                    "database_type": database_type,
                    "database": database,
                    "host": host
                }
            )
            return ToolResult(
                success=False,
                data=None,
                error=f"Connection test failed: {str(e)}"
            )
