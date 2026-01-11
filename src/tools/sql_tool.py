# =============================================================================
# src/tools/sql_tool.py
# SQL Query Tool
# =============================================================================
"""
Tool for executing SQL queries against supported databases (PostgreSQL, MySQL, Oracle)
"""

from typing import List, Dict, Any, Optional
import json

import httpx
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy import text, select
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker

from src.tools.base_tool import BaseTool, ToolCategory, ToolParameter, ToolResult
from src.utils.logger import get_logger


class SQLTool(BaseTool):
    """Tool for executing SQL queries against databases"""

    def __init__(self):
        self.logger = get_logger(__name__)
        super().__init__()

    # =============================================================================
    # Tool Definition
    # =============================================================================

    @property
    def name(self) -> str:
        return "sql_query"

    @property
    def description(self) -> str:
        return "Execute SQL queries against configured databases (PostgreSQL, MySQL, Oracle)"

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
                description="Type of database: 'postgresql', 'mysql', or 'oracle'",
                required=True,
                enum=["postgresql", "mysql", "oracle"],
                example="postgresql"
            ),
            ToolParameter(
                name="host",
                type="string",
                description="Database host or socket path",
                required=True,
                example="localhost"
            ),
            ToolParameter(
                name="port",
                type="integer",
                description="Database port (default: 5432 for PostgreSQL, 3306 for MySQL)",
                required=False,
                default=None,
                example=5432
            ),
            ToolParameter(
                name="database",
                type="string",
                description="Database name",
                required=True,
                example="mydatabase"
            ),
            ToolParameter(
                name="username",
                type="string",
                description="Database username",
                required=True,
                example="dbuser"
            ),
            ToolParameter(
                name="password",
                type="string",
                description="Database password",
                required=True,
                example="dbpassword"
            ),
            ToolParameter(
                name="query",
                type="string",
                description="SQL query to execute",
                required=True,
                example="SELECT * FROM customers WHERE active = True LIMIT 100"
            ),
            ToolParameter(
                name="parameters",
                type="object",
                description="Optional query parameters (for parameterized queries)",
                required=False,
                default={},
                example={"status": "active", "limit": 100}
            ),
            ToolParameter(
                name="fetch_size",
                type="integer",
                description="Maximum number of rows to fetch (default: 100)",
                required=False,
                default=100,
                example=100
            )
        ]

    # =============================================================================
    # Execution
    # =============================================================================

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
        """Execute SQL query against the specified database"""
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

            # Set default ports
            if port is None:
                if database_type == "postgresql":
                    port = 5432
                elif database_type == "mysql":
                    port = 3306
                elif database_type == "oracle":
                    port = 1521

            # Create database URL
            if database_type == "postgresql":
                db_url = URL.create(
                    drivername="postgresql+asyncpg",
                    username=username,
                    password=password,
                    host=host,
                    port=port,
                    database=database
                )
            elif database_type == "mysql":
                db_url = URL.create(
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
                    db_url = URL.create(
                        drivername="oracle+asyncoracledriver",
                        username=username,
                        password=password,
                        host=host,
                        port=port,
                        database=database
                    )
                else:
                    # SID or service name format
                    db_url = URL.create(
                        drivername="oracle+asyncoracledriver",
                        username=username,
                        password=password,
                        host=host,
                        port=port,
                        database=f"{host}:{port}/{database}"
                    )
            else:
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

    # =============================================================================
    # Helper Methods
    # =============================================================================

    async def test_connection(
        self,
        database_type: str,
        host: str,
        port: Optional[int],
        database: str,
        username: str,
        password: str
    ) -> ToolResult:
        """Test database connection"""
        try:
            # Set default ports
            if port is None:
                if database_type == "postgresql":
                    port = 5432
                elif database_type == "mysql":
                    port = 3306
                elif database_type == "oracle":
                    port = 1521

            # Create database URL
            if database_type == "postgresql":
                db_url = URL.create(
                    drivername="postgresql+asyncpg",
                    username=username,
                    password=password,
                    host=host,
                    port=port,
                    database=database
                )
            elif database_type == "mysql":
                db_url = URL.create(
                    drivername="mysql+asyncmy",
                    username=username,
                    password=password,
                    host=host,
                    port=port,
                    database=database
                )
            elif database_type == "oracle":
                db_url = URL.create(
                    drivername="oracle+asyncoracledriver",
                    username=username,
                    password=password,
                    host=host,
                    port=port,
                    database=database
                )
            else:
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
