#!/usr/bin/env python3
# =============================================================================
# scripts/init_tool_types.py
# Initialize base tool types in the database
# =============================================================================
"""
Script to initialize base tool types (http, sql, custom) in the database.
This should be run once during setup or when no custom tools exist.
"""

import asyncio
from src.database.connection import AsyncSessionLocal
from src.models.models import CustomTool
from sqlalchemy import select
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def init_base_tool_types():
    """
    Initialize base tool types in the database if they don't exist.
    """
    try:
        async with AsyncSessionLocal() as db:
            # Check if any custom tools already exist
            result = await db.execute(select(CustomTool))
            existing_tools = result.scalars().all()
            
            if existing_tools:
                logger.info(f"Found {len(existing_tools)} custom tools in database, skipping initialization")
                print(f"✅ Found {len(existing_tools)} custom tools, no initialization needed")
                return
            
            # Define base tool types with complete configuration and schema
            base_tools = [
                {
                    "name": "HTTP Request",
                    "description": "Make HTTP requests to external APIs",
                    "tool_type": "http_request",
                    "is_template": True,
                    "configuration": {
                        "url": "https://api.example.com/data",
                        "method": "GET",
                        "headers": {"Accept": "application/json"},
                        "params": {"page": 1, "limit": 10},
                        "timeout": 30,
                        "auth_token": None
                    },
                    "config_schema": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "Full URL for the HTTP request (must include http:// or https://)",
                                "required": True
                            },
                            "method": {
                                "type": "string",
                                "description": "HTTP method (GET, POST, PUT, DELETE, PATCH)",
                                "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                                "required": True,
                                "default": "GET"
                            },
                            "headers": {
                                "type": "object",
                                "description": "HTTP headers as key-value pairs",
                                "required": False,
                                "default": {}
                            },
                            "params": {
                                "type": "object",
                                "description": "Query parameters for GET requests",
                                "required": False,
                                "default": {}
                            },
                            "body": {
                                "type": "object",
                                "description": "Request body (for POST, PUT, PATCH)",
                                "required": False,
                                "default": None
                            },
                            "timeout": {
                                "type": "integer",
                                "description": "Request timeout in seconds",
                                "required": False,
                                "default": 30,
                                "minimum": 5,
                                "maximum": 120
                            },
                            "auth_token": {
                                "type": "string",
                                "description": "Optional authentication token (will be added to Authorization header)",
                                "required": False,
                                "default": None
                            }
                        }
                    },
                    "example": {
                        "url": "https://api.example.com/data",
                        "method": "GET",
                        "headers": {"Accept": "application/json"},
                        "params": {"page": 1, "limit": 10},
                        "timeout": 30
                    }
                },
                {
                    "name": "SQL Query",
                    "description": "Execute SQL queries against databases (PostgreSQL, MySQL, Oracle)",
                    "tool_type": "sql_query",
                    "is_template": True,
                    "configuration": {
                        "database_type": "postgresql",
                        "host": "localhost",
                        "port": 5432,
                        "database": "mydb",
                        "username": "user",
                        "password": "password",
                        "query": "SELECT * FROM customers WHERE active = True LIMIT 100",
                        "parameters": {},
                        "fetch_size": 100
                    },
                    "config_schema": {
                        "type": "object",
                        "properties": {
                            "database_type": {
                                "type": "string",
                                "description": "Type of database",
                                "enum": ["postgresql", "mysql", "oracle"],
                                "required": True
                            },
                            "host": {
                                "type": "string",
                                "description": "Database host or socket path",
                                "required": True
                            },
                            "port": {
                                "type": "integer",
                                "description": "Database port",
                                "required": False,
                                "default": None
                            },
                            "database": {
                                "type": "string",
                                "description": "Database name",
                                "required": True
                            },
                            "username": {
                                "type": "string",
                                "description": "Database username",
                                "required": True
                            },
                            "password": {
                                "type": "string",
                                "description": "Database password",
                                "required": True,
                                "format": "password"
                            },
                            "query": {
                                "type": "string",
                                "description": "SQL query to execute",
                                "required": True
                            },
                            "parameters": {
                                "type": "object",
                                "description": "Optional query parameters (for parameterized queries)",
                                "required": False,
                                "default": {}
                            },
                            "fetch_size": {
                                "type": "integer",
                                "description": "Maximum number of rows to fetch",
                                "required": False,
                                "default": 100,
                                "minimum": 1,
                                "maximum": 10000
                            }
                        }
                    },
                    "example": {
                        "database_type": "postgresql",
                        "host": "localhost",
                        "port": 5432,
                        "database": "mydb",
                        "username": "user",
                        "password": "password",
                        "query": "SELECT * FROM customers WHERE active = True LIMIT 100"
                    }
                },
                {
                    "name": "RAG Search",
                    "description": "Search for relevant information in documentation collections using semantic search",
                    "tool_type": "rag_search",
                    "is_template": True,
                    "configuration": {
                        "collections": ["documentation"],
                        "k": 5,
                        "score_threshold": 0.5,
                        "search_mode": "semantic"
                    },
                    "config_schema": {
                        "type": "object",
                        "properties": {
                            "collections": {
                                "type": "array",
                                "description": "List of collection names to search in",
                                "required": True
                            },
                            "k": {
                                "type": "integer",
                                "description": "Number of results to return",
                                "required": False,
                                "default": 5
                            },
                            "score_threshold": {
                                "type": "number",
                                "description": "Minimum similarity score (0.0-1.0)",
                                "required": False,
                                "default": 0.5
                            },
                            "search_mode": {
                                "type": "string",
                                "description": "Search mode: 'semantic', 'lexical', or 'hybrid'",
                                "required": False,
                                "default": "semantic",
                                "enum": ["semantic", "lexical", "hybrid"]
                            }
                        }
                    },
                    "example": {
                        "collections": ["documentation", "api_guide"],
                        "k": 5,
                        "score_threshold": 0.5,
                        "search_mode": "semantic"
                    }
                },
                {
                    "name": "Custom Tool",
                    "description": "Custom tool with flexible configuration",
                    "tool_type": "custom",
                    "is_template": True,
                    "configuration": {
                        "url": "https://api.example.com/custom",
                        "method": "POST",
                        "headers": {"Authorization": "Bearer token"},
                        "parameters": [
                            {"name": "param1", "type": "string", "description": "Parameter 1", "required": True}
                        ]
                    },
                    "config_schema": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "Endpoint URL for the custom tool",
                                "required": True
                            },
                            "method": {
                                "type": "string",
                                "description": "HTTP method (GET or POST)",
                                "enum": ["GET", "POST"],
                                "required": True,
                                "default": "POST"
                            },
                            "headers": {
                                "type": "object",
                                "description": "Custom headers",
                                "required": False,
                                "default": {}
                            },
                            "parameters": {
                                "type": "array",
                                "description": "Parameter definitions for the tool",
                                "required": False,
                                "default": []
                            }
                        }
                    },
                    "example": {
                        "url": "https://api.example.com/custom",
                        "method": "POST",
                        "headers": {"Authorization": "Bearer token"},
                        "parameters": [
                            {"name": "param1", "type": "string", "description": "Parameter 1", "required": True}
                        ]
                    }
                }
            ]
            
            # Create base tool types in database
            created_tools = []
            for tool_data in base_tools:
                custom_tool = CustomTool(
                    name=tool_data["name"],
                    description=tool_data["description"],
                    tool_type=tool_data["tool_type"],
                    configuration=tool_data["configuration"],
                    is_active=True
                )
                db.add(custom_tool)
                created_tools.append(custom_tool)
            
            await db.commit()
            
            for tool in created_tools:
                await db.refresh(tool)
                logger.info(f"Created base tool type: {tool.name} ({tool.tool_type})")
            
            logger.info(f"✅ Successfully initialized {len(created_tools)} base tool types")
            print(f"✅ Successfully initialized {len(created_tools)} base tool types")
            
    except Exception as e:
        logger.error(f"Failed to initialize base tool types: {e}", exc_info=True)
        print(f"❌ Failed to initialize base tool types: {e}")
        raise


if __name__ == "__main__":
    print("🚀 Initializing base tool types...")
    asyncio.run(init_base_tool_types())
    print("✅ Base tool types initialization complete")