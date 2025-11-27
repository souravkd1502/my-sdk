"""

"""

# Import necessary modules
import asyncio
import logging
import sqlite3
from mcp.server.fastmcp import FastMCP

# Initialize logger
logger = logging.getLogger("MCPServerInfo")
logger.setLevel(logging.INFO)

class MCPServerInfo:
    def __init__(self, server: FastMCP):
        self.server = server

    def get_server_info(self):
        """
        
        """
        tools_info = []
        tools = self.server._tool_manager.list_tools()
        
        for tool in tools:
            tools_info.append({
                "name": tool.name,
                "description": tool.description,
                "type": "async" if asyncio.iscoroutinefunction(tool.fn) else "sync",
                "parameters": tool.parameters
            })
            
        logger.info(f"Tools: {[tool['name'] for tool in tools_info]}")
            
        resources_info = []
        resources = self.server._resource_manager.list_resources()
        logger.info(f"Found {len(resources)} resources.")
        for resource in resources:
            resources_info.append({
                "name": resource.name,
                "description": resource.description,
                "title": resource.title,
                "uri": resource.uri
            })
            
        logger.info(f"Resources: {[res['name'] for res in resources_info]}")
            
        prompts_info = []
        prompts = self.server._prompt_manager.list_prompts()
        for prompt in prompts:
            prompts_info.append({
                "name": prompt.name,
                "description": prompt.description,
                "arguments": prompt.arguments,
                "title": prompt.title
            })
            
        logger.info(f"Prompts: {[prompt['name'] for prompt in prompts_info]}")
        
        resources_template_info = []
        resource_templates= self.server._resource_manager.list_templates()
        for template in resource_templates:
            resources_template_info.append({
                "name": template.name,
                "description": template.description,
                "title": template.title,
                "uri_template": template.uri_template
            })
            
        logger.info(f"Resource Templates: {[template['name'] for template in resources_template_info]}")
            
        return {
            "tools": tools_info,
            "resources": resources_info,
            "resource_templates": resources_template_info,
            "prompts": prompts_info
        }
        

class MCPDatabaseHandler:
    def __init__(self, db_path="mcp_info.db", server=None):
        self.db_path = db_path
        self.server = server
        self._init_db()

    def _init_db(self):
        """Initialize SQLite tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Tools table
        c.execute("""
        CREATE TABLE IF NOT EXISTS tools (
            name TEXT PRIMARY KEY,
            description TEXT,
            type TEXT,
            parameters TEXT
        )
        """)

        # Resources table
        c.execute("""
        CREATE TABLE IF NOT EXISTS resources (
            name TEXT PRIMARY KEY,
            description TEXT,
            title TEXT,
            uri TEXT
        )
        """)
        
        # Resource Templates table
        c.execute("""
        CREATE TABLE IF NOT EXISTS resource_templates (
            name TEXT PRIMARY KEY,
            description TEXT,
            title TEXT,
            uri TEXT
        )
        """)

        # Prompts table
        c.execute("""
        CREATE TABLE IF NOT EXISTS prompts (
            name TEXT PRIMARY KEY,
            description TEXT,
            arguments TEXT,
            title TEXT
        )
        """)

        conn.commit()
        conn.close()

    def _push_info_to_db(self, info: dict):
        """Push MCP server info to SQLite database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Insert/update tools
        for tool in info.get("tools", []):
            c.execute("""
            INSERT OR REPLACE INTO tools (name, description, type, parameters)
            VALUES (?, ?, ?, ?)
            """, (
                tool["name"],
                tool.get("description"),
                tool.get("type"),
                str(tool.get("parameters"))
            ))

        # Insert/update resources
        for res in info.get("resources", []):
            c.execute("""
            INSERT OR REPLACE INTO resources (name, description, title, uri)
            VALUES (?, ?, ?, ?)
            """, (
                res["name"],
                res.get("description"),
                res.get("title"),
                res.get("uri")
            ))
            
        # Insert/update resources_templates
        for res in info.get("resource_templates", []):
            c.execute("""
            INSERT OR REPLACE INTO resource_templates (name, description, title, uri)
            VALUES (?, ?, ?, ?)
            """, (
                res["name"],
                res.get("description"),
                res.get("title"),
                res.get("uri")
            ))

        # Insert/update prompts
        for prompt in info.get("prompts", []):
            c.execute("""
            INSERT OR REPLACE INTO prompts (name, description, arguments, title)
            VALUES (?, ?, ?, ?)
            """, (
                prompt["name"],
                prompt.get("description"),
                str(prompt.get("arguments")),
                prompt.get("title")
            ))

        conn.commit()
        conn.close()

# Initialize MCP server
mcp = FastMCP(name="demo", port="8001")

# Add an addition tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


# Add a dynamic greeting resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"


# Add a prompt
@mcp.prompt()
def greet_user(name: str, style: str = "friendly") -> str:
    """Generate a greeting prompt"""
    styles = {
        "friendly": "Please write a warm, friendly greeting",
        "formal": "Please write a formal, professional greeting",
        "casual": "Please write a casual, relaxed greeting",
    }

    return f"{styles.get(style, styles['friendly'])} for someone named {name}."

mcp_info = MCPServerInfo(mcp).get_server_info()
db_handler = MCPDatabaseHandler(server=mcp)

db_handler._push_info_to_db(mcp_info)

run_as_local = False

if run_as_local:
    mcp.run(transport='stdio')
else:
    mcp.run(transport='sse')