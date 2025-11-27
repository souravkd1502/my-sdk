import time
import logging
from collections import defaultdict
from functools import wraps
from mcp.server.fastmcp import FastMCP
import asyncio

class MetricsCollector:
    def __init__(self, storage_type="memory", log_file="metrics.log"):
        # Only memory storage supported for now
        if storage_type != "memory":
            raise NotImplementedError("Only memory storage is supported for now")
        self.metrics = defaultdict(lambda: {"calls": 0, "total_time": 0.0})
        self.server = None

        # Configure logger
        self.logger = logging.getLogger("MetricsCollector")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
        self.logger.addHandler(handler)

    def init_app(self, server: FastMCP):
        """Attach metrics collector to a FastMCP server"""
        self.server = server
        self._wrap_tools()
        # You can later add _wrap_resources() and _wrap_prompts() similarly

    def _wrap_tools(self):
        tools = self.server._tool_manager.list_tools()

        for tool in tools:
            original_fn = tool.fn
            tool_name = tool.name

            # Check if the tool is async
            if asyncio.iscoroutinefunction(original_fn):
                @wraps(original_fn)
                async def wrapped_tool(*args, _tool_name=tool_name, **kwargs):
                    start = time.time()
                    result = await original_fn(*args, **kwargs)
                    elapsed = time.time() - start
                    self._record_metric("tool", _tool_name, elapsed)
                    return result
            else:
                @wraps(original_fn)
                def wrapped_tool(*args, _tool_name=tool_name, **kwargs):
                    start = time.time()
                    result = original_fn(*args, **kwargs)
                    elapsed = time.time() - start
                    self._record_metric("tool", _tool_name, elapsed)
                    return result

            tool.fn = wrapped_tool

    def _record_metric(self, category, name, elapsed_time):
        key = f"{category}:{name}"
        self.metrics[key]["calls"] += 1
        self.metrics[key]["total_time"] += elapsed_time

        # Write to log file
        log_msg = (
            f"{category} | {name} | "
            f"call_count={self.metrics[key]['calls']} | "
            f"total_time={self.metrics[key]['total_time']:.4f}s | "
            f"last_elapsed={elapsed_time:.4f}s"
        )
        self.logger.info(log_msg)
        
        

    def get_metrics(self):
        """Return current metrics snapshot"""
        return dict(self.metrics)




# Initialize MCP server
server = FastMCP(name="demo", port="8001")

# Add a tool
@server.tool()
def hello(name: str):
    return f"Hello, {name}!"


# Attach metrics collector
metrics = MetricsCollector(storage_type="memory")
metrics.init_app(server)

server.run(transport="sse")
