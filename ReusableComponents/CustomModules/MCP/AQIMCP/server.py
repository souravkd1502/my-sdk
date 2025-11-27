""" """

import os
import httpx
import logging
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# ---------------------------------
# Logging Configuration
# ---------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------
# Loading Environment Variables
# ---------------------------------
load_dotenv(override=True)

AQI_API_URL = os.getenv("AQI_API_URL", "")
AQI_API_TOKEN = os.getenv("AQI_API_TOKEN", "")
print("AQI_API_URL:", AQI_API_URL)

if not AQI_API_TOKEN:
    raise ValueError("AQI_API_TOKEN is not set")
if not AQI_API_URL:
    raise ValueError("AQI_API_URL is not set")


# ---------------------------------
# MCP Server Setup
# ---------------------------------

mcp = FastMCP(name="aqi-mcp-server", port=8001)


# ---------------------------------
# Helpers
# ---------------------------------
def make_request(path: str, params: dict | None = None) -> dict:
    """
    Make a generic request to the WAQI API.

    Args:
        path (str): The dynamic API path, e.g.:
            - "feed/shanghai"
            - "feed/geo:10.3;20.7"
            - "feed/here"
            - "v2/map/bounds"
            - "search"
        params (dict | None): Optional query parameters.

    Returns:
        dict: Parsed JSON response.
    """
    if params is None:
        params = {}

    # Always include token
    params["token"] = AQI_API_TOKEN

    url = f"{AQI_API_URL}/{path}"
    print("Path:", path)
    print("Request URL:", url)

    headers = {"Accept": "application/json"}

    try:
        response = httpx.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error occurred: {e.response.status_code} {e.response.text}")
        raise

    except httpx.RequestError as e:
        logger.error(f"Network error during request: {e}")
        raise


