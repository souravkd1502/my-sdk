"""
A module for creating and maintaining all the configuration for the chatbot.
Including LLMs, Embeddings, etc.

* Features:
- `Config`: A class for configuring different components of the chatbot, including LLMs and embeddings.
- `SharedData`: Singleton class for managing shared data across modules.

* Environment Variables:
- `MODEL`: Specifies the LLM model.
- `ANTHROPIC_API_KEY`: API key for accessing the Anthropic API.
- `MODEL_TYPE`: Specifies the the type of model (e.g. claude or openAI)
- `OPENAI_API_KEY`: API key for accessing the OpenAI API.
- `AZURE_OPENAI_ENDPOINT`: OpenAI endpoint to connect to Azure OpenAI.
- `OPEN_API_VERSION`: Open API version for the model.
- `AZURE_DEPLOYMENT`: Azure deployment to connect to Azure OpenAI.


* Dependencies:
- `langchain_anthropic`: Integrates ChatAnthropic for anthropic interactions.
- `langchain_community`: Embeddings for handling Hugging Face embeddings.
- `langchain_openai`: Integrates Azure OpenAI APIs.

TODO:

FIXME:

Author:
Sourav Das

Date:
2025-07-01
"""

# Adding directories to system path to allow importing custom modules
import sys

sys.path.append("./")
sys.path.append("../")

# Import dependencies
import os
import logging
from dotenv import load_dotenv
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Get environment variables
MODEL_TYPE = os.getenv("MODEL_TYPE")
MODEL = os.getenv("MODEL")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPEN_API_VERSION = os.getenv("OPEN_API_VERSION")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT")
GPT_4O_API_KEY = os.getenv("GPT_4O_API_KEY")
GPT_MODEL = os.getenv("GPT_MODEL")

# Set up logging
_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Config:
    """
    A class for configuring different components.

    Attributes:
        llm (ChatAnthropic): An instance of ChatAnthropic for managing anthropic interactions.
            temperature (float): The temperature parameter controlling the randomness of responses.
            anthropic_api_key (str): The API key for accessing the Anthropic API.
            model_name (str): The name of the LLM model.
        model_name (str): The name of the Sentence Transformers model.
        model_kwargs (dict): Keyword arguments for configuring the Sentence Transformers model.
        encode_kwargs (dict): Keyword arguments for encoding with the Sentence Transformers model.
        hf (HuggingFaceEmbeddings): An instance of HuggingFaceEmbeddings for handling Hugging Face embeddings.
    """

    # LLM Configurations --------------------------------
    if MODEL_TYPE == "CLAUDE":
        if (
            ANTHROPIC_API_KEY is None
            or ANTHROPIC_API_KEY == ""
            and MODEL is None
            or MODEL == ""
        ):
            raise ValueError("You must specify both MODEL and ANTHROPIC_API_KEY")

        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "langchain_anthropic is not installed. Please install it with `pip install langchain_anthropic`."
            )

        llm = ChatAnthropic(
            temperature=0.2,
            anthropic_api_key=ANTHROPIC_API_KEY,
            model_name=MODEL,
        )
        _logger.info(f"Loading Claude model - {MODEL} as default LLM.")

    elif MODEL_TYPE == "OPENAI":
        if (
            OPENAI_API_KEY is None
            or OPENAI_API_KEY == ""
            and MODEL is None
            or MODEL == ""
            and OPEN_API_VERSION is None
            or OPEN_API_VERSION == ""
            and AZURE_DEPLOYMENT is None
            or AZURE_DEPLOYMENT == ""
        ):
            raise ValueError(
                "You must specify MODEL, OPENAI_API_KEY, OPEN_API_VERSION, and AZURE_DEPLOYMENT"
            )

        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError:
            raise ImportError(
                "langchain_openai is not installed. Please install it with `pip install langchain_openai`."
            )

        llm = AzureChatOpenAI(
            api_key=OPENAI_API_KEY,
            api_version=OPEN_API_VERSION,
            azure_deployment=AZURE_DEPLOYMENT,
        )
        _logger.info(f"Loading OpenAI {AZURE_DEPLOYMENT} model as default LLM.")

    elif MODEL_TYPE == "GPT-4O":
        if (
            GPT_4O_API_KEY is None
            or GPT_4O_API_KEY == ""
            and MODEL is None
            or MODEL == ""
        ):
            raise ValueError("You must specify MODEL and GPT_4O_API_KEY")
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "langchain_openai is not installed. Please install it with `pip install langchain-openai`."
            )

        llm = ChatOpenAI(
            model=MODEL,
            api_key=GPT_4O_API_KEY,
        )
    else:
        raise ValueError(f"Model type {MODEL_TYPE} is not supported.")
    
    try:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "langchain_openai is not installed. Please install it with `pip install langchain-openai`."
            )
        small_llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=GPT_4O_API_KEY,
        )
        
        reasoning_llm = ChatOpenAI(
            model="o1-preview",
            api_key=GPT_4O_API_KEY,
            temperature=1,
        )
    except Exception as e:
        _logger.error(f"Error loading gpt-4o model: {e}")
        

    # Set up LLM Cache
    set_llm_cache(InMemoryCache())

    # Embeddings Configuration --------------------------------
    _logger.info("Loading HuggingFace Embeddings as default Embedding.")
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": False}

    # Load HuggingFaceEmbeddings model
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        cache_folder="./models",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    _logger.info(f"Loading {model_name} model as default Embedding model.")
