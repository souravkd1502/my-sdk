"""
LiveKit Observability Module

This module provides comprehensive observability components for LiveKit-based voice AI
applications, enabling detailed monitoring, logging, and analysis of agent sessions.

Overview:
    This module offers plug-and-play observability components that can be easily integrated
    into LiveKit agent applications to track conversations, monitor resource usage, and
    maintain detailed audit logs for compliance and optimization purposes.

Components:

    1. Transcription Components:
        - TranscriptionHandler: Logs detailed conversation transcripts with timestamps
            and participant information, enabling conversation analysis and compliance auditing.

    2. Usage Metrics Components:
        - UsageMetricsHandler: Tracks and logs resource consumption metrics including
            STT duration, LLM token usage, and TTS character counts for cost monitoring
            and optimization.

    3. Example Agent:
        - ExampleAgent: Reference implementation demonstrating how to integrate
            observability components into a LiveKit voice agent with proper setup and
            lifecycle management.

Key Features:
    - Automated transcript generation with conversation history
    - Real-time usage metrics collection (STT, LLM, TTS)
    - Structured JSON export for easy integration with analytics tools
    - Configurable audit folder and file naming conventions
    - Comprehensive error handling and logging
    - Zero-configuration plug-and-play design
    - Separation of concerns for better maintainability

Observability Data Collected:

    Transcription Data:
        - Complete conversation history with speaker identification
        - Message timestamps and content
        - Participant metadata

    Usage Metrics:
        - LLM: Input/output tokens, completion tokens, cached tokens, prompt tokens
        - STT: Total audio duration processed
        - TTS: Audio duration generated, character count

Environment Variables Required:
    - LIVEKIT_URL: URL of the LiveKit server
    - LIVEKIT_API_KEY: API key for LiveKit authentication
    - LIVEKIT_API_SECRET: API secret for LiveKit authentication
    - OPENAI_API_KEY: OpenAI API key for GPT model access
    - ASSEMBLYAI_API_KEY: AssemblyAI API key for speech recognition
    - CARTESIA_API_KEY: Cartesia API key for text-to-speech

Usage Example:
    ```python
    from observability import (
        ObservableTranscription,
        ObservableUsageMetrics,
        usage_collector
    )

    # In your agent handler
    async def my_agent(ctx: JobContext):
        session = AgentSession(...)

        # Initialize observability components
        transcription = ObservableTranscription()
        usage_metrics = ObservableUsageMetrics(usage_collector)

        # Register shutdown callbacks
        ctx.add_shutdown_callback(transcription.create_shutdown_callback(session, ctx))
        ctx.add_shutdown_callback(usage_metrics.create_shutdown_callback(ctx))

        # Start session
        await session.start(...)
    ```

Running the Example Agent:
    ```bash
    python observability.py start
    ```

Author: Sourav Das
Version: 3.0.0
Last Updated: 2025-12-18
"""

# Standard library imports
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Third-party imports - LiveKit Core
from livekit import agents, rtc
from livekit.agents import (
    AgentServer,
    AgentSession,
    Agent,
    room_io,
    metrics,
    JobContext,
    MetricsCollectedEvent,
)

# LiveKit plugins for audio processing
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("agent_server.log")],
)

# Load environment variables
load_dotenv(override=True)

# ============================================================================
# CONSTANTS
# ============================================================================

AUDIT_FOLDER = Path("audit")
TRANSCRIPT_PREFIX = "transcript"
USAGE_PREFIX = "usage_summary"

# ============================================================================
# PRICING CONSTANTS
# ============================================================================

# All prices are in USD
PRICING = {
    "stt": {  # Prices per hour
        "assemblyai": {
            "universal-streaming": 0.150,
            "universal-streaming-multilingual": 0.150,
        },
        "cartesia": {
            "ink-whisper": 0.180,
        },
        "deepgram": {
            "flux": 0.462,
            "nova-3": 0.462,
            "nova-3-multilingual": 0.552,
            "nova-3-medical": 0.462,
            "nova-2": 0.348,
            "nova-2-medical": 0.348,
            "nova-2-conversational-ai": 0.348,
            "nova-2-phonecall": 0.348,
        },
        "elevenlabs": {
            "scribe-v2-realtime": 0.480,
        },
    },
    "llm": {  # Prices per 1M tokens
        "openai": {
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4.1": {"input": 2.00, "output": 8.00},
            "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
            "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
            "gpt-5": {"input": 1.25, "output": 10.00},
            "gpt-5-mini": {"input": 0.25, "output": 2.00},
            "gpt-5-nano": {"input": 0.05, "output": 0.40},
            "gpt-oss-120b": {"input": 0.10, "output": 0.50},
        },
        "azure": {
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4.1": {"input": 2.00, "output": 8.00},
            "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
            "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
            "gpt-5": {"input": 1.25, "output": 10.00},
            "gpt-5-mini": {"input": 0.25, "output": 2.00},
            "gpt-5-nano": {"input": 0.05, "output": 0.40},
        },
        "baseten": {
            "gpt-oss-120b": {"input": 0.10, "output": 0.50},
            "qwen3-235b-a22b-instruct": {"input": 0.22, "output": 0.80},
            "kimi-k2-instruct": {"input": 0.60, "output": 2.50},
            "deepseek-v3": {"input": 0.77, "output": 0.77},
            "deepseek-v3.2": {"input": 0.30, "output": 0.45},
        },
        "groq": {
            "gpt-oss-120b": {"input": 0.15, "output": 0.60},
        },
        "cerebras": {
            "gpt-oss-120b": {"input": 0.35, "output": 0.75},
        },
        "google": {
            "gemini-2.5-pro": {"input": 2.50, "output": 15.00},
            "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
            "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
            "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
            "gemini-2.0-flash-lite": {"input": 0.07, "output": 0.30},
        },
    },
    "tts": {  # Prices per 1M characters
        "cartesia": {
            "sonic-3": 50.0,
            "sonic-2": 50.0,
            "sonic-turbo": 50.0,
            "sonic": 50.0,
        },
        "deepgram": {
            "aura-1": 15.0,
            "aura-2": 30.0,
        },
        "elevenlabs": {
            "eleven-flash-v2": 150.0,
            "eleven-flash-v2.5": 150.0,
            "eleven-turbo-v2": 150.0,
            "eleven-turbo-v2.5": 150.0,
            "eleven-multilingual-v2": 300.0,
        },
        "inworld": {
            "inworld-tts-1-max": 10.0,
            "inworld-tts-1": 5.0,
        },
        "rime": {
            "arcana-v2": 50.0,
            "mist-v2": 50.0,
        },
    },
}


# ============================================================================
# TRANSCRIPTION OBSERVABILITY COMPONENT
# ============================================================================


class TranscriptionHandler:
    """
    Observability component for logging conversation transcripts.

    This plug-and-play component automatically captures and saves AgentSession
    conversation history to structured JSON files. It provides detailed transcripts
    including all messages, timestamps, and participant information for compliance,
    analysis, and debugging purposes.

    Attributes:
        audit_folder (Path): Directory path where transcript files are saved
        prefix (str): Filename prefix for generated transcript files

    Features:
        - Automatic conversation history capture
        - Timestamped transcript files
        - Structured JSON format for easy parsing
        - Comprehensive error handling
        - Zero-configuration operation

    Example:
        >>> transcription = ObservableTranscription()
        >>> ctx.add_shutdown_callback(transcription.create_shutdown_callback(session, ctx))
    """

    def __init__(
        self, audit_folder: Path = AUDIT_FOLDER, prefix: str = TRANSCRIPT_PREFIX
    ) -> None:
        """
        Initialize the transcription observability component.

        Args:
            audit_folder: Directory path for saving transcripts (default: 'audit')
            prefix: Filename prefix for transcript files (default: 'transcript')
        """
        self.audit_folder = audit_folder
        self.prefix = prefix
        self._ensure_folder()
        logger.info(f"ObservableTranscription initialized with folder: {audit_folder}")

    def _ensure_folder(self) -> None:
        """Ensure the audit folder exists."""
        try:
            self.audit_folder.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create transcription folder: {e}", exc_info=True)
            raise

    def _safe_json_dump(self, data: Dict[str, Any], filepath: Path) -> bool:
        """
        Safely write JSON data to a file with error handling.

        Args:
            data: Dictionary to be serialized to JSON
            filepath: Path object pointing to the target file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Successfully saved data to {filepath}")
            return True
        except (IOError, OSError) as e:
            logger.error(
                f"File I/O error while saving to {filepath}: {e}", exc_info=True
            )
            return False
        except (TypeError, ValueError) as e:
            logger.error(f"JSON serialization error for {filepath}: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(
                f"Unexpected error while saving to {filepath}: {e}", exc_info=True
            )
            return False

    def create_shutdown_callback(self, session: AgentSession, ctx: JobContext):
        """
        Create a shutdown callback function for the given session.

        This method returns an async callback that can be registered with
        ctx.add_shutdown_callback(). The callback captures the session and
        context in its closure.

        Args:
            session: The AgentSession whose history will be logged
            ctx: The JobContext containing room information

        Returns:
            Async callback function compatible with add_shutdown_callback

        Example:
            >>> callback = transcript_logger.create_callback(session, ctx)
            >>> ctx.add_shutdown_callback(callback)
        """

        async def write_transcript(reason: str) -> None:
            """
            Write conversation transcript to file.

            Args:
                reason: Shutdown reason provided by LiveKit
            """
            try:
                # Create filename with timestamp
                current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = (
                    self.audit_folder
                    / f"{self.prefix}_{ctx.room.name}_{current_date}.json"
                )

                # Get transcript data
                transcript_data = session.history.to_dict()

                # Save to file
                if self._safe_json_dump(transcript_data, filename):
                    logger.info(f"Transcript for {ctx.room.name} saved to {filename}")
                else:
                    logger.warning(f"Failed to save transcript for {ctx.room.name}")

            except Exception as e:
                logger.error(f"Error in ObservableTranscription: {e}", exc_info=True)

        return write_transcript


# ============================================================================
# USAGE METRICS OBSERVABILITY COMPONENT
# ============================================================================


class UsageMetricsHandler:
    """
    Observability component for tracking and logging usage metrics.

    This plug-and-play component automatically collects and saves detailed usage
    metrics for STT (Speech-to-Text), LLM (Language Model), and TTS (Text-to-Speech)
    services. It provides comprehensive resource consumption tracking for cost
    monitoring, optimization, and billing purposes.

    Attributes:
        usage_collector (UsageCollector): LiveKit metrics collector instance
        audit_folder (Path): Directory path where usage metric files are saved
        prefix (str): Filename prefix for generated usage metric files

    Features:
        - Real-time metrics collection for all AI services
        - Detailed token usage tracking for LLMs
        - Audio duration tracking for STT/TTS
        - Structured JSON export for analytics
        - Automatic categorization by service type
        - Comprehensive error handling

    Metrics Tracked:
        LLM: Input/output tokens, completion tokens, cached tokens, prompt tokens
        STT: Total audio processing duration
        TTS: Audio generation duration, character count

    Example:
        >>> usage_metrics = ObservableUsageMetrics(usage_collector)
        >>> ctx.add_shutdown_callback(usage_metrics.create_shutdown_callback(ctx))
    """

    def __init__(
        self,
        usage_collector: metrics.UsageCollector,
        audit_folder: Path | str = AUDIT_FOLDER,
        prefix: str = USAGE_PREFIX,
        stt_model: str = None,
        llm_model: str = None,
        tts_model: str = None,
    ) -> None:
        """
        Initialize the usage metrics observability component.

        Args:
            usage_collector: LiveKit UsageCollector instance for metrics
            audit_folder: Directory path or string of the path for saving usage logs (default: 'audit')
            prefix: Filename prefix for usage files (default: 'usage_summary')
            stt_model: STT model string in format "provider/model" (e.g., "assemblyai/universal-streaming")
            llm_model: LLM model string in format "provider/model" (e.g., "openai/gpt-4.1-mini")
            tts_model: TTS model string in format "provider/model" (e.g., "cartesia/sonic-3")
        """
        self.usage_collector = usage_collector
        self.audit_folder = audit_folder
        self.prefix = prefix
        self.stt_model = stt_model
        self.llm_model = llm_model
        self.tts_model = tts_model
        self._ensure_folder()
        logger.info(f"ObservableUsageMetrics initialized with folder: {audit_folder}")

    def _ensure_folder(self) -> None:
        """Ensure the audit folder exists."""
        try:
            self.audit_folder.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create usage logs folder: {e}", exc_info=True)
            raise

    def _safe_json_dump(self, data: Dict[str, Any], filepath: Path) -> bool:
        """
        Safely write JSON data to a file with error handling.

        Args:
            data: Dictionary to be serialized to JSON
            filepath: Path object pointing to the target file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Successfully saved data to {filepath}")
            return True
        except (IOError, OSError) as e:
            logger.error(
                f"File I/O error while saving to {filepath}: {e}", exc_info=True
            )
            return False
        except (TypeError, ValueError) as e:
            logger.error(f"JSON serialization error for {filepath}: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(
                f"Unexpected error while saving to {filepath}: {e}", exc_info=True
            )
            return False

    def _parse_model_string(self, model_string: str) -> tuple[str, str]:
        """
        Parse model string to extract provider and model name.
        
        Args:
            model_string: Model string in format "provider/model" or "provider/model:variant"
            
        Returns:
            Tuple of (provider, model_name)
        """
        if not model_string:
            return None, None
        
        # Split by '/' to get provider and model
        parts = model_string.split('/')
        if len(parts) < 2:
            return None, None
            
        provider = parts[0].lower()
        model_part = parts[1]
        
        # Remove variant/voice ID if present (after ':')
        model_name = model_part.split(':')[0].lower()
        
        return provider, model_name
    
    def _calculate_stt_cost(self, audio_duration: float) -> float:
        """
        Calculate STT cost based on audio duration.
        
        Args:
            audio_duration: Audio duration in seconds
            
        Returns:
            Cost in USD
        """
        if not self.stt_model or audio_duration <= 0:
            return 0.0
            
        provider, model_name = self._parse_model_string(self.stt_model)
        
        if not provider or not model_name:
            logger.warning(f"Unable to parse STT model: {self.stt_model}")
            return 0.0
        
        # Get pricing for provider/model
        try:
            price_per_hour = PRICING["stt"].get(provider, {}).get(model_name, 0.0)
            
            # Convert seconds to hours and calculate cost
            duration_hours = audio_duration / 3600.0
            cost = duration_hours * price_per_hour
            
            return round(cost, 6)
        except (KeyError, TypeError) as e:
            logger.warning(f"Pricing not found for STT model {provider}/{model_name}: {e}")
            return 0.0
    
    def _calculate_llm_cost(self, input_tokens: int, output_tokens: int) -> Dict[str, float]:
        """
        Calculate LLM cost based on input and output tokens.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Dictionary with input_cost, output_cost, and total_cost
        """
        if not self.llm_model or (input_tokens <= 0 and output_tokens <= 0):
            return {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}
            
        provider, model_name = self._parse_model_string(self.llm_model)
        
        if not provider or not model_name:
            logger.warning(f"Unable to parse LLM model: {self.llm_model}")
            return {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}
        
        # Get pricing for provider/model
        try:
            pricing = PRICING["llm"].get(provider, {}).get(model_name, {})
            price_per_million_input = pricing.get("input", 0.0)
            price_per_million_output = pricing.get("output", 0.0)
            
            # Calculate costs (prices are per 1M tokens)
            input_cost = (input_tokens / 1_000_000) * price_per_million_input
            output_cost = (output_tokens / 1_000_000) * price_per_million_output
            total_cost = input_cost + output_cost
            
            return {
                "input_cost": round(input_cost, 6),
                "output_cost": round(output_cost, 6),
                "total_cost": round(total_cost, 6)
            }
        except (KeyError, TypeError) as e:
            logger.warning(f"Pricing not found for LLM model {provider}/{model_name}: {e}")
            return {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}
    
    def _calculate_tts_cost(self, characters_count: int) -> float:
        """
        Calculate TTS cost based on character count.
        
        Args:
            characters_count: Number of characters synthesized
            
        Returns:
            Cost in USD
        """
        if not self.tts_model or characters_count <= 0:
            return 0.0
            
        provider, model_name = self._parse_model_string(self.tts_model)
        
        if not provider or not model_name:
            logger.warning(f"Unable to parse TTS model: {self.tts_model}")
            return 0.0
        
        # Get pricing for provider/model
        try:
            price_per_million_chars = PRICING["tts"].get(provider, {}).get(model_name, 0.0)
            
            # Calculate cost (prices are per 1M characters)
            cost = (characters_count / 1_000_000) * price_per_million_chars
            
            return round(cost, 6)
        except (KeyError, TypeError) as e:
            logger.warning(f"Pricing not found for TTS model {provider}/{model_name}: {e}")
            return 0.0

    def _format_summary(self, summary: metrics.UsageSummary) -> Dict[str, Any]:
        """
        Format usage summary into a structured dictionary with cost calculations.

        Args:
            summary (UsageSummary): UsageSummary object from UsageCollector

        Returns:
            Formatted dictionary with categorized metrics and costs
        """
        # Calculate costs
        stt_cost = self._calculate_stt_cost(summary.stt_audio_duration)
        llm_costs = self._calculate_llm_cost(summary.llm_input_tokens, summary.llm_output_tokens)
        tts_cost = self._calculate_tts_cost(summary.tts_characters_count)
        
        # Calculate total cost
        total_cost = stt_cost + llm_costs["total_cost"] + tts_cost
        
        return {
            "llm_summary": {
                "llm_input_tokens": summary.llm_input_tokens,
                "llm_output_tokens": summary.llm_output_tokens,
                "llm_completion_tokens": summary.llm_completion_tokens,
                "llm_input_cached_audio_tokens": summary.llm_input_cached_audio_tokens,
                "llm_input_cached_image_tokens": summary.llm_input_cached_image_tokens,
                "llm_input_cached_text_tokens": summary.llm_input_cached_text_tokens,
                "llm_input_image_tokens": summary.llm_input_image_tokens,
                "llm_input_text_tokens": summary.llm_input_text_tokens,
                "llm_prompt_cached_tokens": summary.llm_prompt_cached_tokens,
                "llm_prompt_tokens": summary.llm_prompt_tokens,
                "llm_cost_usd": llm_costs["total_cost"],
                "llm_input_cost_usd": llm_costs["input_cost"],
                "llm_output_cost_usd": llm_costs["output_cost"],
            },
            "stt_summary": {
                "stt_audio_duration": summary.stt_audio_duration,
                "stt_cost_usd": stt_cost,
            },
            "tts_summary": {
                "tts_audio_duration": summary.tts_audio_duration,
                "tts_characters_count": summary.tts_characters_count,
                "tts_cost_usd": tts_cost,
            },
            "cost_summary": {
                "stt_cost_usd": stt_cost,
                "llm_cost_usd": llm_costs["total_cost"],
                "tts_cost_usd": tts_cost,
                "total_cost_usd": round(total_cost, 6),
            },
            "models_used": {
                "stt_model": self.stt_model,
                "llm_model": self.llm_model,
                "tts_model": self.tts_model,
            },
        }

    def create_shutdown_callback(self, ctx: JobContext):
        """
        Create a shutdown callback function for usage logging.

        This method returns an async callback that can be registered with
        ctx.add_shutdown_callback(). The callback captures the context
        in its closure.

        Args:
            ctx: The JobContext containing room information

        Returns:
            Async callback function compatible with add_shutdown_callback

        Example:
            >>> callback = usage_logger.create_callback(ctx)
            >>> ctx.add_shutdown_callback(callback)
        """

        async def log_usage(reason: str) -> None:
            """
            Log usage summary to console and file.

            Args:
                reason: Shutdown reason provided by LiveKit
            """
            try:
                # Get and format usage summary
                raw_summary = self.usage_collector.get_summary()
                formatted_summary = self._format_summary(raw_summary)

                # Log to console
                logger.info(
                    f"Usage summary for room {ctx.room.name}: {formatted_summary}"
                )

                # Save to file
                current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = (
                    self.audit_folder
                    / f"{self.prefix}_{ctx.room.name}_{current_date}.json"
                )

                if self._safe_json_dump(formatted_summary, filename):
                    logger.info(f"Usage summary saved to {filename}")
                else:
                    logger.warning(f"Failed to save usage summary for {ctx.room.name}")

            except Exception as e:
                logger.error(f"Error in ObservableUsageMetrics: {e}", exc_info=True)

        return log_usage


# ============================================================================
# EXAMPLE AGENT IMPLEMENTATION
# ============================================================================


class ExampleAgent(Agent):
    """
    Example AI Assistant Agent demonstrating observability integration.

    This reference implementation shows how to integrate observability components
    into a LiveKit voice agent. It extends the base Agent class with predefined
    instructions for conversational AI behavior.

    Attributes:
        instructions (str): System prompt defining the assistant's behavior,
                            personality, and response formatting rules.

    Behavior:
        - Provides helpful, informative responses from extensive knowledge base
        - Uses concise, clear language without complex formatting
        - Avoids emojis, asterisks, and other decorative symbols
        - Maintains a friendly, curious tone with appropriate humor

    Usage:
        This agent is used in the example_observable_agent_handler to demonstrate
        proper integration of transcription and usage metrics observability.
    """

    def __init__(self) -> None:
        """Initialize the ExampleObservableAgent with predefined instructions."""
        super().__init__(
            instructions="""You are a helpful voice AI assistant.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )
        logger.info("ExampleObservableAgent initialized successfully")


# ============================================================================
# SERVER INITIALIZATION
# ============================================================================

# Initialize the agent server
server = AgentServer()
logger.info("AgentServer instance created")

# Initialize usage metrics collector
usage_collector = metrics.UsageCollector()
logger.info("UsageCollector initialized for metrics tracking")


# ============================================================================
# EXAMPLE AGENT HANDLER WITH OBSERVABILITY
# ============================================================================


@server.rtc_session()
async def example_observable_agent_handler(ctx: JobContext) -> None:
    """
    Example agent handler demonstrating observability component integration.

    This handler serves as a reference implementation showing how to properly
    integrate ObservableTranscription and ObservableUsageMetrics into a LiveKit
    voice agent. It demonstrates the complete lifecycle from session setup to
    shutdown with comprehensive observability.

    Args:
        ctx: JobContext object containing room information and utilities

    Implementation Flow:
        1. Initialize AgentSession with AI service providers (STT, LLM, TTS, VAD)
        2. Register metrics collection event handler
        3. Initialize observability components (transcription + usage metrics)
        4. Register shutdown callbacks for data persistence
        5. Start session with noise cancellation configuration
        6. Generate initial greeting to the user

    Observability Components:
        - ObservableTranscription: Captures complete conversation history
        - ObservableUsageMetrics: Tracks resource consumption (tokens, duration)

    Event Handlers:
        - metrics_collected: Continuously tracks usage metrics during session

    Shutdown Callbacks:
        - write_transcript: Saves conversation transcript to audit folder
        - log_usage: Saves usage metrics summary to audit folder

    Error Handling:
        All exceptions are caught and logged; does not propagate errors

    Example:
        This handler is automatically invoked by the LiveKit server for each
        new room connection. To run: `python observability.py start`
    """
    logger.info(f"Starting new observable agent session for room: {ctx.room.name}")

    # Define model configurations
    stt_model = "assemblyai/universal-streaming:en"
    llm_model = "openai/gpt-4.1-mini"
    tts_model = "cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"

    try:
        # Initialize agent session with AI service providers
        session = AgentSession(
            stt=stt_model,  # Speech-to-text
            llm=llm_model,  # Language model
            tts=tts_model,  # Text-to-speech
            vad=silero.VAD.load(),  # Voice activity detection
            turn_detection=MultilingualModel(),  # Turn-taking detection
        )
        logger.info(
            "AgentSession configured with STT, LLM, TTS, VAD, and turn detection"
        )

    except Exception as e:
        logger.error(f"Failed to initialize AgentSession: {e}", exc_info=True)
        return

    # Register metrics collection handler
    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        """Collect usage metrics throughout the session."""
        usage_collector.collect(ev.metrics)

    # Initialize observability components with model information for cost calculation
    transcription = TranscriptionHandler()
    usage_metrics = UsageMetricsHandler(
        usage_collector,
        stt_model=stt_model,
        llm_model=llm_model,
        tts_model=tts_model,
    )
    logger.info("Observability components initialized")

    # Register shutdown callbacks for data persistence
    ctx.add_shutdown_callback(transcription.create_shutdown_callback(session, ctx))
    ctx.add_shutdown_callback(usage_metrics.create_shutdown_callback(ctx))
    logger.debug("Shutdown callbacks registered for observability")

    try:
        # Start the agent session with noise cancellation
        await session.start(
            room=ctx.room,
            agent=ExampleAgent(),
            room_options=room_io.RoomOptions(
                audio_input=room_io.AudioInputOptions(
                    noise_cancellation=lambda params: (
                        # Use telephony-optimized noise cancellation for SIP participants
                        noise_cancellation.BVCTelephony()
                        if params.participant.kind
                        == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                        # Use standard noise cancellation for regular participants
                        else noise_cancellation.BVC()
                    ),
                ),
            ),
        )
        logger.info(f"Agent session started successfully for room: {ctx.room.name}")

    except Exception as e:
        logger.error(f"Error starting agent session: {e}", exc_info=True)
        return

    try:
        # Generate initial greeting to user
        await session.generate_reply(
            instructions="Greet the user and offer your assistance."
        )
        logger.debug("Initial greeting generated")

    except Exception as e:
        logger.error(f"Error generating initial greeting: {e}", exc_info=True)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


if __name__ == "__main__":
    """
    Main entry point for running the example observable agent server.

    Usage:
        python observability.py start

    This starts the LiveKit agent server with observability enabled:
    1. Connects to LiveKit server using environment variables
    2. Listens for incoming room connections
    3. Spawns observable agent instances for each new session
    4. Automatically logs transcripts and usage metrics
    5. Saves all data to the audit folder on session completion

    Environment Variables Required:
        - LIVEKIT_URL: LiveKit server URL
        - LIVEKIT_API_KEY: API key for authentication
        - LIVEKIT_API_SECRET: API secret for authentication
        - OPENAI_API_KEY: OpenAI API key
        - ASSEMBLYAI_API_KEY: AssemblyAI API key
        - CARTESIA_API_KEY: Cartesia API key
    """
    try:
        logger.info("Starting LiveKit Observable Agent Server...")
        logger.info("Observability enabled: Transcription + Usage Metrics")
        agents.cli.run_app(server)
    except KeyboardInterrupt:
        logger.info("Observable agent server stopped by user")
    except Exception as e:
        logger.critical(
            f"Fatal error running observable agent server: {e}", exc_info=True
        )
        raise
