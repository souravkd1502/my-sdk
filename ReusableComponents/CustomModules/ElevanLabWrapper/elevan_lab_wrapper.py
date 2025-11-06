"""
ElevenLabs API Wrapper Module

This module provides a comprehensive Python wrapper for the ElevenLabs API, offering convenient
client classes for text-to-speech, text-to-dialogue, speech-to-text, and AI music generation.

Main Classes:
    TextToSpeechClient: Convert text to natural-sounding speech with customizable voices
    TextToDialogueClient: Generate multi-speaker dialogue audio with emotional annotations
    SpeechToTextClient: Transcribe audio files with diarization and multi-channel support
    MusicClient: Generate AI-powered music compositions from text descriptions

Key Features:
    - Multiple output formats: direct playback, file saving, BytesIO streams, S3 uploads
    - Advanced voice customization and settings
    - Request chaining for seamless paragraph stitching
    - Multi-channel audio transcription with speaker diarization
    - Conversation-style transcript generation
    - Music composition with detailed planning options
    - Comprehensive error handling and logging
    - Type hints for better IDE support and code quality

Environment Variables:
    ELEVENLABS_API_KEY (required): Your ElevenLabs API key for authentication
    AWS_ACCESS_KEY_ID (optional): AWS access key for S3 upload functionality
    AWS_SECRET_ACCESS_KEY (optional): AWS secret key for S3 upload functionality
    AWS_REGION_NAME (optional): AWS region name for S3 operations
    AWS_S3_BUCKET_NAME (optional): S3 bucket name for audio file uploads

Dependencies:
    - elevenlabs: Official ElevenLabs Python SDK
    - boto3: AWS SDK for S3 integration (optional)
    - python-dotenv: Environment variable management
    - requests: HTTP library for downloading audio from URLs

Example Usage:
    >>> from elevan_lab_wrapper import TextToSpeechClient, TextToDialogueClient
    >>> 
    >>> # Text-to-Speech
    >>> tts_client = TextToSpeechClient()
    >>> tts_client.speak("Hello, world!")
    >>> file_path = tts_client.to_file("Generate speech and save to file")
    >>> 
    >>> # Text-to-Dialogue
    >>> dialogue_client = TextToDialogueClient()
    >>> dialogue_inputs = [
    ...     {"text": "[cheerfully] Hello!", "voice_id": "voice_1"},
    ...     {"text": "[calmly] Hi there!", "voice_id": "voice_2"}
    ... ]
    >>> dialogue_client.to_file(dialogue_inputs, "conversation.mp3")
    >>> 
    >>> # Speech-to-Text
    >>> from elevan_lab_wrapper import SpeechToTextClient
    >>> stt_client = SpeechToTextClient()
    >>> result = stt_client.transcribe("audio.wav", diarize=True)
    >>> 
    >>> # AI Music Generation
    >>> from elevan_lab_wrapper import MusicClient
    >>> music_client = MusicClient()
    >>> track = music_client.compose_music(prompt="Upbeat electronic dance music")
    >>> music_client.play_music(track)

Notes:
    - All clients require an ElevenLabs API key (get one at https://elevenlabs.io)
    - S3 functionality is optional and only available if AWS credentials are configured
    - Logging is configured automatically at module import with INFO level
    - For production use, consider implementing rate limiting and request retries

Author: Sourav Das
Date: 06-11-2025
Version: 1.0.0
License: MIT
"""

import os
import uuid
import logging
import requests
from io import BytesIO
from typing import IO, Optional, List, Dict, Sequence, Union

from dotenv import load_dotenv
from elevenlabs.play import play
from elevenlabs import ElevenLabs, VoiceSettings, DialogueInput

import boto3

# Module logger - use proper initialization pattern
logger = logging.getLogger(__name__)

# Configure logging if not already configured
if not logger.handlers:
    # Set log level
    logger.setLevel(logging.INFO)
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    # Add handler to logger
    logger.addHandler(console_handler)
    # Prevent propagation to avoid duplicate logs if parent loggers are configured
    logger.propagate = False

# Load environment variables from .env file if it exists
load_dotenv(override=True)

# Optional: AWS S3 setup for uploading audio files
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION_NAME = os.getenv("AWS_REGION_NAME")
AWS_S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")

if (
    AWS_ACCESS_KEY_ID
    and AWS_SECRET_ACCESS_KEY
    and AWS_REGION_NAME
    and AWS_S3_BUCKET_NAME
):
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION_NAME,
    )
    s3 = session.client("s3")
else:
    s3 = None


class TextToSpeechClient:
    """
    A client wrapper for ElevenLabs Text-to-Speech API.

    Provides methods for converting text to speech, saving to files,
    streaming audio, uploading to S3, and stitching multiple paragraphs.
    """

    def __init__(self, api_key: Optional[str] = None, config: Optional[dict] = None):
        """
        Initialize the TTS client with API key and optional configuration.

        Args:
            api_key (str, optional): ElevenLabs API key. Defaults to environment variable.
            config (dict, optional): Optional configuration for voice, model, output format, and voice settings.

        Raises:
            ValueError: If API key is not provided.
        """
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.config = config or {}

        if not self.api_key:
            logger.error("ELEVENLABS_API_KEY is required.")
            raise ValueError("ELEVENLABS_API_KEY is required.")

        self.client = ElevenLabs(api_key=self.api_key)

        # Default configuration
        self.default_voice_id = self.config.get("voice_id", "pNInz6obpgDQGcFmaJgB")
        self.default_model_id = self.config.get("model_id", "eleven_multilingual_v2")
        self.default_output_format = self.config.get("output_format", "mp3_22050_32")
        self.default_voice_settings: VoiceSettings = self.config.get(
            "voice_settings",
            VoiceSettings(
                stability=0.0,
                similarity_boost=1.0,
                style=0.0,
                use_speaker_boost=True,
                speed=1.0,
            ),
        )

        logger.info("TextToSpeechClient initialized successfully.")

    def _validate_config(self) -> None:
        """
        Validates the TTS configuration to ensure all required fields are present
        and optional parameters have the correct types.

        Raises:
            ValueError: If required parameters are missing or invalid.
            TypeError: If parameters are of incorrect type.
        """
        config = self.config

        # Required parameter
        if not config.get("voice_id"):
            raise ValueError("voice_id is required for text-to-speech conversion.")

        # Optional type checks
        if "text" in config and not isinstance(config["text"], str):
            raise TypeError("text must be a string.")
        if "enable_logging" in config and not isinstance(
            config["enable_logging"], (bool, type(None))
        ):
            raise TypeError("enable_logging must be a boolean or None.")
        if "optimize_streaming_latency" in config and config[
            "optimize_streaming_latency"
        ] not in (None, 0, 1, 2, 3, 4):
            raise ValueError("optimize_streaming_latency must be 0, 1, 2, 3, or 4.")
        if "output_format" in config and not isinstance(
            config["output_format"], (str, type(None))
        ):
            raise TypeError(
                "output_format must be a string like 'mp3_44100_128' or None."
            )
        if "model_id" in config and not isinstance(
            config["model_id"], (str, type(None))
        ):
            raise TypeError("model_id must be a string or None.")
        if "language_code" in config and not isinstance(
            config["language_code"], (str, type(None))
        ):
            raise TypeError("language_code must be a string or None.")

        logger.info("TTS configuration validated successfully.")

    def speak(self, text: str) -> None:
        """
        Convert text to speech and play audio immediately.

        Args:
            text (str): Text to convert.

        Raises:
            RuntimeError: If conversion fails.
        """
        try:
            audio = self.client.text_to_speech.convert(
                text=text,
                voice_id=self.default_voice_id,
                model_id=self.default_model_id,
                output_format=self.default_output_format,
                voice_settings=self.default_voice_settings,
            )
            play(audio)
            logger.info("Audio played successfully.")
        except Exception as e:
            logger.error(f"Failed to play audio: {e}")
            raise RuntimeError(f"Failed to play audio: {e}") from e

    def to_file(self, text: str, save_path: Optional[str] = None) -> str:
        """
        Convert text to speech and save audio to a file.

        Args:
            text (str): Text to convert.
            save_path (str, optional): File path to save audio. Defaults to a UUID-based filename.

        Returns:
            str: Path to saved audio file.

        Raises:
            RuntimeError: If saving fails.
        """
        save_path = save_path or f"{uuid.uuid4()}.mp3"
        try:
            response = self.client.text_to_speech.convert(
                text=text,
                voice_id=self.default_voice_id,
                model_id=self.default_model_id,
                output_format=self.default_output_format,
                voice_settings=self.default_voice_settings,
            )
            with open(save_path, "wb") as f:
                for chunk in response:
                    if chunk:
                        f.write(chunk)
            logger.info(f"Audio saved to file: {save_path}")
            return save_path
        except Exception as e:
            logger.error(f"Failed to save audio to file: {e}")
            raise RuntimeError(f"Failed to save audio to file: {e}") from e

    def to_stream(self, text: str) -> IO[bytes]:
        """
        Convert text to speech and return as a BytesIO stream.

        Args:
            text (str): Text to convert.

        Returns:
            BytesIO: Audio stream of the generated speech.

        Raises:
            RuntimeError: If streaming fails.
        """
        try:
            response = self.client.text_to_speech.stream(
                text=text,
                voice_id=self.default_voice_id,
                model_id=self.default_model_id,
                output_format=self.default_output_format,
                voice_settings=self.default_voice_settings,
            )
            audio_stream = BytesIO()
            for chunk in response:
                if chunk:
                    audio_stream.write(chunk)
            audio_stream.seek(0)
            logger.info("Audio stream created successfully.")
            return audio_stream
        except Exception as e:
            logger.error(f"Failed to generate audio stream: {e}")
            raise RuntimeError(f"Failed to generate audio stream: {e}") from e

    def upload_to_s3(self, audio_stream: IO[bytes]) -> str:
        """
        Upload audio stream to AWS S3 and return the object key.

        Args:
            audio_stream (IO[bytes]): Audio stream to upload.

        Returns:
            str: S3 object key.

        Raises:
            RuntimeError: If S3 client is not configured or upload fails.
        """
        if not s3:
            logger.error("S3 client not configured.")
            raise RuntimeError("S3 client not configured.")
        s3_file_name = f"{uuid.uuid4()}.mp3"
        try:
            s3.upload_fileobj(audio_stream, AWS_S3_BUCKET_NAME, s3_file_name)
            logger.info(f"Audio uploaded to S3: {s3_file_name}")
            return s3_file_name
        except Exception as e:
            logger.error(f"Failed to upload audio to S3: {e}")
            raise RuntimeError(f"Failed to upload audio to S3: {e}") from e

    def stitch_paragraphs(self, paragraphs: List[str]) -> BytesIO:
        """
        Convert multiple paragraphs to speech and combine into a single audio stream.

        Args:
            paragraphs (List[str]): List of paragraphs to convert.

        Returns:
            BytesIO: Combined audio stream.

        Raises:
            RuntimeError: If conversion or stitching fails.
        """
        try:
            request_ids = []
            audio_buffers = []

            for paragraph in paragraphs:
                with self.client.text_to_speech.with_raw_response.convert(
                    text=paragraph,
                    voice_id=self.default_voice_id,
                    model_id=self.default_model_id,
                    previous_request_ids=request_ids,
                ) as response:
                    request_ids.append(response._response.headers.get("request-id"))
                    audio_data = b"".join(response.data)
                    audio_buffers.append(BytesIO(audio_data))

            combined_stream = BytesIO(b"".join(buf.getvalue() for buf in audio_buffers))
            combined_stream.seek(0)
            logger.info("Paragraphs stitched into single audio stream successfully.")
            return combined_stream
        except Exception as e:
            logger.error(f"Failed to stitch paragraphs: {e}")
            raise RuntimeError(f"Failed to stitch paragraphs: {e}") from e

    def example_usage(self):
        """Example usage of the TextToDialogueClient"""
        print(
            """
            # Example - text to speech
            text = "Hello! This is a test of the ElevenLabs TextToSpeechClient."

            # Initialize the TTS client
            tts_client = TextToSpeechClient()

            print("🔊 Playing audio directly...")
            tts_client.speak(text)

            print("💾 Saving audio to file...")
            file_path = tts_client.to_file(text)
            print(f"Audio saved to {file_path}")

            print("🎧 Getting audio as stream...")
            audio_stream = tts_client.to_stream(text)
            print(f"Audio stream ready with {len(audio_stream.getvalue())} bytes")

            # Optional: upload to S3 if configured
            if s3:
                print("☁️ Uploading audio stream to S3...")
                s3_key = tts_client.upload_to_s3(audio_stream)
                print(f"Audio uploaded to S3 with key: {s3_key}")
                
            # Optional: test stitching multiple paragraphs
            paragraphs = [
                "This is the first paragraph.",
                "Here comes the second paragraph.",
                "Finally, this is the third paragraph."
            ]
            print("🔗 Stitching multiple paragraphs into one audio stream...")
            stitched_stream = tts_client.stitch_paragraphs(paragraphs)
            stitched_file = "stitched_audio.mp3"
            with open(stitched_file, "wb") as f:
                f.write(stitched_stream.getvalue())
            print(f"Stitched audio saved to {stitched_file}")
            """
        )


class TextToDialogueClient:
    """
    A client wrapper for ElevenLabs Text-to-Dialogue API.

    Provides methods for converting dialogues to speech, playing audio,
    saving to files, streaming audio, uploading to S3, and stitching multiple dialogues.
    """

    def __init__(self, api_key: Optional[str] = None, config: Optional[dict] = None):
        """
        Initialize the TextToDialogue client with API key and optional configuration.

        Args:
            api_key (str, optional): ElevenLabs API key. Defaults to environment variable.
            config (dict, optional): Optional configuration for output_format.

        Raises:
            ValueError: If API key is not provided.
        """
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.config = config or {}

        if not self.api_key:
            logger.error("ELEVENLABS_API_KEY is required.")
            raise ValueError("ELEVENLABS_API_KEY is required.")

        self.client = ElevenLabs(api_key=self.api_key)
        self.default_output_format: str = self.config.get(
            "output_format", "mp3_22050_32"
        )

        logger.info("TextToDialogueClient initialized successfully.")

    def _validate_config(
        self,
        inputs: Sequence[Union[DialogueInput, Dict]],
        output_format: Optional[str] = None,
        model_id: Optional[str] = None,
        language_code: Optional[str] = None,
        settings: Optional[dict] = None,
        pronunciation_dictionary_locators: Optional[Sequence[dict]] = None,
        seed: Optional[int] = None,
        apply_text_normalization: Optional[str] = None,
        request_options: Optional[dict] = None,
    ) -> None:
        """
        Validate configuration for text_to_dialogue.convert.

        Args:
            inputs (Sequence[Union[DialogueInput, dict]]): Dialogue inputs to validate.
            output_format (str, optional): Output audio format.
            model_id (str, optional): Model ID for dialogue synthesis.
            language_code (str, optional): Language code (ISO 639-1).
            settings (dict, optional): Model settings.
            pronunciation_dictionary_locators (Sequence[dict], optional): Max 3 pronunciation dictionaries.
            seed (int, optional): Random seed (0-4294967295).
            apply_text_normalization (str, optional): 'auto', 'on', or 'off'.
            request_options (dict, optional): Additional request options.

        Raises:
            ValueError, TypeError: If inputs or parameters are invalid.
        """
        # Validate inputs
        if not inputs or not isinstance(inputs, (list, tuple)):
            raise ValueError(
                "inputs must be a non-empty list or tuple of DialogueInput objects or dicts."
            )

        for idx, item in enumerate(inputs):
            if isinstance(item, dict):
                if "text" not in item or "voice_id" not in item:
                    raise ValueError(f"Entry {idx} must include 'text' and 'voice_id'.")
            elif not isinstance(item, DialogueInput):
                raise TypeError(
                    f"Entry {idx} must be a DialogueInput or dict with 'text' and 'voice_id'."
                )

        # Optional parameter type checks
        if output_format is not None and not isinstance(output_format, str):
            raise TypeError("output_format must be a string, e.g., 'mp3_22050_32'.")
        if model_id is not None and not isinstance(model_id, str):
            raise TypeError("model_id must be a string.")
        if language_code is not None and not isinstance(language_code, str):
            raise TypeError("language_code must be a string (ISO 639-1).")
        if settings is not None and not isinstance(settings, dict):
            raise TypeError(
                "settings must be a dict or ModelSettingsResponseModel object."
            )
        if pronunciation_dictionary_locators is not None:
            if not isinstance(pronunciation_dictionary_locators, (list, tuple)):
                raise TypeError(
                    "pronunciation_dictionary_locators must be a list or tuple of dicts."
                )
            if len(pronunciation_dictionary_locators) > 3:
                raise ValueError(
                    "Maximum of 3 pronunciation_dictionary_locators allowed."
                )
        if seed is not None and not (0 <= seed <= 4294967295):
            raise ValueError("seed must be an integer between 0 and 4294967295.")
        if apply_text_normalization is not None and apply_text_normalization not in (
            "auto",
            "on",
            "off",
        ):
            raise ValueError(
                "apply_text_normalization must be one of: 'auto', 'on', 'off'."
            )
        if request_options is not None and not isinstance(request_options, dict):
            raise TypeError("request_options must be a dict.")

        logger.info("Dialogue configuration validated successfully.")

    def play(self, dialogue_inputs: List[Dict]) -> None:
        """
        Convert dialogue inputs to speech and play audio immediately.

        Args:
            dialogue_inputs (List[Dict]): Dialogue input objects.

        Raises:
            RuntimeError: If audio conversion or playback fails.
        """
        self._validate_config(dialogue_inputs)
        try:
            audio = self.client.text_to_dialogue.convert(inputs=dialogue_inputs)
            play(audio)
            logger.info("Dialogue audio played successfully.")
        except Exception as e:
            logger.error(f"Failed to play dialogue audio: {e}")
            raise RuntimeError(f"Failed to play dialogue audio: {e}") from e

    def to_file(
        self, dialogue_inputs: List[Dict], save_path: Optional[str] = None
    ) -> str:
        """
        Convert dialogue to speech and save to a file.

        Args:
            dialogue_inputs (List[Dict]): Dialogue input objects.
            save_path (str, optional): Path to save audio. Defaults to UUID-based filename.

        Returns:
            str: Path to saved audio file.

        Raises:
            RuntimeError: If conversion or file writing fails.
        """
        self._validate_config(dialogue_inputs)
        save_path = save_path or f"{uuid.uuid4()}.mp3"
        try:
            audio = self.client.text_to_dialogue.convert(inputs=dialogue_inputs)
            with open(save_path, "wb") as f:
                for chunk in audio:
                    if chunk:
                        f.write(chunk)
            logger.info(f"Dialogue audio saved to file: {save_path}")
            return save_path
        except Exception as e:
            logger.error(f"Failed to save dialogue audio to file: {e}")
            raise RuntimeError(f"Failed to save dialogue audio to file: {e}") from e

    def to_stream(self, dialogue_inputs: List[Dict]) -> IO[bytes]:
        """
        Convert dialogue to audio stream (BytesIO).

        Args:
            dialogue_inputs (List[Dict]): Dialogue input objects.

        Returns:
            IO[bytes]: Audio stream of the generated dialogue.

        Raises:
            RuntimeError: If streaming fails.
        """
        self._validate_config(dialogue_inputs)
        try:
            audio = self.client.text_to_dialogue.stream(inputs=dialogue_inputs)
            audio_stream = BytesIO()
            for chunk in audio:
                if chunk:
                    audio_stream.write(chunk)
            audio_stream.seek(0)
            logger.info("Dialogue audio stream created successfully.")
            return audio_stream
        except Exception as e:
            logger.error(f"Failed to generate dialogue audio stream: {e}")
            raise RuntimeError(f"Failed to generate dialogue audio stream: {e}") from e

    def upload_to_s3(self, audio_stream: IO[bytes]) -> str:
        """
        Upload audio stream to AWS S3 and return the object key.

        Args:
            audio_stream (IO[bytes]): Audio stream to upload.

        Returns:
            str: S3 object key.

        Raises:
            RuntimeError: If S3 client not configured or upload fails.
        """
        if not s3:
            logger.error("S3 client not configured.")
            raise RuntimeError("S3 client not configured.")
        s3_file_name = f"{uuid.uuid4()}.mp3"
        try:
            s3.upload_fileobj(audio_stream, AWS_S3_BUCKET_NAME, s3_file_name)
            logger.info(f"Dialogue audio uploaded to S3: {s3_file_name}")
            return s3_file_name
        except Exception as e:
            logger.error(f"Failed to upload dialogue audio to S3: {e}")
            raise RuntimeError(f"Failed to upload dialogue audio to S3: {e}") from e

    def stitch_dialogues(self, dialogues_list: List[List[Dict]]) -> BytesIO:
        """
        Stitch multiple dialogue requests together into a single audio stream.

        Args:
            dialogues_list (List[List[Dict]]): List of dialogue input lists.

        Returns:
            BytesIO: Combined audio stream of all dialogues.

        Raises:
            RuntimeError: If conversion or stitching fails.
        """
        combined_stream = BytesIO()
        try:
            for dialogue_inputs in dialogues_list:
                self._validate_config(dialogue_inputs)
                audio = self.client.text_to_dialogue.convert(inputs=dialogue_inputs)
                for chunk in audio:
                    if chunk:
                        combined_stream.write(chunk)
            combined_stream.seek(0)
            logger.info(
                "All dialogues stitched successfully into a single audio stream."
            )
            return combined_stream
        except Exception as e:
            logger.error(f"Failed to stitch dialogues: {e}")
            raise RuntimeError(f"Failed to stitch dialogues: {e}") from e

    def example_usage(self):
        """Example usage of the TextToDialogueClient"""
        print(
            """
            # Example - text to dialogue
            dialogue_inputs = [
                DialogueInput(text="[cheerfully] Hello, how are you?", voice_id="9BWtsMINqrJLrRacOk9x"),
                DialogueInput(text="[stuttering] I'm... I'm doing well, thank you", voice_id="IKne3meq5aSn9XLyUdCD"),
            ]

            # Initialize the dialogue client
            dialogue_client = TextToDialogueClient()

            # Play audio directly
            print("🔊 Playing dialogue audio...")
            dialogue_client.play(dialogue_inputs)

            # Save dialogue to file
            print("💾 Saving dialogue audio to file...")
            file_path = dialogue_client.to_file(dialogue_inputs)
            print(f"Dialogue audio saved to {file_path}")

            # Get dialogue as a BytesIO stream
            print("🎧 Converting dialogue to audio stream...")
            audio_stream = dialogue_client.to_stream(dialogue_inputs)
            print(f"Dialogue audio stream ready with {len(audio_stream.getvalue())} bytes")

            # Optional: upload to S3 if configured
            if s3:
                print("☁️ Uploading dialogue audio stream to S3...")
                s3_key = dialogue_client.upload_to_s3(audio_stream)
                print(f"Dialogue audio uploaded to S3 with key: {s3_key}")

            # Test stitching multiple dialogues
            stitched_dialogues = [
                [
                    DialogueInput(text="First dialogue block, line 1.", voice_id="9BWtsMINqrJLrRacOk9x"),
                    DialogueInput(text="First dialogue block, line 2.", voice_id="IKne3meq5aSn9XLyUdCD"),
                ],
                [
                    DialogueInput(text="Second dialogue block, line 1.", voice_id="9BWtsMINqrJLrRacOk9x"),
                    DialogueInput(text="Second dialogue block, line 2.", voice_id="IKne3meq5aSn9XLyUdCD"),
                ]
            ]

            print("🔗 Stitching multiple dialogue blocks...")
            stitched_stream = dialogue_client.stitch_dialogues(stitched_dialogues)
            stitched_file = "stitched_dialogue.mp3"
            with open(stitched_file, "wb") as f:
                f.write(stitched_stream.getvalue())
            print(f"Stitched dialogue audio saved to {stitched_file}")
            """
        )


class SpeechToTextClient:
    """
    A client wrapper for ElevenLabs Speech-to-Text API.

    Provides methods for transcribing audio files or streams, handling
    multi-channel audio, diarization, and generating conversation-style transcripts.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the SpeechToText client.

        Args:
            api_key (str, optional): ElevenLabs API key. Defaults to environment variable.

        Raises:
            ValueError: If API key is not provided.
        """
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            logger.error("ELEVENLABS_API_KEY is required.")
            raise ValueError("ELEVENLABS_API_KEY is required.")

        self.client = ElevenLabs(api_key=self.api_key)
        logger.info("SpeechToTextClient initialized successfully.")

    def _validate_config(
        self,
        file: Union[BytesIO, str],
        model_id: str = "scribe_v1",
        use_multi_channel: bool = False,
        diarize: bool = False,
        timestamps_granularity: Optional[str] = None,
        language_code: Optional[str] = None,
    ) -> None:
        """
        Validate parameters for audio transcription.

        Args:
            file (BytesIO or str): Audio file object or path.
            model_id (str): Model ID to use.
            use_multi_channel (bool): Enable multi-channel mode.
            diarize (bool): Enable speaker diarization.
            timestamps_granularity (str, optional): 'none', 'word', or 'character'.
            language_code (str, optional): Language code (ISO 639-1).

        Raises:
            ValueError, TypeError: If parameters are invalid.
        """
        if file is None:
            raise ValueError("Audio file must be provided.")
        if use_multi_channel and diarize:
            raise ValueError("Diarization cannot be used with multi-channel mode.")
        if timestamps_granularity not in (None, "word", "character"):
            raise ValueError(
                "timestamps_granularity must be 'word', 'character', or None."
            )
        if timestamps_granularity == "":
            raise ValueError(
                "timestamps_granularity cannot be an empty string. Must be 'word', 'character', or None."
            )
        if not isinstance(model_id, str):
            raise TypeError("model_id must be a string.")
        if language_code is not None and not isinstance(language_code, str):
            raise TypeError("language_code must be a string (ISO 639-1).")

        logger.info("Transcription configuration validated successfully.")

    def transcribe(
        self,
        file: Union[BytesIO, str],
        model_id: str = "scribe_v1",
        use_multi_channel: bool = False,
        diarize: bool = True,
        timestamps_granularity: Optional[str] = None,
        language_code: Optional[str] = None,
        tag_audio_events: bool = False,
    ) -> Dict:
        """
        Transcribe a single audio file or stream.

        Args:
            file (BytesIO or str): Audio file object or local path.
            model_id (str): Model ID to use.
            use_multi_channel (bool): Enable multi-channel transcription.
            diarize (bool): Enable speaker diarization.
            timestamps_granularity (str, optional): 'word', 'character', or None.
            language_code (str, optional): Language code.
            tag_audio_events (bool): Include audio event tags.

        Returns:
            Dict: Transcription result.

        Raises:
            RuntimeError: If transcription fails.
        """
        self._validate_config(
            file,
            model_id,
            use_multi_channel,
            diarize,
            timestamps_granularity,
            language_code,
        )

        # Load file if a path is provided
        try:
            if isinstance(file, str):
                with open(file, "rb") as f:
                    audio_file = BytesIO(f.read())
            else:
                audio_file = file

            result = self.client.speech_to_text.convert(
                file=audio_file,
                model_id=model_id,
                use_multi_channel=use_multi_channel,
                diarize=diarize,
                timestamps_granularity=timestamps_granularity,
                language_code=language_code,
                tag_audio_events=tag_audio_events,
            )
            logger.info("Audio transcription completed successfully.")
            return result
        except Exception as e:
            logger.error(f"Failed to transcribe audio: {e}")
            raise RuntimeError(f"Failed to transcribe audio: {e}") from e

    def create_conversation_transcript(self, transcription_result) -> List[Dict]:
        """
        Generate a conversation-style transcript from multi-channel transcription.

        Args:
            transcription_result: Result returned by `transcribe`.

        Returns:
            List[Dict]: List of dictionaries with 'speaker' and 'text'.

        Notes:
            - Single-channel transcriptions return a default speaker 0.
        """
        all_words = []

        # Multi-channel transcription
        if hasattr(transcription_result, "transcripts"):
            for transcript in transcription_result.transcripts:
                for word in transcript.words or []:
                    if word.type == "word":
                        all_words.append(
                            {
                                "text": word.text,
                                "start": word.start,
                                "speaker_id": word.speaker_id,
                                "channel": transcript.channel_index,
                            }
                        )
        # Single-channel fallback
        elif hasattr(transcription_result, "text"):
            return [{"speaker": 0, "text": transcription_result.text}]

        # Sort by timestamp
        all_words.sort(key=lambda w: w["start"])

        # Group consecutive words by speaker
        conversation = []
        current_speaker = None
        current_text = []
        for word in all_words:
            if word["speaker_id"] != current_speaker:
                if current_text:
                    conversation.append(
                        {"speaker": current_speaker, "text": " ".join(current_text)}
                    )
                current_speaker = word["speaker_id"]
                current_text = [word["text"]]
            else:
                current_text.append(word["text"])
        if current_text:
            conversation.append(
                {"speaker": current_speaker, "text": " ".join(current_text)}
            )

        logger.info("Conversation-style transcript generated successfully.")
        return conversation

    def transcribe_from_url(self, audio_url: str, **kwargs) -> Dict:
        """
        Download audio from a URL and transcribe.

        Args:
            audio_url (str): URL to audio file.
            **kwargs: Additional arguments passed to `transcribe`.

        Returns:
            Dict: Transcription result.

        Raises:
            RuntimeError: If download or transcription fails.
        """
        try:
            response = requests.get(audio_url)
            response.raise_for_status()
            audio_data = BytesIO(response.content)
            logger.info(f"Audio downloaded successfully from {audio_url}")
            return self.transcribe(audio_data, **kwargs)
        except requests.RequestException as e:
            logger.error(f"Failed to download audio from URL: {e}")
            raise RuntimeError(f"Failed to download audio from URL: {e}") from e
        except Exception as e:
            logger.error(f"Failed to transcribe audio from URL: {e}")
            raise RuntimeError(f"Failed to transcribe audio from URL: {e}") from e

    def example_usage(self):
        """Example usage of the SpeechToTextClient"""
        print(
            """
            # Example - speech to text
            # Initialize the SpeechToTextClient
            stt_client = SpeechToTextClient()

            # Example 1: Transcribe audio from URL
            audio_url = (
                "https://storage.googleapis.com/eleven-public-cdn/audio/marketing/nicole.mp3"
            )
            print("🔊 Transcribing audio from URL...")
            result = stt_client.transcribe_from_url(
                audio_url,
                model_id="scribe_v1",
                diarize=True,
                language_code="eng",
                tag_audio_events=True,
                timestamps_granularity="word",
            )
            print("✅ Transcription result (URL):")
            print(result)

            # Example 2: Transcribe local audio file (single-channel)
            local_file = "example_audio.wav"  # Replace with your file path
            if os.path.exists(local_file):
                print(f"🔊 Transcribing local file: {local_file}...")
                result_local = stt_client.transcribe(
                    file=local_file,
                    model_id="scribe_v1",
                    diarize=True,
                    language_code="eng",
                    timestamps_granularity="word",
                )
                print("✅ Transcription result (local file):")
                print(result_local)
            else:
                print(
                    f"⚠️ Local file {local_file} not found. Skipping local transcription test."
                )

            # Example 3: Multi-channel transcription
            multichannel_file = "stereo_interview.wav"  # Replace with your file path
            if os.path.exists(multichannel_file):
                print(f"🔊 Transcribing multi-channel file: {multichannel_file}...")
                result_multi = stt_client.transcribe(
                    file=multichannel_file,
                    model_id="scribe_v1",
                    use_multi_channel=True,
                    diarize=False,
                    timestamps_granularity="word",
                )
                # Create conversation-style transcript
                conversation = stt_client.create_conversation_transcript(result_multi)
                print("✅ Multi-channel conversation transcript:")
                for turn in conversation:
                    print(f"Speaker {turn['speaker']}: {turn['text']}")
            else:
                print(
                    f"⚠️ Multi-channel file {multichannel_file} not found. Skipping multi-channel test."
                )
            """
        )


class MusicClient:
    """
    A client wrapper for ElevenLabs Music API.

    Provides methods for generating music from prompts or composition plans,
    validating composition plans, and playing generated music.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the MusicClient.

        Args:
            api_key (str, optional): ElevenLabs API key. Defaults to environment variable.

        Raises:
            ValueError: If API key is not provided.
        """
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            logger.error("ELEVENLABS_API_KEY is required.")
            raise ValueError("ELEVENLABS_API_KEY is required.")
        self.client = ElevenLabs(api_key=self.api_key)
        logger.info("MusicClient initialized successfully.")

    def create_composition_plan(
        self, prompt: str, music_length_ms: int = 10000
    ) -> Dict:
        """
        Create a composition plan from a prompt.

        Args:
            prompt (str): Text description of the desired music.
            music_length_ms (int): Length of music in milliseconds.

        Returns:
            Dict: Composition plan.

        Raises:
            ValueError: If prompt is empty or music length is non-positive.
        """
        if not prompt:
            raise ValueError("Prompt must not be empty.")
        if music_length_ms <= 0:
            raise ValueError("music_length_ms must be a positive integer.")

        plan = self.client.music.composition_plan.create(
            prompt=prompt, music_length_ms=music_length_ms
        )
        logger.info("Composition plan created successfully.")
        return plan

    def validate_composition_plan(self, composition_plan: Dict) -> None:
        """
        Validate a composition plan to ensure it contains required fields.

        Args:
            composition_plan (Dict): The composition plan to validate.

        Raises:
            ValueError: If the composition plan is invalid.
        """
        if not isinstance(composition_plan, dict):
            raise ValueError("Composition plan must be a dictionary.")

        if "sections" not in composition_plan:
            raise ValueError("Composition plan missing required key: 'sections'.")
        if (
            not isinstance(composition_plan["sections"], list)
            or not composition_plan["sections"]
        ):
            raise ValueError("Composition plan must contain at least one section.")

        # Validate each section structure
        for idx, section in enumerate(composition_plan["sections"]):
            if "sectionName" not in section or "durationMs" not in section:
                raise ValueError(
                    f"Section {idx} must have 'sectionName' and 'durationMs'."
                )
        logger.info("Composition plan validated successfully.")

    def compose_music(
        self, prompt: Optional[str] = None, composition_plan: Optional[Dict] = None
    ) -> bytes:
        """
        Generate music from either a prompt or a validated composition plan.

        Args:
            prompt (str, optional): Text description for music generation.
            composition_plan (Dict, optional): Validated composition plan.

        Returns:
            bytes: Generated music audio bytes.

        Raises:
            ValueError: If both or neither of prompt and composition_plan are provided.
            ValueError: If prompt is rejected by ElevenLabs API.
        """
        if bool(prompt) == bool(composition_plan):
            raise ValueError("Provide either a prompt or a composition plan, not both.")

        try:
            if composition_plan:
                self.validate_composition_plan(composition_plan)
                track = self.client.music.compose(composition_plan=composition_plan)
            else:
                track = self.client.music.compose(prompt=prompt)
            logger.info("Music composed successfully.")
            return track
        except Exception as e:
            # Handle bad prompts with suggestion if available
            if (
                hasattr(e, "body")
                and e.body.get("detail", {}).get("status") == "bad_prompt"
            ):
                suggestion = e.body["detail"]["data"].get("prompt_suggestion", "")
                logger.error(f"Prompt rejected. Suggested alternative: {suggestion}")
                raise ValueError(
                    f"Prompt rejected. Suggested alternative: {suggestion}"
                )
            logger.error(f"Music composition failed: {e}")
            raise e

    def compose_music_detailed(self, prompt: str, music_length_ms: int = 10000) -> Dict:
        """
        Generate music with detailed metadata, including the composition plan.

        Args:
            prompt (str): Text description of the music.
            music_length_ms (int): Length of music in milliseconds.

        Returns:
            Dict: Detailed music response with metadata.

        Raises:
            ValueError: If prompt is empty.
        """
        if not prompt:
            raise ValueError("Prompt must not be empty.")

        track_details = self.client.music.compose_detailed(
            prompt=prompt, music_length_ms=music_length_ms
        )
        logger.info("Detailed music composition generated successfully.")
        return track_details

    def play_music(self, track_bytes: bytes) -> None:
        """
        Play generated music audio bytes.

        Args:
            track_bytes (bytes): Music audio data.

        Raises:
            ValueError: If track_bytes is empty.
        """
        if not track_bytes:
            raise ValueError("No music data to play.")
        play(track_bytes)
        logger.info("Music playback started.")

    def example_usage(self):
        """Example usage of the MusicClient"""
        print(
            """
            # Example - music generation
            # Initialize MusicClient
            music_client = MusicClient()

            # Example 1: Generate music directly from a prompt
            prompt = (
                "Create an intense, fast-paced electronic track for a high-adrenaline video game scene. "
                "Use driving synth arpeggios, punchy drums, distorted bass, glitch effects, and aggressive rhythmic textures. "
                "The tempo should be fast, 130–150 bpm, with rising tension, quick transitions, and dynamic energy bursts."
            )
            print("🎵 Generating music from prompt...")
            try:
                track_bytes = music_client.compose_music(prompt=prompt)
                music_client.play_music(track_bytes)
                print("✅ Music generated and played successfully!")
            except ValueError as e:
                print(f"⚠️ Failed to generate music from prompt: {e}")

            # Example 2: Generate and use a composition plan
            print("📝 Creating a composition plan...")
            composition_plan = music_client.create_composition_plan(prompt=prompt, music_length_ms=10000)
            print("Composition plan created:")
            print(composition_plan)

            print("🎵 Generating music from composition plan...")
            try:
                track_bytes_plan = music_client.compose_music(composition_plan=composition_plan)
                music_client.play_music(track_bytes_plan)
                print("✅ Music from composition plan generated and played successfully!")
            except ValueError as e:
                print(f"⚠️ Failed to generate music from composition plan: {e}")

            # Example 3: Generate detailed music response
            print("🔍 Generating detailed music response...")
            track_details = music_client.compose_music_detailed(prompt=prompt, music_length_ms=10000)
            print("Track details (JSON):")
            print(track_details.json)
            print("Track file name:", track_details.filename)
            music_client.play_music(track_details.audio)
            print("✅ Detailed music generated and played successfully!")
            """
        )
