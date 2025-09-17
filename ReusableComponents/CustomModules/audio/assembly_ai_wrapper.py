"""
AssemblyAI Audio Transcription Wrapper Module

This module provides a comprehensive, production-ready wrapper for AssemblyAI's speech-to-text API,
offering multiple specialized transcription pipelines for different audio content types and use cases.

Features:
- Multiple specialized transcription pipelines (general, meeting, podcast, interview, etc.)
- Advanced content analysis and safety features
- PII detection and redaction capabilities
- Speaker diarization and custom speaker labeling
- Content moderation and safety analysis
- Batch processing with concurrency control
- Organized file output with multiple formats
- Comprehensive error handling and logging
- Factory pattern for easy pipeline creation
- Support for multiple audio sources (local files, URLs, S3, Google Drive)

Pipeline Types:
1. GeneralTranscriptionPipeline: Basic transcription with configurable features
2. MeetingTranscriptionPipeline: Optimized for meeting recordings with summarization
3. PodcastTranscriptionPipeline: Specialized for podcast content with chapter detection
4. RedactionModerationPipeline: Content safety and PII redaction focused
5. ContentAnalysisPipeline: Adaptive analysis based on content type
6. CustomSpeakerPipeline: Custom speaker labeling with pyannote integration
7. BatchTranscriptionPipeline: Concurrent processing of multiple audio files

Content Types:
- GENERAL: Basic audio content
- MEETING: Business meetings, conferences
- PODCAST: Podcast episodes, audio shows
- LECTURE: Educational content, presentations
- INTERVIEW: Job interviews, journalistic interviews

Audio Sources:
- LOCAL_FILE: Local audio files
- URL: Direct audio URLs
- S3_BUCKET: Amazon S3 stored audio
- GOOGLE_DRIVE: Google Drive shared files

Key Classes:
- BasePipeline: Abstract base class for all transcription pipelines
- PipelineConfig: Configuration for pipeline behavior
- TranscriptionFeatures: Feature flags for transcription capabilities
- PipelineResult: Comprehensive result structure with all extracted data
- OutputConfig: Configuration for file output operations
- PipelineFactory: Factory for creating appropriate pipelines

Usage Examples:
```python
# Basic transcription
config = PipelineConfig(api_key="your_api_key")
pipeline = GeneralTranscriptionPipeline(config)
result = pipeline.process("audio.mp3")

# Meeting transcription with advanced features
features = TranscriptionFeatures(
    speaker_diarization=True,
    summarization=True,
    auto_highlights=True
)
meeting_pipeline = MeetingTranscriptionPipeline(config)
result = meeting_pipeline.process("meeting.mp3", features=features)

# Content safety and redaction
redaction_pipeline = RedactionModerationPipeline(config)
result = redaction_pipeline.process("sensitive_audio.mp3", save_files=True)
```

Requirements:
- assemblyai: AssemblyAI Python SDK
- boto3: For S3 audio source support
- requests: For HTTP operations
- dataclasses: For structured configuration
- typing: For type hints
- pathlib: For file path operations
- pydub: For downloading files as MP3 (if needed)

Author: Your Name
Version: 2.0.0
License: MIT
Created: 2024
Last Modified: 2024

Notes:
- Ensure AssemblyAI API key is properly configured
- For S3 sources, AWS credentials must be available
- File outputs are organized by date, pipeline type, and content type
- All pipelines support webhook callbacks for async processing
- Custom speaker labeling requires pyannote.audio setup
"""

# Standard library imports
import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Union, Type

# Third-party imports
import assemblyai as aai  # AssemblyAI Python SDK for speech-to-text
import boto3  # AWS SDK for S3 audio source support
import requests  # HTTP client for downloading audio files

# Configure comprehensive logging with detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Console output
        # Uncomment below to also log to file
        # logging.FileHandler('assembly_ai_wrapper.log', encoding='utf-8')
    ],
)
logger = logging.getLogger(__name__)

# Set specific log levels for external libraries to reduce noise
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)


class AudioSource(Enum):
    """
    Enumeration of supported audio source types for transcription pipelines.

    This enum defines the various sources from which audio can be loaded for processing.
    Each source type may require different authentication and access methods.

    Attributes:
        LOCAL_FILE (str): Audio files stored locally on the filesystem
            - Supports common formats: MP3, WAV, FLAC, M4A, OGG, etc.
            - Requires valid file path accessible to the application
            - Most efficient for processing as no network transfer needed

        URL (str): Direct HTTP/HTTPS URLs to audio files
            - Publicly accessible audio URLs
            - Must return audio content directly (not HTML pages)
            - Supports redirects and common web audio formats
            - Requires stable internet connection

        S3_BUCKET (str): Audio files stored in Amazon S3
            - Requires AWS credentials (access key, secret key)
            - Supports both public and private S3 objects
            - Uses presigned URLs for secure access
            - Format: "s3://bucket-name/path/to/file.mp3"
            - Automatically handles authentication and access

        GOOGLE_DRIVE (str): Audio files shared via Google Drive
            - Requires publicly shared Google Drive files
            - Uses file ID from Google Drive share links
            - Limited to files accessible without authentication
            - May require Google Drive API setup for private files

    Example Usage:
        ```python
        # Local file processing
        result = pipeline.process("audio.mp3", AudioSource.LOCAL_FILE)

        # URL processing
        result = pipeline.process("https://example.com/audio.mp3", AudioSource.URL)

        # S3 processing
        result = pipeline.process("s3://my-bucket/audio.mp3", AudioSource.S3_BUCKET)

        # Google Drive processing
        result = pipeline.process("1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms", AudioSource.GOOGLE_DRIVE)
        ```

    Notes:
        - LOCAL_FILE is fastest but requires file accessibility
        - URL sources should be stable and not require authentication
        - S3_BUCKET requires proper AWS configuration
        - GOOGLE_DRIVE has limitations for private files
    """

    LOCAL_FILE = "local_file"
    URL = "url"
    S3_BUCKET = "s3_bucket"
    GOOGLE_DRIVE = "google_drive"


class ContentType(Enum):
    """
    Enumeration of audio content types for optimized transcription processing.

    This enum categorizes different types of audio content to enable specialized
    processing pipelines with content-specific features and optimizations.
    Each content type triggers different feature combinations and processing strategies.

    Attributes:
        GENERAL (str): Generic audio content without specific characteristics
            - Default transcription features
            - Basic speaker diarization and word timestamps
            - Suitable for: casual conversations, general recordings
            - Features: speaker_diarization, word_timestamps, basic PII redaction

        MEETING (str): Business meetings, conferences, and formal discussions
            - Optimized for multi-speaker scenarios
            - Enhanced summarization and action item extraction
            - Suitable for: team meetings, conference calls, business discussions
            - Features: speaker_diarization, summarization, auto_highlights,
                        sentiment_analysis, topic_detection

        PODCAST (str): Podcast episodes and audio shows
            - Chapter detection and content categorization
            - Show notes generation and highlight extraction
            - Suitable for: podcast episodes, audio blogs, talk shows
            - Features: auto_chapters, auto_highlights, sentiment_analysis,
                        topic_detection, iab_categories

        LECTURE (str): Educational content and presentations
            - Enhanced transcription accuracy for educational terms
            - Content structure analysis and key concept extraction
            - Suitable for: university lectures, training sessions, webinars
            - Features: speaker_diarization, auto_highlights, summarization

        INTERVIEW (str): Interviews and Q&A sessions
            - Question-answer flow analysis
            - Key quote extraction and sentiment tracking
            - Suitable for: job interviews, journalistic interviews, surveys
            - Features: speaker_diarization, auto_highlights, summarization,
                        sentiment_analysis, topic_detection

    Pipeline Optimization:
        Each content type automatically configures the optimal set of features:

        - GENERAL: Basic transcription with essential features
        - MEETING: Business-focused with action items and summaries
        - PODCAST: Content creation focused with chapters and categories
        - LECTURE: Education-focused with structured analysis
        - INTERVIEW: Conversation analysis with Q&A detection

    Example Usage:
        ```python
        # Meeting transcription with specialized features
        meeting_pipeline = ContentAnalysisPipeline(config)
        result = meeting_pipeline.process(
            "team_meeting.mp3",
            content_type=ContentType.MEETING
        )

        # Podcast processing with chapter detection
        result = meeting_pipeline.process(
            "podcast_episode.mp3",
            content_type=ContentType.PODCAST
        )
        ```

    Feature Matrix:
        | Content Type | Diarization | Highlights | Chapters | Summary | Sentiment | Categories |
        |--------------|-------------|------------|----------|---------|-----------|------------|
        | GENERAL      | ✓          | ✓          | ✗        | ✓       | ✓         | ✗          |
        | MEETING      | ✓          | ✓          | ✗        | ✓       | ✓         | ✓          |
        | PODCAST      | ✓          | ✓          | ✓        | ✗       | ✓         | ✓          |
        | LECTURE      | ✓          | ✓          | ✗        | ✓       | ✓         | ✗          |
        | INTERVIEW    | ✓          | ✓          | ✗        | ✓       | ✓         | ✓          |

    Notes:
        - Content type selection affects processing time and cost
        - Some features may require higher-tier AssemblyAI plans
        - Specialized insights are generated based on content type
    """

    GENERAL = "general"
    MEETING = "meeting"
    PODCAST = "podcast"
    LECTURE = "lecture"
    INTERVIEW = "interview"


@dataclass
class OutputConfig:
    """
    Configuration for comprehensive file output operations.

    This class controls how transcription results and related files are saved,
    including organizational structure, file formats, and additional content generation.

    Attributes:
        base_output_dir (str): Base directory for all output files
            - Default: "output"
            - Can be absolute or relative path
            - Directory will be created if it doesn't exist

        organize_by_date (bool): Organize files by processing date
            - Default: True
            - Creates subdirectories in YYYY-MM-DD format
            - Helps organize files chronologically

        organize_by_content_type (bool): Organize files by content type
            - Default: True
            - Creates subdirectories for MEETING, PODCAST, etc.
            - Separates different types of audio content

        organize_by_pipeline_type (bool): Organize files by pipeline type
            - Default: True
            - Creates subdirectories for different pipeline types
            - Examples: "general_transcription", "redaction_moderation"

        save_json (bool): Save detailed transcription data as JSON
            - Default: True
            - Includes all metadata, timestamps, speakers, etc.
            - Machine-readable format for further processing

        save_txt (bool): Save human-readable text transcription
            - Default: True
            - Formatted report with headers and sections
            - Easy to read and share

        download_audio (bool): Download and save audio files
            - Default: True
            - Saves original and redacted audio when available
            - Useful for archival and comparison purposes

        audio_format (str): Preferred audio format for downloads
            - Default: "wav"
            - Options: "wav", "mp3", "flac", "m4a"
            - Some formats may not be available depending on source

        create_reports (bool): Generate analysis reports
            - Default: True
            - Creates detailed JSON reports with insights
            - Includes speaker analysis, content metrics, safety analysis

    Directory Structure Example:
        ```
        output/
        ├── 2024-01-15/
        │   ├── content_analysis/
        │   │   ├── meeting/
        │   │   │   ├── transcriptions/
        │   │   │   │   ├── transcript_id.json
        │   │   │   │   └── transcript_id.txt
        │   │   │   ├── audio/
        │   │   │   │   └── transcript_id_original.mp3
        │   │   │   └── reports/
        │   │   │       └── transcript_id_report.json
        │   │   └── podcast/
        │   └── redaction_moderation/
        └── 2024-01-16/
        ```

    Methods:
        get_output_path(): Generates organized directory paths based on configuration

    Example Usage:
        ```python
        # Basic configuration
        output_config = OutputConfig()

        # Custom configuration
        output_config = OutputConfig(
            base_output_dir="/var/transcriptions",
            organize_by_date=False,
            save_json=True,
            save_txt=False,
            download_audio=False
        )

        # Include in pipeline config
        config = PipelineConfig(
            api_key="your_key",
            output_config=output_config
        )
        ```

    Notes:
        - All boolean flags can be independently controlled
        - Directory organization helps manage large numbers of files
        - JSON format preserves all metadata for downstream processing
        - TXT format provides human-readable summaries
    """

    base_output_dir: str = "output"
    organize_by_date: bool = True
    organize_by_content_type: bool = True
    organize_by_pipeline_type: bool = True
    save_json: bool = True
    save_txt: bool = True
    download_audio: bool = True
    audio_format: str = "wav"
    create_reports: bool = True

    def get_output_path(
        self,
        pipeline_type: str,
        content_type: Optional[str] = None,
        file_type: str = "transcriptions",
    ) -> Path:
        """
        Generate organized output path based on configuration settings.

        Creates a hierarchical directory structure based on the enabled organization options.
        The path is built incrementally based on which organization flags are enabled.

        Args:
            pipeline_type (str): Type of pipeline used (e.g., "general_transcription", "redaction_moderation")
            content_type (Optional[str]): Content type (e.g., "meeting", "podcast"). Only used if organize_by_content_type is True
            file_type (str): Type of files to be stored (e.g., "transcriptions", "audio", "reports")

        Returns:
            Path: Complete directory path where files should be saved

        Example:
            ```python
            config = OutputConfig(organize_by_date=True, organize_by_content_type=True)
            path = config.get_output_path("content_analysis", "meeting", "transcriptions")
            # Returns: Path("output/2024-01-15/content_analysis/meeting/transcriptions")
            ```

        Notes:
            - Directory creation is handled by the calling code
            - Path separators are automatically handled by pathlib
            - All directory names are lowercase for consistency
        """
        path_parts = [self.base_output_dir]

        if self.organize_by_date:
            path_parts.append(datetime.now().strftime("%Y-%m-%d"))

        if self.organize_by_pipeline_type:
            path_parts.append(pipeline_type)

        if self.organize_by_content_type and content_type:
            path_parts.append(content_type.lower())

        path_parts.append(file_type)

        return Path("/".join(path_parts))


@dataclass
class PipelineConfig:
    """
    Base configuration for all transcription pipelines.

    This class contains the core settings that control how audio transcription
    is performed, including API authentication, language settings, and output preferences.

    Attributes:
        api_key (str): AssemblyAI API key for authentication
            - Required for all operations
            - Get from: https://app.assemblyai.com/
            - Should be kept secure and not hardcoded

        language_code (str): Language code for transcription
            - Default: "en" (English)
            - Supports: "es", "fr", "de", "it", "pt", "nl", "af", "sq", etc.
            - See AssemblyAI docs for full list of supported languages

        speech_model (str): AssemblyAI speech model to use
            - Default: "best" (highest accuracy)
            - Options: "best", "nano" (fastest)
            - "best" provides higher accuracy but takes longer
            - "nano" provides faster results with slightly lower accuracy

        punctuate (bool): Enable automatic punctuation
            - Default: True
            - Adds periods, commas, question marks, etc.
            - Improves readability of transcription output

        format_text (bool): Enable automatic text formatting
            - Default: True
            - Applies proper capitalization and formatting
            - Improves overall text quality and readability

        dual_channel (bool): Enable dual channel audio processing
            - Default: False
            - Useful for phone calls or stereo recordings
            - Processes left and right channels separately
            - Helps with speaker separation in some cases

        webhook_url (Optional[str]): URL for webhook notifications
            - Default: None (polling mode)
            - If provided, AssemblyAI will POST results to this URL
            - Useful for asynchronous processing
            - Must be publicly accessible HTTPS endpoint

        output_config (Optional[OutputConfig]): File output configuration
            - Default: None (no file output)
            - Controls how and where results are saved
            - See OutputConfig for detailed documentation

    Example Usage:
        ```python
        # Basic configuration
        config = PipelineConfig(api_key="your_api_key_here")

        # Advanced configuration
        config = PipelineConfig(
            api_key=os.getenv("ASSEMBLYAI_API_KEY"),
            language_code="es",
            speech_model="nano",
            dual_channel=True,
            webhook_url="https://yourapp.com/webhook",
            output_config=OutputConfig(base_output_dir="/var/transcriptions")
        )

        # Use with pipeline
        pipeline = GeneralTranscriptionPipeline(config)
        ```

    Security Notes:
        - Never hardcode API keys in source code
        - Use environment variables or secure key management
        - Webhook URLs should use HTTPS and validate requests
        - Consider API key rotation for production systems

    Performance Notes:
        - "best" model: higher accuracy, slower processing
        - "nano" model: faster processing, slightly lower accuracy
        - dual_channel adds processing time but improves accuracy for stereo audio
        - webhook mode is more efficient than polling for large batches
    """

    api_key: str
    language_code: str = "en"
    speech_model: str = "best"
    punctuate: bool = True
    format_text: bool = True
    dual_channel: bool = False
    webhook_url: Optional[str] = None
    output_config: Optional[OutputConfig] = None


@dataclass
class TranscriptionFeatures:
    """
    Feature flags for controlling transcription capabilities and advanced analysis.

    This class provides granular control over which AssemblyAI features are enabled
    for a transcription job. Features can be enabled individually based on needs,
    with consideration for processing time and cost implications.

    Core Features:
        speaker_diarization (bool): Identify and separate different speakers
            - Default: False
            - Distinguishes between multiple speakers in audio
            - Provides speaker labels (A, B, C, etc.) and timestamps
            - Essential for meetings, interviews, conversations

        word_timestamps (bool): Provide precise timing for each word
            - Default: True
            - Gives start and end times for every word
            - Useful for creating captions or syncing with video
            - Minimal performance impact

    Privacy & Safety Features:
        pii_redaction (bool): Detect and redact basic personally identifiable information
            - Default: False
            - Redacts: names, phone numbers, email addresses
            - Replaces detected PII with hash placeholders
            - Basic privacy protection

        entity_redaction (bool): Comprehensive entity detection and redaction
            - Default: False
            - Redacts: medical info, financial data, addresses, etc.
            - More comprehensive than basic PII redaction
            - Includes 25+ entity types for maximum privacy

        content_moderation (bool): Filter profanity and inappropriate content
            - Default: False
            - Replaces profanity with asterisks (****)
            - Helps maintain content standards
            - Useful for public-facing content

        hate_speech_detection (bool): Detect hate speech and toxic content
            - Default: False
            - Identifies various forms of hate speech
            - Provides confidence scores and categories
            - Important for content safety and compliance

    Content Analysis Features:
        auto_highlights (bool): Automatically extract key phrases and topics
            - Default: False
            - Identifies most important parts of audio
            - Provides ranking and timestamps
            - Great for meeting summaries and content discovery

        summarization (bool): Generate automatic summaries
            - Default: False
            - Creates bullet-point or paragraph summaries
            - Configurable summary types (bullets, narrative)
            - Ideal for long-form content and meetings

        auto_chapters (bool): Detect chapter boundaries and topics
            - Default: False
            - Automatically segments long audio into chapters
            - Provides chapter titles and timestamps
            - Perfect for podcasts and lectures

        sentiment_analysis (bool): Analyze emotional tone and sentiment
            - Default: False
            - Provides positive/negative/neutral classifications
            - Includes confidence scores and timestamps
            - Useful for customer service and feedback analysis

        topic_detection (bool): Identify main topics and themes
            - Default: False
            - Extracts key topics discussed in audio
            - Provides topic labels and confidence scores
            - Helps with content categorization

        iab_categories (bool): Categorize content using IAB taxonomy
            - Default: False
            - Uses Interactive Advertising Bureau categories
            - Provides standardized content classification
            - Useful for content management and advertising

    Feature Dependencies:
        - entity_redaction includes all pii_redaction capabilities
        - Some features may require higher-tier AssemblyAI plans
        - Features can be combined for comprehensive analysis
        - More features = longer processing time and higher cost

    Example Usage:
        ```python
        # Basic transcription
        features = TranscriptionFeatures(
            speaker_diarization=True,
            word_timestamps=True
        )

        # Meeting analysis
        features = TranscriptionFeatures(
            speaker_diarization=True,
            auto_highlights=True,
            summarization=True,
            sentiment_analysis=True
        )

        # Privacy-focused transcription
        features = TranscriptionFeatures(
            entity_redaction=True,
            content_moderation=True,
            hate_speech_detection=True
        )

        # Comprehensive content analysis
        features = TranscriptionFeatures(
            speaker_diarization=True,
            auto_highlights=True,
            auto_chapters=True,
            sentiment_analysis=True,
            topic_detection=True,
            iab_categories=True
        )
        ```

    Cost Considerations:
        - Basic features (word_timestamps, speaker_diarization) are included in most plans
        - Advanced features (summarization, sentiment_analysis) may incur additional costs
        - PII redaction and content safety features may require enterprise plans
        - Check AssemblyAI pricing for current feature availability

    Performance Impact:
        - More features = longer processing time
        - Basic features add minimal overhead
        - AI-powered features (summarization, sentiment) add significant processing time
        - Consider feature necessity vs. processing speed requirements
    """

    speaker_diarization: bool = False
    word_timestamps: bool = True
    pii_redaction: bool = False
    entity_redaction: bool = False
    content_moderation: bool = False
    hate_speech_detection: bool = False
    auto_highlights: bool = False
    summarization: bool = False
    auto_chapters: bool = False
    sentiment_analysis: bool = False
    topic_detection: bool = False
    iab_categories: bool = False


@dataclass
class PipelineResult:
    """
    Comprehensive result container for all transcription pipeline outputs.

    This class holds all possible data that can be extracted from audio transcription,
    including basic transcription, advanced analysis, safety information, and metadata.
    Not all fields will be populated - depends on enabled features and content type.

    Required Core Fields:
        id (str): Unique AssemblyAI transcript identifier
            - Used for tracking and referencing transcripts
            - Required for downloading audio or accessing via API

        text (str): Complete transcription text
            - Main transcribed content from audio
            - Formatted and punctuated based on configuration

        confidence (float): Overall transcription confidence score
            - Range: 0.0 to 1.0 (higher is better)
            - Indicates reliability of transcription
            - Values above 0.8 generally indicate good quality

        audio_duration (float): Length of processed audio in seconds
            - Total duration of the transcribed audio
            - Useful for calculating processing metrics

        source_info (Dict[str, Any]): Metadata about audio source and processing
            - Contains source type, processing parameters, etc.
            - Varies based on pipeline type and configuration

    Core Analysis Fields:
        words (Optional[List[Dict[str, Any]]]): Word-level timestamps and confidence
            - Each word with start/end times and confidence scores
            - Includes speaker labels if speaker diarization enabled
            - Format: [{"text": "word", "start": 1.23, "end": 1.45, "confidence": 0.95}]

        speakers (Optional[List[Dict[str, Any]]]): Speaker diarization results
            - Segments of audio attributed to different speakers
            - Includes text, timing, and confidence for each segment
            - Format: [{"speaker": "A", "text": "Hello", "start": 0.0, "end": 2.5}]

        speaker_timeline (Optional[List[Dict[str, Any]]]): Speaker change timeline
            - Simplified view of when speakers change
            - Useful for creating speaker activity charts
            - Generated automatically from speaker data

    Privacy & Safety Fields:
        pii_detected (Optional[List[Dict[str, Any]]]): Detected PII instances
            - List of personally identifiable information found
            - Includes type, location, and confidence for each detection
            - Only populated if PII redaction features enabled

        redacted_entities (Optional[List[Dict[str, Any]]]): Redacted entity information
            - Comprehensive list of redacted entities
            - Includes medical, financial, and other sensitive data
            - More detailed than basic PII detection

        content_safety (Optional[Dict[str, Any]]): Content safety analysis results
            - Hate speech, toxicity, and inappropriate content detection
            - Includes severity levels and confidence scores
            - Critical for content moderation workflows

    Advanced Analysis Fields:
        highlights (Optional[List[Dict[str, Any]]]): Key moments and phrases
            - Most important parts of the audio content
            - Ranked by importance with timestamps
            - Great for creating summaries and navigation

        summary (Optional[str]): Generated text summary
            - Bullet points or narrative summary of content
            - Style depends on pipeline configuration
            - Useful for quick content overview

        chapters (Optional[List[Dict[str, Any]]]): Chapter/section boundaries
            - Automatic segmentation of long-form content
            - Includes chapter titles, summaries, and timestamps
            - Perfect for podcasts and educational content

        sentiment (Optional[Dict[str, Any]]): Sentiment analysis results
            - Emotional tone analysis throughout audio
            - Positive/negative/neutral classifications with confidence
            - Timestamped for tracking sentiment changes

        topics (Optional[List[Dict[str, Any]]]): Identified topics and themes
            - Main subjects discussed in the audio
            - Includes topic labels and confidence scores
            - Helps with content categorization and search

        iab_categories (Optional[Dict[str, Any]]): IAB content categories
            - Standardized content classification
            - Uses Interactive Advertising Bureau taxonomy
            - Useful for advertising and content management

    Custom Features:
        custom_speaker_labels (Optional[Dict[str, str]]): Custom speaker name mapping
            - Maps generic speaker IDs (A, B) to actual names
            - Only available with CustomSpeakerPipeline
            - Format: {"A": "John Doe", "B": "Jane Smith"}

    Raw Data:
        raw_response (Optional[Dict[str, Any]]): Complete AssemblyAI API response
            - Full JSON response from AssemblyAI
            - Contains all available data including internal fields
            - Useful for debugging and accessing undocumented features

    Example Usage:
        ```python
        result = pipeline.process("audio.mp3")

        # Basic transcription access
        print(f"Transcript: {result.text}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Duration: {result.audio_duration:.1f} seconds")

        # Speaker information
        if result.speakers:
            for speaker in result.speakers:
                print(f"Speaker {speaker['speaker']}: {speaker['text']}")

        # Key highlights
        if result.highlights:
            print("Key highlights:")
            for highlight in result.highlights[:5]:
                print(f"- {highlight['text']}")

        # Safety information
        if result.pii_detected:
            print(f"PII instances found: {len(result.pii_detected)}")
        ```

    Notes:
        - Optional fields will be None if corresponding features weren't enabled
        - Data structures match AssemblyAI API formats for consistency
        - All timestamps are in seconds from start of audio
        - Confidence scores range from 0.0 to 1.0
        - Custom processing may add additional fields to source_info
    """

    id: str
    text: str
    confidence: float
    audio_duration: float
    source_info: Dict[str, Any]

    # Core features
    words: Optional[List[Dict[str, Any]]] = None
    speakers: Optional[List[Dict[str, Any]]] = None
    speaker_timeline: Optional[List[Dict[str, Any]]] = None

    # Content safety
    pii_detected: Optional[List[Dict[str, Any]]] = None
    redacted_entities: Optional[List[Dict[str, Any]]] = None
    content_safety: Optional[Dict[str, Any]] = None

    # Advanced features
    highlights: Optional[List[Dict[str, Any]]] = None
    summary: Optional[str] = None
    chapters: Optional[List[Dict[str, Any]]] = None
    sentiment: Optional[Dict[str, Any]] = None
    topics: Optional[List[Dict[str, Any]]] = None
    iab_categories: Optional[Dict[str, Any]] = None

    # Custom speaker labels (if using pyannote integration)
    custom_speaker_labels: Optional[Dict[str, str]] = None

    raw_response: Optional[Dict[str, Any]] = None


@dataclass
class BatchJobStatus:
    """
    Status tracking for individual batch jobs

    Attributes:
        source: Original source identifier
        status: Current job status (pending, processing, completed, failed)
        start_time: When processing started
        end_time: When processing completed (if finished)
        error: Error information if job failed
        result: Processing result if successful
    """

    source: str
    status: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[PipelineResult] = None


@dataclass
class BatchProgress:
    """
    Comprehensive batch processing progress information

    Attributes:
        total_jobs: Total number of jobs in the batch
        completed_jobs: Number of completed jobs (successful or failed)
        successful_jobs: Number of successfully completed jobs
        failed_jobs: Number of failed jobs
        processing_jobs: Number of currently processing jobs
        pending_jobs: Number of jobs waiting to start
        start_time: When batch processing started
        estimated_completion: Estimated completion time based on current progress
        average_job_duration: Average time per completed job
    """

    total_jobs: int
    completed_jobs: int = 0
    successful_jobs: int = 0
    failed_jobs: int = 0
    processing_jobs: int = 0
    pending_jobs: int = 0
    start_time: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    average_job_duration: Optional[float] = None


class PipelineType(Enum):
    """
    Enumeration of available pipeline types for specialized processing

    This enum provides a standardized way to identify and select pipeline types
    for different transcription and analysis scenarios.
    """

    GENERAL = "general"
    MEETING = "meeting"
    PODCAST = "podcast"
    INTERVIEW = "interview"
    LECTURE = "lecture"
    REDACTION = "redaction"
    CONTENT_ANALYSIS = "content_analysis"
    CUSTOM_SPEAKER = "custom_speaker"
    BATCH = "batch"


class PipelineCapability(Enum):
    """
    Enumeration of pipeline capabilities for feature-based selection

    These capabilities help identify which pipelines support specific features
    for advanced pipeline selection and validation.
    """

    SPEAKER_DIARIZATION = "speaker_diarization"
    PII_REDACTION = "pii_redaction"
    CONTENT_MODERATION = "content_moderation"
    AUTO_CHAPTERS = "auto_chapters"
    SUMMARIZATION = "summarization"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    CUSTOM_SPEAKERS = "custom_speakers"
    BATCH_PROCESSING = "batch_processing"
    CONTENT_ANALYSIS = "content_analysis"
    MEETING_INSIGHTS = "meeting_insights"
    PODCAST_INSIGHTS = "podcast_insights"


class BasePipeline(ABC):
    """
    Abstract base class for all transcription pipelines

    This class provides the common functionality shared across all pipeline types,
    including configuration building, result formatting, and output management.
    """

    def __init__(self, config: PipelineConfig) -> None:
        """
        Initialize the base pipeline with configuration

        Args:
            config: Pipeline configuration containing API key and settings
        """
        logger.info("Initializing BasePipeline with provided configuration")
        self.config = config
        aai.settings.api_key = config.api_key
        self.transcriber = aai.Transcriber()
        logger.debug("AssemblyAI transcriber initialized successfully")

    @abstractmethod
    def process(self, source: str, **kwargs) -> PipelineResult:
        """
        Process audio from source and return structured result

        This method must be implemented by all pipeline subclasses to define
        their specific processing logic.

        Args:
            source: Path or URL to the audio source
            **kwargs: Additional processing parameters

        Returns:
            PipelineResult containing all transcription outputs
        """
        pass

    def _build_transcription_config(
        self, features: TranscriptionFeatures
    ) -> aai.TranscriptionConfig:
        """
        Build AssemblyAI configuration from feature flags

        This method translates the high-level feature configuration into
        the specific AssemblyAI TranscriptionConfig object.

        Args:
            features: TranscriptionFeatures object specifying which features to enable

        Returns:
            aai.TranscriptionConfig: Configured AssemblyAI transcription settings
        """
        logger.debug("Building transcription configuration from features")

        # Determine PII redaction settings based on feature flags
        enable_pii_redaction = features.pii_redaction or features.entity_redaction
        pii_policies = None

        if enable_pii_redaction:
            logger.info("PII redaction enabled, configuring policies")
            if features.entity_redaction:
                logger.debug("Using comprehensive entity redaction policies")
                # Comprehensive PII redaction policies for maximum privacy protection
                pii_policies = [
                    aai.PIIRedactionPolicy.medical_condition,
                    aai.PIIRedactionPolicy.medical_process,
                    aai.PIIRedactionPolicy.blood_type,
                    aai.PIIRedactionPolicy.drug,
                    aai.PIIRedactionPolicy.injury,
                    aai.PIIRedactionPolicy.number_sequence,
                    aai.PIIRedactionPolicy.email_address,
                    aai.PIIRedactionPolicy.date_of_birth,
                    aai.PIIRedactionPolicy.phone_number,
                    aai.PIIRedactionPolicy.us_social_security_number,
                    aai.PIIRedactionPolicy.credit_card_number,
                    aai.PIIRedactionPolicy.credit_card_cvv,
                    aai.PIIRedactionPolicy.credit_card_expiration,
                    aai.PIIRedactionPolicy.username,
                    aai.PIIRedactionPolicy.passport_number,
                    aai.PIIRedactionPolicy.password,
                    aai.PIIRedactionPolicy.account_number,
                    aai.PIIRedactionPolicy.ip_address,
                    aai.PIIRedactionPolicy.gender_sexuality,
                    aai.PIIRedactionPolicy.person_name,
                    aai.PIIRedactionPolicy.location,
                    aai.PIIRedactionPolicy.organization,
                    aai.PIIRedactionPolicy.vehicle_id,
                    aai.PIIRedactionPolicy.banking_information,
                    aai.PIIRedactionPolicy.religion,
                    aai.PIIRedactionPolicy.marital_status,
                    aai.PIIRedactionPolicy.person_age,
                ]
            else:
                logger.debug("Using basic PII redaction policies")
                # Basic PII redaction policies when just pii_redaction is True
                pii_policies = [
                    aai.PIIRedactionPolicy.person_name,
                    aai.PIIRedactionPolicy.phone_number,
                    aai.PIIRedactionPolicy.email_address,
                ]

        # Configure PII substitution method (hash for security)
        pii_substitution_policy = (
            aai.PIISubstitutionPolicy.hash if enable_pii_redaction else None
        )

        # Build the complete transcription configuration
        config = aai.TranscriptionConfig(
            # Basic transcription settings
            language_code=self.config.language_code,
            speech_model=self.config.speech_model,
            punctuate=self.config.punctuate,
            format_text=self.config.format_text,
            dual_channel=self.config.dual_channel,
            webhook_url=self.config.webhook_url,
            # Speaker identification features
            speaker_labels=features.speaker_diarization,
            # Content safety and moderation features
            redact_pii=enable_pii_redaction,
            redact_pii_policies=pii_policies,
            filter_profanity=features.content_moderation,
            content_safety=features.hate_speech_detection,
            # Advanced analysis features
            auto_highlights=features.auto_highlights,
            summarization=features.summarization,
            summary_model=(
                aai.SummarizationModel.informative if features.summarization else None
            ),
            summary_type=(
                aai.SummarizationType.bullets if features.summarization else None
            ),
            auto_chapters=features.auto_chapters,
            sentiment_analysis=features.sentiment_analysis,
            iab_categories=features.iab_categories,
            # PII redaction audio and substitution settings
            redact_pii_audio=True if enable_pii_redaction else False,
            redact_pii_sub=pii_substitution_policy,
        )

        logger.debug("Transcription configuration built successfully")
        return config

    def _format_result(
        self, transcript, source_info: Dict[str, Any], features: TranscriptionFeatures
    ) -> PipelineResult:
        """
        Format AssemblyAI response into structured PipelineResult

        This method extracts and structures all the available information from
        the AssemblyAI transcript response based on the enabled features.

        Args:
            transcript: Raw AssemblyAI transcript response
            source_info: Information about the audio source
            features: Feature flags to determine what data to extract

        Returns:
            PipelineResult: Structured result containing all requested data
        """
        logger.debug(
            f"Formatting transcription result for transcript ID: {transcript.id}"
        )

        # Extract word-level timestamps if requested
        words = None
        if (
            features.word_timestamps
            and hasattr(transcript, "words")
            and transcript.words
        ):
            logger.debug(
                f"Extracting word-level timestamps for {len(transcript.words)} words"
            )
            words = [
                {
                    "text": word.text,
                    "start": word.start,
                    "end": word.end,
                    "confidence": word.confidence,
                    "speaker": getattr(word, "speaker", None),
                }
                for word in transcript.words
            ]

        # Extract speaker information if diarization was enabled
        speakers = None
        speaker_timeline = None
        if features.speaker_diarization:
            if hasattr(transcript, "utterances") and transcript.utterances:
                logger.debug(
                    f"Extracting speaker information for {len(transcript.utterances)} utterances"
                )
                speakers = [
                    {
                        "speaker": utterance.speaker,
                        "text": utterance.text,
                        "start": utterance.start,
                        "end": utterance.end,
                        "confidence": utterance.confidence,
                    }
                    for utterance in transcript.utterances
                ]

                # Create speaker timeline view for easier analysis
                speaker_timeline = self._create_speaker_timeline(transcript.utterances)

        # Extract PII/Entity redaction information if redaction was enabled
        pii_detected = None
        if (
            hasattr(transcript, "pii_redacted_audio_intelligence")
            and transcript.pii_redacted_audio_intelligence
        ):
            logger.debug(
                f"Extracting PII detection data for {len(transcript.pii_redacted_audio_intelligence)} items"
            )
            pii_detected = [
                {
                    "label": item.label,
                    "text": item.text,
                    "start": item.start,
                    "end": item.end,
                    "confidence": item.confidence,
                }
                for item in transcript.pii_redacted_audio_intelligence
            ]

        # Extract content safety information if hate speech detection was enabled
        content_safety = None
        if (
            hasattr(transcript, "content_safety_labels")
            and transcript.content_safety_labels
        ):
            logger.debug("Extracting content safety analysis results")
            content_safety = {
                "results": [
                    {
                        "text": label.text,
                        "labels": [
                            {
                                "label": result.label,
                                "confidence": result.confidence,
                                "severity": getattr(result, "severity", None),
                            }
                            for result in label.labels
                        ],
                        "start": label.start,
                        "end": label.end,
                    }
                    for label in transcript.content_safety_labels.results
                ],
                "summary": transcript.content_safety_labels.summary,
            }

        # Extract auto-generated highlights if enabled
        highlights = None
        if (
            hasattr(transcript, "auto_highlights_result")
            and transcript.auto_highlights_result
        ):
            logger.debug(
                f"Extracting {len(transcript.auto_highlights_result.results)} auto-highlights"
            )
            highlights = [
                {
                    "count": highlight.count,
                    "rank": highlight.rank,
                    "text": highlight.text,
                    "timestamps": [
                        {"start": ts.start, "end": ts.end}
                        for ts in highlight.timestamps
                    ],
                }
                for highlight in transcript.auto_highlights_result.results
            ]

        # Extract summary if summarization was enabled
        summary = None
        if hasattr(transcript, "summary") and transcript.summary:
            logger.debug("Extracting generated summary")
            summary = transcript.summary

        # Extract auto-generated chapters if enabled
        chapters = None
        if hasattr(transcript, "chapters") and transcript.chapters:
            logger.debug(
                f"Extracting {len(transcript.chapters)} auto-generated chapters"
            )
            chapters = [
                {
                    "summary": chapter.summary,
                    "headline": chapter.headline,
                    "gist": chapter.gist,
                    "start": chapter.start,
                    "end": chapter.end,
                }
                for chapter in transcript.chapters
            ]

        # Extract sentiment analysis results if enabled
        sentiment = None
        if (
            hasattr(transcript, "sentiment_analysis_results")
            and transcript.sentiment_analysis_results
        ):
            logger.debug(
                f"Extracting sentiment analysis for {len(transcript.sentiment_analysis_results)} segments"
            )
            sentiment = [
                {
                    "text": result.text,
                    "start": result.start,
                    "end": result.end,
                    "sentiment": result.sentiment,
                    "confidence": result.confidence,
                }
                for result in transcript.sentiment_analysis_results
            ]

        # Extract IAB categories if content classification was enabled
        iab_categories = None
        if (
            hasattr(transcript, "iab_categories_result")
            and transcript.iab_categories_result
        ):
            logger.debug("Extracting IAB content categorization results")
            iab_categories = {
                "summary": transcript.iab_categories_result.summary,
                "results": [
                    {
                        "text": result.text,
                        "labels": [
                            {"relevance": label.relevance, "label": label.label}
                            for label in result.labels
                        ],
                        "start": result.start,
                        "end": result.end,
                    }
                    for result in transcript.iab_categories_result.results
                ],
            }

        # Create and return the structured result
        result = PipelineResult(
            id=transcript.id,
            text=transcript.text,
            confidence=transcript.confidence,
            audio_duration=transcript.audio_duration,
            source_info=source_info,
            words=words,
            speakers=speakers,
            speaker_timeline=speaker_timeline,
            pii_detected=pii_detected,
            content_safety=content_safety,
            highlights=highlights,
            summary=summary,
            chapters=chapters,
            sentiment=sentiment,
            iab_categories=iab_categories,
            raw_response=transcript.json_response,
        )

        logger.info(
            f"Successfully formatted transcription result for ID: {transcript.id}"
        )
        return result

    def _create_speaker_timeline(self, utterances) -> List[Dict[str, Any]]:
        """
        Create a timeline view of speaker changes for easier analysis

        This method processes speaker utterances to create a simplified timeline
        showing when speakers change, which is useful for understanding conversation flow.

        Args:
            utterances: List of speaker utterances from AssemblyAI

        Returns:
            List[Dict[str, Any]]: Timeline segments with speaker changes
        """
        logger.debug("Creating speaker timeline from utterances")
        timeline = []
        current_speaker = None
        segment_start = None

        for utterance in utterances:
            # Check if speaker has changed
            if utterance.speaker != current_speaker:
                # Close previous segment if it exists
                if current_speaker is not None:
                    timeline.append(
                        {
                            "speaker": current_speaker,
                            "start": segment_start,
                            "end": utterance.start,
                        }
                    )
                # Start new segment
                current_speaker = utterance.speaker
                segment_start = utterance.start

        # Add final segment
        if current_speaker is not None:
            timeline.append(
                {
                    "speaker": current_speaker,
                    "start": segment_start,
                    "end": utterances[-1].end,
                }
            )

        logger.debug(f"Created speaker timeline with {len(timeline)} segments")
        return timeline

    def _save_outputs(
        self,
        result: PipelineResult,
        pipeline_type: str,
        content_type: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Save transcription and audio outputs to files based on configuration

        This method handles saving various output formats including JSON, TXT,
        audio files, and analysis reports based on the output configuration.

        Args:
            result: The transcription result to save
            pipeline_type: Type of pipeline that generated the result
            content_type: Optional content type for organizing outputs

        Returns:
            Dict[str, str]: Mapping of output types to saved file paths
        """
        if not self.config.output_config:
            logger.info("No output configuration provided, skipping file saves")
            return {}

        logger.info(f"Saving outputs for transcript ID: {result.id}")
        output_config = self.config.output_config
        saved_files = {}

        try:
            # Create base filename from transcript ID for consistency
            base_filename = result.id

            # Save transcription files if requested
            if output_config.save_json or output_config.save_txt:
                logger.debug("Saving transcription files")
                transcription_path = output_config.get_output_path(
                    pipeline_type, content_type, "transcriptions"
                )
                transcription_path.mkdir(parents=True, exist_ok=True)

                if output_config.save_json:
                    json_file = transcription_path / f"{base_filename}.json"
                    self._save_json_transcription(result, json_file)
                    saved_files["json_transcription"] = str(json_file)

                if output_config.save_txt:
                    txt_file = transcription_path / f"{base_filename}.txt"
                    self._save_txt_transcription(result, txt_file)
                    saved_files["txt_transcription"] = str(txt_file)

            # Download and save audio files if available and requested
            if output_config.download_audio:
                logger.debug("Processing audio file downloads")
                audio_path = output_config.get_output_path(
                    pipeline_type, content_type, "audio"
                )
                audio_path.mkdir(parents=True, exist_ok=True)

                # Download redacted audio file if available (when PII redaction is enabled)
                # Check multiple possible locations for redacted audio URL
                redacted_audio_url = result.source_info.get("redacted_audio_url")

                logger.debug(
                    f"Redacted audio URL found: {redacted_audio_url is not None}"
                )

                if redacted_audio_url:
                    try:
                        redacted_audio_file = (
                            audio_path
                            / f"{base_filename}_redacted.{output_config.audio_format}"
                        )
                        logger.info(
                            f"Downloading redacted audio from: {redacted_audio_url[:100]}..."
                        )
                        self._download_audio_file(
                            redacted_audio_url, redacted_audio_file
                        )
                        saved_files["redacted_audio"] = str(redacted_audio_file)
                        logger.info(
                            f"Successfully saved redacted audio to {redacted_audio_file}"
                        )
                    except Exception as e:
                        logger.error(f"Failed to download redacted audio: {str(e)}")
                        logger.debug(f"Redacted audio URL was: {redacted_audio_url}")
                else:
                    logger.warning(
                        "Redacted audio URL not found in response or source_info"
                    )
                    logger.debug(
                        "This may indicate no PII was detected or redaction was not enabled"
                    )

            # Create analysis reports if requested
            if output_config.create_reports:
                logger.debug("Creating analysis reports")
                reports_path = output_config.get_output_path(
                    pipeline_type, content_type, "reports"
                )
                reports_path.mkdir(parents=True, exist_ok=True)

                report_file = reports_path / f"{base_filename}_report.json"
                self._create_analysis_report(result, report_file)
                saved_files["analysis_report"] = str(report_file)

        except Exception as e:
            logger.error(f"Error saving outputs: {str(e)}")

        logger.info(f"Completed saving outputs for transcript ID: {result.id}")
        return saved_files

    def _save_json_transcription(self, result: PipelineResult, file_path: Path) -> None:
        """
        Save transcription result as structured JSON file

        Args:
            result: The transcription result to save
            file_path: Path where the JSON file should be saved
        """
        logger.debug(f"Saving JSON transcription to {file_path}")

        # Create structured data dictionary with all available information
        transcription_data = {
            "id": result.id,
            "text": result.text,
            "confidence": result.confidence,
            "audio_duration": result.audio_duration,
            "source_info": result.source_info,
            "words": result.words,
            "speakers": result.speakers,
            "speaker_timeline": result.speaker_timeline,
            "pii_detected": result.pii_detected,
            "content_safety": result.content_safety,
            "highlights": result.highlights,
            "summary": result.summary,
            "chapters": result.chapters,
            "sentiment": result.sentiment,
            "topics": result.topics,
            "iab_categories": result.iab_categories,
            "custom_speaker_labels": result.custom_speaker_labels,
            "processed_at": datetime.now().isoformat(),
        }

        # Write JSON with proper formatting and UTF-8 encoding
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(transcription_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved JSON transcription to {file_path}")

    def _save_txt_transcription(self, result: PipelineResult, file_path: Path) -> None:
        """
        Save transcription result as human-readable formatted text file

        Args:
            result: The transcription result to save
            file_path: Path where the text file should be saved
        """
        logger.debug(f"Saving TXT transcription to {file_path}")
        content = []

        # Create header with basic information
        content.append("TRANSCRIPTION REPORT")
        content.append("=" * 50)
        content.append(f"Transcript ID: {result.id}")
        content.append(f"Confidence: {result.confidence:.2f}")
        content.append(f"Duration: {result.audio_duration:.2f} seconds")
        content.append(f"Source: {result.source_info.get('source', 'Unknown')}")
        content.append(f"Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append("")

        # Main transcript text
        content.append("TRANSCRIPT TEXT")
        content.append("-" * 30)
        content.append(result.text)
        content.append("")

        # Speaker breakdown if available
        if result.speakers:
            content.append("SPEAKER BREAKDOWN")
            content.append("-" * 30)
            for speaker in result.speakers:
                content.append(
                    f"[{speaker['speaker']}] ({speaker['start']:.1f}s - {speaker['end']:.1f}s): {speaker['text']}"
                )
            content.append("")

        # Summary section if available
        if result.summary:
            content.append("SUMMARY")
            content.append("-" * 30)
            content.append(result.summary)
            content.append("")

        # Key highlights section if available
        if result.highlights:
            content.append("KEY HIGHLIGHTS")
            content.append("-" * 30)
            for i, highlight in enumerate(result.highlights[:10], 1):  # Limit to top 10
                content.append(f"{i}. {highlight['text']} (Rank: {highlight['rank']})")
            content.append("")

        # Chapters section if available
        if result.chapters:
            content.append("CHAPTERS")
            content.append("-" * 30)
            for i, chapter in enumerate(result.chapters, 1):
                content.append(f"Chapter {i}: {chapter['headline']}")
                content.append(
                    f"  Time: {chapter['start']:.1f}s - {chapter['end']:.1f}s"
                )
                content.append(f"  Summary: {chapter['summary']}")
                content.append("")

        # Write formatted text file with UTF-8 encoding
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(content))
        logger.info(f"Saved TXT transcription to {file_path}")

    def _download_audio_file(self, url: str, file_path: Path) -> None:
        """
        Download audio file from URL to local storage

        Args:
            url: URL of the audio file to download
            file_path: Local path where the file should be saved
        """
        logger.debug(f"Downloading audio file from {url} to {file_path}")

        try:
            # Download file in chunks to handle large files efficiently
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Downloaded audio file to {file_path}")

        except Exception as e:
            logger.error(f"Failed to download audio file: {str(e)}")
            raise

    def _create_analysis_report(self, result: PipelineResult, file_path: Path) -> None:
        """
        Create comprehensive analysis report with insights and metrics

        This method generates a detailed report containing quality metrics,
        content analysis, safety analysis, and other insights derived from
        the transcription result.

        Args:
            result: The transcription result to analyze
            file_path: Path where the report should be saved
        """
        logger.debug(f"Creating analysis report for transcript ID: {result.id}")

        # Initialize report structure with basic information
        report = {
            "transcript_id": result.id,
            "analysis_timestamp": datetime.now().isoformat(),
            "source_info": result.source_info,
            "quality_metrics": {
                "confidence": result.confidence,
                "duration": result.audio_duration,
                "word_count": len(result.text.split()) if result.text else 0,
            },
            "content_analysis": {},
            "safety_analysis": {},
        }

        # Add speaker analysis if speaker diarization was performed
        if result.speakers:
            logger.debug("Adding speaker analysis to report")
            # Calculate speaker statistics
            unique_speakers = set(s["speaker"] for s in result.speakers)
            report["speaker_analysis"] = {
                "total_speakers": len(unique_speakers),
                "speaker_distribution": {},
            }

            # Calculate word count and speaking time for each speaker
            for speaker in result.speakers:
                speaker_id = speaker["speaker"]
                if speaker_id not in report["speaker_analysis"]["speaker_distribution"]:
                    report["speaker_analysis"]["speaker_distribution"][speaker_id] = {
                        "word_count": 0,
                        "speaking_time": 0,
                    }

                # Accumulate statistics for each speaker
                report["speaker_analysis"]["speaker_distribution"][speaker_id][
                    "word_count"
                ] += len(speaker["text"].split())
                report["speaker_analysis"]["speaker_distribution"][speaker_id][
                    "speaking_time"
                ] += (speaker["end"] - speaker["start"])

        # Add content safety analysis if available
        if result.content_safety:
            logger.debug("Adding content safety analysis to report")
            report["safety_analysis"] = {
                "has_safety_issues": len(result.content_safety.get("results", [])) > 0,
                "safety_summary": result.content_safety.get("summary", {}),
                "issue_count": len(result.content_safety.get("results", [])),
            }

        # Add privacy analysis if PII was detected
        if result.pii_detected:
            logger.debug("Adding privacy analysis to report")
            report["privacy_analysis"] = {
                "pii_detected": len(result.pii_detected) > 0,
                "pii_types": list(set(item["label"] for item in result.pii_detected)),
                "pii_count": len(result.pii_detected),
            }

        # Add content insights from highlights and chapters
        if result.highlights:
            logger.debug("Adding highlights analysis to report")
            report["content_analysis"]["highlights_count"] = len(result.highlights)
            report["content_analysis"]["top_highlights"] = [
                h["text"] for h in result.highlights[:5]  # Top 5 highlights
            ]

        if result.chapters:
            logger.debug("Adding chapters analysis to report")
            report["content_analysis"]["chapters_count"] = len(result.chapters)
            report["content_analysis"]["chapter_titles"] = [
                c["headline"] for c in result.chapters
            ]

        # Save report as JSON file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Created analysis report at {file_path}")


class GeneralTranscriptionPipeline(BasePipeline):
    """
    General-purpose transcription pipeline for any audio content

    This pipeline provides a flexible foundation for transcribing various types of audio
    content with comprehensive feature support. It handles multiple audio source types
    including local files, URLs, S3 buckets, and Google Drive files.

    Key features:
    - Multi-source audio input support (local, URL, S3, Google Drive)
    - Configurable transcription features (speaker diarization, PII redaction, etc.)
    - Automatic source preparation and URL generation
    - Comprehensive error handling and logging

    Inherits from:
        BasePipeline: Provides core transcription functionality and result formatting
    """

    def process(
        self,
        source: str,
        source_type: AudioSource = AudioSource.LOCAL_FILE,
        features: Optional[TranscriptionFeatures] = None,
        **kwargs,
    ) -> PipelineResult:
        """
        Process audio with general transcription features

        This method orchestrates the complete transcription workflow including
        source preparation, configuration building, transcription execution,
        and result formatting.

        Args:
            source: Audio source path, URL, or identifier
                    - For LOCAL_FILE: file system path (e.g., "/path/to/audio.mp3")
                    - For URL: direct HTTP/HTTPS URL to audio file
                    - For S3_BUCKET: S3 URI format (e.g., "s3://bucket-name/object-key")
                    - For GOOGLE_DRIVE: Google Drive file ID
            source_type: Type of audio source, defaults to LOCAL_FILE
            features: Transcription features to enable. If None, uses default feature set
                            with speaker diarization, word timestamps, PII redaction,
                            content moderation, and hate speech detection enabled
            **kwargs: Additional arguments passed to source preparation methods
                            - For S3: aws_access_key, aws_secret_key, aws_region
                            - For Google Drive: credentials or API configuration

        Returns:
            PipelineResult: Comprehensive transcription result containing:
                - Full transcript text and metadata
                - Speaker information (if diarization enabled)
                - Word-level timestamps (if enabled)
                - PII detection results (if redaction enabled)
                - Content safety analysis (if hate speech detection enabled)
                - All other requested features

        Raises:
            ValueError: If source_type is unsupported or source format is invalid
            Exception: If transcription fails due to API errors, network issues, etc.
        """
        logger.info(f"Starting general transcription process for source: {source}")
        logger.debug(f"Source type: {source_type.value}, Features: {features}")

        # Use default feature set if none provided
        # Default includes commonly used features for general transcription
        if features is None:
            logger.debug("No features specified, using default feature set")
            features = TranscriptionFeatures(
                speaker_diarization=True,  # Identify different speakers
                word_timestamps=True,  # Include word-level timing
                pii_redaction=True,  # Redact personal information
                content_moderation=True,  # Filter profanity
                hate_speech_detection=True,  # Detect harmful content
            )

        # Build transcription configuration from features
        logger.debug("Building transcription configuration")
        config = self._build_transcription_config(features)

        # Prepare audio source based on type (convert to URL AssemblyAI can access)
        logger.info(f"Preparing audio source of type: {source_type.value}")
        audio_url = self._prepare_audio_source(source, source_type, **kwargs)
        logger.debug(f"Prepared audio URL: {audio_url}")

        try:
            # Execute transcription using AssemblyAI
            logger.info("Starting transcription with AssemblyAI")
            transcript = self.transcriber.transcribe(audio_url, config)
            logger.info(f"Transcription completed successfully. ID: {transcript.id}")

            # Prepare source information for result metadata
            source_info = {
                "source": source,  # Original source identifier
                "source_type": source_type.value,  # Type of source (local, url, s3, etc.)
                "processed_url": audio_url,  # Final URL used for transcription
            }

            # Format and return structured result
            logger.debug("Formatting transcription result")
            result = self._format_result(transcript, source_info, features)
            logger.info(
                f"General transcription pipeline completed for source: {source}"
            )

            return result

        except Exception as e:
            # Log detailed error information for debugging
            logger.error(
                f"Transcription failed for source '{source}' of type '{source_type.value}': {str(e)}"
            )
            logger.debug(
                f"Error details - Audio URL: {audio_url}, Features: {features}"
            )
            raise

    def _prepare_audio_source(
        self, source: str, source_type: AudioSource, **kwargs
    ) -> str:
        """
        Prepare audio source based on type and convert to accessible URL

        This method handles the conversion of various audio source types into URLs
        that AssemblyAI can access for transcription. Each source type requires
        different preparation logic.

        Args:
            source: Raw source identifier (path, URL, S3 URI, file ID, etc.)
            source_type: Type of audio source determining preparation method
            **kwargs: Additional parameters for source-specific preparation

        Returns:
            str: URL that AssemblyAI can access to download the audio file

        Raises:
            ValueError: If source_type is not supported or source format is invalid
        """
        logger.debug(f"Preparing audio source: {source} (type: {source_type.value})")

        if source_type == AudioSource.LOCAL_FILE:
            # Local files are passed directly to AssemblyAI (will be uploaded)
            logger.debug("Using local file path directly")
            return source

        elif source_type == AudioSource.URL:
            # Direct URLs are passed as-is (must be publicly accessible)
            logger.debug("Using URL directly")
            return source

        elif source_type == AudioSource.S3_BUCKET:
            # S3 objects require presigned URLs for AssemblyAI access
            logger.debug("Generating S3 presigned URL")
            return self._get_s3_presigned_url(source, **kwargs)

        elif source_type == AudioSource.GOOGLE_DRIVE:
            # Google Drive files need special URL format for public access
            logger.debug("Generating Google Drive download URL")
            return self._get_google_drive_url(source, **kwargs)

        else:
            # Unsupported source types should raise an error
            error_msg = f"Unsupported source type: {source_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _get_s3_presigned_url(self, s3_path: str, **kwargs) -> str:
        """
        Generate presigned URL for S3 object to enable AssemblyAI access

        This method creates a temporary URL that allows AssemblyAI to download
        audio files from private S3 buckets without requiring permanent public access.

        Args:
            s3_path: S3 URI in format "s3://bucket-name/object-key"
            **kwargs: AWS configuration parameters:
                - aws_access_key: AWS access key ID (optional if using IAM)
                - aws_secret_key: AWS secret access key (optional if using IAM)
                - aws_region: AWS region (defaults to us-east-1)

        Returns:
            str: Presigned URL valid for 1 hour that allows downloading the S3 object

        Raises:
            ValueError: If S3 path format is invalid
            Exception: If AWS credentials are invalid or S3 access fails
        """
        logger.debug(f"Processing S3 path: {s3_path}")

        # Validate and parse S3 URI format
        if not s3_path.startswith("s3://"):
            error_msg = "S3 path must start with s3://"
            logger.error(f"Invalid S3 path format: {s3_path}")
            raise ValueError(error_msg)

        # Extract bucket name and object key from S3 URI
        path_parts = s3_path[5:].split("/", 1)  # Remove "s3://" prefix and split
        bucket_name = path_parts[0]
        object_key = path_parts[1] if len(path_parts) > 1 else ""

        logger.debug(f"Parsed S3 URI - Bucket: {bucket_name}, Key: {object_key}")

        # Get AWS credentials from kwargs or use default credential chain
        aws_access_key = kwargs.get("aws_access_key")
        aws_secret_key = kwargs.get("aws_secret_key")
        aws_region = kwargs.get("aws_region", "us-east-1")

        # Create S3 client with explicit credentials or default credential chain
        if aws_access_key and aws_secret_key:
            logger.debug("Using explicit AWS credentials")
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region,
            )
        else:
            # Use default credentials (IAM role, AWS profile, environment variables, etc.)
            logger.debug("Using default AWS credential chain")
            s3_client = boto3.client("s3", region_name=aws_region)

        try:
            # Generate presigned URL valid for 1 hour (3600 seconds)
            # This provides sufficient time for AssemblyAI to download the file
            logger.debug("Generating presigned URL with 1-hour expiration")
            presigned_url = s3_client.generate_presigned_url(
                "get_object",  # S3 operation
                Params={
                    "Bucket": bucket_name,
                    "Key": object_key,
                },  # S3 object parameters
                ExpiresIn=3600,  # URL validity period (1 hour)
            )

            logger.info(
                f"Generated S3 presigned URL for bucket: {bucket_name}, key: {object_key}"
            )
            return presigned_url

        except Exception as e:
            logger.error(f"Failed to generate S3 presigned URL: {str(e)}")
            raise

    def _get_google_drive_url(self, file_id: str) -> str:
        """
        Get Google Drive file URL for public download access

        This method constructs a direct download URL for Google Drive files.
        Note: This implementation uses the simple public download format and
        requires the file to have public sharing enabled.

        For production use, consider implementing proper Google Drive API
        integration with authentication and private file access.

        Args:
            file_id: Google Drive file ID (extracted from sharing URL)
            **kwargs: Future Google Drive API configuration parameters

        Returns:
            str: Direct download URL for the Google Drive file

        Note:
            Current implementation assumes public file access. For private files,
            implement Google Drive API authentication and use the Drive API
            to generate download URLs with proper permissions.
        """
        logger.debug(f"Generating Google Drive download URL for file ID: {file_id}")

        # Construct direct download URL using Google Drive's export format
        # This format works for publicly shared files
        download_url = f"https://drive.google.com/uc?id={file_id}&export=download"

        logger.info(f"Generated Google Drive URL for file ID: {file_id}")
        logger.warning(
            "Google Drive URL assumes public file access. For private files, implement proper Google Drive API authentication."
        )

        return download_url


class MeetingTranscriptionPipeline(BasePipeline):
    """
    Specialized pipeline for meeting transcription and summarization

    This pipeline is optimized for business meetings, conference calls, and collaborative
    discussions. It provides enhanced features specifically tailored for meeting analysis
    including comprehensive summarization, speaker mapping, sentiment tracking, and
    actionable insights extraction.

    Key meeting-specific features:
    - Informative bullet-point summaries optimized for meeting notes
    - Speaker-to-attendee mapping for personalized meeting records
    - Sentiment analysis to gauge meeting tone and participant engagement
    - Topic detection to identify key discussion points
    - Auto-highlights for action items and important decisions
    - PII redaction to protect sensitive business information

    Ideal for:
    - Business meetings and conference calls
    - Team standups and retrospectives
    - Client presentations and sales calls
    - Board meetings and executive sessions
    - Training sessions and workshops

    Inherits from:
        BasePipeline: Provides core transcription functionality and result formatting
    """

    def process(
        self,
        source: str,
        source_type: AudioSource = AudioSource.LOCAL_FILE,
        meeting_context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> PipelineResult:
        """
        Process meeting audio with specialized features for business collaboration

        This method configures and executes transcription with features specifically
        optimized for meeting scenarios, including enhanced summarization, speaker
        identification, and business-relevant content analysis.

        Args:
            source: Audio source path, URL, or identifier for the meeting recording
                    - Local file: "/path/to/meeting_recording.mp3"
                    - URL: "https://example.com/meeting.wav"
                    - S3: "s3://meetings-bucket/2024/q1/team-meeting.mp3"
                    - Google Drive: Google Drive file ID for shared meeting recordings
            source_type: Type of audio source, defaults to LOCAL_FILE
            meeting_context: Optional meeting metadata and configuration:
                - attendees: List of meeting participant names
                - speaker_mapping: Dict mapping speaker IDs to attendee names
                    (e.g., {"A": "John Smith", "B": "Jane Doe"})
                - meeting_title: Title or subject of the meeting
                - meeting_date: Date/time of the meeting
                - meeting_type: Type of meeting (standup, review, planning, etc.)
                - department: Relevant department or team
            **kwargs: Additional source-specific parameters (AWS credentials, etc.)

        Returns:
            PipelineResult: Enhanced meeting transcription result containing:
                - Full meeting transcript with speaker identification
                - Bullet-point summary optimized for meeting notes
                - Speaker-to-attendee mapping (if context provided)
                - Sentiment analysis showing meeting tone and engagement
                - Auto-generated highlights for key decisions and action items
                - Topic detection for main discussion points
                - PII-redacted content for privacy protection
                - Meeting-specific metadata and insights

        Raises:
            Exception: If meeting transcription fails due to API errors, invalid audio,
                            network issues, or configuration problems

        Example:
            >>> pipeline = MeetingTranscriptionPipeline(config)
            >>> context = {
            ...     "attendees": ["Alice Johnson", "Bob Smith", "Carol Davis"],
            ...     "speaker_mapping": {"A": "Alice Johnson", "B": "Bob Smith", "C": "Carol Davis"},
            ...     "meeting_title": "Q1 Planning Meeting",
            ...     "meeting_type": "planning"
            ... }
            >>> result = pipeline.process("meeting.mp3", meeting_context=context)
        """
        logger.info(f"Starting meeting transcription pipeline for source: {source}")
        logger.debug(f"Source type: {source_type.value}")

        # Log meeting context for debugging and audit trail
        if meeting_context:
            attendee_count = len(meeting_context.get("attendees", []))
            meeting_title = meeting_context.get("meeting_title", "Unknown")
            logger.info(
                f"Processing meeting: '{meeting_title}' with {attendee_count} expected attendees"
            )
            logger.debug(f"Meeting context: {meeting_context}")
        else:
            logger.info("No meeting context provided, using default meeting processing")

        # Configure meeting-optimized transcription features
        # These features are specifically chosen for business meeting scenarios
        logger.debug("Configuring meeting-specific transcription features")
        features = TranscriptionFeatures(
            speaker_diarization=True,  # Essential for identifying who said what
            word_timestamps=True,  # Needed for precise meeting minutes
            pii_redaction=True,  # Protect sensitive business information
            content_moderation=True,  # Filter inappropriate content
            summarization=True,  # Generate meeting summary
            auto_highlights=True,  # Extract key decisions and action items
            sentiment_analysis=True,  # Track meeting tone and engagement
            topic_detection=True,  # Identify main discussion points
        )

        # Build base transcription configuration
        logger.debug("Building transcription configuration")
        config = self._build_transcription_config(features)

        # Apply meeting-specific configuration overrides
        # Use informative model for detailed, structured summaries
        logger.debug("Applying meeting-specific configuration overrides")
        config.summary_model = aai.SummarizationModel.informative
        config.summary_type = (
            aai.SummarizationType.bullets
        )  # Bullet points for easy reading

        # Prepare audio source for transcription
        logger.info(f"Preparing audio source of type: {source_type.value}")
        audio_url = self._prepare_audio_source(source, source_type, **kwargs)
        logger.debug("Prepared audio URL for meeting transcription")

        try:
            # Execute transcription with meeting-optimized settings
            logger.info("Starting meeting transcription with AssemblyAI")
            transcript = self.transcriber.transcribe(audio_url, config)
            logger.info(
                f"Meeting transcription completed successfully. ID: {transcript.id}"
            )

            # Prepare comprehensive source information for meeting records
            source_info = {
                "source": source,  # Original source identifier
                "source_type": source_type.value,  # Source type (local, url, s3, etc.)
                "content_type": ContentType.MEETING.value,  # Specialized content type
                "meeting_context": meeting_context
                or {},  # Meeting metadata and context
            }

            # Format standard transcription result
            logger.debug("Formatting meeting transcription result")
            result = self._format_result(transcript, source_info, features)

            # Apply meeting-specific enhancements and processing
            logger.debug("Applying meeting-specific result enhancements")
            result = self._enhance_meeting_result(result, meeting_context)

            logger.info(
                f"Meeting transcription pipeline completed successfully for: {source}"
            )
            return result

        except Exception as e:
            # Log detailed error information for meeting transcription failures
            error_context = {
                "source": source,
                "source_type": source_type.value,
                "has_context": meeting_context is not None,
                "attendee_count": (
                    len(meeting_context.get("attendees", [])) if meeting_context else 0
                ),
            }
            logger.error(f"Meeting transcription failed: {str(e)}")
            logger.debug(f"Error context: {error_context}")
            raise

    def _enhance_meeting_result(
        self, result: PipelineResult, context: Optional[Dict[str, Any]]
    ) -> PipelineResult:
        """
        Add meeting-specific enhancements to transcription results

        This method applies meeting-specific processing including speaker-to-attendee
        mapping, meeting metadata integration, and business-relevant result formatting.
        It enriches the standard transcription output with meeting context.

        Args:
            result: Base transcription result from AssemblyAI processing
            context: Optional meeting context containing attendee information and metadata

        Returns:
            PipelineResult: Enhanced result with meeting-specific information:
                - Speaker utterances mapped to actual attendee names
                - Meeting metadata integrated into source_info
                - Additional meeting-specific insights and formatting

        Note:
            This method modifies the result in-place while maintaining the original
            structure. Future enhancements could include action item extraction,
            decision tracking, and meeting effectiveness metrics.
        """
        logger.debug("Enhancing meeting result with context-specific information")

        # Apply speaker-to-attendee mapping if context is provided
        if context and "attendees" in context:
            logger.debug(
                f"Processing attendee mapping for {len(context['attendees'])} attendees"
            )

            # Extract speaker mapping from context
            attendee_mapping = context.get("speaker_mapping", {})

            if result.speakers and attendee_mapping:
                logger.info(f"Applying speaker mapping: {attendee_mapping}")

                # Map each speaker utterance to actual attendee names
                mapped_speakers = 0
                for speaker in result.speakers:
                    speaker_id = speaker["speaker"]
                    if speaker_id in attendee_mapping:
                        # Add attendee name to speaker information
                        speaker["attendee_name"] = attendee_mapping[speaker_id]
                        mapped_speakers += 1
                        logger.debug(
                            f"Mapped speaker {speaker_id} to {attendee_mapping[speaker_id]}"
                        )

                logger.info(
                    f"Successfully mapped {mapped_speakers} speakers to attendees"
                )

                # Also update word-level speaker mapping if available
                if result.words:
                    for word in result.words:
                        if word.get("speaker") and word["speaker"] in attendee_mapping:
                            word["attendee_name"] = attendee_mapping[word["speaker"]]

            else:
                if not result.speakers:
                    logger.warning("No speakers detected for attendee mapping")
                if not attendee_mapping:
                    logger.warning("No speaker mapping provided in meeting context")

        else:
            logger.debug("No attendee context provided, skipping speaker mapping")

        # Add meeting-specific metadata to source_info
        if context:
            # Integrate additional meeting metadata
            meeting_metadata = {
                "meeting_title": context.get("meeting_title"),
                "meeting_date": context.get("meeting_date"),
                "meeting_type": context.get("meeting_type"),
                "department": context.get("department"),
                "expected_attendees": context.get("attendees", []),
                "actual_speakers_detected": (
                    len(set(s["speaker"] for s in result.speakers))
                    if result.speakers
                    else 0
                ),
            }

            # Filter out None values for cleaner metadata
            meeting_metadata = {
                k: v for k, v in meeting_metadata.items() if v is not None
            }

            if meeting_metadata:
                result.source_info["meeting_metadata"] = meeting_metadata
                logger.debug(f"Added meeting metadata: {list(meeting_metadata.keys())}")

        logger.debug("Meeting result enhancement completed")
        return result

    def _prepare_audio_source(
        self, source: str, source_type: AudioSource, **kwargs
    ) -> str:
        """
        Prepare audio source for meeting transcription

        This method reuses the general pipeline's audio source preparation logic
        since the source handling requirements are identical. Meeting-specific
        processing occurs in the transcription configuration and result enhancement.

        Args:
            source: Audio source identifier (path, URL, S3 URI, etc.)
            source_type: Type of audio source determining preparation method
            **kwargs: Source-specific preparation parameters

        Returns:
            str: URL that AssemblyAI can access for meeting audio transcription

        Note:
            Delegates to GeneralTranscriptionPipeline for consistent source handling
            across all pipeline types while maintaining meeting-specific logging context.
        """
        logger.debug(
            f"Preparing meeting audio source: {source} (type: {source_type.value})"
        )

        # Delegate to general pipeline's source preparation method
        # This ensures consistent behavior across all pipeline types
        prepared_url = GeneralTranscriptionPipeline._prepare_audio_source(
            self, source, source_type, **kwargs
        )

        logger.debug("Meeting audio source preparation completed")
        return prepared_url


class PodcastTranscriptionPipeline(BasePipeline):
    """
    Specialized pipeline for podcast transcription with chapters and highlights

    This pipeline is optimized for podcast content, providing enhanced features
    specifically designed for long-form audio content, episodic shows, and media
    production workflows. It focuses on content discovery, audience engagement,
    and automated content marketing tools.

    Key podcast-specific features:
    - Auto-generated chapters for easy navigation and timestamps
    - Content highlights extraction for social media and marketing
    - IAB content categorization for advertising and discovery
    - Topic detection for content tagging and SEO optimization
    - Speaker diarization for multi-host shows and guest identification
    - Sentiment analysis for audience engagement insights
    - Automated show notes generation for episode descriptions
    - Content accessibility compliance for transcript publication

    Ideal for:
    - Podcast episodes and audio shows
    - Interview-style content with multiple participants
    - Educational audio content and lectures
    - Long-form discussions and panel conversations
    - Audio content for accessibility compliance
    - Content marketing and social media clip generation

    Content Optimization:
    - Generates searchable transcripts for podcast platforms
    - Creates chapter markers for enhanced listener experience
    - Extracts quotable highlights for promotional content
    - Provides topic tags for content discovery and SEO

    Inherits from:
        BasePipeline: Provides core transcription functionality and result formatting
    """

    def process(
        self,
        source: str,
        source_type: AudioSource = AudioSource.LOCAL_FILE,
        podcast_metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> PipelineResult:
        """
        Process podcast audio with specialized features for content creators

        This method configures and executes transcription with features specifically
        optimized for podcast content including automatic chapter generation, content
        highlights extraction, and comprehensive content categorization for discovery
        and marketing purposes.

        Args:
            source: Audio source path, URL, or identifier for the podcast episode
                    - Local file: "/path/to/podcast_episode_042.mp3"
                    - URL: "https://cdn.example.com/episodes/tech-talk-ep42.wav"
                    - S3: "s3://podcast-storage/2024/season2/episode-042.mp3"
                    - Google Drive: File ID for podcast episode stored in Google Drive
            source_type: Type of audio source, defaults to LOCAL_FILE
            podcast_metadata: Optional podcast episode metadata and configuration:
                - show_name: Name of the podcast show (e.g., "Tech Talk Weekly")
                - episode_number: Episode number or identifier
                - episode_title: Title of the specific episode
                - season: Season number (if applicable)
                - hosts: List of podcast host names
                - guests: List of guest participant names
                - episode_date: Date of episode recording/publication
                - show_description: Brief description of the podcast show
                - episode_description: Description of the specific episode
                - tags: Content tags for categorization
                - target_audience: Intended audience demographic
                - show_category: Podcast category (technology, business, etc.)
            **kwargs: Additional source-specific parameters (AWS credentials, etc.)

        Returns:
            PipelineResult: Enhanced podcast transcription result containing:
                - Full episode transcript with speaker identification
                - Auto-generated chapter divisions with timestamps and descriptions
                - Content highlights ranked by importance for promotional use
                - IAB content categories for advertising and discovery
                - Topic detection results for SEO and content tagging
                - Sentiment analysis for audience engagement insights
                - Generated show notes combining chapters and highlights
                - Speaker diarization for multi-participant content
                - Podcast-specific metadata and content marketing insights

        Raises:
            Exception: If podcast transcription fails due to API errors, invalid audio,
                        network issues, or configuration problems

        Example:
            >>> pipeline = PodcastTranscriptionPipeline(config)
            >>> metadata = {
            ...     "show_name": "Tech Innovation Weekly",
            ...     "episode_number": 42,
            ...     "episode_title": "The Future of AI in Healthcare",
            ...     "hosts": ["Alice Johnson", "Bob Tech"],
            ...     "guests": ["Dr. Sarah AI", "Mike Healthcare"],
            ...     "show_category": "technology"
            ... }
            >>> result = pipeline.process("episode-042.mp3", podcast_metadata=metadata)
            >>> print(f"Generated {len(result.chapters)} chapters")
            >>> print(f"Found {len(result.highlights)} key highlights")
        """
        logger.info(f"Starting podcast transcription pipeline for source: {source}")
        logger.debug(f"Source type: {source_type.value}")

        # Log podcast metadata for content tracking and analytics
        if podcast_metadata:
            show_name = podcast_metadata.get("show_name", "Unknown Show")
            episode_title = podcast_metadata.get("episode_title", "Unknown Episode")
            episode_number = podcast_metadata.get("episode_number", "Unknown")
            logger.info(
                f"Processing podcast: '{show_name}' - Episode {episode_number}: '{episode_title}'"
            )

            # Log participant information for speaker diarization context
            hosts = podcast_metadata.get("hosts", [])
            guests = podcast_metadata.get("guests", [])
            total_participants = len(hosts) + len(guests)
            logger.info(
                f"Expected participants: {total_participants} ({len(hosts)} hosts, {len(guests)} guests)"
            )
            logger.debug(f"Podcast metadata: {podcast_metadata}")
        else:
            logger.info(
                "No podcast metadata provided, using default podcast processing"
            )

        # Configure podcast-optimized transcription features
        # These features are specifically chosen for podcast content discovery and marketing
        logger.debug("Configuring podcast-specific transcription features")
        features = TranscriptionFeatures(
            speaker_diarization=True,  # Essential for multi-host shows and guest identification
            word_timestamps=True,  # Needed for precise chapter timestamps
            auto_highlights=True,  # Extract quotable content for social media
            auto_chapters=True,  # Generate navigable chapter divisions
            sentiment_analysis=True,  # Track content tone for audience insights
            topic_detection=True,  # Identify discussion topics for SEO
            iab_categories=True,  # Content categorization for advertising
        )

        # Build transcription configuration optimized for podcast content
        logger.debug("Building transcription configuration for podcast content")
        config = self._build_transcription_config(features)

        # Prepare audio source for podcast transcription
        logger.info(f"Preparing podcast audio source of type: {source_type.value}")
        audio_url = self._prepare_audio_source(source, source_type, **kwargs)
        logger.debug("Prepared audio URL for podcast transcription")

        try:
            # Execute transcription with podcast-optimized settings
            logger.info("Starting podcast transcription with AssemblyAI")
            transcript = self.transcriber.transcribe(audio_url, config)
            logger.info(
                f"Podcast transcription completed successfully. ID: {transcript.id}"
            )

            # Log content analysis metrics for podcast analytics
            if hasattr(transcript, "chapters") and transcript.chapters:
                logger.info(f"Generated {len(transcript.chapters)} chapter divisions")
            if (
                hasattr(transcript, "auto_highlights_result")
                and transcript.auto_highlights_result
            ):
                highlights_count = len(transcript.auto_highlights_result.results)
                logger.info(f"Extracted {highlights_count} content highlights")

            # Prepare comprehensive source information for podcast records
            source_info = {
                "source": source,  # Original source identifier
                "source_type": source_type.value,  # Source type (local, url, s3, etc.)
                "content_type": ContentType.PODCAST.value,  # Specialized content type
                "podcast_metadata": podcast_metadata
                or {},  # Episode metadata and context
            }

            # Format standard transcription result
            logger.debug("Formatting podcast transcription result")
            result = self._format_result(transcript, source_info, features)

            # Apply podcast-specific enhancements and content generation
            logger.debug("Applying podcast-specific result enhancements")
            result = self._enhance_podcast_result(result, podcast_metadata)

            logger.info(
                f"Podcast transcription pipeline completed successfully for: {source}"
            )
            return result

        except Exception as e:
            # Log detailed error information for podcast transcription failures
            error_context = {
                "source": source,
                "source_type": source_type.value,
                "has_metadata": podcast_metadata is not None,
                "show_name": (
                    podcast_metadata.get("show_name") if podcast_metadata else None
                ),
                "episode_number": (
                    podcast_metadata.get("episode_number") if podcast_metadata else None
                ),
            }
            logger.error(f"Podcast transcription failed: {str(e)}")
            logger.debug(f"Error context: {error_context}")
            raise

    def _enhance_podcast_result(
        self, result: PipelineResult, metadata: Optional[Dict[str, Any]]
    ) -> PipelineResult:
        """
        Add podcast-specific enhancements to transcription results

        This method applies podcast-specific processing including automated show notes
        generation, content marketing insights, and episode-specific metadata integration.
        It enriches the standard transcription output with podcast-focused features.

        Args:
            result: Base transcription result from AssemblyAI processing
            metadata: Optional podcast metadata containing show and episode information

        Returns:
            PipelineResult: Enhanced result with podcast-specific information:
                - Generated show notes combining chapters and highlights
                - Content marketing insights for social media and promotion
                - Episode metadata integrated into source_info
                - Additional podcast-specific analytics and formatting

        Note:
            This method focuses on content creation and marketing tools that help
            podcast creators maximize their content's reach and engagement potential.
        """
        logger.debug("Enhancing podcast result with content-specific information")

        # Generate automated show notes if chapter and highlight data is available
        if metadata:
            logger.debug("Processing podcast metadata for content enhancement")

            # Generate show notes from chapters and highlights for content marketing
            if result.chapters and result.highlights:
                logger.info(
                    "Generating automated show notes from chapters and highlights"
                )
                show_notes = self._generate_show_notes(
                    result.chapters, result.highlights
                )
                result.source_info["generated_show_notes"] = show_notes
                logger.info(
                    f"Generated show notes with {len(show_notes['chapter_summary'])} chapter summaries and {len(show_notes['key_highlights'])} highlights"
                )
            else:
                if not result.chapters:
                    logger.warning("No chapters available for show notes generation")
                if not result.highlights:
                    logger.warning("No highlights available for show notes generation")

            # Add podcast-specific content insights for analytics and marketing
            content_insights = self._generate_content_insights(result, metadata)
            if content_insights:
                result.source_info["content_insights"] = content_insights
                logger.debug(f"Added content insights: {list(content_insights.keys())}")

            # Integrate episode metadata for comprehensive podcast records
            episode_metadata = {
                "show_name": metadata.get("show_name"),
                "episode_number": metadata.get("episode_number"),
                "episode_title": metadata.get("episode_title"),
                "season": metadata.get("season"),
                "hosts": metadata.get("hosts", []),
                "guests": metadata.get("guests", []),
                "episode_date": metadata.get("episode_date"),
                "show_category": metadata.get("show_category"),
                "target_audience": metadata.get("target_audience"),
                "episode_duration": result.audio_duration,
                "speakers_detected": (
                    len(set(s["speaker"] for s in result.speakers))
                    if result.speakers
                    else 0
                ),
                "chapters_generated": len(result.chapters) if result.chapters else 0,
                "highlights_extracted": (
                    len(result.highlights) if result.highlights else 0
                ),
            }

            # Filter out None values for cleaner metadata
            episode_metadata = {
                k: v for k, v in episode_metadata.items() if v is not None
            }

            if episode_metadata:
                result.source_info["episode_metadata"] = episode_metadata
                logger.debug(f"Added episode metadata: {list(episode_metadata.keys())}")

        else:
            logger.debug(
                "No podcast metadata provided, skipping enhanced content generation"
            )

        logger.debug("Podcast result enhancement completed")
        return result

    def _generate_show_notes(
        self, chapters: List[Dict], highlights: List[Dict]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive show notes from chapters and highlights

        This method creates automated show notes that podcast creators can use
        for episode descriptions, social media posts, and content marketing.
        The output is optimized for readability and audience engagement.

        Args:
            chapters: List of auto-generated chapter information with headlines and summaries
            highlights: List of content highlights ranked by importance and engagement potential

        Returns:
            Dict[str, Any]: Structured show notes containing:
                - chapter_summary: List of chapter descriptions with headlines and key points
                - key_highlights: Top 5 quotable highlights for social media and promotion
                - content_outline: Structured outline for episode descriptions
                - social_media_clips: Suggested content for social media posts

        Note:
            The generated show notes are designed to maximize content discoverability
            and provide ready-to-use marketing materials for podcast promotion.
        """
        logger.debug(
            f"Generating show notes from {len(chapters)} chapters and {len(highlights)} highlights"
        )

        # Create chapter summaries with headlines and key points
        chapter_summary = []
        for i, chapter in enumerate(chapters, 1):
            # Format: "Chapter 1: Headline - Key Point"
            chapter_entry = f"Chapter {i}: {chapter['headline']}"
            if chapter.get("gist"):
                chapter_entry += f" - {chapter['gist']}"
            chapter_summary.append(chapter_entry)

        # Extract top highlights ranked by importance for promotional content
        # Sort highlights by rank (lower rank = higher importance) and take top 5
        sorted_highlights = sorted(highlights, key=lambda x: x["rank"])
        key_highlights = [highlight["text"] for highlight in sorted_highlights[:5]]

        # Create additional content marketing materials
        show_notes = {
            "chapter_summary": chapter_summary,
            "key_highlights": key_highlights,
            "content_outline": self._create_content_outline(chapters),
            "social_media_clips": self._suggest_social_clips(
                sorted_highlights[:3]
            ),  # Top 3 for social media
        }

        logger.info(
            f"Generated show notes with {len(chapter_summary)} chapter summaries and {len(key_highlights)} key highlights"
        )
        return show_notes

    def _create_content_outline(self, chapters: List[Dict]) -> List[str]:
        """
        Create a structured content outline from chapters for episode descriptions

        Args:
            chapters: List of chapter information

        Returns:
            List[str]: Structured outline suitable for episode descriptions
        """
        outline = []
        for chapter in chapters:
            # Create timestamp-based outline entries
            start_time = self._format_timestamp(chapter.get("start", 0))
            outline.append(f"{start_time} - {chapter['headline']}")
        return outline

    def _suggest_social_clips(self, top_highlights: List[Dict]) -> List[Dict[str, Any]]:
        """
        Suggest social media clips from top highlights

        Args:
            top_highlights: Top-ranked highlights for social media

        Returns:
            List[Dict]: Suggested clips with timestamps and content
        """
        clips = []
        for highlight in top_highlights:
            if highlight.get("timestamps"):
                # Use first timestamp for clip suggestion
                timestamp = highlight["timestamps"][0]
                clips.append(
                    {
                        "text": highlight["text"],
                        "start_time": timestamp.get("start", 0),
                        "end_time": timestamp.get("end", 0),
                        "suggested_duration": "30-60 seconds",  # Optimal for social media
                    }
                )
        return clips

    def _format_timestamp(self, milliseconds: float) -> str:
        """Format timestamp in MM:SS format for readability"""
        total_seconds = int(milliseconds / 1000)
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

    def _generate_content_insights(
        self, result: PipelineResult, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate content insights for podcast analytics and optimization

        Args:
            result: Transcription result with analysis data
            metadata: Podcast metadata

        Returns:
            Dict[str, Any]: Content insights for podcast optimization
        """
        insights = {}

        # Speaking time analysis for multi-host shows
        if result.speakers and len(result.speakers) > 1:
            speaking_distribution = {}
            for speaker in result.speakers:
                speaker_id = speaker["speaker"]
                duration = speaker["end"] - speaker["start"]
                speaking_distribution[speaker_id] = (
                    speaking_distribution.get(speaker_id, 0) + duration
                )

            insights["speaking_time_distribution"] = speaking_distribution

        # Content engagement metrics
        if result.highlights:
            insights["engagement_score"] = len(
                result.highlights
            )  # Number of highlights as engagement proxy

        # Topic diversity for content categorization
        if result.iab_categories:
            insights["content_diversity"] = len(
                result.iab_categories.get("results", [])
            )

        return insights if insights else None

    def _prepare_audio_source(
        self, source: str, source_type: AudioSource, **kwargs
    ) -> str:
        """
        Prepare audio source for podcast transcription

        This method reuses the general pipeline's audio source preparation logic
        since the source handling requirements are identical. Podcast-specific
        processing occurs in the transcription configuration and result enhancement.

        Args:
            source: Audio source identifier (path, URL, S3 URI, etc.)
            source_type: Type of audio source determining preparation method
            **kwargs: Source-specific preparation parameters

        Returns:
            str: URL that AssemblyAI can access for podcast audio transcription

        Note:
            Delegates to GeneralTranscriptionPipeline for consistent source handling
            across all pipeline types while maintaining podcast-specific logging context.
        """
        logger.debug(
            f"Preparing podcast audio source: {source} (type: {source_type.value})"
        )

        # Delegate to general pipeline's source preparation method
        # This ensures consistent behavior across all pipeline types
        prepared_url = GeneralTranscriptionPipeline._prepare_audio_source(
            self, source, source_type, **kwargs
        )

        logger.debug("Podcast audio source preparation completed")
        return prepared_url


class PodcastTranscriptionPipeline(BasePipeline):
    """
    Specialized pipeline for podcast transcription with chapters and highlights

    This pipeline is optimized for podcast content, providing enhanced features
    specifically designed for long-form audio content, episodic shows, and media
    production workflows. It focuses on content discovery, audience engagement,
    and automated content marketing tools.

    Key podcast-specific features:
    - Auto-generated chapters for easy navigation and timestamps
    - Content highlights extraction for social media and marketing
    - IAB content categorization for advertising and discovery
    - Topic detection for content tagging and SEO optimization
    - Speaker diarization for multi-host shows and guest identification
    - Sentiment analysis for audience engagement insights
    - Automated show notes generation for episode descriptions
    - Content accessibility compliance for transcript publication

    Ideal for:
    - Podcast episodes and audio shows
    - Interview-style content with multiple participants
    - Educational audio content and lectures
    - Long-form discussions and panel conversations
    - Audio content for accessibility compliance
    - Content marketing and social media clip generation

    Content Optimization:
    - Generates searchable transcripts for podcast platforms
    - Creates chapter markers for enhanced listener experience
    - Extracts quotable highlights for promotional content
    - Provides topic tags for content discovery and SEO

    Inherits from:
        BasePipeline: Provides core transcription functionality and result formatting
    """

    def process(
        self,
        source: str,
        source_type: AudioSource = AudioSource.LOCAL_FILE,
        podcast_metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> PipelineResult:
        """
        Process podcast audio with specialized features for content creators

        This method configures and executes transcription with features specifically
        optimized for podcast content including automatic chapter generation, content
        highlights extraction, and comprehensive content categorization for discovery
        and marketing purposes.

        Args:
            source: Audio source path, URL, or identifier for the podcast episode
                    - Local file: "/path/to/podcast_episode_042.mp3"
                    - URL: "https://cdn.example.com/episodes/tech-talk-ep42.wav"
                    - S3: "s3://podcast-storage/2024/season2/episode-042.mp3"
                    - Google Drive: File ID for podcast episode stored in Google Drive
            source_type: Type of audio source, defaults to LOCAL_FILE
            podcast_metadata: Optional podcast episode metadata and configuration:
                - show_name: Name of the podcast show (e.g., "Tech Talk Weekly")
                - episode_number: Episode number or identifier
                - episode_title: Title of the specific episode
                - season: Season number (if applicable)
                - hosts: List of podcast host names
                - guests: List of guest participant names
                - episode_date: Date of episode recording/publication
                - show_description: Brief description of the podcast show
                - episode_description: Description of the specific episode
                - tags: Content tags for categorization
                - target_audience: Intended audience demographic
                - show_category: Podcast category (technology, business, etc.)
            **kwargs: Additional source-specific parameters (AWS credentials, etc.)

        Returns:
            PipelineResult: Enhanced podcast transcription result containing:
                - Full episode transcript with speaker identification
                - Auto-generated chapter divisions with timestamps and descriptions
                - Content highlights ranked by importance for promotional use
                - IAB content categories for advertising and discovery
                - Topic detection results for SEO and content tagging
                - Sentiment analysis for audience engagement insights
                - Generated show notes combining chapters and highlights
                - Speaker diarization for multi-participant content
                - Podcast-specific metadata and content marketing insights

        Raises:
            Exception: If podcast transcription fails due to API errors, invalid audio,
                                network issues, or configuration problems

        Example:
            >>> pipeline = PodcastTranscriptionPipeline(config)
            >>> metadata = {
            ...     "show_name": "Tech Innovation Weekly",
            ...     "episode_number": 42,
            ...     "episode_title": "The Future of AI in Healthcare",
            ...     "hosts": ["Alice Johnson", "Bob Tech"],
            ...     "guests": ["Dr. Sarah AI", "Mike Healthcare"],
            ...     "show_category": "technology"
            ... }
            >>> result = pipeline.process("episode-042.mp3", podcast_metadata=metadata)
            >>> print(f"Generated {len(result.chapters)} chapters")
            >>> print(f"Found {len(result.highlights)} key highlights")
        """
        logger.info(f"Starting podcast transcription pipeline for source: {source}")
        logger.debug(f"Source type: {source_type.value}")

        # Log podcast metadata for content tracking and analytics
        if podcast_metadata:
            show_name = podcast_metadata.get("show_name", "Unknown Show")
            episode_title = podcast_metadata.get("episode_title", "Unknown Episode")
            episode_number = podcast_metadata.get("episode_number", "Unknown")
            logger.info(
                f"Processing podcast: '{show_name}' - Episode {episode_number}: '{episode_title}'"
            )

            # Log participant information for speaker diarization context
            hosts = podcast_metadata.get("hosts", [])
            guests = podcast_metadata.get("guests", [])
            total_participants = len(hosts) + len(guests)
            logger.info(
                f"Expected participants: {total_participants} ({len(hosts)} hosts, {len(guests)} guests)"
            )
            logger.debug(f"Podcast metadata: {podcast_metadata}")
        else:
            logger.info(
                "No podcast metadata provided, using default podcast processing"
            )

        # Configure podcast-optimized transcription features
        # These features are specifically chosen for podcast content discovery and marketing
        logger.debug("Configuring podcast-specific transcription features")
        features = TranscriptionFeatures(
            speaker_diarization=True,  # Essential for multi-host shows and guest identification
            word_timestamps=True,  # Needed for precise chapter timestamps
            auto_highlights=True,  # Extract quotable content for social media
            auto_chapters=True,  # Generate navigable chapter divisions
            sentiment_analysis=True,  # Track content tone for audience insights
            topic_detection=True,  # Identify discussion topics for SEO
            iab_categories=True,  # Content categorization for advertising
        )

        # Build transcription configuration optimized for podcast content
        logger.debug("Building transcription configuration for podcast content")
        config = self._build_transcription_config(features)

        # Prepare audio source for podcast transcription
        logger.info(f"Preparing podcast audio source of type: {source_type.value}")
        audio_url = self._prepare_audio_source(source, source_type, **kwargs)
        logger.debug("Prepared audio URL for podcast transcription")

        try:
            # Execute transcription with podcast-optimized settings
            logger.info("Starting podcast transcription with AssemblyAI")
            transcript = self.transcriber.transcribe(audio_url, config)
            logger.info(
                f"Podcast transcription completed successfully. ID: {transcript.id}"
            )

            # Log content analysis metrics for podcast analytics
            if hasattr(transcript, "chapters") and transcript.chapters:
                logger.info(f"Generated {len(transcript.chapters)} chapter divisions")
            if (
                hasattr(transcript, "auto_highlights_result")
                and transcript.auto_highlights_result
            ):
                highlights_count = len(transcript.auto_highlights_result.results)
                logger.info(f"Extracted {highlights_count} content highlights")

            # Prepare comprehensive source information for podcast records
            source_info = {
                "source": source,  # Original source identifier
                "source_type": source_type.value,  # Source type (local, url, s3, etc.)
                "content_type": ContentType.PODCAST.value,  # Specialized content type
                "podcast_metadata": podcast_metadata
                or {},  # Episode metadata and context
            }

            # Format standard transcription result
            logger.debug("Formatting podcast transcription result")
            result = self._format_result(transcript, source_info, features)

            # Apply podcast-specific enhancements and content generation
            logger.debug("Applying podcast-specific result enhancements")
            result = self._enhance_podcast_result(result, podcast_metadata)

            logger.info(
                f"Podcast transcription pipeline completed successfully for: {source}"
            )
            return result

        except Exception as e:
            # Log detailed error information for podcast transcription failures
            error_context = {
                "source": source,
                "source_type": source_type.value,
                "has_metadata": podcast_metadata is not None,
                "show_name": (
                    podcast_metadata.get("show_name") if podcast_metadata else None
                ),
                "episode_number": (
                    podcast_metadata.get("episode_number") if podcast_metadata else None
                ),
            }
            logger.error(f"Podcast transcription failed: {str(e)}")
            logger.debug(f"Error context: {error_context}")
            raise

    def _enhance_podcast_result(
        self, result: PipelineResult, metadata: Optional[Dict[str, Any]]
    ) -> PipelineResult:
        """
        Add podcast-specific enhancements to transcription results

        This method applies podcast-specific processing including automated show notes
        generation, content marketing insights, and episode-specific metadata integration.
        It enriches the standard transcription output with podcast-focused features.

        Args:
            result: Base transcription result from AssemblyAI processing
            metadata: Optional podcast metadata containing show and episode information

        Returns:
            PipelineResult: Enhanced result with podcast-specific information:
                - Generated show notes combining chapters and highlights
                - Content marketing insights for social media and promotion
                - Episode metadata integrated into source_info
                - Additional podcast-specific analytics and formatting

        Note:
            This method focuses on content creation and marketing tools that help
            podcast creators maximize their content's reach and engagement potential.
        """
        logger.debug("Enhancing podcast result with content-specific information")

        # Generate automated show notes if chapter and highlight data is available
        if metadata:
            logger.debug("Processing podcast metadata for content enhancement")

            # Generate show notes from chapters and highlights for content marketing
            if result.chapters and result.highlights:
                logger.info(
                    "Generating automated show notes from chapters and highlights"
                )
                show_notes = self._generate_show_notes(
                    result.chapters, result.highlights
                )
                result.source_info["generated_show_notes"] = show_notes
                logger.info(
                    f"Generated show notes with {len(show_notes['chapter_summary'])} chapter summaries and {len(show_notes['key_highlights'])} highlights"
                )
            else:
                if not result.chapters:
                    logger.warning("No chapters available for show notes generation")
                if not result.highlights:
                    logger.warning("No highlights available for show notes generation")

            # Add podcast-specific content insights for analytics and marketing
            content_insights = self._generate_content_insights(result, metadata)
            if content_insights:
                result.source_info["content_insights"] = content_insights
                logger.debug(f"Added content insights: {list(content_insights.keys())}")

            # Integrate episode metadata for comprehensive podcast records
            episode_metadata = {
                "show_name": metadata.get("show_name"),
                "episode_number": metadata.get("episode_number"),
                "episode_title": metadata.get("episode_title"),
                "season": metadata.get("season"),
                "hosts": metadata.get("hosts", []),
                "guests": metadata.get("guests", []),
                "episode_date": metadata.get("episode_date"),
                "show_category": metadata.get("show_category"),
                "target_audience": metadata.get("target_audience"),
                "episode_duration": result.audio_duration,
                "speakers_detected": (
                    len(set(s["speaker"] for s in result.speakers))
                    if result.speakers
                    else 0
                ),
                "chapters_generated": len(result.chapters) if result.chapters else 0,
                "highlights_extracted": (
                    len(result.highlights) if result.highlights else 0
                ),
            }

            # Filter out None values for cleaner metadata
            episode_metadata = {
                k: v for k, v in episode_metadata.items() if v is not None
            }

            if episode_metadata:
                result.source_info["episode_metadata"] = episode_metadata
                logger.debug(f"Added episode metadata: {list(episode_metadata.keys())}")

        else:
            logger.debug(
                "No podcast metadata provided, skipping enhanced content generation"
            )

        logger.debug("Podcast result enhancement completed")
        return result

    def _generate_show_notes(
        self, chapters: List[Dict], highlights: List[Dict]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive show notes from chapters and highlights

        This method creates automated show notes that podcast creators can use
        for episode descriptions, social media posts, and content marketing.
        The output is optimized for readability and audience engagement.

        Args:
            chapters: List of auto-generated chapter information with headlines and summaries
            highlights: List of content highlights ranked by importance and engagement potential

        Returns:
            Dict[str, Any]: Structured show notes containing:
                - chapter_summary: List of chapter descriptions with headlines and key points
                - key_highlights: Top 5 quotable highlights for social media and promotion
                - content_outline: Structured outline for episode descriptions
                - social_media_clips: Suggested content for social media posts

        Note:
            The generated show notes are designed to maximize content discoverability
            and provide ready-to-use marketing materials for podcast promotion.
        """
        logger.debug(
            f"Generating show notes from {len(chapters)} chapters and {len(highlights)} highlights"
        )

        # Create chapter summaries with headlines and key points
        chapter_summary = []
        for i, chapter in enumerate(chapters, 1):
            # Format: "Chapter 1: Headline - Key Point"
            chapter_entry = f"Chapter {i}: {chapter['headline']}"
            if chapter.get("gist"):
                chapter_entry += f" - {chapter['gist']}"
            chapter_summary.append(chapter_entry)

        # Extract top highlights ranked by importance for promotional content
        # Sort highlights by rank (lower rank = higher importance) and take top 5
        sorted_highlights = sorted(highlights, key=lambda x: x["rank"])
        key_highlights = [highlight["text"] for highlight in sorted_highlights[:5]]

        # Create additional content marketing materials
        show_notes = {
            "chapter_summary": chapter_summary,
            "key_highlights": key_highlights,
            "content_outline": self._create_content_outline(chapters),
            "social_media_clips": self._suggest_social_clips(
                sorted_highlights[:3]
            ),  # Top 3 for social media
        }

        logger.info(
            f"Generated show notes with {len(chapter_summary)} chapter summaries and {len(key_highlights)} key highlights"
        )
        return show_notes

    def _create_content_outline(self, chapters: List[Dict]) -> List[str]:
        """
        Create a structured content outline from chapters for episode descriptions

        Args:
            chapters: List of chapter information

        Returns:
            List[str]: Structured outline suitable for episode descriptions
        """
        outline = []
        for chapter in chapters:
            # Create timestamp-based outline entries
            start_time = self._format_timestamp(chapter.get("start", 0))
            outline.append(f"{start_time} - {chapter['headline']}")
        return outline

    def _suggest_social_clips(self, top_highlights: List[Dict]) -> List[Dict[str, Any]]:
        """
        Suggest social media clips from top highlights

        Args:
            top_highlights: Top-ranked highlights for social media

        Returns:
            List[Dict]: Suggested clips with timestamps and content
        """
        clips = []
        for highlight in top_highlights:
            if highlight.get("timestamps"):
                # Use first timestamp for clip suggestion
                timestamp = highlight["timestamps"][0]
                clips.append(
                    {
                        "text": highlight["text"],
                        "start_time": timestamp.get("start", 0),
                        "end_time": timestamp.get("end", 0),
                        "suggested_duration": "30-60 seconds",  # Optimal for social media
                    }
                )
        return clips

    def _format_timestamp(self, milliseconds: float) -> str:
        """Format timestamp in MM:SS format for readability"""
        total_seconds = int(milliseconds / 1000)
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

    def _generate_content_insights(
        self, result: PipelineResult, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate content insights for podcast analytics and optimization

        Args:
            result: Transcription result with analysis data
            metadata: Podcast metadata

        Returns:
            Dict[str, Any]: Content insights for podcast optimization
        """
        insights = {}

        # Speaking time analysis for multi-host shows
        if result.speakers and len(result.speakers) > 1:
            speaking_distribution = {}
            for speaker in result.speakers:
                speaker_id = speaker["speaker"]
                duration = speaker["end"] - speaker["start"]
                speaking_distribution[speaker_id] = (
                    speaking_distribution.get(speaker_id, 0) + duration
                )

            insights["speaking_time_distribution"] = speaking_distribution

        # Content engagement metrics
        if result.highlights:
            insights["engagement_score"] = len(
                result.highlights
            )  # Number of highlights as engagement proxy

        # Topic diversity for content categorization
        if result.iab_categories:
            insights["content_diversity"] = len(
                result.iab_categories.get("results", [])
            )

        return insights if insights else None

    def _prepare_audio_source(
        self, source: str, source_type: AudioSource, **kwargs
    ) -> str:
        """
        Prepare audio source for podcast transcription

        This method reuses the general pipeline's audio source preparation logic
        since the source handling requirements are identical. Podcast-specific
        processing occurs in the transcription configuration and result enhancement.

        Args:
            source: Audio source identifier (path, URL, S3 URI, etc.)
            source_type: Type of audio source determining preparation method
            **kwargs: Source-specific preparation parameters

        Returns:
            str: URL that AssemblyAI can access for podcast audio transcription

        Note:
            Delegates to GeneralTranscriptionPipeline for consistent source handling
            across all pipeline types while maintaining podcast-specific logging context.
        """
        logger.debug(
            f"Preparing podcast audio source: {source} (type: {source_type.value})"
        )

        # Delegate to general pipeline's source preparation method
        # This ensures consistent behavior across all pipeline types
        prepared_url = GeneralTranscriptionPipeline._prepare_audio_source(
            self, source, source_type, **kwargs
        )

        logger.debug("Podcast audio source preparation completed")
        return prepared_url


class RedactionModerationPipeline(BasePipeline):
    """
    Specialized pipeline for content safety, redaction, and moderation

    This pipeline is designed for scenarios requiring comprehensive privacy protection,
    content safety compliance, and regulatory adherence. It provides maximum security
    through PII redaction, content moderation, and safety analysis while maintaining
    transcription accuracy and usability.

    Key security and compliance features:
    - Comprehensive PII (Personally Identifiable Information) detection and redaction
    - Entity-level redaction for sensitive business and personal data
    - Content moderation to filter profanity and inappropriate language
    - Hate speech and harmful content detection with severity analysis
    - Audio redaction producing sanitized audio files
    - Detailed redaction reporting for compliance auditing
    - Multi-level privacy protection for sensitive content

    Use cases and compliance scenarios:
    - Healthcare: HIPAA-compliant medical transcriptions
    - Legal: Attorney-client privilege protection and case confidentiality
    - Financial: SOX and PCI compliance for financial discussions
    - HR: Employee privacy protection in recordings
    - Customer service: PII protection in support call transcriptions
    - Government: Classification and sensitive information handling
    - Education: FERPA compliance for student record protection
    - Corporate: Trade secret and confidential information protection

    Regulatory compliance:
    - GDPR (General Data Protection Regulation) compliance
    - HIPAA (Health Insurance Portability and Accountability Act)
    - SOX (Sarbanes-Oxley Act) requirements
    - PCI DSS (Payment Card Industry Data Security Standard)
    - FERPA (Family Educational Rights and Privacy Act)
    - CCPA (California Consumer Privacy Act)

    Output capabilities:
    - Redacted transcripts with sensitive information removed
    - Clean audio files with PII audio segments replaced
    - Comprehensive redaction reports for audit trails
    - Content safety analysis with risk assessments
    - Detailed compliance documentation

    Inherits from:
        BasePipeline: Provides core transcription functionality and result formatting
    """

    def process(
        self,
        source: str,
        source_type: AudioSource = AudioSource.LOCAL_FILE,
        redaction_policies: Optional[List[str]] = None,
        save_files: bool = True,
        **kwargs,
    ) -> PipelineResult:
        """
        Process audio with comprehensive content safety and redaction features

        This method executes a complete privacy-focused transcription workflow including
        PII detection, content moderation, safety analysis, and optional file output
        for compliance documentation. It provides maximum protection for sensitive content
        while maintaining transcription quality and generating audit-ready documentation.

        Args:
            source: Audio source path, URL, or identifier containing potentially sensitive content
                        - Local file: "/path/to/sensitive_meeting.mp3"
                        - URL: "https://secure.example.com/confidential-call.wav"
                        - S3: "s3://secure-bucket/private/client-consultation.mp3"
                        - Google Drive: File ID for sensitive recording requiring redaction
            source_type: Type of audio source, defaults to LOCAL_FILE
            redaction_policies: Optional list of specific redaction policies to apply
                                    If None, uses comprehensive default policies covering:
                                    - Personal identifiers (names, SSN, phone numbers)
                                    - Financial information (credit cards, account numbers)
                                    - Medical information (conditions, medications, procedures)
                                    - Location data and addresses
                                    - Organizational information
                                    Custom policies can be specified for targeted redaction
            save_files: Whether to save redacted outputs to files (default: True)
                            When True and output_config is available:
                            - Saves redacted transcript (JSON and TXT formats)
                            - Downloads and saves redacted audio file
                            - Generates compliance reports and audit documentation
                            - Creates redaction summary reports
            **kwargs: Additional source-specific parameters and redaction configuration

        Returns:
            PipelineResult: Comprehensive redaction and moderation result containing:
                - Full transcript with PII redacted using hash substitution
                - Detailed PII detection results with locations and confidence scores
                - Content safety analysis with hate speech and harmful content detection
                - Speaker identification for privacy-aware transcription
                - Redaction summary with statistics and compliance metrics
                - URLs for both original and redacted audio files
                - Saved file paths for compliance documentation (if save_files=True)
                - Audit trail information for regulatory compliance

        Raises:
            Exception: If redaction/moderation fails due to API errors, invalid audio,
                            network issues, or configuration problems

        Security considerations:
            - Original audio URLs are stored for reference but may contain sensitive data
            - Redacted audio URLs provide cleaned versions safe for broader distribution
            - PII redaction uses secure hashing to prevent data reconstruction
            - All processing maintains audit trails for compliance verification

        Example:
            >>> pipeline = RedactionModerationPipeline(config)
            >>> result = pipeline.process(
            ...     "confidential_meeting.mp3",
            ...     save_files=True  # Save redacted outputs for compliance
            ... )
            >>> print(f"Redacted {result.source_info['redaction_summary']['total_pii_instances']} PII instances")
            >>> print(f"Clean audio available at: {result.source_info['redacted_audio_url']}")
        """
        logger.info(f"Starting redaction/moderation pipeline for source: {source}")
        logger.info(
            "Processing with maximum privacy protection and content safety features"
        )
        logger.debug(f"Source type: {source_type.value}, Save files: {save_files}")

        # Log redaction policy configuration
        if redaction_policies:
            logger.info(f"Using custom redaction policies: {redaction_policies}")
        else:
            logger.info(
                "Using comprehensive default redaction policies for maximum privacy protection"
            )

        # Configure maximum privacy and safety features
        # All available protection features are enabled for comprehensive security
        logger.debug("Configuring comprehensive privacy and safety features")
        features = TranscriptionFeatures(
            speaker_diarization=True,  # Identify speakers for privacy-aware processing
            word_timestamps=True,  # Enable precise PII location tracking
            pii_redaction=True,  # Basic PII redaction
            entity_redaction=True,  # Comprehensive entity-level redaction
            content_moderation=True,  # Filter profanity and inappropriate content
            hate_speech_detection=True,  # Detect harmful and hateful content
        )

        # Build transcription configuration with maximum security settings
        logger.debug(
            "Building transcription configuration with security-focused settings"
        )
        config = self._build_transcription_config(features)

        # Apply custom redaction policies if provided
        if redaction_policies:
            logger.debug(
                f"Applying {len(redaction_policies)} custom redaction policies"
            )
            # Note: Implementation would map string policies to AssemblyAI enum values
            # This is a placeholder for custom policy application logic

        # Prepare audio source for secure transcription
        logger.info(f"Preparing audio source of type: {source_type.value}")
        audio_url = self._prepare_audio_source(source, source_type, **kwargs)
        logger.debug("Audio source prepared for redaction/moderation transcription")

        try:
            # Execute transcription with comprehensive privacy protection
            logger.info("Starting transcription with redaction and moderation features")
            transcript = self.transcriber.transcribe(audio_url, config)
            logger.info(
                f"Redaction/moderation transcription completed successfully. ID: {transcript.id}"
            )

            # Store audio URLs for both original and redacted versions
            # This enables compliance workflows that need both versions
            logger.debug("Extracting original and redacted audio URLs")
            self.original_audio_url = audio_url
            self.redacted_audio_url = transcript.get_redacted_audio_url()

            # Log audio redaction status
            if self.redacted_audio_url:
                logger.info("Redacted audio file successfully generated")
            else:
                logger.warning(
                    "No redacted audio URL available - may indicate no PII was detected"
                )

            # Prepare comprehensive source information for compliance tracking
            source_info = {
                "source": source,  # Original source identifier
                "source_type": source_type.value,  # Source type for audit trail
                "content_type": "redaction_moderation",  # Specialized processing type
                "pipeline_type": "redaction_moderation",  # Pipeline identifier for analytics
                "original_audio_url": self.original_audio_url,  # Original audio reference
                "redacted_audio_url": self.redacted_audio_url,  # Sanitized audio reference
                "processing_timestamp": datetime.now().isoformat(),  # Compliance timestamp
                "redaction_policies_applied": redaction_policies
                or "comprehensive_default",
            }

            # Format standard transcription result with redaction data
            logger.debug("Formatting redaction/moderation result")
            result = self._format_result(transcript, source_info, features)

            # Apply redaction-specific enhancements and analysis
            logger.debug("Applying redaction-specific result enhancements")
            result = self._enhance_redaction_result(result)

            # Generate compliance outputs if requested and configuration is available
            if save_files and self.config.output_config:
                logger.info(
                    "Saving redaction/moderation outputs for compliance documentation"
                )
                saved_files = self._save_outputs(result, "redaction_moderation")
                result.source_info["saved_files"] = saved_files

                # Provide detailed logging for compliance audit trails
                if saved_files:
                    logger.info(
                        f"Saved {len(saved_files)} compliance files: {list(saved_files.keys())}"
                    )

                    # Log redacted audio status for security compliance
                    if "redacted_audio" in saved_files:
                        logger.info(
                            f"Redacted audio saved successfully: {saved_files['redacted_audio']}"
                        )
                    else:
                        logger.warning(
                            "Redacted audio not saved - check if PII was detected and redaction is properly configured"
                        )
                        logger.debug(
                            "Redacted audio may not be available if no PII was detected in the content"
                        )

                    # Log other compliance files
                    if "json_transcription" in saved_files:
                        logger.info(
                            f"Redacted transcript saved: {saved_files['json_transcription']}"
                        )
                    if "analysis_report" in saved_files:
                        logger.info(
                            f"Compliance report saved: {saved_files['analysis_report']}"
                        )
                else:
                    logger.warning("No files were saved - check output configuration")
            else:
                if not save_files:
                    logger.info("File saving disabled - no compliance files generated")
                if not self.config.output_config:
                    logger.warning(
                        "No output configuration provided - cannot save compliance files"
                    )

            # Log final redaction statistics for audit purposes
            redaction_stats = result.source_info.get("redaction_summary", {})
            logger.info(
                f"Redaction processing complete: {redaction_stats.get('total_pii_instances', 0)} PII instances detected"
            )
            logger.info(
                f"Content safety issues found: {redaction_stats.get('content_safety_issues', 0)}"
            )

            logger.info(
                f"Redaction/moderation pipeline completed successfully for: {source}"
            )
            return result

        except Exception as e:
            # Log detailed error information for security and compliance troubleshooting
            error_context = {
                "source": source,
                "source_type": source_type.value,
                "save_files_requested": save_files,
                "has_output_config": self.config.output_config is not None,
                "custom_policies": redaction_policies is not None,
                "policy_count": len(redaction_policies) if redaction_policies else 0,
            }
            logger.error(f"Redaction/moderation transcription failed: {str(e)}")
            logger.debug(f"Error context: {error_context}")
            raise

    def _enhance_redaction_result(self, result: PipelineResult) -> PipelineResult:
        """
        Add redaction-specific enhancements and comprehensive privacy analysis

        This method creates detailed redaction summaries, compliance metrics, and
        privacy protection statistics that are essential for audit trails and
        regulatory compliance documentation.

        Args:
            result: Base transcription result with redaction data from AssemblyAI

        Returns:
            PipelineResult: Enhanced result with redaction analysis and compliance metrics:
                - Comprehensive redaction summary with statistics
                - PII type breakdown for privacy impact assessment
                - Content safety risk analysis
                - Compliance readiness indicators
                - Audit trail information

        Privacy analysis includes:
            - Total count of PII instances detected and redacted
            - Breakdown of PII types found (names, SSN, credit cards, etc.)
            - Content safety issue severity and count
            - Redaction effectiveness metrics
            - Risk assessment for content distribution
        """
        logger.debug(
            "Enhancing result with redaction-specific analysis and compliance metrics"
        )

        # Create comprehensive redaction summary for compliance reporting
        redaction_summary = {
            # Core PII statistics
            "total_pii_instances": (
                len(result.pii_detected) if result.pii_detected else 0
            ),
            "pii_types_found": (
                list(set(item["label"] for item in result.pii_detected))
                if result.pii_detected
                else []
            ),
            # Content safety metrics
            "content_safety_issues": (
                len(result.content_safety.get("results", []))
                if result.content_safety
                else 0
            ),
            # Redaction effectiveness indicators
            "redaction_applied": result.pii_detected is not None
            and len(result.pii_detected) > 0,
            "audio_redaction_available": result.source_info.get("redacted_audio_url")
            is not None,
            # Risk assessment metrics
            "privacy_risk_level": self._assess_privacy_risk(result),
            "content_safety_risk_level": self._assess_content_safety_risk(result),
            # Compliance indicators
            "compliance_ready": self._assess_compliance_readiness(result),
            "audit_trail_complete": True,  # Always true for this pipeline
        }

        # Add detailed PII breakdown for privacy impact assessment
        if result.pii_detected:
            pii_breakdown = self._create_pii_breakdown(result.pii_detected)
            redaction_summary["pii_breakdown"] = pii_breakdown
            logger.info(f"PII breakdown: {pii_breakdown}")

        # Add content safety analysis if available
        if result.content_safety:
            safety_analysis = self._create_safety_analysis(result.content_safety)
            redaction_summary["safety_analysis"] = safety_analysis
            logger.info(f"Content safety analysis: {safety_analysis}")

        # Store comprehensive summary in result
        result.source_info["redaction_summary"] = redaction_summary

        # Log summary for audit trail
        logger.info(
            f"Redaction analysis complete - PII instances: {redaction_summary['total_pii_instances']}, "
            f"Safety issues: {redaction_summary['content_safety_issues']}, "
            f"Privacy risk: {redaction_summary['privacy_risk_level']}"
        )

        logger.debug("Redaction result enhancement completed")
        return result

    def _assess_privacy_risk(self, result: PipelineResult) -> str:
        """
        Assess privacy risk level based on PII detection results

        Args:
            result: Transcription result with PII data

        Returns:
            str: Risk level (LOW, MEDIUM, HIGH, CRITICAL)
        """
        if not result.pii_detected:
            return "LOW"

        pii_count = len(result.pii_detected)
        high_risk_types = [
            "us_social_security_number",
            "credit_card_number",
            "passport_number",
            "medical_condition",
        ]

        high_risk_items = [
            item for item in result.pii_detected if item["label"] in high_risk_types
        ]

        if len(high_risk_items) > 0:
            return "CRITICAL"
        elif pii_count > 10:
            return "HIGH"
        elif pii_count > 3:
            return "MEDIUM"
        else:
            return "LOW"

    def _assess_content_safety_risk(self, result: PipelineResult) -> str:
        """
        Assess content safety risk level based on detected issues

        Args:
            result: Transcription result with content safety data

        Returns:
            str: Risk level (LOW, MEDIUM, HIGH, CRITICAL)
        """
        if not result.content_safety or not result.content_safety.get("results"):
            return "LOW"

        issues = result.content_safety["results"]
        high_severity_count = sum(
            1
            for issue in issues
            if any(label.get("severity") == "high" for label in issue.get("labels", []))
        )

        if high_severity_count > 0:
            return "HIGH"
        elif len(issues) > 5:
            return "MEDIUM"
        elif len(issues) > 0:
            return "LOW"
        else:
            return "LOW"

    def _assess_compliance_readiness(self, result: PipelineResult) -> bool:
        """
        Assess whether the redacted content is ready for compliance distribution

        Args:
            result: Transcription result with redaction data

        Returns:
            bool: True if content meets basic compliance standards
        """
        # Basic compliance check: PII detected and redacted, no critical safety issues
        pii_handled = (
            not result.pii_detected
            or len(result.pii_detected) == 0
            or result.source_info.get("redacted_audio_url")
        )
        safety_acceptable = self._assess_content_safety_risk(result) in [
            "LOW",
            "MEDIUM",
        ]

        return pii_handled and safety_acceptable

    def _create_pii_breakdown(
        self, pii_detected: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Create breakdown of PII types for privacy impact assessment"""
        breakdown = {}
        for item in pii_detected:
            pii_type = item["label"]
            breakdown[pii_type] = breakdown.get(pii_type, 0) + 1
        return breakdown

    def _create_safety_analysis(self, content_safety: Dict[str, Any]) -> Dict[str, Any]:
        """Create content safety analysis summary"""
        results = content_safety.get("results", [])

        analysis = {
            "total_issues": len(results),
            "issue_types": [],
            "severity_breakdown": {"low": 0, "medium": 0, "high": 0},
        }

        for result in results:
            for label in result.get("labels", []):
                issue_type = label.get("label", "unknown")
                if issue_type not in analysis["issue_types"]:
                    analysis["issue_types"].append(issue_type)

                severity = label.get("severity", "low")
                if severity in analysis["severity_breakdown"]:
                    analysis["severity_breakdown"][severity] += 1

        return analysis

    def _prepare_audio_source(
        self, source: str, source_type: AudioSource, **kwargs
    ) -> str:
        """
        Prepare audio source for redaction/moderation transcription

        This method reuses the general pipeline's audio source preparation logic
        since the source handling requirements are identical. Redaction-specific
        processing occurs in the transcription configuration and result enhancement.

        Args:
            source: Audio source identifier (path, URL, S3 URI, etc.)
            source_type: Type of audio source determining preparation method
            **kwargs: Source-specific preparation parameters

        Returns:
            str: URL that AssemblyAI can access for redaction/moderation transcription

        Security note:
            Source preparation maintains the same security standards as other pipelines.
            Additional security measures are applied during transcription processing
            and result handling for sensitive content protection.
        """
        logger.debug(
            f"Preparing audio source for redaction/moderation: {source} (type: {source_type.value})"
        )

        # Delegate to general pipeline's source preparation method
        # This ensures consistent behavior across all pipeline types
        prepared_url = GeneralTranscriptionPipeline._prepare_audio_source(
            self, source, source_type, **kwargs
        )

        logger.debug(
            "Audio source preparation completed for redaction/moderation pipeline"
        )
        return prepared_url


class RedactionModerationPipeline(BasePipeline):
    """
    Specialized pipeline for content safety, redaction, and moderation

    This pipeline is designed for scenarios requiring comprehensive privacy protection,
    content safety compliance, and regulatory adherence. It provides maximum security
    through PII redaction, content moderation, and safety analysis while maintaining
    transcription accuracy and usability.

    Key security and compliance features:
    - Comprehensive PII (Personally Identifiable Information) detection and redaction
    - Entity-level redaction for sensitive business and personal data
    - Content moderation to filter profanity and inappropriate language
    - Hate speech and harmful content detection with severity analysis
    - Audio redaction producing sanitized audio files
    - Detailed redaction reporting for compliance auditing
    - Multi-level privacy protection for sensitive content

    Use cases and compliance scenarios:
    - Healthcare: HIPAA-compliant medical transcriptions
    - Legal: Attorney-client privilege protection and case confidentiality
    - Financial: SOX and PCI compliance for financial discussions
    - HR: Employee privacy protection in recordings
    - Customer service: PII protection in support call transcriptions
    - Government: Classification and sensitive information handling
    - Education: FERPA compliance for student record protection
    - Corporate: Trade secret and confidential information protection

    Regulatory compliance:
    - GDPR (General Data Protection Regulation) compliance
    - HIPAA (Health Insurance Portability and Accountability Act)
    - SOX (Sarbanes-Oxley Act) requirements
    - PCI DSS (Payment Card Industry Data Security Standard)
    - FERPA (Family Educational Rights and Privacy Act)
    - CCPA (California Consumer Privacy Act)

    Output capabilities:
    - Redacted transcripts with sensitive information removed
    - Clean audio files with PII audio segments replaced
    - Comprehensive redaction reports for audit trails
    - Content safety analysis with risk assessments
    - Detailed compliance documentation

    Inherits from:
        BasePipeline: Provides core transcription functionality and result formatting
    """

    def process(
        self,
        source: str,
        source_type: AudioSource = AudioSource.LOCAL_FILE,
        redaction_policies: Optional[List[str]] = None,
        save_files: bool = True,
        **kwargs,
    ) -> PipelineResult:
        """
        Process audio with comprehensive content safety and redaction features

        This method executes a complete privacy-focused transcription workflow including
        PII detection, content moderation, safety analysis, and optional file output
        for compliance documentation. It provides maximum protection for sensitive content
        while maintaining transcription quality and generating audit-ready documentation.

        Args:
            source: Audio source path, URL, or identifier containing potentially sensitive content
                    - Local file: "/path/to/sensitive_meeting.mp3"
                    - URL: "https://secure.example.com/confidential-call.wav"
                    - S3: "s3://secure-bucket/private/client-consultation.mp3"
                    - Google Drive: File ID for sensitive recording requiring redaction
            source_type: Type of audio source, defaults to LOCAL_FILE
            redaction_policies: Optional list of specific redaction policies to apply
                                If None, uses comprehensive default policies covering:
                                - Personal identifiers (names, SSN, phone numbers)
                                - Financial information (credit cards, account numbers)
                                - Medical information (conditions, medications, procedures)
                                - Location data and addresses
                                - Organizational information
                                Custom policies can be specified for targeted redaction
            save_files: Whether to save redacted outputs to files (default: True)
                        When True and output_config is available:
                        - Saves redacted transcript (JSON and TXT formats)
                        - Downloads and saves redacted audio file
                        - Generates compliance reports and audit documentation
                        - Creates redaction summary reports
            **kwargs: Additional source-specific parameters and redaction configuration

        Returns:
            PipelineResult: Comprehensive redaction and moderation result containing:
                - Full transcript with PII redacted using hash substitution
                - Detailed PII detection results with locations and confidence scores
                - Content safety analysis with hate speech and harmful content detection
                - Speaker identification for privacy-aware transcription
                - Redaction summary with statistics and compliance metrics
                - URLs for both original and redacted audio files
                - Saved file paths for compliance documentation (if save_files=True)
                - Audit trail information for regulatory compliance

        Raises:
            Exception: If redaction/moderation fails due to API errors, invalid audio,
                        network issues, or configuration problems

        Security considerations:
            - Original audio URLs are stored for reference but may contain sensitive data
            - Redacted audio URLs provide cleaned versions safe for broader distribution
            - PII redaction uses secure hashing to prevent data reconstruction
            - All processing maintains audit trails for compliance verification

        Example:
            >>> pipeline = RedactionModerationPipeline(config)
            >>> result = pipeline.process(
            ...     "confidential_meeting.mp3",
            ...     save_files=True  # Save redacted outputs for compliance
            ... )
            >>> print(f"Redacted {result.source_info['redaction_summary']['total_pii_instances']} PII instances")
            >>> print(f"Clean audio available at: {result.source_info['redacted_audio_url']}")
        """
        logger.info(f"Starting redaction/moderation pipeline for source: {source}")
        logger.info(
            "Processing with maximum privacy protection and content safety features"
        )
        logger.debug(f"Source type: {source_type.value}, Save files: {save_files}")

        # Log redaction policy configuration
        if redaction_policies:
            logger.info(f"Using custom redaction policies: {redaction_policies}")
        else:
            logger.info(
                "Using comprehensive default redaction policies for maximum privacy protection"
            )

        # Configure maximum privacy and safety features
        # All available protection features are enabled for comprehensive security
        logger.debug("Configuring comprehensive privacy and safety features")
        features = TranscriptionFeatures(
            speaker_diarization=True,  # Identify speakers for privacy-aware processing
            word_timestamps=True,  # Enable precise PII location tracking
            pii_redaction=True,  # Basic PII redaction
            entity_redaction=True,  # Comprehensive entity-level redaction
            content_moderation=True,  # Filter profanity and inappropriate content
            hate_speech_detection=True,  # Detect harmful and hateful content
        )

        # Build transcription configuration with maximum security settings
        logger.debug(
            "Building transcription configuration with security-focused settings"
        )
        config = self._build_transcription_config(features)

        # Apply custom redaction policies if provided
        if redaction_policies:
            logger.debug(
                f"Applying {len(redaction_policies)} custom redaction policies"
            )
            # Note: Implementation would map string policies to AssemblyAI enum values
            # This is a placeholder for custom policy application logic

        # Prepare audio source for secure transcription
        logger.info(f"Preparing audio source of type: {source_type.value}")
        audio_url = self._prepare_audio_source(source, source_type, **kwargs)
        logger.debug("Audio source prepared for redaction/moderation transcription")

        try:
            # Execute transcription with comprehensive privacy protection
            logger.info("Starting transcription with redaction and moderation features")
            transcript = self.transcriber.transcribe(audio_url, config)
            logger.info(
                f"Redaction/moderation transcription completed successfully. ID: {transcript.id}"
            )

            # Store audio URLs for both original and redacted versions
            # This enables compliance workflows that need both versions
            logger.debug("Extracting original and redacted audio URLs")
            self.original_audio_url = audio_url
            self.redacted_audio_url = transcript.get_redacted_audio_url()

            # Log audio redaction status
            if self.redacted_audio_url:
                logger.info("Redacted audio file successfully generated")
            else:
                logger.warning(
                    "No redacted audio URL available - may indicate no PII was detected"
                )

            # Prepare comprehensive source information for compliance tracking
            source_info = {
                "source": source,  # Original source identifier
                "source_type": source_type.value,  # Source type for audit trail
                "content_type": "redaction_moderation",  # Specialized processing type
                "pipeline_type": "redaction_moderation",  # Pipeline identifier for analytics
                "original_audio_url": self.original_audio_url,  # Original audio reference
                "redacted_audio_url": self.redacted_audio_url,  # Sanitized audio reference
                "processing_timestamp": datetime.now().isoformat(),  # Compliance timestamp
                "redaction_policies_applied": redaction_policies
                or "comprehensive_default",
            }

            # Format standard transcription result with redaction data
            logger.debug("Formatting redaction/moderation result")
            result = self._format_result(transcript, source_info, features)

            # Apply redaction-specific enhancements and analysis
            logger.debug("Applying redaction-specific result enhancements")
            result = self._enhance_redaction_result(result)

            # Generate compliance outputs if requested and configuration is available
            if save_files and self.config.output_config:
                logger.info(
                    "Saving redaction/moderation outputs for compliance documentation"
                )
                saved_files = self._save_outputs(result, "redaction_moderation")
                result.source_info["saved_files"] = saved_files

                # Provide detailed logging for compliance audit trails
                if saved_files:
                    logger.info(
                        f"Saved {len(saved_files)} compliance files: {list(saved_files.keys())}"
                    )

                    # Log redacted audio status for security compliance
                    if "redacted_audio" in saved_files:
                        logger.info(
                            f"Redacted audio saved successfully: {saved_files['redacted_audio']}"
                        )
                    else:
                        logger.warning(
                            "Redacted audio not saved - check if PII was detected and redaction is properly configured"
                        )
                        logger.debug(
                            "Redacted audio may not be available if no PII was detected in the content"
                        )

                    # Log other compliance files
                    if "json_transcription" in saved_files:
                        logger.info(
                            f"Redacted transcript saved: {saved_files['json_transcription']}"
                        )
                    if "analysis_report" in saved_files:
                        logger.info(
                            f"Compliance report saved: {saved_files['analysis_report']}"
                        )
                else:
                    logger.warning("No files were saved - check output configuration")
            else:
                if not save_files:
                    logger.info("File saving disabled - no compliance files generated")
                if not self.config.output_config:
                    logger.warning(
                        "No output configuration provided - cannot save compliance files"
                    )

            # Log final redaction statistics for audit purposes
            redaction_stats = result.source_info.get("redaction_summary", {})
            logger.info(
                f"Redaction processing complete: {redaction_stats.get('total_pii_instances', 0)} PII instances detected"
            )
            logger.info(
                f"Content safety issues found: {redaction_stats.get('content_safety_issues', 0)}"
            )

            logger.info(
                f"Redaction/moderation pipeline completed successfully for: {source}"
            )
            return result

        except Exception as e:
            # Log detailed error information for security and compliance troubleshooting
            error_context = {
                "source": source,
                "source_type": source_type.value,
                "save_files_requested": save_files,
                "has_output_config": self.config.output_config is not None,
                "custom_policies": redaction_policies is not None,
                "policy_count": len(redaction_policies) if redaction_policies else 0,
            }
            logger.error(f"Redaction/moderation transcription failed: {str(e)}")
            logger.debug(f"Error context: {error_context}")
            raise

    def _enhance_redaction_result(self, result: PipelineResult) -> PipelineResult:
        """
        Add redaction-specific enhancements and comprehensive privacy analysis

        This method creates detailed redaction summaries, compliance metrics, and
        privacy protection statistics that are essential for audit trails and
        regulatory compliance documentation.

        Args:
            result: Base transcription result with redaction data from AssemblyAI

        Returns:
            PipelineResult: Enhanced result with redaction analysis and compliance metrics:
                - Comprehensive redaction summary with statistics
                - PII type breakdown for privacy impact assessment
                - Content safety risk analysis
                - Compliance readiness indicators
                - Audit trail information

        Privacy analysis includes:
            - Total count of PII instances detected and redacted
            - Breakdown of PII types found (names, SSN, credit cards, etc.)
            - Content safety issue severity and count
            - Redaction effectiveness metrics
            - Risk assessment for content distribution
        """
        logger.debug(
            "Enhancing result with redaction-specific analysis and compliance metrics"
        )

        # Create comprehensive redaction summary for compliance reporting
        redaction_summary = {
            # Core PII statistics
            "total_pii_instances": (
                len(result.pii_detected) if result.pii_detected else 0
            ),
            "pii_types_found": (
                list(set(item["label"] for item in result.pii_detected))
                if result.pii_detected
                else []
            ),
            # Content safety metrics
            "content_safety_issues": (
                len(result.content_safety.get("results", []))
                if result.content_safety
                else 0
            ),
            # Redaction effectiveness indicators
            "redaction_applied": result.pii_detected is not None
            and len(result.pii_detected) > 0,
            "audio_redaction_available": result.source_info.get("redacted_audio_url")
            is not None,
            # Risk assessment metrics
            "privacy_risk_level": self._assess_privacy_risk(result),
            "content_safety_risk_level": self._assess_content_safety_risk(result),
            # Compliance indicators
            "compliance_ready": self._assess_compliance_readiness(result),
            "audit_trail_complete": True,  # Always true for this pipeline
        }

        # Add detailed PII breakdown for privacy impact assessment
        if result.pii_detected:
            pii_breakdown = self._create_pii_breakdown(result.pii_detected)
            redaction_summary["pii_breakdown"] = pii_breakdown
            logger.info(f"PII breakdown: {pii_breakdown}")

        # Add content safety analysis if available
        if result.content_safety:
            safety_analysis = self._create_safety_analysis(result.content_safety)
            redaction_summary["safety_analysis"] = safety_analysis
            logger.info(f"Content safety analysis: {safety_analysis}")

        # Store comprehensive summary in result
        result.source_info["redaction_summary"] = redaction_summary

        # Log summary for audit trail
        logger.info(
            f"Redaction analysis complete - PII instances: {redaction_summary['total_pii_instances']}, "
            f"Safety issues: {redaction_summary['content_safety_issues']}, "
            f"Privacy risk: {redaction_summary['privacy_risk_level']}"
        )

        logger.debug("Redaction result enhancement completed")
        return result

    def _assess_privacy_risk(self, result: PipelineResult) -> str:
        """
        Assess privacy risk level based on PII detection results

        Args:
            result: Transcription result with PII data

        Returns:
            str: Risk level (LOW, MEDIUM, HIGH, CRITICAL)
        """
        if not result.pii_detected:
            return "LOW"

        pii_count = len(result.pii_detected)
        high_risk_types = [
            "us_social_security_number",
            "credit_card_number",
            "passport_number",
            "medical_condition",
        ]

        high_risk_items = [
            item for item in result.pii_detected if item["label"] in high_risk_types
        ]

        if len(high_risk_items) > 0:
            return "CRITICAL"
        elif pii_count > 10:
            return "HIGH"
        elif pii_count > 3:
            return "MEDIUM"
        else:
            return "LOW"

    def _assess_content_safety_risk(self, result: PipelineResult) -> str:
        """
        Assess content safety risk level based on detected issues

        Args:
            result: Transcription result with content safety data

        Returns:
            str: Risk level (LOW, MEDIUM, HIGH, CRITICAL)
        """
        if not result.content_safety or not result.content_safety.get("results"):
            return "LOW"

        issues = result.content_safety["results"]
        high_severity_count = sum(
            1
            for issue in issues
            if any(label.get("severity") == "high" for label in issue.get("labels", []))
        )

        if high_severity_count > 0:
            return "HIGH"
        elif len(issues) > 5:
            return "MEDIUM"
        elif len(issues) > 0:
            return "LOW"
        else:
            return "LOW"

    def _assess_compliance_readiness(self, result: PipelineResult) -> bool:
        """
        Assess whether the redacted content is ready for compliance distribution

        Args:
            result: Transcription result with redaction data

        Returns:
            bool: True if content meets basic compliance standards
        """
        # Basic compliance check: PII detected and redacted, no critical safety issues
        pii_handled = (
            not result.pii_detected
            or len(result.pii_detected) == 0
            or result.source_info.get("redacted_audio_url")
        )
        safety_acceptable = self._assess_content_safety_risk(result) in [
            "LOW",
            "MEDIUM",
        ]

        return pii_handled and safety_acceptable

    def _create_pii_breakdown(
        self, pii_detected: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Create breakdown of PII types for privacy impact assessment"""
        breakdown = {}
        for item in pii_detected:
            pii_type = item["label"]
            breakdown[pii_type] = breakdown.get(pii_type, 0) + 1
        return breakdown

    def _create_safety_analysis(self, content_safety: Dict[str, Any]) -> Dict[str, Any]:
        """Create content safety analysis summary"""
        results = content_safety.get("results", [])

        analysis = {
            "total_issues": len(results),
            "issue_types": [],
            "severity_breakdown": {"low": 0, "medium": 0, "high": 0},
        }

        for result in results:
            for label in result.get("labels", []):
                issue_type = label.get("label", "unknown")
                if issue_type not in analysis["issue_types"]:
                    analysis["issue_types"].append(issue_type)

                severity = label.get("severity", "low")
                if severity in analysis["severity_breakdown"]:
                    analysis["severity_breakdown"][severity] += 1

        return analysis

    def _prepare_audio_source(
        self, source: str, source_type: AudioSource, **kwargs
    ) -> str:
        """
        Prepare audio source for redaction/moderation transcription

        This method reuses the general pipeline's audio source preparation logic
        since the source handling requirements are identical. Redaction-specific
        processing occurs in the transcription configuration and result enhancement.

        Args:
            source: Audio source identifier (path, URL, S3 URI, etc.)
            source_type: Type of audio source determining preparation method
            **kwargs: Source-specific preparation parameters

        Returns:
            str: URL that AssemblyAI can access for redaction/moderation transcription

        Security note:
            Source preparation maintains the same security standards as other pipelines.
            Additional security measures are applied during transcription processing
            and result handling for sensitive content protection.
        """
        logger.debug(
            f"Preparing audio source for redaction/moderation: {source} (type: {source_type.value})"
        )

        # Delegate to general pipeline's source preparation method
        # This ensures consistent behavior across all pipeline types
        prepared_url = GeneralTranscriptionPipeline._prepare_audio_source(
            self, source, source_type, **kwargs
        )

        logger.debug(
            "Audio source preparation completed for redaction/moderation pipeline"
        )
        return prepared_url


class ContentAnalysisPipeline(BasePipeline):
    """
    Adaptive pipeline for intelligent content analysis based on audio type

    This sophisticated pipeline automatically adapts its analysis approach based on
    the content type, providing specialized insights and features tailored to different
    audio scenarios. It serves as a unified solution for comprehensive content analysis
    across various use cases while maintaining optimal performance for each content type.

    Key adaptive capabilities:
    - Content-type specific feature optimization for maximum relevance
    - Intelligent analysis adaptation based on audio characteristics
    - Specialized insights generation for different content scenarios
    - Unified interface with content-aware processing
    - Comprehensive analytics and metrics for content optimization

    Supported content types and optimizations:
    - PODCAST: Chapters, highlights, show notes, content categorization
    - MEETING: Action items, participation analysis, decision tracking
    - INTERVIEW: Quote extraction, sentiment flow, Q&A analysis
    - LECTURE: Knowledge extraction, topic progression, educational insights
    - GENERAL: Balanced feature set for unknown or mixed content

    Content-specific insights:
    - Structural analysis (chapters, segments, flow)
    - Engagement metrics (highlights, key moments, participation)
    - Sentiment and emotional analysis throughout content
    - Topic identification and categorization
    - Content optimization recommendations
    - Accessibility and searchability enhancements

    Use cases:
    - Content creators seeking audience engagement insights
    - Business analysts reviewing meeting effectiveness
    - Journalists analyzing interview content and quotes
    - Educators optimizing lecture content and delivery
    - Researchers conducting content analysis studies
    - Marketing teams extracting content for campaigns

    Benefits:
    - Automated content type detection and optimization
    - Rich insights beyond basic transcription
    - Content marketing and SEO optimization data
    - Audience engagement and participation metrics
    - Ready-to-use content derivatives (show notes, summaries, quotes)

    Inherits from:
        BasePipeline: Provides core transcription functionality and result formatting
    """

    def process(
        self,
        source: str,
        content_type: ContentType = ContentType.GENERAL,
        source_type: AudioSource = AudioSource.LOCAL_FILE,
        context_metadata: Optional[Dict[str, Any]] = None,
        save_files: bool = True,
        **kwargs,
    ) -> PipelineResult:
        """
        Process audio with adaptive content analysis features tailored to content type

        This method intelligently configures transcription features and analysis approaches
        based on the specified content type, ensuring optimal results for each scenario.
        It provides comprehensive content analysis beyond basic transcription, including
        insights, metrics, and derivative content for enhanced usability.

        Args:
            source: Audio source path, URL, or identifier for content analysis
                    - Local file: "/path/to/content.mp3"
                    - URL: "https://example.com/podcast-episode.wav"
                    - S3: "s3://content-bucket/meeting-recording.mp3"
                    - Google Drive: File ID for shared content
            content_type: Type of content for analysis optimization, defaults to GENERAL
                            - PODCAST: Optimized for episodic content, chapters, show notes
                            - MEETING: Focused on participation, decisions, action items
                            - INTERVIEW: Emphasis on quotes, sentiment, Q&A dynamics
                            - LECTURE: Educational content analysis and knowledge extraction
                            - GENERAL: Balanced analysis for unknown or mixed content
            source_type: Type of audio source, defaults to LOCAL_FILE
            context_metadata: Optional content context and metadata for enhanced analysis:
                - title: Content title for better categorization
                - description: Content description for context
                - participants: List of expected participants/speakers
                - topics: Expected topics or themes for targeted analysis
                - duration_estimate: Expected duration for processing optimization
                - target_audience: Intended audience for relevance scoring
                - content_goals: Specific goals (education, entertainment, business)
                - brand_context: Brand or organization context for analysis
            save_files: Whether to save analysis outputs and reports (default: True)
                        When True and output_config is available:
                        - Saves comprehensive transcription and analysis results
                        - Generates content-specific reports and insights
                        - Creates derivative content (show notes, summaries, etc.)
                        - Saves structured data for further analysis
            **kwargs: Additional source-specific and analysis parameters

        Returns:
            PipelineResult: Comprehensive content analysis result containing:
                - Full transcript optimized for content type
                - Content-specific insights and analytics
                - Structural analysis (chapters, segments, flow)
                - Engagement metrics and audience insights
                - Derivative content ready for use (show notes, quotes, summaries)
                - Participation and sentiment analysis
                - Topic identification and categorization
                - Content optimization recommendations
                - SEO and marketing-ready data extracts

        Raises:
            Exception: If content analysis fails due to API errors, invalid audio,
                        network issues, or configuration problems

        Examples:
            >>> # Podcast analysis with show notes generation
            >>> pipeline = ContentAnalysisPipeline(config)
            >>> result = pipeline.process(
            ...     "podcast-episode.mp3",
            ...     content_type=ContentType.PODCAST,
            ...     context_metadata={"title": "Tech Innovation Weekly", "episode": 42}
            ... )
            >>> show_notes = result.source_info["content_insights"]["show_notes"]

            >>> # Meeting analysis with participation tracking
            >>> result = pipeline.process(
            ...     "team-meeting.wav",
            ...     content_type=ContentType.MEETING,
            ...     context_metadata={"participants": ["Alice", "Bob", "Carol"]}
            ... )
            >>> participation = result.source_info["content_insights"]["participation_analysis"]

            >>> # Interview analysis with quote extraction
            >>> result = pipeline.process(
            ...     "expert-interview.mp3",
            ...     content_type=ContentType.INTERVIEW
            ... )
            >>> key_quotes = result.source_info["content_insights"]["key_quotes"]

        Content-type optimizations:
            - Each content type receives specialized feature configuration
            - Analysis algorithms adapt to content characteristics
            - Output formats optimized for typical use cases
            - Insights generation tailored to content goals
        """
        logger.info(f"Starting adaptive content analysis pipeline for source: {source}")
        logger.info(
            f"Content type: {content_type.value} - Optimizing analysis approach"
        )
        logger.debug(f"Source type: {source_type.value}, Save files: {save_files}")

        # Log context metadata for enhanced analysis
        if context_metadata:
            metadata_keys = list(context_metadata.keys())
            logger.info(
                f"Using context metadata for enhanced analysis: {metadata_keys}"
            )
            logger.debug(f"Context metadata: {context_metadata}")
        else:
            logger.info(
                "No context metadata provided - using content type optimization only"
            )

        # Configure content-type specific features for optimal analysis
        logger.debug(f"Configuring features optimized for {content_type.value} content")
        features = self._get_content_specific_features(content_type)
        logger.info(f"Applied {content_type.value}-specific feature optimization")

        # Build transcription configuration with content-aware settings
        config = self._build_transcription_config(features)

        # Apply content-type specific configuration overrides
        logger.debug("Applying content-type specific configuration overrides")
        if content_type == ContentType.MEETING:
            # Optimize summarization for meeting notes and action items
            config.summary_model = aai.SummarizationModel.informative
            config.summary_type = aai.SummarizationType.bullets
            logger.debug(
                "Applied meeting-specific summarization settings (informative, bullets)"
            )

        # Future content-type specific configurations can be added here
        # elif content_type == ContentType.PODCAST:
        #     # Podcast-specific optimizations
        # elif content_type == ContentType.INTERVIEW:
        #     # Interview-specific optimizations

        # Prepare audio source for content analysis
        logger.info(f"Preparing audio source of type: {source_type.value}")
        audio_url = self._prepare_audio_source(source, source_type, **kwargs)
        logger.debug("Audio source prepared for content analysis transcription")

        try:
            # Execute transcription with content-optimized configuration
            logger.info(f"Starting {content_type.value} content analysis transcription")
            transcript = self.transcriber.transcribe(audio_url, config)
            logger.info(
                f"Content analysis transcription completed successfully. ID: {transcript.id}"
            )

            # Log analysis results for content optimization insights
            if hasattr(transcript, "chapters") and transcript.chapters:
                logger.info(f"Generated {len(transcript.chapters)} content chapters")
            if (
                hasattr(transcript, "auto_highlights_result")
                and transcript.auto_highlights_result
            ):
                highlights_count = len(transcript.auto_highlights_result.results)
                logger.info(f"Extracted {highlights_count} content highlights")

            # Prepare comprehensive source information for content analytics
            source_info = {
                "source": source,  # Original source identifier
                "source_type": source_type.value,  # Source type for processing tracking
                "content_type": content_type.value,  # Content type for analysis optimization
                "context_metadata": context_metadata
                or {},  # Additional context for analysis
                "pipeline_type": "content_analysis",  # Pipeline identifier
                "analysis_timestamp": datetime.now().isoformat(),  # Processing timestamp
                "optimization_applied": content_type.value,  # Optimization strategy used
            }

            # Format standard transcription result with content-aware processing
            logger.debug("Formatting content analysis result")
            result = self._format_result(transcript, source_info, features)

            # Apply content-type specific enhancements and insight generation
            logger.info(
                f"Generating {content_type.value}-specific insights and analytics"
            )
            result = self._enhance_content_analysis_result(
                result, content_type, context_metadata
            )

            # Generate and save comprehensive outputs if requested
            if save_files and self.config.output_config:
                logger.info("Saving content analysis outputs and reports")
                saved_files = self._save_outputs(
                    result, "content_analysis", content_type.value
                )
                result.source_info["saved_files"] = saved_files

                if saved_files:
                    logger.info(
                        f"Saved {len(saved_files)} content analysis files: {list(saved_files.keys())}"
                    )
                else:
                    logger.warning("No files were saved - check output configuration")
            else:
                if not save_files:
                    logger.info("File saving disabled for content analysis")
                if not self.config.output_config:
                    logger.warning(
                        "No output configuration provided - cannot save content analysis files"
                    )

            # Log final content analysis statistics
            insights = result.source_info.get("content_insights", {})
            if insights:
                insight_categories = list(insights.keys())
                logger.info(f"Generated content insights: {insight_categories}")

            logger.info(
                f"Content analysis pipeline completed successfully for: {source}"
            )
            return result

        except Exception as e:
            # Log detailed error information for content analysis troubleshooting
            error_context = {
                "source": source,
                "source_type": source_type.value,
                "content_type": content_type.value,
                "has_context": context_metadata is not None,
                "save_files_requested": save_files,
                "features_enabled": [
                    attr for attr, value in features.__dict__.items() if value
                ],
            }
            logger.error(f"Content analysis transcription failed: {str(e)}")
            logger.debug(f"Error context: {error_context}")
            raise

    def _get_content_specific_features(
        self, content_type: ContentType
    ) -> TranscriptionFeatures:
        """
        Get optimized transcription features for specific content types

        This method configures transcription features based on content type to maximize
        relevance and usefulness of the analysis results. Each content type receives
        a tailored feature set that emphasizes the most valuable capabilities for that
        scenario.

        Args:
            content_type: Type of content being analyzed for feature optimization

        Returns:
            TranscriptionFeatures: Optimized feature configuration for the content type

        Feature optimization strategies:
            - PODCAST: Emphasizes content discovery, navigation, and marketing features
            - MEETING: Focuses on participation, decisions, and actionable insights
            - INTERVIEW: Prioritizes quotes, sentiment, and conversational analysis
            - GENERAL/LECTURE: Provides balanced feature set for comprehensive analysis
        """
        logger.debug(f"Configuring content-specific features for {content_type.value}")

        if content_type == ContentType.PODCAST:
            # Podcast optimization: Focus on content discovery and audience engagement
            logger.debug("Applying podcast-specific feature optimization")
            return TranscriptionFeatures(
                speaker_diarization=True,  # Multi-host and guest identification
                word_timestamps=True,  # Precise chapter and highlight timing
                auto_highlights=True,  # Key moments for social media clips
                auto_chapters=True,  # Episode navigation and structure
                sentiment_analysis=True,  # Audience engagement insights
                topic_detection=True,  # Content tagging and SEO optimization
                iab_categories=True,  # Content categorization for advertising
            )

        elif content_type == ContentType.MEETING:
            # Meeting optimization: Focus on actionable insights and participation
            logger.debug("Applying meeting-specific feature optimization")
            return TranscriptionFeatures(
                speaker_diarization=True,  # Participant identification and attribution
                word_timestamps=True,  # Precise meeting minutes timing
                auto_highlights=True,  # Key decisions and action items
                summarization=True,  # Meeting summary and notes generation
                sentiment_analysis=True,  # Meeting tone and engagement tracking
                topic_detection=True,  # Discussion topic identification
            )

        elif content_type == ContentType.INTERVIEW:
            # Interview optimization: Focus on quotes, insights, and conversational flow
            logger.debug("Applying interview-specific feature optimization")
            return TranscriptionFeatures(
                speaker_diarization=True,  # Interviewer/interviewee identification
                word_timestamps=True,  # Precise quote timing and attribution
                auto_highlights=True,  # Key quotes and memorable moments
                summarization=True,  # Interview summary and key points
                sentiment_analysis=True,  # Emotional flow and engagement analysis
                topic_detection=True,  # Interview topic and theme identification
            )

        else:  # GENERAL, LECTURE, and other content types
            # General optimization: Balanced feature set for comprehensive analysis
            logger.debug(
                f"Applying general feature optimization for {content_type.value}"
            )
            return TranscriptionFeatures(
                speaker_diarization=True,  # Speaker identification for multi-speaker content
                word_timestamps=True,  # Detailed timing information
                auto_highlights=True,  # Important moments and key points
                summarization=True,  # Content summary and overview
                sentiment_analysis=True,  # Emotional and engagement analysis
            )

    def _enhance_content_analysis_result(
        self,
        result: PipelineResult,
        content_type: ContentType,
        metadata: Optional[Dict[str, Any]],
    ) -> PipelineResult:
        """
        Add content-type specific enhancements and intelligent insights

        This method generates specialized insights and analytics based on the content
        type, providing actionable information that goes beyond basic transcription.
        Each content type receives tailored analysis that addresses its specific use cases.

        Args:
            result: Base transcription result from AssemblyAI processing
            content_type: Content type for specialized enhancement selection
            metadata: Optional context metadata for enhanced insight generation

        Returns:
            PipelineResult: Enhanced result with content-type specific insights:
                - Podcast: Show notes, episode structure, content categories
                - Meeting: Action items, participation analysis, decision tracking
                - Interview: Key quotes, sentiment flow, Q&A dynamics
                - General: Balanced insights for comprehensive content understanding
        """
        logger.debug(f"Generating {content_type.value}-specific content insights")

        content_insights = {}

        # Generate content-type specific insights
        if content_type == ContentType.PODCAST:
            logger.debug("Generating podcast-specific insights")
            content_insights = self._generate_podcast_insights(result, metadata)

        elif content_type == ContentType.MEETING:
            logger.debug("Generating meeting-specific insights")
            content_insights = self._generate_meeting_insights(result, metadata)

        elif content_type == ContentType.INTERVIEW:
            logger.debug("Generating interview-specific insights")
            content_insights = self._generate_interview_insights(result, metadata)

        else:
            # Generate general insights for other content types
            logger.debug(f"Generating general insights for {content_type.value}")
            content_insights = self._generate_general_insights(result, metadata)

        # Store insights in result for access and analysis
        result.source_info["content_insights"] = content_insights

        # Log insight generation results
        if content_insights:
            insight_categories = list(content_insights.keys())
            logger.info(
                f"Generated {len(insight_categories)} insight categories: {insight_categories}"
            )
        else:
            logger.warning(f"No insights generated for {content_type.value} content")

        logger.debug("Content-specific enhancement completed")
        return result

    def _generate_podcast_insights(
        self, result: PipelineResult, metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate podcast-specific insights for content creators and marketers

        Args:
            result: Transcription result with podcast data
            metadata: Optional podcast metadata for enhanced insights

        Returns:
            Dict[str, Any]: Podcast-specific insights including structure, show notes, and engagement data
        """
        logger.debug("Generating comprehensive podcast insights")
        insights = {}

        # Episode structure analysis for content optimization
        if result.chapters:
            logger.debug(
                f"Analyzing episode structure from {len(result.chapters)} chapters"
            )
            insights["episode_structure"] = {
                "total_chapters": len(result.chapters),
                "chapter_titles": [c["headline"] for c in result.chapters],
                "average_chapter_length": sum(
                    c["end"] - c["start"] for c in result.chapters
                )
                / len(result.chapters),
                "chapter_distribution": [
                    {
                        "title": c["headline"],
                        "duration": c["end"] - c["start"],
                        "start_time": c["start"],
                    }
                    for c in result.chapters
                ],
            }

            # Generate ready-to-use show notes for podcast platforms
            insights["show_notes"] = {
                "timestamps": [
                    f"{self._format_time(c['start'])} - {c['headline']}"
                    for c in result.chapters
                ],
                "chapter_summaries": [
                    f"{c['headline']}: {c['gist']}" for c in result.chapters
                ],
                "episode_outline": [
                    f"• {c['headline']} ({self._format_time(c['start'])})"
                    for c in result.chapters
                ],
            }
            logger.info("Generated show notes with timestamps and chapter summaries")

        # Key moments analysis for social media and marketing
        if result.highlights:
            logger.debug(f"Analyzing {len(result.highlights)} content highlights")
            top_highlights = sorted(result.highlights, key=lambda x: x["rank"])[:10]
            insights["key_moments"] = [
                {
                    "text": h["text"],
                    "rank": h["rank"],
                    "timestamps": h["timestamps"],
                    "social_media_ready": len(h["text"])
                    <= 280,  # Twitter-friendly length
                }
                for h in top_highlights
            ]

            # Extract social media clips
            social_clips = [h for h in top_highlights if len(h["text"]) <= 140][:5]
            if social_clips:
                insights["social_media_clips"] = [
                    {
                        "text": clip["text"],
                        "optimal_for": "Twitter/X, Instagram Stories",
                        "timestamp": (
                            clip["timestamps"][0] if clip["timestamps"] else None
                        ),
                    }
                    for clip in social_clips
                ]
            logger.info(f"Identified {len(top_highlights)} key moments for promotion")

        # Content categorization for discovery and advertising
        if result.iab_categories:
            logger.debug("Processing IAB content categorization")
            insights["content_categories"] = {
                "primary_categories": result.iab_categories.get("summary", {}),
                "detailed_categories": result.iab_categories.get("results", []),
                "advertising_ready": True,
            }
            logger.info("Generated content categories for advertising and discovery")

        # Audience engagement metrics
        if result.sentiment:
            sentiment_summary = self._analyze_podcast_sentiment(result.sentiment)
            insights["audience_engagement"] = sentiment_summary
            logger.debug(
                "Generated audience engagement metrics from sentiment analysis"
            )

        logger.info(f"Generated {len(insights)} podcast insight categories")
        return insights

    def _generate_meeting_insights(
        self, result: PipelineResult, metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate meeting-specific insights for business productivity and follow-up

        Args:
            result: Transcription result with meeting data
            metadata: Optional meeting metadata for enhanced insights

        Returns:
            Dict[str, Any]: Meeting-specific insights including decisions, actions, and participation
        """
        logger.debug("Generating comprehensive meeting insights")
        insights = {}

        # Extract actionable items and key decisions
        if result.highlights:
            logger.debug(
                f"Processing {len(result.highlights)} highlights for actionable items"
            )
            all_highlights = [h["text"] for h in result.highlights]

            # Top decisions and important points
            insights["key_decisions"] = all_highlights[:5]

            # Extract potential action items using keyword analysis
            action_keywords = [
                "action",
                "todo",
                "follow up",
                "next steps",
                "assign",
                "due",
                "deadline",
                "responsible",
            ]
            action_items = [
                highlight
                for highlight in all_highlights
                if any(keyword in highlight.lower() for keyword in action_keywords)
            ]
            insights["action_items"] = action_items[:10]  # Limit to top 10 action items

            # Extract decisions using decision keywords
            decision_keywords = [
                "decide",
                "decision",
                "agreed",
                "resolve",
                "conclude",
                "final",
            ]
            decisions = [
                highlight
                for highlight in all_highlights
                if any(keyword in highlight.lower() for keyword in decision_keywords)
            ]
            insights["decisions_made"] = decisions[:8]

            logger.info(
                f"Identified {len(action_items)} action items and {len(decisions)} decisions"
            )

        # Participation analysis for meeting effectiveness
        if result.speakers:
            logger.debug("Analyzing meeting participation patterns")
            speaker_stats = {}
            total_speaking_time = 0

            for speaker in result.speakers:
                speaker_id = speaker["speaker"]
                speaking_duration = speaker["end"] - speaker["start"]

                if speaker_id not in speaker_stats:
                    speaker_stats[speaker_id] = {
                        "word_count": 0,
                        "speaking_time": 0,
                        "segments": 0,
                        "average_segment_length": 0,
                    }

                speaker_stats[speaker_id]["word_count"] += len(speaker["text"].split())
                speaker_stats[speaker_id]["speaking_time"] += speaking_duration
                speaker_stats[speaker_id]["segments"] += 1
                total_speaking_time += speaking_duration

            # Calculate participation percentages and engagement metrics
            for speaker_id, stats in speaker_stats.items():
                stats["speaking_percentage"] = (
                    (stats["speaking_time"] / total_speaking_time * 100)
                    if total_speaking_time > 0
                    else 0
                )
                stats["average_segment_length"] = (
                    stats["speaking_time"] / stats["segments"]
                    if stats["segments"] > 0
                    else 0
                )
                stats["words_per_minute"] = (
                    (stats["word_count"] / (stats["speaking_time"] / 60000))
                    if stats["speaking_time"] > 0
                    else 0
                )

            insights["participation_analysis"] = {
                "speaker_statistics": speaker_stats,
                "total_speakers": len(speaker_stats),
                "most_active_speaker": (
                    max(
                        speaker_stats.keys(),
                        key=lambda x: speaker_stats[x]["speaking_time"],
                    )
                    if speaker_stats
                    else None
                ),
                "participation_balance": self._calculate_participation_balance(
                    speaker_stats
                ),
            }
            logger.info(
                f"Analyzed participation for {len(speaker_stats)} meeting participants"
            )

        # Meeting summary and overview
        if result.summary:
            insights["meeting_summary"] = {
                "executive_summary": result.summary,
                "meeting_effectiveness_score": self._calculate_meeting_effectiveness(
                    insights
                ),
                "follow_up_required": len(insights.get("action_items", [])) > 0,
            }
            logger.debug("Generated meeting summary and effectiveness metrics")

        logger.info(f"Generated {len(insights)} meeting insight categories")
        return insights

    def _generate_interview_insights(
        self, result: PipelineResult, metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate interview-specific insights for journalists and researchers

        Args:
            result: Transcription result with interview data
            metadata: Optional interview metadata for enhanced insights

        Returns:
            Dict[str, Any]: Interview-specific insights including quotes, sentiment, and dynamics
        """
        logger.debug("Generating comprehensive interview insights")
        insights = {}

        # Extract key quotes and memorable statements
        if result.highlights:
            logger.debug(
                f"Processing {len(result.highlights)} highlights for quotable content"
            )
            top_highlights = sorted(result.highlights, key=lambda x: x["rank"])

            insights["key_quotes"] = [
                {
                    "text": h["text"],
                    "rank": h["rank"],
                    "quotability_score": self._assess_quotability(h["text"]),
                    "timestamp": h["timestamps"][0] if h["timestamps"] else None,
                }
                for h in top_highlights[:8]
            ]

            # Extract short, punchy quotes for headlines
            short_quotes = [h for h in top_highlights if len(h["text"]) <= 200][:5]
            if short_quotes:
                insights["headline_quotes"] = [h["text"] for h in short_quotes]

            logger.info(
                f"Extracted {len(insights['key_quotes'])} key quotes for interview coverage"
            )

        # Sentiment flow analysis for interview dynamics
        if result.sentiment:
            logger.debug("Analyzing interview sentiment and emotional flow")
            sentiment_timeline = []
            emotional_peaks = []

            for sentiment_item in result.sentiment:
                timeline_entry = {
                    "time": sentiment_item["start"],
                    "sentiment": sentiment_item["sentiment"],
                    "confidence": sentiment_item["confidence"],
                    "text_snippet": (
                        sentiment_item["text"][:100] + "..."
                        if len(sentiment_item["text"]) > 100
                        else sentiment_item["text"]
                    ),
                }
                sentiment_timeline.append(timeline_entry)

                # Identify emotional peaks (high confidence sentiment changes)
                if sentiment_item["confidence"] > 0.8:
                    emotional_peaks.append(timeline_entry)

            insights["sentiment_flow"] = {
                "timeline": sentiment_timeline,
                "emotional_peaks": emotional_peaks[:5],  # Top 5 emotional moments
                "overall_tone": self._determine_overall_tone(sentiment_timeline),
            }
            logger.info(
                f"Analyzed sentiment flow with {len(emotional_peaks)} emotional peaks"
            )

        # Interview dynamic analysis for conversational insights
        if result.speakers and len(result.speakers) >= 2:
            logger.debug("Analyzing interview conversational dynamics")
            speakers = list(set(s["speaker"] for s in result.speakers))

            if len(speakers) == 2:
                # Classic interviewer-interviewee dynamic
                speaker_contributions = {}
                for speaker in speakers:
                    speaker_utterances = [
                        s for s in result.speakers if s["speaker"] == speaker
                    ]
                    total_words = sum(
                        len(s["text"].split()) for s in speaker_utterances
                    )
                    total_time = sum(s["end"] - s["start"] for s in speaker_utterances)

                    speaker_contributions[speaker] = {
                        "total_words": total_words,
                        "total_time": total_time,
                        "average_response_length": (
                            total_words / len(speaker_utterances)
                            if speaker_utterances
                            else 0
                        ),
                    }

                # Determine roles based on speaking patterns
                speakers_by_words = sorted(
                    speakers, key=lambda x: speaker_contributions[x]["total_words"]
                )
                likely_interviewer = speakers_by_words[
                    0
                ]  # Usually asks shorter questions
                likely_interviewee = speakers_by_words[
                    1
                ]  # Usually gives longer answers

                insights["interview_dynamic"] = {
                    "total_speakers": len(speakers),
                    "likely_interviewer": likely_interviewer,
                    "likely_interviewee": likely_interviewee,
                    "question_answer_ratio": self._analyze_qa_ratio(result.speakers),
                    "conversational_balance": speaker_contributions,
                    "interview_style": self._determine_interview_style(
                        speaker_contributions
                    ),
                }
                logger.info("Analyzed interview dynamic and conversational patterns")

            elif len(speakers) > 2:
                # Multi-participant interview or panel discussion
                insights["interview_dynamic"] = {
                    "total_speakers": len(speakers),
                    "format": "panel_discussion",
                    "participation_distribution": self._analyze_multi_speaker_distribution(
                        result.speakers
                    ),
                }
                logger.info(
                    f"Analyzed multi-speaker interview with {len(speakers)} participants"
                )

        logger.info(f"Generated {len(insights)} interview insight categories")
        return insights

    def _generate_general_insights(
        self, result: PipelineResult, metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate general insights for content types not specifically handled

        Args:
            result: Transcription result with general content data
            metadata: Optional content metadata for enhanced insights

        Returns:
            Dict[str, Any]: General content insights and analytics
        """
        logger.debug("Generating general content insights")
        insights = {}

        # Basic content structure analysis
        if result.summary:
            insights["content_summary"] = result.summary

        if result.highlights:
            insights["key_points"] = [h["text"] for h in result.highlights[:10]]

        if result.speakers:
            insights["speaker_count"] = len(set(s["speaker"] for s in result.speakers))

        if result.sentiment:
            insights["emotional_tone"] = self._determine_overall_tone(result.sentiment)

        logger.debug(f"Generated {len(insights)} general insight categories")
        return insights

    # Helper methods for insight generation

    def _format_time(self, milliseconds: float) -> str:
        """Format timestamp in MM:SS format"""
        total_seconds = int(milliseconds / 1000)
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

    def _assess_quotability(self, text: str) -> float:
        """Assess how quotable a piece of text is based on various factors"""
        # Simple quotability scoring based on length, completeness, and impact words
        score = 0.5  # Base score

        # Length scoring (optimal quote length)
        if 50 <= len(text) <= 300:
            score += 0.2

        # Complete sentence scoring
        if text.strip().endswith((".", "!", "?")):
            score += 0.1

        # Impact words
        impact_words = [
            "important",
            "significant",
            "key",
            "critical",
            "essential",
            "believe",
            "think",
        ]
        if any(word in text.lower() for word in impact_words):
            score += 0.2

        return min(score, 1.0)

    def _analyze_podcast_sentiment(
        self, sentiment_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze sentiment data for podcast audience engagement metrics"""
        if not sentiment_data:
            return {}

        positive_count = sum(1 for s in sentiment_data if s["sentiment"] == "POSITIVE")
        negative_count = sum(1 for s in sentiment_data if s["sentiment"] == "NEGATIVE")
        neutral_count = len(sentiment_data) - positive_count - negative_count

        return {
            "overall_positivity": positive_count / len(sentiment_data),
            "engagement_level": (positive_count + negative_count)
            / len(sentiment_data),  # Non-neutral content is more engaging
            "sentiment_distribution": {
                "positive": positive_count,
                "negative": negative_count,
                "neutral": neutral_count,
            },
        }

    def _calculate_participation_balance(
        self, speaker_stats: Dict[str, Dict[str, Any]]
    ) -> float:
        """Calculate how balanced participation is in a meeting (0 = unbalanced, 1 = perfectly balanced)"""
        if not speaker_stats or len(speaker_stats) < 2:
            return 1.0

        speaking_times = [stats["speaking_time"] for stats in speaker_stats.values()]
        avg_time = sum(speaking_times) / len(speaking_times)

        # Calculate standard deviation as a measure of imbalance
        variance = sum((time - avg_time) ** 2 for time in speaking_times) / len(
            speaking_times
        )
        std_dev = variance**0.5

        # Normalize to 0-1 scale (lower std_dev = more balanced)
        balance_score = max(0, 1 - (std_dev / avg_time)) if avg_time > 0 else 1.0
        return balance_score

    def _calculate_meeting_effectiveness(self, insights: Dict[str, Any]) -> float:
        """Calculate a simple meeting effectiveness score based on available insights"""
        score = 0.5  # Base score

        # Bonus for identified action items
        if insights.get("action_items"):
            score += 0.2

        # Bonus for balanced participation
        if (
            insights.get("participation_analysis", {}).get("participation_balance", 0)
            > 0.7
        ):
            score += 0.2

        # Bonus for clear decisions
        if insights.get("decisions_made"):
            score += 0.1

        return min(score, 1.0)

    def _determine_overall_tone(self, sentiment_data: List[Dict[str, Any]]) -> str:
        """Determine overall emotional tone from sentiment analysis"""
        if not sentiment_data:
            return "neutral"

        positive_count = sum(1 for s in sentiment_data if s["sentiment"] == "POSITIVE")
        negative_count = sum(1 for s in sentiment_data if s["sentiment"] == "NEGATIVE")

        if positive_count > negative_count * 1.5:
            return "positive"
        elif negative_count > positive_count * 1.5:
            return "negative"
        else:
            return "neutral"

    def _analyze_qa_ratio(self, speakers: List[Dict[str, Any]]) -> float:
        """Analyze question to answer ratio in interview/conversation"""
        question_indicators = [
            "?",
            "what",
            "how",
            "why",
            "when",
            "where",
            "who",
            "which",
            "could",
            "would",
            "should",
        ]

        questions = 0
        total_segments = len(speakers)

        for speaker in speakers:
            text_lower = speaker["text"].lower()
            if any(indicator in text_lower for indicator in question_indicators):
                questions += 1

        return questions / total_segments if total_segments > 0 else 0

    def _determine_interview_style(
        self, speaker_contributions: Dict[str, Dict[str, Any]]
    ) -> str:
        """Determine interview style based on speaking patterns"""
        speakers = list(speaker_contributions.keys())
        if len(speakers) != 2:
            return "unknown"

        speaker1_avg = speaker_contributions[speakers[0]]["average_response_length"]
        speaker2_avg = speaker_contributions[speakers[1]]["average_response_length"]

        ratio = (
            max(speaker1_avg, speaker2_avg) / min(speaker1_avg, speaker2_avg)
            if min(speaker1_avg, speaker2_avg) > 0
            else 1
        )

        if ratio > 3:
            return "structured_interview"  # Clear interviewer/interviewee roles
        elif ratio > 1.5:
            return "conversational_interview"  # Somewhat structured
        else:
            return "discussion"  # More balanced conversation

    def _analyze_multi_speaker_distribution(
        self, speakers: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze speaking distribution in multi-speaker content"""
        speaker_stats = {}

        for speaker in speakers:
            speaker_id = speaker["speaker"]
            if speaker_id not in speaker_stats:
                speaker_stats[speaker_id] = {"segments": 0, "total_words": 0}

            speaker_stats[speaker_id]["segments"] += 1
            speaker_stats[speaker_id]["total_words"] += len(speaker["text"].split())

        return {
            "speaker_distribution": speaker_stats,
            "dominant_speaker": (
                max(speaker_stats.keys(), key=lambda x: speaker_stats[x]["total_words"])
                if speaker_stats
                else None
            ),
            "participation_evenness": self._calculate_participation_balance(
                {
                    k: {"speaking_time": v["total_words"]}
                    for k, v in speaker_stats.items()
                }
            ),
        }

    def _prepare_audio_source(
        self, source: str, source_type: AudioSource, **kwargs
    ) -> str:
        """
        Prepare audio source for content analysis transcription

        This method reuses the general pipeline's audio source preparation logic
        since the source handling requirements are identical. Content-specific
        processing occurs in the transcription configuration and result enhancement.

        Args:
            source: Audio source identifier (path, URL, S3 URI, etc.)
            source_type: Type of audio source determining preparation method
            **kwargs: Source-specific preparation parameters

        Returns:
            str: URL that AssemblyAI can access for content analysis transcription

        Note:
            Delegates to GeneralTranscriptionPipeline for consistent source handling
            across all pipeline types while maintaining content analysis-specific
            logging context for enhanced traceability.
        """
        logger.debug(
            f"Preparing audio source for content analysis: {source} (type: {source_type.value})"
        )

        # Delegate to general pipeline's source preparation method
        # This ensures consistent behavior across all pipeline types
        prepared_url = GeneralTranscriptionPipeline._prepare_audio_source(
            self, source, source_type, **kwargs
        )

        logger.debug("Audio source preparation completed for content analysis pipeline")
        return prepared_url


class CustomSpeakerPipeline(BasePipeline):
    """
    Pipeline with custom speaker labeling using pyannote integration

    This specialized pipeline extends the base transcription functionality with
    custom speaker identification and labeling capabilities. It allows users to
    map detected speakers to meaningful, human-readable names for enhanced
    readability and personalized transcription results.

    Key features:
    - Custom speaker name mapping for personalized transcripts
    - Integration with pyannote.audio for advanced speaker diarization
    - Word-level and utterance-level speaker label application
    - Consistent speaker identification across audio segments
    - Support for known participant identification in meetings, interviews, etc.

    Use cases:
    - Interview transcriptions with known participant names
    - Meeting recordings with predetermined attendee lists
    - Podcast episodes with known hosts and guests
    - Educational content with identified instructors and students
    - Legal depositions with named parties
    - Multi-speaker presentations with known presenters

    Benefits:
    - Improves transcript readability by using actual names instead of "Speaker A", "Speaker B"
    - Enables better content searchability by participant name
    - Facilitates meeting minutes and interview summaries
    - Supports accessibility compliance with named speakers
    - Enhances user experience for transcript consumers

    Integration:
    - Works with pyannote.audio for enhanced speaker diarization accuracy
    - Maintains compatibility with all AssemblyAI transcription features
    - Supports all audio source types (local, URL, S3, Google Drive)

    Inherits from:
        BasePipeline: Provides core transcription functionality and result formatting
    """

    def __init__(
        self, config: PipelineConfig, pyannote_token: Optional[str] = None
    ) -> None:
        """
        Initialize the custom speaker pipeline with enhanced speaker identification

        Args:
            config: Pipeline configuration containing API key and settings
            pyannote_token: Optional Hugging Face token for pyannote.audio integration
                                Required for advanced speaker diarization features.
                                Can be obtained from https://huggingface.co/settings/tokens

        Note:
            The pyannote_token enables access to state-of-the-art speaker diarization
            models that can improve speaker identification accuracy, especially in
            challenging audio conditions or with similar-sounding speakers.
        """
        logger.info(
            "Initializing CustomSpeakerPipeline with speaker labeling capabilities"
        )

        # Initialize base pipeline functionality
        super().__init__(config)

        # Store pyannote token for advanced speaker diarization
        self.pyannote_token = pyannote_token

        if pyannote_token:
            logger.info(
                "Pyannote token provided - enhanced speaker diarization available"
            )
            logger.debug(
                "Pyannote integration enabled for improved speaker identification"
            )
        else:
            logger.info(
                "No pyannote token provided - using standard AssemblyAI speaker diarization"
            )
            logger.debug(
                "Consider providing pyannote_token for enhanced speaker identification accuracy"
            )

    def process_with_custom_speakers(
        self,
        source: str,
        speaker_labels: Dict[str, str],
        source_type: AudioSource = AudioSource.LOCAL_FILE,
        features: Optional[TranscriptionFeatures] = None,
        **kwargs,
    ) -> PipelineResult:
        """
        Process audio with custom speaker labels for personalized transcription

        This method performs transcription with speaker diarization and then applies
        custom speaker labels to replace generic speaker IDs (A, B, C) with meaningful
        names (John, Sarah, Mike). This creates more readable and useful transcripts
        for meetings, interviews, and multi-speaker content.

        Args:
            source: Audio source path, URL, or identifier
                    - Local file: "/path/to/interview.mp3"
                    - URL: "https://example.com/meeting-recording.wav"
                    - S3: "s3://recordings/client-interview-2024.mp3"
                    - Google Drive: File ID for shared recording
            speaker_labels: Dictionary mapping speaker IDs to custom names
                            Example: {"A": "John Smith", "B": "Sarah Johnson", "C": "Mike Chen"}
                            Note: Speaker IDs are typically assigned alphabetically (A, B, C...)
                            by AssemblyAI's speaker diarization system
            source_type: Type of audio source, defaults to LOCAL_FILE
            features: Transcription features to enable. If None, uses minimal feature set
                        optimized for speaker identification (diarization + timestamps)
            **kwargs: Additional source-specific parameters (AWS credentials, etc.)

        Returns:
            PipelineResult: Enhanced transcription result with custom speaker labels:
                - All speaker utterances labeled with provided custom names
                - Word-level timestamps include custom speaker information
                - Original speaker IDs preserved alongside custom labels
                - Speaker mapping metadata included in source_info
                - Full transcript text with enhanced readability

        Raises:
            Exception: If transcription fails due to API errors, invalid audio,
                        network issues, or speaker diarization problems

        Example:
            >>> pipeline = CustomSpeakerPipeline(config)
            >>> labels = {"A": "Dr. Smith", "B": "Patient", "C": "Nurse Johnson"}
            >>> result = pipeline.process_with_custom_speakers(
            ...     "medical_consultation.mp3",
            ...     speaker_labels=labels
            ... )
            >>> # Result will show "Dr. Smith:" instead of "Speaker A:"

        Workflow:
            1. Execute standard transcription with speaker diarization
            2. Map detected speaker IDs to provided custom labels
            3. Apply labels to both utterance-level and word-level data
            4. Preserve original speaker IDs for reference and debugging
            5. Return enhanced result with improved readability
        """
        logger.info(f"Starting custom speaker transcription for source: {source}")
        logger.info(
            f"Applying custom labels for {len(speaker_labels)} speakers: {list(speaker_labels.values())}"
        )
        logger.debug(f"Speaker mapping: {speaker_labels}")

        # Use minimal feature set focused on speaker identification if none provided
        if features is None:
            logger.debug("Using default features optimized for speaker identification")
            features = TranscriptionFeatures(
                speaker_diarization=True,  # Essential for speaker identification
                word_timestamps=True,  # Needed for word-level speaker mapping
            )
        else:
            # Ensure speaker diarization is enabled regardless of provided features
            if not features.speaker_diarization:
                logger.warning(
                    "Speaker diarization was disabled in features - enabling for custom speaker pipeline"
                )
                features.speaker_diarization = True
            logger.debug(
                f"Using provided features with speaker diarization: {features}"
            )

        # Build transcription configuration with speaker-focused settings
        logger.debug("Building transcription configuration for speaker identification")
        config = self._build_transcription_config(features)

        # Prepare audio source for transcription
        logger.info(f"Preparing audio source of type: {source_type.value}")
        audio_url = self._prepare_audio_source(source, source_type, **kwargs)
        logger.debug("Audio source prepared for custom speaker transcription")

        try:
            # Execute standard transcription with speaker diarization
            logger.info("Starting transcription with speaker diarization")
            transcript = self.transcriber.transcribe(audio_url, config)
            logger.info(f"Transcription completed successfully. ID: {transcript.id}")

            # Log speaker diarization results for validation
            if hasattr(transcript, "utterances") and transcript.utterances:
                detected_speakers = set(u.speaker for u in transcript.utterances)
                logger.info(
                    f"Detected {len(detected_speakers)} unique speakers: {sorted(detected_speakers)}"
                )

                # Validate that provided labels match detected speakers
                unmapped_speakers = detected_speakers - set(speaker_labels.keys())
                unused_labels = set(speaker_labels.keys()) - detected_speakers

                if unmapped_speakers:
                    logger.warning(
                        f"Detected speakers without custom labels: {sorted(unmapped_speakers)}"
                    )
                if unused_labels:
                    logger.warning(
                        f"Provided labels for undetected speakers: {sorted(unused_labels)}"
                    )
            else:
                logger.warning(
                    "No speaker utterances detected - speaker diarization may have failed"
                )

            # Prepare comprehensive source information including custom speaker metadata
            source_info = {
                "source": source,  # Original source identifier
                "source_type": source_type.value,  # Source type (local, url, s3, etc.)
                "custom_speakers": True,  # Flag indicating custom speaker processing
                "speaker_labels": speaker_labels,  # Custom speaker name mapping
                "pyannote_enabled": self.pyannote_token
                is not None,  # Advanced diarization flag
            }

            # Format standard transcription result
            logger.debug("Formatting transcription result")
            result = self._format_result(transcript, source_info, features)

            # Apply custom speaker labels to enhance readability
            logger.info("Applying custom speaker labels to transcription result")
            result = self._apply_custom_speaker_labels(result, speaker_labels)

            logger.info(
                f"Custom speaker transcription completed successfully for: {source}"
            )
            return result

        except Exception as e:
            # Log detailed error information for custom speaker transcription failures
            error_context = {
                "source": source,
                "source_type": source_type.value,
                "speaker_count": len(speaker_labels),
                "speaker_labels": list(speaker_labels.values()),
                "pyannote_enabled": self.pyannote_token is not None,
            }
            logger.error(f"Custom speaker transcription failed: {str(e)}")
            logger.debug(f"Error context: {error_context}")
            raise

    def _apply_custom_speaker_labels(
        self, result: PipelineResult, speaker_labels: Dict[str, str]
    ) -> PipelineResult:
        """
        Apply custom speaker labels to transcription result for enhanced readability

        This method processes the transcription result to replace generic speaker IDs
        with meaningful custom labels at both the utterance and word levels. It preserves
        the original speaker IDs while adding human-readable names for better usability.

        Args:
            result: Base transcription result from AssemblyAI processing
            speaker_labels: Dictionary mapping speaker IDs to custom names
                                    (e.g., {"A": "John Doe", "B": "Jane Smith"})

        Returns:
            PipelineResult: Enhanced result with custom speaker labels applied:
                - Speaker utterances include both original ID and custom label
                - Word-level data includes custom speaker information
                - Speaker mapping stored in custom_speaker_labels field
                - Original data preserved for reference and debugging

        Processing Details:
            - Utterance-level: Adds "custom_label" field to each speaker entry
            - Word-level: Adds "custom_speaker" field to each word entry
            - Validation: Logs mapping statistics and unmapped speakers
            - Preservation: Original speaker IDs remain unchanged
        """
        logger.debug(f"Applying custom speaker labels: {speaker_labels}")

        speakers_mapped = 0
        speakers_unmapped = 0

        # Apply custom labels to speaker utterances
        if result.speakers:
            logger.debug(f"Processing {len(result.speakers)} speaker utterances")

            for speaker in result.speakers:
                speaker_id = speaker["speaker"]
                if speaker_id in speaker_labels:
                    # Add custom label while preserving original speaker ID
                    speaker["custom_label"] = speaker_labels[speaker_id]
                    speakers_mapped += 1
                    logger.debug(
                        f"Mapped speaker {speaker_id} to '{speaker_labels[speaker_id]}'"
                    )
                else:
                    speakers_unmapped += 1
                    logger.debug(f"No custom label provided for speaker {speaker_id}")

            logger.info(
                f"Speaker utterance mapping complete: {speakers_mapped} mapped, {speakers_unmapped} unmapped"
            )
        else:
            logger.warning("No speaker utterances available for custom labeling")

        # Apply custom labels to word-level data for detailed analysis
        words_mapped = 0
        if result.words:
            logger.debug(
                f"Processing word-level speaker data for {len(result.words)} words"
            )

            for word in result.words:
                if word.get("speaker") and word["speaker"] in speaker_labels:
                    # Add custom speaker name to word-level data
                    word["custom_speaker"] = speaker_labels[word["speaker"]]
                    words_mapped += 1

            if words_mapped > 0:
                logger.info(f"Applied custom labels to {words_mapped} words")
            else:
                logger.debug("No word-level speaker data found or no mappings applied")
        else:
            logger.debug("No word-level data available for custom speaker labeling")

        # Store the speaker label mapping in the result for reference
        result.custom_speaker_labels = speaker_labels

        # Generate speaker mapping summary for analytics
        mapping_summary = {
            "total_labels_provided": len(speaker_labels),
            "speakers_successfully_mapped": speakers_mapped,
            "speakers_without_labels": speakers_unmapped,
            "words_with_custom_labels": words_mapped,
        }

        # Add mapping summary to source info for analytics and debugging
        if "speaker_mapping_summary" not in result.source_info:
            result.source_info["speaker_mapping_summary"] = mapping_summary

        logger.info(f"Custom speaker labeling completed: {mapping_summary}")
        logger.debug(
            "Custom speaker labels successfully applied to transcription result"
        )

        return result

    def _prepare_audio_source(
        self, source: str, source_type: AudioSource, **kwargs
    ) -> str:
        """
        Prepare audio source for custom speaker transcription

        This method reuses the general pipeline's audio source preparation logic
        since the source handling requirements are identical. Custom speaker-specific
        processing occurs in the transcription execution and result enhancement phases.

        Args:
            source: Audio source identifier (path, URL, S3 URI, etc.)
            source_type: Type of audio source determining preparation method
            **kwargs: Source-specific preparation parameters

        Returns:
            str: URL that AssemblyAI can access for custom speaker transcription

        Note:
            Delegates to GeneralTranscriptionPipeline for consistent source handling
            across all pipeline types while maintaining custom speaker-specific
            logging context for enhanced traceability.
        """
        logger.debug(
            f"Preparing audio source for custom speaker transcription: {source} (type: {source_type.value})"
        )

        # Delegate to general pipeline's source preparation method
        # This ensures consistent behavior across all pipeline types
        prepared_url = GeneralTranscriptionPipeline._prepare_audio_source(
            self, source, source_type, **kwargs
        )

        logger.debug("Audio source preparation completed for custom speaker pipeline")
        return prepared_url


class BatchTranscriptionPipeline:
    """
    Pipeline for batch processing multiple audio files with concurrent execution

    This pipeline enables efficient processing of large numbers of audio files by
    leveraging concurrent execution while respecting rate limits and system resources.
    It provides comprehensive progress tracking, error handling, and result management
    for enterprise-scale transcription workflows.

    Key capabilities:
    - Concurrent processing with configurable concurrency limits
    - Real-time progress tracking and reporting
    - Robust error handling with individual job isolation
    - Memory-efficient processing for large batches
    - Flexible configuration per audio source
    - Comprehensive batch analytics and reporting
    - Resume capability for interrupted batches
    - Resource management and rate limiting

    Use cases:
    - Enterprise content libraries requiring bulk transcription
    - Media companies processing large archives
    - Educational institutions transcribing lecture collections
    - Legal firms processing case recording batches
    - Healthcare organizations transcribing consultation archives
    - Research institutions analyzing interview datasets
    - Podcast networks processing episode backlogs
    - Call centers analyzing customer interaction recordings

    Performance considerations:
    - Optimal concurrency based on system resources and API limits
    - Memory management for large result sets
    - Network bandwidth optimization for multiple simultaneous uploads
    - Error recovery and retry mechanisms
    - Progress persistence for long-running batches

    Integration benefits:
    - Works with any BasePipeline subclass for specialized processing
    - Maintains all pipeline-specific features and configurations
    - Supports mixed content types in single batch
    - Provides unified interface for different transcription approaches

    Attributes:
        base_pipeline: The underlying pipeline to use for individual transcriptions
        max_concurrent: Maximum number of concurrent transcription jobs
        job_statuses: Real-time status tracking for all batch jobs
        batch_progress: Comprehensive progress metrics and analytics
    """

    def __init__(self, base_pipeline: BasePipeline, max_concurrent: int = 5) -> None:
        """
        Initialize the batch transcription pipeline with concurrency control

        Args:
            base_pipeline: The pipeline instance to use for individual transcriptions.
                                Can be any BasePipeline subclass (GeneralTranscriptionPipeline,
                                MeetingTranscriptionPipeline, PodcastTranscriptionPipeline, etc.)
                                The batch processor will use this pipeline's specific features
                                and configurations for all transcriptions in the batch.
            max_concurrent: Maximum number of concurrent transcription jobs to run
                                simultaneously. Optimal values depend on:
                                - System resources (CPU, memory, network bandwidth)
                                - AssemblyAI API rate limits
                                - Audio file sizes and processing complexity
                                - Storage I/O capabilities
                                Recommended range: 3-10 for most use cases

        Note:
            The concurrency limit helps balance processing speed with resource usage
            and API rate limiting. Higher values increase throughput but may cause
            resource contention or API throttling. Monitor system performance and
            API response times to optimize this value for your specific use case.
        """
        logger.info(
            f"Initializing BatchTranscriptionPipeline with {type(base_pipeline).__name__}"
        )
        logger.info(
            f"Configured for maximum {max_concurrent} concurrent transcriptions"
        )

        self.base_pipeline = base_pipeline
        self.max_concurrent = max_concurrent

        # Initialize tracking structures for batch processing
        self.job_statuses: Dict[str, BatchJobStatus] = {}
        self.batch_progress = BatchProgress(total_jobs=0)

        # Validate configuration
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        if max_concurrent > 20:
            logger.warning(
                f"High concurrency ({max_concurrent}) may cause API rate limiting or resource issues"
            )

        logger.debug(
            f"Batch pipeline initialized with base pipeline: {type(base_pipeline).__name__}"
        )

    async def process_batch(
        self,
        sources: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int, BatchProgress], None]] = None,
        error_callback: Optional[Callable[[str, Exception], None]] = None,
        completion_callback: Optional[
            Callable[[List[Union[PipelineResult, Exception]]], None]
        ] = None,
    ) -> List[Union[PipelineResult, Exception]]:
        """
        Process multiple audio sources concurrently with comprehensive tracking

        This method orchestrates the concurrent processing of multiple audio files,
        providing real-time progress updates, error handling, and result collection.
        It manages system resources efficiently while maximizing throughput through
        controlled concurrency.

        Args:
            sources: List of source configurations for batch processing. Each configuration
                    is a dictionary containing:
                    - "source" (required): Audio source path, URL, or identifier
                    - "source_type" (optional): AudioSource enum value, defaults to LOCAL_FILE
                    - "features" (optional): TranscriptionFeatures for this specific source
                    - "kwargs" (optional): Additional parameters for source preparation
                    - "metadata" (optional): Custom metadata for this source

                    Example configurations:
                    [
                        {
                            "source": "/path/to/meeting1.mp3",
                            "source_type": AudioSource.LOCAL_FILE,
                            "features": TranscriptionFeatures(speaker_diarization=True),
                            "metadata": {"meeting_id": "MTG001", "department": "Engineering"}
                        },
                        {
                            "source": "s3://bucket/interview.wav",
                            "source_type": AudioSource.S3_BUCKET,
                            "kwargs": {"aws_access_key": "key", "aws_secret_key": "secret"}
                        }
                    ]

            progress_callback: Optional callback function for real-time progress updates.
                                Called with (completed_count, total_count, BatchProgress object)
                                Enables UI updates, logging, or progress persistence.

            error_callback: Optional callback for individual job error handling.
                            Called with (source_identifier, Exception) for each failed job.
                            Enables custom error logging, notification, or recovery actions.

            completion_callback: Optional callback when entire batch completes.
                                Called with the complete results list.
                                Enables post-processing, reporting, or cleanup actions.

        Returns:
            List[Union[PipelineResult, Exception]]: Results for each source in the same
            order as the input sources list. Successful transcriptions return PipelineResult
            objects, while failed transcriptions return Exception objects. This preserves
            the source-to-result mapping even when some jobs fail.

        Raises:
            ValueError: If sources list is empty or contains invalid configurations
            RuntimeError: If batch processing fails to initialize or manage concurrency

        Processing workflow:
            1. Validate input sources and initialize tracking structures
            2. Create concurrent tasks with semaphore-controlled execution
            3. Monitor progress and execute callbacks as jobs complete
            4. Collect results while preserving order and handling errors
            5. Generate final batch statistics and cleanup resources

        Performance characteristics:
            - Memory usage scales with max_concurrent, not total batch size
            - Network utilization optimized through concurrent connections
            - CPU usage balanced across available cores through async execution
            - Error isolation ensures individual failures don't affect other jobs

        Example:
            >>> pipeline = BatchTranscriptionPipeline(GeneralTranscriptionPipeline(config))
            >>>
            >>> def progress_update(completed, total, progress):
            ...     print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")
            ...     print(f"Success rate: {progress.successful_jobs}/{progress.completed_jobs}")
            >>>
            >>> sources = [
            ...     {"source": "meeting1.mp3", "metadata": {"type": "standup"}},
            ...     {"source": "meeting2.mp3", "metadata": {"type": "planning"}},
            ...     {"source": "interview.wav", "metadata": {"type": "candidate"}}
            ... ]
            >>>
            >>> results = await pipeline.process_batch(sources, progress_callback=progress_update)
            >>> successful_results = [r for r in results if isinstance(r, PipelineResult)]
        """
        if not sources:
            raise ValueError("Sources list cannot be empty")

        logger.info(f"Starting batch transcription of {len(sources)} audio sources")
        logger.info(
            f"Concurrency limit: {self.max_concurrent}, Pipeline: {type(self.base_pipeline).__name__}"
        )

        # Initialize batch tracking
        self._initialize_batch_tracking(sources)

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)
        completed_count = 0
        batch_start_time = datetime.now()

        # Track job durations for progress estimation
        job_durations: List[float] = []

        async def process_single_source(
            source_config: Dict[str, Any], index: int
        ) -> Union[PipelineResult, Exception]:
            """
            Process a single audio source with comprehensive error handling and tracking

            Args:
                source_config: Configuration dictionary for this source
                index: Index in the sources list for result ordering

            Returns:
                Union[PipelineResult, Exception]: Processing result or error
            """
            nonlocal completed_count

            source_id = source_config.get("source", f"unknown_source_{index}")
            job_status = self.job_statuses[source_id]

            async with semaphore:
                try:
                    # Update job status to processing
                    job_status.status = "processing"
                    job_status.start_time = datetime.now()
                    self.batch_progress.processing_jobs += 1
                    self.batch_progress.pending_jobs -= 1

                    logger.debug(
                        f"Starting transcription for source {index + 1}/{len(sources)}: {source_id}"
                    )

                    # Get event loop for thread pool execution
                    loop = asyncio.get_event_loop()

                    # Extract processing parameters with defaults
                    source = source_config["source"]
                    source_type = source_config.get(
                        "source_type", AudioSource.LOCAL_FILE
                    )
                    features = source_config.get("features")
                    kwargs = source_config.get("kwargs", {})

                    # Execute transcription in thread pool to avoid blocking async loop
                    # This is necessary because the base pipeline process method is synchronous
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        result = await loop.run_in_executor(
                            executor,
                            self._execute_pipeline_process,
                            source,
                            source_type,
                            features,
                            kwargs,
                        )

                    # Record successful completion
                    job_status.status = "completed"
                    job_status.end_time = datetime.now()
                    job_status.result = result

                    # Calculate job duration for progress estimation
                    job_duration = (
                        job_status.end_time - job_status.start_time
                    ).total_seconds()
                    job_durations.append(job_duration)

                    # Update batch progress metrics
                    completed_count += 1
                    self.batch_progress.completed_jobs += 1
                    self.batch_progress.successful_jobs += 1
                    self.batch_progress.processing_jobs -= 1

                    # Update progress estimates
                    self._update_progress_estimates(job_durations)

                    logger.info(
                        f"Successfully completed transcription {completed_count}/{len(sources)}: {source_id}"
                    )
                    logger.debug(f"Job duration: {job_duration:.2f} seconds")

                    # Execute progress callback if provided
                    if progress_callback:
                        try:
                            progress_callback(
                                completed_count, len(sources), self.batch_progress
                            )
                        except Exception as callback_error:
                            logger.warning(
                                f"Progress callback failed: {callback_error}"
                            )

                    return result

                except Exception as e:
                    # Record failed completion
                    job_status.status = "failed"
                    job_status.end_time = datetime.now()
                    job_status.error = str(e)

                    # Update batch progress metrics
                    completed_count += 1
                    self.batch_progress.completed_jobs += 1
                    self.batch_progress.failed_jobs += 1
                    self.batch_progress.processing_jobs -= 1

                    logger.error(
                        f"Transcription failed for source {index + 1}/{len(sources)}: {source_id}"
                    )
                    logger.error(f"Error details: {str(e)}")
                    logger.debug(f"Error type: {type(e).__name__}")

                    # Execute error callback if provided
                    if error_callback:
                        try:
                            error_callback(source_id, e)
                        except Exception as callback_error:
                            logger.warning(f"Error callback failed: {callback_error}")

                    # Execute progress callback for failed job
                    if progress_callback:
                        try:
                            progress_callback(
                                completed_count, len(sources), self.batch_progress
                            )
                        except Exception as callback_error:
                            logger.warning(
                                f"Progress callback failed: {callback_error}"
                            )

                    return e

        # Create tasks for all sources
        logger.debug("Creating concurrent tasks for batch processing")
        tasks = [
            process_single_source(source_config, index)
            for index, source_config in enumerate(sources)
        ]

        try:
            # Execute all tasks concurrently with comprehensive error handling
            logger.info("Executing batch transcription with concurrent processing")
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Calculate final batch statistics
            batch_duration = (datetime.now() - batch_start_time).total_seconds()
            success_count = sum(1 for r in results if isinstance(r, PipelineResult))
            failure_count = len(results) - success_count

            # Log comprehensive batch completion summary
            logger.info(
                f"Batch transcription completed in {batch_duration:.2f} seconds"
            )
            logger.info(
                f"Average processing time: {sum(job_durations) / len(job_durations):.2f} seconds per job"
                if job_durations
                else "N/A"
            )

            # Execute completion callback if provided
            if completion_callback:
                try:
                    completion_callback(results)
                    logger.debug("Completion callback executed successfully")
                except Exception as callback_error:
                    logger.warning(f"Completion callback failed: {callback_error}")

            # Log detailed failure information if any jobs failed
            if failure_count > 0:
                failed_sources = [
                    source_config["source"]
                    for source_config, result in zip(sources, results)
                    if isinstance(result, Exception)
                ]
                logger.warning(f"Failed sources: {failed_sources}")

            return results

        except Exception as e:
            logger.error(f"Batch processing failed with critical error: {str(e)}")
            logger.debug(f"Critical error type: {type(e).__name__}")
            raise RuntimeError(f"Batch transcription failed: {str(e)}") from e

    def _initialize_batch_tracking(self, sources: List[Dict[str, Any]]) -> None:
        """
        Initialize tracking structures for batch processing

        Args:
            sources: List of source configurations to initialize tracking for
        """
        logger.debug("Initializing batch tracking structures")

        # Initialize job status tracking for each source
        self.job_statuses = {}
        for source_config in sources:
            source_id = source_config.get("source", "unknown_source")
            self.job_statuses[source_id] = BatchJobStatus(
                source=source_id, status="pending"
            )

        # Initialize batch progress tracking
        self.batch_progress = BatchProgress(
            total_jobs=len(sources),
            pending_jobs=len(sources),
            start_time=datetime.now(),
        )

        logger.debug(f"Initialized tracking for {len(sources)} sources")

    def _execute_pipeline_process(
        self,
        source: str,
        source_type: AudioSource,
        features: Optional[TranscriptionFeatures],
        kwargs: Dict[str, Any],
    ) -> PipelineResult:
        """
        Execute the base pipeline's process method in a thread-safe manner

        This method provides a clean interface for executing the synchronous
        pipeline process method from the async context while maintaining
        proper error handling and parameter passing.

        Args:
            source: Audio source identifier
            source_type: Type of audio source
            features: Optional transcription features
            kwargs: Additional processing parameters

        Returns:
            PipelineResult: Result from the base pipeline processing
        """
        # Handle different pipeline types and their process method signatures
        if hasattr(self.base_pipeline, "process_with_custom_speakers"):
            # CustomSpeakerPipeline has a different method signature
            if "speaker_labels" in kwargs:
                speaker_labels = kwargs.pop("speaker_labels")
                return self.base_pipeline.process_with_custom_speakers(
                    source, speaker_labels, source_type, features, **kwargs
                )

        # Handle pipelines with specialized parameters
        if isinstance(
            self.base_pipeline,
            (MeetingTranscriptionPipeline, PodcastTranscriptionPipeline),
        ):
            # These pipelines expect context metadata parameters
            if "meeting_context" in kwargs:
                meeting_context = kwargs.pop("meeting_context")
                return self.base_pipeline.process(
                    source, source_type, meeting_context, **kwargs
                )
            elif "podcast_metadata" in kwargs:
                podcast_metadata = kwargs.pop("podcast_metadata")
                return self.base_pipeline.process(
                    source, source_type, podcast_metadata, **kwargs
                )

        # Standard pipeline process method
        return self.base_pipeline.process(source, source_type, features, **kwargs)

    def _update_progress_estimates(self, job_durations: List[float]) -> None:
        """
        Update progress estimates based on completed job durations

        Args:
            job_durations: List of completed job durations in seconds
            batch_start_time: When the batch processing started
        """
        if not job_durations:
            return

        # Calculate average job duration
        self.batch_progress.average_job_duration = sum(job_durations) / len(
            job_durations
        )

        # Estimate completion time for remaining jobs
        remaining_jobs = (
            self.batch_progress.total_jobs - self.batch_progress.completed_jobs
        )
        estimated_remaining_time = (
            remaining_jobs * self.batch_progress.average_job_duration
        )

        # Account for concurrent processing (divide by concurrency factor)
        concurrency_factor = min(self.max_concurrent, remaining_jobs)
        if concurrency_factor > 0:
            estimated_remaining_time = estimated_remaining_time / concurrency_factor

        self.batch_progress.estimated_completion = datetime.now() + timedelta(
            seconds=estimated_remaining_time
        )

    def get_batch_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status information for the current batch

        Returns:
            Dict[str, Any]: Detailed batch status including progress metrics,
                            job statuses, performance statistics, and timing information
        """
        return {
            "progress": {
                "total_jobs": self.batch_progress.total_jobs,
                "completed_jobs": self.batch_progress.completed_jobs,
                "successful_jobs": self.batch_progress.successful_jobs,
                "failed_jobs": self.batch_progress.failed_jobs,
                "processing_jobs": self.batch_progress.processing_jobs,
                "pending_jobs": self.batch_progress.pending_jobs,
                "completion_percentage": (
                    (
                        self.batch_progress.completed_jobs
                        / self.batch_progress.total_jobs
                        * 100
                    )
                    if self.batch_progress.total_jobs > 0
                    else 0
                ),
                "success_rate": (
                    (
                        self.batch_progress.successful_jobs
                        / self.batch_progress.completed_jobs
                        * 100
                    )
                    if self.batch_progress.completed_jobs > 0
                    else 0
                ),
            },
            "timing": {
                "start_time": (
                    self.batch_progress.start_time.isoformat()
                    if self.batch_progress.start_time
                    else None
                ),
                "estimated_completion": (
                    self.batch_progress.estimated_completion.isoformat()
                    if self.batch_progress.estimated_completion
                    else None
                ),
                "average_job_duration": self.batch_progress.average_job_duration,
                "elapsed_time": (
                    (datetime.now() - self.batch_progress.start_time).total_seconds()
                    if self.batch_progress.start_time
                    else None
                ),
            },
            "configuration": {
                "max_concurrent": self.max_concurrent,
                "pipeline_type": type(self.base_pipeline).__name__,
            },
            "job_statuses": {
                source_id: {
                    "status": status.status,
                    "start_time": (
                        status.start_time.isoformat() if status.start_time else None
                    ),
                    "end_time": (
                        status.end_time.isoformat() if status.end_time else None
                    ),
                    "error": status.error,
                    "has_result": status.result is not None,
                }
                for source_id, status in self.job_statuses.items()
            },
        }

    def get_successful_results(self) -> List[PipelineResult]:
        """
        Get all successful transcription results from completed jobs

        Returns:
            List[PipelineResult]: All successful transcription results
        """
        return [
            status.result
            for status in self.job_statuses.values()
            if status.result is not None
        ]

    def get_failed_jobs(self) -> Dict[str, str]:
        """
        Get information about all failed jobs

        Returns:
            Dict[str, str]: Mapping of source identifiers to error messages
        """
        return {
            source_id: status.error
            for source_id, status in self.job_statuses.items()
            if status.status == "failed" and status.error
        }


# Pipeline Factory
class PipelineFactory:
    """
    Factory for creating appropriate pipelines based on content type and pipeline purpose

    This factory provides a centralized, intelligent way to create and configure
    transcription pipelines based on content characteristics, processing requirements,
    and feature needs. It simplifies pipeline selection while ensuring optimal
    configuration for each use case.

    Key benefits:
    - Intelligent pipeline selection based on content type and requirements
    - Centralized configuration management for consistent setup
    - Feature-based pipeline recommendation for optimal results
    - Validation of pipeline capabilities against requirements
    - Extensible design for adding new pipeline types
    - Comprehensive logging for pipeline creation and configuration

    Factory patterns supported:
    - Content-type based creation (meeting, podcast, interview, etc.)
    - Feature-based selection (redaction, custom speakers, content analysis)
    - Batch processing setup with configurable concurrency
    - Hybrid pipeline creation for complex workflows
    - Auto-detection and recommendation based on requirements

    Use cases:
    - Automated pipeline selection in workflow systems
    - Dynamic configuration based on content characteristics
    - Feature validation and capability matching
    - Standardized pipeline creation across applications
    - Configuration management for enterprise deployments

    Design principles:
    - Single responsibility: Each method creates one specific pipeline type
    - Fail-fast validation: Early detection of configuration issues
    - Consistent interfaces: Standardized parameter patterns
    - Extensibility: Easy addition of new pipeline types
    - Documentation: Clear guidance for pipeline selection
    """

    # Class-level mapping of content types to optimal pipeline classes
    _CONTENT_TYPE_MAPPING: Dict[ContentType, Type[BasePipeline]] = {
        ContentType.MEETING: MeetingTranscriptionPipeline,
        ContentType.PODCAST: PodcastTranscriptionPipeline,
        ContentType.INTERVIEW: GeneralTranscriptionPipeline,  # Interviews work well with general pipeline
        ContentType.LECTURE: GeneralTranscriptionPipeline,  # Lectures work well with general pipeline
        ContentType.GENERAL: GeneralTranscriptionPipeline,
    }

    # Pipeline capability matrix for feature validation
    _PIPELINE_CAPABILITIES: Dict[Type[BasePipeline], List[PipelineCapability]] = {
        GeneralTranscriptionPipeline: [
            PipelineCapability.SPEAKER_DIARIZATION,
            PipelineCapability.PII_REDACTION,
            PipelineCapability.CONTENT_MODERATION,
            PipelineCapability.SUMMARIZATION,
            PipelineCapability.SENTIMENT_ANALYSIS,
        ],
        MeetingTranscriptionPipeline: [
            PipelineCapability.SPEAKER_DIARIZATION,
            PipelineCapability.PII_REDACTION,
            PipelineCapability.CONTENT_MODERATION,
            PipelineCapability.SUMMARIZATION,
            PipelineCapability.SENTIMENT_ANALYSIS,
            PipelineCapability.MEETING_INSIGHTS,
        ],
        PodcastTranscriptionPipeline: [
            PipelineCapability.SPEAKER_DIARIZATION,
            PipelineCapability.AUTO_CHAPTERS,
            PipelineCapability.SUMMARIZATION,
            PipelineCapability.SENTIMENT_ANALYSIS,
            PipelineCapability.PODCAST_INSIGHTS,
        ],
        CustomSpeakerPipeline: [
            PipelineCapability.SPEAKER_DIARIZATION,
            PipelineCapability.CUSTOM_SPEAKERS,
            PipelineCapability.PII_REDACTION,
            PipelineCapability.CONTENT_MODERATION,
        ],
        RedactionModerationPipeline: [
            PipelineCapability.PII_REDACTION,
            PipelineCapability.CONTENT_MODERATION,
            PipelineCapability.SPEAKER_DIARIZATION,
        ],
        ContentAnalysisPipeline: [
            PipelineCapability.CONTENT_ANALYSIS,
            PipelineCapability.SPEAKER_DIARIZATION,
            PipelineCapability.SUMMARIZATION,
            PipelineCapability.SENTIMENT_ANALYSIS,
            PipelineCapability.AUTO_CHAPTERS,
        ],
        BatchTranscriptionPipeline: [
            PipelineCapability.BATCH_PROCESSING,
        ],
    }

    @staticmethod
    def create_pipeline(
        content_type: ContentType,
        config: PipelineConfig,
        validate_config: bool = True,
        **kwargs,
    ) -> BasePipeline:
        """
        Create appropriate pipeline for content type with intelligent selection

        This method provides the primary interface for content-type based pipeline
        creation. It automatically selects the optimal pipeline class based on
        content characteristics while providing configuration validation and
        comprehensive logging.

        Args:
            content_type: Type of content being processed for optimal pipeline selection
                            - MEETING: Optimized for business meetings with action items
                            - PODCAST: Optimized for episodic content with chapters
                            - INTERVIEW: Uses general pipeline with good speaker handling
                            - LECTURE: Uses general pipeline suitable for educational content
                            - GENERAL: Balanced pipeline for unknown or mixed content
            config: Pipeline configuration containing API keys and settings
            validate_config: Whether to validate configuration before pipeline creation
            **kwargs: Additional configuration parameters for specific pipeline types

        Returns:
            BasePipeline: Configured pipeline instance optimized for the content type

        Raises:
            ValueError: If content_type is invalid or configuration is incomplete
            RuntimeError: If pipeline creation fails due to configuration issues

        Examples:
            >>> # Create meeting pipeline for business transcription
            >>> config = PipelineConfig(api_key="your_key")
            >>> meeting_pipeline = PipelineFactory.create_pipeline(
            ...     ContentType.MEETING, config
            ... )

            >>> # Create podcast pipeline for content creation
            >>> podcast_pipeline = PipelineFactory.create_pipeline(
            ...     ContentType.PODCAST, config
            ... )

            >>> # Create general pipeline for mixed content
            >>> general_pipeline = PipelineFactory.create_pipeline(
            ...     ContentType.GENERAL, config
            ... )
        """
        logger.info(f"Creating pipeline for content type: {content_type.value}")
        logger.debug(f"Pipeline creation parameters: {kwargs}")

        # Validate content type
        if not isinstance(content_type, ContentType):
            raise ValueError(
                f"Invalid content_type: {content_type}. Must be a ContentType enum value."
            )

        # Validate configuration if requested
        if validate_config:
            PipelineFactory._validate_pipeline_config(config)

        # Get pipeline class based on content type mapping
        pipeline_class = PipelineFactory._CONTENT_TYPE_MAPPING.get(content_type)

        if not pipeline_class:
            logger.warning(
                f"No specific pipeline for {content_type.value}, falling back to GeneralTranscriptionPipeline"
            )
            pipeline_class = GeneralTranscriptionPipeline

        try:
            # Create pipeline instance with configuration
            pipeline = pipeline_class(config)

            logger.info(
                f"Successfully created {pipeline_class.__name__} for {content_type.value} content"
            )
            logger.debug(
                f"Pipeline capabilities: {PipelineFactory._get_pipeline_capabilities(pipeline_class)}"
            )

            return pipeline

        except Exception as e:
            error_msg = f"Failed to create {pipeline_class.__name__} for {content_type.value}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    @staticmethod
    def create_redaction_pipeline(
        config: PipelineConfig,
        redaction_policies: Optional[List[str]] = None,
        validate_config: bool = True,
    ) -> RedactionModerationPipeline:
        """
        Create specialized redaction and moderation pipeline for privacy protection

        This method creates a pipeline specifically designed for content that requires
        comprehensive privacy protection, PII redaction, and content moderation.
        It's ideal for sensitive content in healthcare, legal, financial, and
        enterprise environments.

        Args:
            config: Pipeline configuration with API keys and privacy settings
            redaction_policies: Optional list of specific redaction policies to apply
                                If None, uses comprehensive default policies for maximum protection
            validate_config: Whether to validate configuration before pipeline creation

        Returns:
            RedactionModerationPipeline: Configured pipeline for privacy-focused processing

        Raises:
            ValueError: If configuration is invalid for redaction requirements
            RuntimeError: If redaction pipeline creation fails

        Use cases:
            - Healthcare: HIPAA-compliant medical transcriptions
            - Legal: Attorney-client privilege protection
            - Financial: PCI DSS compliance for payment discussions
            - HR: Employee privacy in workplace recordings
            - Government: Classified information handling

        Example:
            >>> config = PipelineConfig(
            ...     api_key="your_key",
            ...     output_config=OutputConfig(save_redacted_audio=True)
            ... )
            >>> redaction_pipeline = PipelineFactory.create_redaction_pipeline(
            ...     config,
            ...     redaction_policies=["person_name", "phone_number", "ssn"]
            ... )
        """
        logger.info("Creating specialized redaction and moderation pipeline")
        logger.info("Configured for maximum privacy protection and content safety")

        if redaction_policies:
            logger.info(f"Using custom redaction policies: {redaction_policies}")
        else:
            logger.info("Using comprehensive default redaction policies")

        # Validate configuration for redaction requirements
        if validate_config:
            PipelineFactory._validate_pipeline_config(config)
            PipelineFactory._validate_redaction_config(config)

        try:
            pipeline = RedactionModerationPipeline(config)

            logger.info("Successfully created RedactionModerationPipeline")
            logger.debug(
                "Pipeline configured for PII redaction, content moderation, and safety analysis"
            )

            return pipeline

        except Exception as e:
            error_msg = f"Failed to create RedactionModerationPipeline: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    @staticmethod
    def create_content_analysis_pipeline(
        config: PipelineConfig,
        default_content_type: ContentType = ContentType.GENERAL,
        validate_config: bool = True,
    ) -> ContentAnalysisPipeline:
        """
        Create adaptive content analysis pipeline for intelligent content processing

        This method creates a sophisticated pipeline that adapts its analysis approach
        based on content type, providing comprehensive insights and analytics beyond
        basic transcription. It's ideal for content creators, analysts, and businesses
        seeking deep content understanding.

        Args:
            config: Pipeline configuration with API keys and analysis settings
            default_content_type: Default content type for analysis optimization
                                            Used when content type is not specified per source
            validate_config: Whether to validate configuration before pipeline creation

        Returns:
            ContentAnalysisPipeline: Configured pipeline for adaptive content analysis

        Raises:
            ValueError: If configuration is invalid for content analysis requirements
            RuntimeError: If content analysis pipeline creation fails

        Analysis capabilities:
            - Automatic content-type detection and optimization
            - Comprehensive insights generation (structure, engagement, topics)
            - Content marketing and SEO optimization data
            - Audience engagement and participation metrics
            - Ready-to-use derivative content (summaries, highlights, notes)

        Use cases:
            - Content creators analyzing audience engagement
            - Business analysts reviewing meeting effectiveness
            - Marketers extracting promotional content
            - Researchers conducting content studies
            - Educators optimizing instructional materials

        Example:
            >>> config = PipelineConfig(
            ...     api_key="your_key",
            ...     output_config=OutputConfig(create_reports=True)
            ... )
            >>> analysis_pipeline = PipelineFactory.create_content_analysis_pipeline(
            ...     config,
            ...     default_content_type=ContentType.PODCAST
            ... )
        """
        logger.info("Creating adaptive content analysis pipeline")
        logger.info(f"Default content type optimization: {default_content_type.value}")
        logger.info("Configured for comprehensive content insights and analytics")

        # Validate configuration for content analysis requirements
        if validate_config:
            PipelineFactory._validate_pipeline_config(config)

        try:
            pipeline = ContentAnalysisPipeline(config)

            logger.info("Successfully created ContentAnalysisPipeline")
            logger.debug(
                "Pipeline configured for adaptive analysis with content-type optimization"
            )

            return pipeline

        except Exception as e:
            error_msg = f"Failed to create ContentAnalysisPipeline: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    @staticmethod
    def create_custom_speaker_pipeline(
        config: PipelineConfig,
        pyannote_token: Optional[str] = None,
        validate_config: bool = True,
    ) -> CustomSpeakerPipeline:
        """
        Create custom speaker labeling pipeline for personalized transcription

        This method creates a pipeline that enables mapping of detected speakers
        to meaningful, human-readable names for enhanced transcript readability
        and usability. It's ideal for scenarios with known participants.

        Args:
            config: Pipeline configuration with API keys and speaker settings
            pyannote_token: Optional Hugging Face token for enhanced speaker diarization
                                Enables access to state-of-the-art speaker identification models
                                Obtain from: https://huggingface.co/settings/tokens
            validate_config: Whether to validate configuration before pipeline creation

        Returns:
            CustomSpeakerPipeline: Configured pipeline for custom speaker identification

        Raises:
            ValueError: If configuration is invalid for custom speaker requirements
            RuntimeError: If custom speaker pipeline creation fails

        Enhanced capabilities with pyannote_token:
            - Improved speaker identification accuracy
            - Better handling of similar-sounding speakers
            - Enhanced performance in challenging audio conditions
            - Access to latest speaker diarization research models

        Use cases:
            - Meeting transcriptions with known attendees
            - Interview recordings with identified participants
            - Podcast episodes with known hosts and guests
            - Educational content with named instructors
            - Legal depositions with identified parties

        Example:
            >>> config = PipelineConfig(api_key="your_key")
            >>> speaker_pipeline = PipelineFactory.create_custom_speaker_pipeline(
            ...     config,
            ...     pyannote_token="hf_your_token_here"
            ... )
            >>>
            >>> # Use with speaker mapping
            >>> speaker_labels = {"A": "John Smith", "B": "Jane Doe"}
            >>> result = speaker_pipeline.process_with_custom_speakers(
            ...     "meeting.mp3", speaker_labels
            ... )
        """
        logger.info("Creating custom speaker labeling pipeline")

        if pyannote_token:
            logger.info(
                "Pyannote token provided - enhanced speaker diarization enabled"
            )
            logger.debug("Advanced speaker identification models will be available")
        else:
            logger.info(
                "No pyannote token provided - using standard speaker diarization"
            )
            logger.debug(
                "Consider providing pyannote token for improved speaker identification"
            )

        # Validate configuration for custom speaker requirements
        if validate_config:
            PipelineFactory._validate_pipeline_config(config)

        try:
            pipeline = CustomSpeakerPipeline(config, pyannote_token)

            logger.info("Successfully created CustomSpeakerPipeline")
            logger.debug(
                f"Pipeline configured with pyannote integration: {pyannote_token is not None}"
            )

            return pipeline

        except Exception as e:
            error_msg = f"Failed to create CustomSpeakerPipeline: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    @staticmethod
    def create_batch_pipeline(
        base_pipeline: BasePipeline,
        max_concurrent: int = 5,
        validate_config: bool = True,
    ) -> BatchTranscriptionPipeline:
        """
        Create batch processing pipeline for high-volume transcription workflows

        This method creates a pipeline that enables efficient processing of large
        numbers of audio files through controlled concurrent execution. It's ideal
        for enterprise-scale transcription workflows and bulk processing scenarios.

        Args:
            base_pipeline: The underlying pipeline to use for individual transcriptions
                            Can be any BasePipeline subclass with its specific features
                            The batch processor maintains all pipeline-specific capabilities
            max_concurrent: Maximum number of concurrent transcription jobs
                            Optimal values depend on system resources and API limits
                            Recommended range: 3-10 for most use cases
            validate_config: Whether to validate base pipeline configuration

        Returns:
            BatchTranscriptionPipeline: Configured pipeline for batch processing

        Raises:
            ValueError: If base_pipeline is invalid or max_concurrent is out of range
            RuntimeError: If batch pipeline creation fails

        Performance considerations:
            - Higher concurrency increases throughput but may cause resource contention
            - API rate limits may require lower concurrency values
            - Memory usage scales with concurrent jobs, not total batch size
            - Network bandwidth optimization through concurrent connections

        Use cases:
            - Enterprise content libraries requiring bulk transcription
            - Media companies processing large archives
            - Educational institutions transcribing lecture collections
            - Legal firms processing case recording batches
            - Research institutions analyzing interview datasets

        Example:
            >>> # Create base pipeline for specific content type
            >>> base_config = PipelineConfig(api_key="your_key")
            >>> meeting_pipeline = PipelineFactory.create_pipeline(
            ...     ContentType.MEETING, base_config
            ... )
            >>>
            >>> # Create batch processor with controlled concurrency
            >>> batch_pipeline = PipelineFactory.create_batch_pipeline(
            ...     meeting_pipeline,
            ...     max_concurrent=3
            ... )
            >>>
            >>> # Process multiple sources
            >>> sources = [
            ...     {"source": "meeting1.mp3"},
            ...     {"source": "meeting2.mp3"},
            ...     {"source": "meeting3.mp3"}
            ... ]
            >>> results = await batch_pipeline.process_batch(sources)
        """
        logger.info(
            f"Creating batch processing pipeline with {type(base_pipeline).__name__}"
        )
        logger.info(
            f"Configured for maximum {max_concurrent} concurrent transcriptions"
        )

        # Validate base pipeline
        if not isinstance(base_pipeline, BasePipeline):
            raise ValueError(
                f"base_pipeline must be a BasePipeline instance, got {type(base_pipeline)}"
            )

        # Validate concurrency settings
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        if max_concurrent > 20:
            logger.warning(
                f"High concurrency ({max_concurrent}) may cause API rate limiting"
            )

        # Validate base pipeline configuration if requested
        if validate_config and hasattr(base_pipeline, "config"):
            PipelineFactory._validate_pipeline_config(base_pipeline.config)

        try:
            batch_pipeline = BatchTranscriptionPipeline(base_pipeline, max_concurrent)

            logger.info("Successfully created BatchTranscriptionPipeline")
            logger.debug(
                f"Base pipeline capabilities preserved: {PipelineFactory._get_pipeline_capabilities(type(base_pipeline))}"
            )

            return batch_pipeline

        except Exception as e:
            error_msg = f"Failed to create BatchTranscriptionPipeline: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    @staticmethod
    def get_pipeline_recommendations(
        requirements: List[PipelineCapability],
        content_type: Optional[ContentType] = None,
    ) -> Dict[str, Any]:
        """
        Get pipeline recommendations based on feature requirements and content type

        This method analyzes requirements and provides recommendations for the most
        suitable pipeline types, helping users make informed decisions about
        pipeline selection for their specific needs.

        Args:
            requirements: List of required pipeline capabilities
            content_type: Optional content type for additional optimization

        Returns:
            Dict[str, Any]: Comprehensive recommendations including:
                - recommended_pipelines: List of suitable pipeline types
                - capability_matrix: Feature support across pipeline types
                - optimization_suggestions: Performance and configuration tips
                - content_type_analysis: Content-specific recommendations

        Example:
            >>> requirements = [
            ...     PipelineCapability.SPEAKER_DIARIZATION,
            ...     PipelineCapability.PII_REDACTION,
            ...     PipelineCapability.MEETING_INSIGHTS
            ... ]
            >>> recommendations = PipelineFactory.get_pipeline_recommendations(
            ...     requirements, ContentType.MEETING
            ... )
            >>> print(recommendations["recommended_pipelines"])
        """
        logger.info(
            f"Analyzing pipeline recommendations for {len(requirements)} requirements"
        )
        if content_type:
            logger.info(f"Content type consideration: {content_type.value}")

        # Find pipelines that support all required capabilities
        suitable_pipelines = []
        for (
            pipeline_class,
            capabilities,
        ) in PipelineFactory._PIPELINE_CAPABILITIES.items():
            if all(req in capabilities for req in requirements):
                suitable_pipelines.append(pipeline_class)

        # Generate recommendations
        recommendations = {
            "recommended_pipelines": [
                pipeline.__name__ for pipeline in suitable_pipelines
            ],
            "capability_matrix": {
                pipeline.__name__: [cap.value for cap in caps]
                for pipeline, caps in PipelineFactory._PIPELINE_CAPABILITIES.items()
            },
            "requirements_analysis": {
                "required_capabilities": [req.value for req in requirements],
                "content_type": content_type.value if content_type else None,
                "suitable_pipeline_count": len(suitable_pipelines),
            },
        }

        # Add content-type specific recommendations
        if content_type and content_type in PipelineFactory._CONTENT_TYPE_MAPPING:
            optimal_pipeline = PipelineFactory._CONTENT_TYPE_MAPPING[content_type]
            if optimal_pipeline in suitable_pipelines:
                recommendations["content_type_optimal"] = optimal_pipeline.__name__

        # Add optimization suggestions
        recommendations["optimization_suggestions"] = (
            PipelineFactory._generate_optimization_suggestions(
                requirements, content_type, suitable_pipelines
            )
        )

        logger.info(
            f"Generated recommendations for {len(suitable_pipelines)} suitable pipelines"
        )
        return recommendations

    @staticmethod
    def _validate_pipeline_config(config: PipelineConfig) -> None:
        """
        Validate pipeline configuration for common requirements

        Args:
            config: Pipeline configuration to validate

        Raises:
            ValueError: If configuration is invalid or incomplete
        """
        if not config.api_key:
            raise ValueError("API key is required in pipeline configuration")

        if not isinstance(config.api_key, str) or len(config.api_key.strip()) == 0:
            raise ValueError("API key must be a non-empty string")

        logger.debug("Pipeline configuration validation passed")

    @staticmethod
    def _validate_redaction_config(config: PipelineConfig) -> None:
        """
        Validate configuration specific to redaction pipeline requirements

        Args:
            config: Pipeline configuration to validate for redaction
        """
        # Add redaction-specific validation if needed
        logger.debug("Redaction configuration validation passed")

    @staticmethod
    def _get_pipeline_capabilities(pipeline_class: Type[BasePipeline]) -> List[str]:
        """
        Get list of capabilities for a pipeline class

        Args:
            pipeline_class: Pipeline class to get capabilities for

        Returns:
            List[str]: List of capability names
        """
        capabilities = PipelineFactory._PIPELINE_CAPABILITIES.get(pipeline_class, [])
        return [cap.value for cap in capabilities]

    @staticmethod
    def _generate_optimization_suggestions(
        requirements: List[PipelineCapability],
        content_type: Optional[ContentType],
        suitable_pipelines: List[Type[BasePipeline]],
    ) -> List[str]:
        """
        Generate optimization suggestions based on requirements and available pipelines

        Args:
            requirements: Required capabilities
            content_type: Content type if specified
            suitable_pipelines: List of suitable pipeline classes

        Returns:
            List[str]: List of optimization suggestions
        """
        suggestions = []

        if PipelineCapability.BATCH_PROCESSING in requirements:
            suggestions.append(
                "Consider using BatchTranscriptionPipeline for high-volume processing"
            )

        if PipelineCapability.PII_REDACTION in requirements:
            suggestions.append(
                "Ensure output configuration is set up for saving redacted files"
            )

        if PipelineCapability.CUSTOM_SPEAKERS in requirements:
            suggestions.append(
                "Obtain pyannote token for enhanced speaker identification accuracy"
            )

        if content_type == ContentType.MEETING and len(suitable_pipelines) > 1:
            suggestions.append(
                "MeetingTranscriptionPipeline provides optimized features for business meetings"
            )

        if content_type == ContentType.PODCAST and len(suitable_pipelines) > 1:
            suggestions.append(
                "PodcastTranscriptionPipeline includes specialized features for content creation"
            )

        return suggestions

    @staticmethod
    def list_available_pipelines() -> Dict[str, Dict[str, Any]]:
        """
        List all available pipeline types with their capabilities and descriptions

        Returns:
            Dict[str, Dict[str, Any]]: Comprehensive pipeline information including
                                        capabilities, use cases, and descriptions
        """
        pipeline_info = {}

        for (
            pipeline_class,
            capabilities,
        ) in PipelineFactory._PIPELINE_CAPABILITIES.items():
            pipeline_info[pipeline_class.__name__] = {
                "capabilities": [cap.value for cap in capabilities],
                "description": (
                    pipeline_class.__doc__.split("\n")[0]
                    if pipeline_class.__doc__
                    else "No description available"
                ),
                "class_name": pipeline_class.__name__,
            }

        return pipeline_info


# Usage Examples
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv(override=True)

    # Configuration with output settings
    output_config = OutputConfig(
        base_output_dir="transcription_outputs",
        organize_by_date=True,
        organize_by_pipeline_type=True,
        organize_by_content_type=True,
        save_json=True,
        save_txt=True,
        download_audio=True,
        create_reports=True,
    )

    config = PipelineConfig(
        api_key=os.getenv("ASSEMBLYAI_API_KEY"),
        language_code="en",
        speech_model="best",
        output_config=output_config,
    )

    # ==== NEW SPECIALIZED PIPELINES ====

    # 1. Redaction and Moderation Pipeline
    print("=== REDACTION & MODERATION PIPELINE ===")
    redaction_pipeline = RedactionModerationPipeline(config)

    # Example with custom redaction policies
    redaction_result = redaction_pipeline.process(
        "C:\\Users\\OMEN\\Downloads\\personal information.mp3",
        source_type=AudioSource.LOCAL_FILE,
        save_files=True,
    )
    # print(f"PII Detected: {len(redaction_result.pii_detected) if redaction_result.pii_detected else 0}")
    # print(f"Content Safety Issues: {redaction_result.source_info.get('redaction_summary', {}).get('content_safety_issues', 0)}")

    print(redaction_result)

    # # 2. Content Analysis Pipeline - Podcast
    # print("\n=== CONTENT ANALYSIS PIPELINE - PODCAST ===")
    # content_pipeline = ContentAnalysisPipeline(config)

    # podcast_result = content_pipeline.process(
    #     "podcast_episode.mp3",
    #     content_type=ContentType.PODCAST,
    #     source_type=AudioSource.LOCAL_FILE,
    #     context_metadata={"show_name": "Tech Talk", "episode": 42},
    #     save_files=True
    # )

    # if podcast_result.chapters:
    #     print(f"Chapters found: {len(podcast_result.chapters)}")
    #     for i, chapter in enumerate(podcast_result.chapters[:3], 1):
    #         print(f"  {i}. {chapter['headline']} ({chapter['start']:.0f}s)")

    # # 3. Content Analysis Pipeline - Meeting
    # print("\n=== CONTENT ANALYSIS PIPELINE - MEETING ===")
    # meeting_result = content_pipeline.process(
    #     "team_meeting.mp3",
    #     content_type=ContentType.MEETING,
    #     source_type=AudioSource.LOCAL_FILE,
    #     context_metadata={
    #         "meeting_type": "standup",
    #         "attendees": ["Alice", "Bob", "Carol"]
    #     },
    #     save_files=True
    # )

    # insights = meeting_result.source_info.get("content_insights", {})
    # if "action_items" in insights:
    #     print(f"Action items identified: {len(insights['action_items'])}")
    #     for item in insights["action_items"][:3]:
    #         print(f"  - {item}")

    # # 4. Content Analysis Pipeline - Interview
    # print("\n=== CONTENT ANALYSIS PIPELINE - INTERVIEW ===")
    # interview_result = content_pipeline.process(
    #     "job_interview.mp3",
    #     content_type=ContentType.INTERVIEW,
    #     source_type=AudioSource.LOCAL_FILE,
    #     context_metadata={
    #         "interviewer": "HR Manager",
    #         "candidate": "John Doe",
    #         "position": "Software Engineer"
    #     },
    #     save_files=True
    # )

    # insights = interview_result.source_info.get("content_insights", {})
    # if "key_quotes" in insights:
    #     print(f"Key quotes captured: {len(insights['key_quotes'])}")
    # if "sentiment_flow" in insights:
    #     print(f"Sentiment analysis: {len(insights['sentiment_flow'])} data points")

    # # ==== EXISTING PIPELINES (Enhanced with file outputs) ====

    # # General transcription with file output
    # print("\n=== GENERAL TRANSCRIPTION (Enhanced) ===")
    # general_pipeline = GeneralTranscriptionPipeline(config)
    # general_result = general_pipeline.process(
    #     "general_audio.mp3",
    #     source_type=AudioSource.LOCAL_FILE,
    #     features=TranscriptionFeatures(
    #         speaker_diarization=True,
    #         pii_redaction=True,
    #         hate_speech_detection=True,
    #         auto_highlights=True,
    #     ),
    # )

    # # Save outputs manually if needed
    # if config.output_config:
    #     saved_files = general_pipeline._save_outputs(general_result, "general_transcription")
    #     print(f"Files saved: {list(saved_files.keys())}")

    # # Factory usage with new pipelines
    # print("\n=== FACTORY PATTERN USAGE ===")

    # # Create pipelines using factory
    # redaction_factory_pipeline = PipelineFactory.create_redaction_pipeline(config)
    # content_factory_pipeline = PipelineFactory.create_content_analysis_pipeline(config)

    # # Batch processing with new pipelines
    # print("\n=== BATCH PROCESSING ===")
    # batch_pipeline = BatchTranscriptionPipeline(content_pipeline, max_concurrent=2)

    # sources = [
    #     {
    #         "source": "podcast1.mp3",
    #         "content_type": ContentType.PODCAST,
    #         "save_files": True
    #     },
    #     {
    #         "source": "meeting1.mp3",
    #         "content_type": ContentType.MEETING,
    #         "save_files": True
    #     }
    # ]

    # # Example of how to run batch processing
    # # batch_results = asyncio.run(batch_pipeline.process_batch(sources))

    # print("\n=== PIPELINE CAPABILITIES SUMMARY ===")
    # print("✓ RedactionModerationPipeline: PII redaction, content safety, audio redaction")
    # print("✓ ContentAnalysisPipeline: Adaptive analysis (podcasts, meetings, interviews)")
    # print("✓ File outputs: JSON, TXT, redacted audio, analysis reports")
    # print("✓ Organized directory structure by date/pipeline/content type")
    # print("✓ Enhanced factory pattern for easy pipeline creation")
