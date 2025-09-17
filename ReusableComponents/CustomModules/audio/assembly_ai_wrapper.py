""" """

import boto3
import logging
import asyncio
import json
import requests
import os
from pathlib import Path
from datetime import datetime
from enum import Enum
import assemblyai as aai
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Callable, Union


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioSource(Enum):
    LOCAL_FILE = "local_file"
    URL = "url"
    S3_BUCKET = "s3_bucket"
    GOOGLE_DRIVE = "google_drive"


class ContentType(Enum):
    GENERAL = "general"
    MEETING = "meeting"
    PODCAST = "podcast"
    LECTURE = "lecture"
    INTERVIEW = "interview"


@dataclass
class OutputConfig:
    """Configuration for file output operations"""

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
        """Generate organized output path based on configuration"""
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
    """Base configuration for all pipelines"""

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
    """Feature flags for transcription capabilities"""

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
    """Comprehensive result from any pipeline"""

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


class BasePipeline(ABC):
    """Abstract base class for all transcription pipelines"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        aai.settings.api_key = config.api_key
        self.transcriber = aai.Transcriber()

    @abstractmethod
    def process(self, source: str, **kwargs) -> PipelineResult:
        """Process audio from source and return structured result"""
        pass

    def _build_transcription_config(
        self, features: TranscriptionFeatures
    ) -> aai.TranscriptionConfig:
        """Build AssemblyAI configuration from features"""

        # Determine PII redaction settings
        enable_pii_redaction = features.pii_redaction or features.entity_redaction
        pii_policies = None

        if enable_pii_redaction:
            if features.entity_redaction:
                # Comprehensive PII redaction policies
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
                # Basic PII redaction policies when just pii_redaction is True
                pii_policies = [
                    aai.PIIRedactionPolicy.person_name,
                    aai.PIIRedactionPolicy.phone_number,
                    aai.PIIRedactionPolicy.email_address,
                ]
                
        pii_substitution_policy = aai.PIISubstitutionPolicy.hash if enable_pii_redaction else None  

        config = aai.TranscriptionConfig(
            language_code=self.config.language_code,
            speech_model=self.config.speech_model,
            punctuate=self.config.punctuate,
            format_text=self.config.format_text,
            dual_channel=self.config.dual_channel,
            webhook_url=self.config.webhook_url,
            # Speaker features
            speaker_labels=features.speaker_diarization,
            # Content safety
            redact_pii=enable_pii_redaction,
            redact_pii_policies=pii_policies,
            filter_profanity=features.content_moderation,
            content_safety=features.hate_speech_detection,
            # Advanced features
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
            # Redaction settings
            redact_pii_audio=True if enable_pii_redaction else False,
            redact_pii_sub=pii_substitution_policy,
        )

        return config

    def _format_result(
        self, transcript, source_info: Dict[str, Any], features: TranscriptionFeatures
    ) -> PipelineResult:
        """Format AssemblyAI response into structured result"""

        # Extract word-level timestamps
        words = None
        if (
            features.word_timestamps
            and hasattr(transcript, "words")
            and transcript.words
        ):
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

        # Extract speaker information
        speakers = None
        speaker_timeline = None
        if features.speaker_diarization:
            if hasattr(transcript, "utterances") and transcript.utterances:
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

                # Create speaker timeline
                speaker_timeline = self._create_speaker_timeline(transcript.utterances)

        # Extract PII/Entity redaction information
        pii_detected = None
        if (
            hasattr(transcript, "pii_redacted_audio_intelligence")
            and transcript.pii_redacted_audio_intelligence
        ):
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

        # Extract content safety information
        content_safety = None
        if (
            hasattr(transcript, "content_safety_labels")
            and transcript.content_safety_labels
        ):
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

        # Extract highlights
        highlights = None
        if (
            hasattr(transcript, "auto_highlights_result")
            and transcript.auto_highlights_result
        ):
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

        # Extract summary
        summary = None
        if hasattr(transcript, "summary") and transcript.summary:
            summary = transcript.summary

        # Extract chapters
        chapters = None
        if hasattr(transcript, "chapters") and transcript.chapters:
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

        # Extract sentiment analysis
        sentiment = None
        if (
            hasattr(transcript, "sentiment_analysis_results")
            and transcript.sentiment_analysis_results
        ):
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

        # Extract IAB categories
        iab_categories = None
        if (
            hasattr(transcript, "iab_categories_result")
            and transcript.iab_categories_result
        ):
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

        return PipelineResult(
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

    def _create_speaker_timeline(self, utterances) -> List[Dict[str, Any]]:
        """Create a timeline view of speaker changes"""
        timeline = []
        current_speaker = None
        segment_start = None

        for utterance in utterances:
            if utterance.speaker != current_speaker:
                if current_speaker is not None:
                    timeline.append(
                        {
                            "speaker": current_speaker,
                            "start": segment_start,
                            "end": utterance.start,
                        }
                    )
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

        return timeline

    def _save_outputs(
        self,
        result: PipelineResult,
        pipeline_type: str,
        content_type: Optional[str] = None,
    ) -> Dict[str, str]:
        """Save transcription and audio outputs to files"""
        if not self.config.output_config:
            logger.info("No output configuration provided, skipping file saves")
            return {}

        output_config = self.config.output_config
        saved_files = {}

        try:
            # Create base filename from transcript ID
            base_filename = result.id

            # Download the redacted/ moderated audio URL if available

            # Save transcription files
            if output_config.save_json or output_config.save_txt:
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

            # Download and save audio files if available
            if output_config.download_audio:
                audio_path = output_config.get_output_path(
                    pipeline_type, content_type, "audio"
                )
                audio_path.mkdir(parents=True, exist_ok=True)

                # Download original audio file if available
                original_audio_url = result.source_info.get("original_audio_url")
                if original_audio_url:
                    original_audio_file = audio_path / f"{base_filename}_original.mp3"
                    self._download_audio_file(original_audio_url, original_audio_file)
                    saved_files["original_audio"] = str(original_audio_file)
                    logger.info(f"Saved original audio to {original_audio_file}")

                # Download redacted audio file if available
                if result.raw_response:
                    # Check for redacted audio URL in various possible formats
                    redacted_audio_url = None
                    if "redacted_audio_url" in result.raw_response:
                        redacted_audio_url = result.raw_response["redacted_audio_url"]
                    elif hasattr(result.raw_response, "redacted_audio_url"):
                        redacted_audio_url = result.raw_response.redacted_audio_url

                    # Also check source_info for redacted URL
                    if not redacted_audio_url:
                        redacted_audio_url = result.source_info.get(
                            "redacted_audio_url"
                        )

                    if redacted_audio_url:
                        redacted_audio_file = (
                            audio_path
                            / f"{base_filename}_redacted.{output_config.audio_format}"
                        )
                        self._download_audio_file(
                            redacted_audio_url, redacted_audio_file
                        )
                        saved_files["redacted_audio"] = str(redacted_audio_file)
                        logger.info(f"Saved redacted audio to {redacted_audio_file}")
                    else:
                        logger.warning("Redacted audio URL not found in response")

            # Create reports if requested
            if output_config.create_reports:
                reports_path = output_config.get_output_path(
                    pipeline_type, content_type, "reports"
                )
                reports_path.mkdir(parents=True, exist_ok=True)

                report_file = reports_path / f"{base_filename}_report.json"
                self._create_analysis_report(result, report_file)
                saved_files["analysis_report"] = str(report_file)

        except Exception as e:
            logger.error(f"Error saving outputs: {str(e)}")

        return saved_files

    def _save_json_transcription(self, result: PipelineResult, file_path: Path):
        """Save transcription result as JSON"""
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

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(transcription_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved JSON transcription to {file_path}")

    def _save_txt_transcription(self, result: PipelineResult, file_path: Path):
        """Save transcription result as formatted text"""
        content = []

        # Header information
        content.append("TRANSCRIPTION REPORT")
        content.append("=" * 50)
        content.append(f"Transcript ID: {result.id}")
        content.append(f"Confidence: {result.confidence:.2f}")
        content.append(f"Duration: {result.audio_duration:.2f} seconds")
        content.append(f"Source: {result.source_info.get('source', 'Unknown')}")
        content.append(f"Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append("")

        # Main transcript
        content.append("TRANSCRIPT TEXT")
        content.append("-" * 30)
        content.append(result.text)
        content.append("")

        # Speaker information if available
        if result.speakers:
            content.append("SPEAKER BREAKDOWN")
            content.append("-" * 30)
            for speaker in result.speakers:
                content.append(
                    f"[{speaker['speaker']}] ({speaker['start']:.1f}s - {speaker['end']:.1f}s): {speaker['text']}"
                )
            content.append("")

        # Summary if available
        if result.summary:
            content.append("SUMMARY")
            content.append("-" * 30)
            content.append(result.summary)
            content.append("")

        # Highlights if available
        if result.highlights:
            content.append("KEY HIGHLIGHTS")
            content.append("-" * 30)
            for i, highlight in enumerate(result.highlights[:10], 1):
                content.append(f"{i}. {highlight['text']} (Rank: {highlight['rank']})")
            content.append("")

        # Chapters if available
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

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(content))

        logger.info(f"Saved TXT transcription to {file_path}")

    def _download_audio_file(self, url: str, file_path: Path):
        """Download audio file from URL"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Downloaded audio file to {file_path}")

        except Exception as e:
            logger.error(f"Failed to download audio file: {str(e)}")

    def _create_analysis_report(self, result: PipelineResult, file_path: Path):
        """Create analysis report with insights"""
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

        # Add speaker analysis
        if result.speakers:
            report["speaker_analysis"] = {
                "total_speakers": len(set(s["speaker"] for s in result.speakers)),
                "speaker_distribution": {},
            }

            for speaker in result.speakers:
                speaker_id = speaker["speaker"]
                if speaker_id not in report["speaker_analysis"]["speaker_distribution"]:
                    report["speaker_analysis"]["speaker_distribution"][speaker_id] = {
                        "word_count": 0,
                        "speaking_time": 0,
                    }

                report["speaker_analysis"]["speaker_distribution"][speaker_id][
                    "word_count"
                ] += len(speaker["text"].split())
                report["speaker_analysis"]["speaker_distribution"][speaker_id][
                    "speaking_time"
                ] += (speaker["end"] - speaker["start"])

        # Add content safety analysis
        if result.content_safety:
            report["safety_analysis"] = {
                "has_safety_issues": len(result.content_safety.get("results", [])) > 0,
                "safety_summary": result.content_safety.get("summary", {}),
                "issue_count": len(result.content_safety.get("results", [])),
            }

        # Add PII analysis
        if result.pii_detected:
            report["privacy_analysis"] = {
                "pii_detected": len(result.pii_detected) > 0,
                "pii_types": list(set(item["label"] for item in result.pii_detected)),
                "pii_count": len(result.pii_detected),
            }

        # Add content insights
        if result.highlights:
            report["content_analysis"]["highlights_count"] = len(result.highlights)
            report["content_analysis"]["top_highlights"] = [
                h["text"] for h in result.highlights[:5]
            ]

        if result.chapters:
            report["content_analysis"]["chapters_count"] = len(result.chapters)
            report["content_analysis"]["chapter_titles"] = [
                c["headline"] for c in result.chapters
            ]

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Created analysis report at {file_path}")


class GeneralTranscriptionPipeline(BasePipeline):
    """General-purpose transcription pipeline for any audio content"""

    def process(
        self,
        source: str,
        source_type: AudioSource = AudioSource.LOCAL_FILE,
        features: Optional[TranscriptionFeatures] = None,
        **kwargs,
    ) -> PipelineResult:
        """Process audio with general transcription features"""

        if features is None:
            features = TranscriptionFeatures(
                speaker_diarization=True,
                word_timestamps=True,
                pii_redaction=True,
                content_moderation=True,
                hate_speech_detection=True,
            )

        config = self._build_transcription_config(features)

        # Handle different source types
        audio_url = self._prepare_audio_source(source, source_type, **kwargs)

        try:
            transcript = self.transcriber.transcribe(audio_url, config)

            source_info = {
                "source": source,
                "source_type": source_type.value,
                "processed_url": audio_url,
            }

            return self._format_result(transcript, source_info, features)

        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise

    def _prepare_audio_source(
        self, source: str, source_type: AudioSource, **kwargs
    ) -> str:
        """Prepare audio source based on type"""
        if source_type == AudioSource.LOCAL_FILE:
            return source
        elif source_type == AudioSource.URL:
            return source
        elif source_type == AudioSource.S3_BUCKET:
            return self._get_s3_presigned_url(source, **kwargs)
        elif source_type == AudioSource.GOOGLE_DRIVE:
            return self._get_google_drive_url(source, **kwargs)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

    def _get_s3_presigned_url(self, s3_path: str, **kwargs) -> str:
        """Generate presigned URL for S3 object"""
        # Parse s3://bucket/key format
        if not s3_path.startswith("s3://"):
            raise ValueError("S3 path must start with s3://")

        path_parts = s3_path[5:].split("/", 1)
        bucket_name = path_parts[0]
        object_key = path_parts[1] if len(path_parts) > 1 else ""

        # Get AWS credentials from kwargs or environment
        aws_access_key = kwargs.get("aws_access_key")
        aws_secret_key = kwargs.get("aws_secret_key")
        aws_region = kwargs.get("aws_region", "us-east-1")

        if aws_access_key and aws_secret_key:
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region,
            )
        else:
            # Use default credentials (IAM role, profile, etc.)
            s3_client = boto3.client("s3", region_name=aws_region)

        # Generate presigned URL (valid for 1 hour)
        presigned_url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": object_key},
            ExpiresIn=3600,
        )

        return presigned_url

    def _get_google_drive_url(self, file_id: str, **kwargs) -> str:
        """Get Google Drive file URL"""
        # This would require Google Drive API setup
        # For now, return the direct download link format
        return f"https://drive.google.com/uc?id={file_id}&export=download"


class MeetingTranscriptionPipeline(BasePipeline):
    """Specialized pipeline for meeting transcription and summarization"""

    def process(
        self,
        source: str,
        source_type: AudioSource = AudioSource.LOCAL_FILE,
        meeting_context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> PipelineResult:
        """Process meeting audio with specialized features"""

        features = TranscriptionFeatures(
            speaker_diarization=True,
            word_timestamps=True,
            pii_redaction=True,
            content_moderation=True,
            summarization=True,
            auto_highlights=True,
            sentiment_analysis=True,
            topic_detection=True,
        )

        config = self._build_transcription_config(features)

        # Meeting-specific configuration
        config.summary_model = aai.SummarizationModel.informative
        config.summary_type = aai.SummarizationType.bullets

        audio_url = self._prepare_audio_source(source, source_type, **kwargs)

        try:
            transcript = self.transcriber.transcribe(audio_url, config)

            source_info = {
                "source": source,
                "source_type": source_type.value,
                "content_type": ContentType.MEETING.value,
                "meeting_context": meeting_context or {},
            }

            result = self._format_result(transcript, source_info, features)

            # Add meeting-specific processing
            result = self._enhance_meeting_result(result, meeting_context)

            return result

        except Exception as e:
            logger.error(f"Meeting transcription failed: {str(e)}")
            raise

    def _enhance_meeting_result(
        self, result: PipelineResult, context: Optional[Dict[str, Any]]
    ) -> PipelineResult:
        """Add meeting-specific enhancements"""
        if context and "attendees" in context:
            # Map speakers to attendees if provided
            attendee_mapping = context.get("speaker_mapping", {})
            if result.speakers and attendee_mapping:
                for speaker in result.speakers:
                    speaker_id = speaker["speaker"]
                    if speaker_id in attendee_mapping:
                        speaker["attendee_name"] = attendee_mapping[speaker_id]

        return result

    def _prepare_audio_source(
        self, source: str, source_type: AudioSource, **kwargs
    ) -> str:
        """Prepare audio source (same as general pipeline)"""
        return GeneralTranscriptionPipeline._prepare_audio_source(
            self, source, source_type, **kwargs
        )


class PodcastTranscriptionPipeline(BasePipeline):
    """Specialized pipeline for podcast transcription with chapters and highlights"""

    def process(
        self,
        source: str,
        source_type: AudioSource = AudioSource.LOCAL_FILE,
        podcast_metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> PipelineResult:
        """Process podcast audio with specialized features"""

        features = TranscriptionFeatures(
            speaker_diarization=True,
            word_timestamps=True,
            auto_highlights=True,
            auto_chapters=True,
            sentiment_analysis=True,
            topic_detection=True,
            iab_categories=True,
        )

        config = self._build_transcription_config(features)
        audio_url = self._prepare_audio_source(source, source_type, **kwargs)

        try:
            transcript = self.transcriber.transcribe(audio_url, config)

            source_info = {
                "source": source,
                "source_type": source_type.value,
                "content_type": ContentType.PODCAST.value,
                "podcast_metadata": podcast_metadata or {},
            }

            result = self._format_result(transcript, source_info, features)

            # Add podcast-specific processing
            result = self._enhance_podcast_result(result, podcast_metadata)

            return result

        except Exception as e:
            logger.error(f"Podcast transcription failed: {str(e)}")
            raise

    def _enhance_podcast_result(
        self, result: PipelineResult, metadata: Optional[Dict[str, Any]]
    ) -> PipelineResult:
        """Add podcast-specific enhancements"""
        if metadata:
            # Add show notes generation based on chapters and highlights
            if result.chapters and result.highlights:
                show_notes = self._generate_show_notes(
                    result.chapters, result.highlights
                )
                result.source_info["generated_show_notes"] = show_notes

        return result

    def _generate_show_notes(
        self, chapters: List[Dict], highlights: List[Dict]
    ) -> Dict[str, Any]:
        """Generate show notes from chapters and highlights"""
        return {
            "chapter_summary": [
                f"{chapter['headline']}: {chapter['gist']}" for chapter in chapters
            ],
            "key_highlights": [
                highlight["text"]
                for highlight in sorted(highlights, key=lambda x: x["rank"])[:5]
            ],
        }

    def _prepare_audio_source(
        self, source: str, source_type: AudioSource, **kwargs
    ) -> str:
        """Prepare audio source (same as general pipeline)"""
        return GeneralTranscriptionPipeline._prepare_audio_source(
            self, source, source_type, **kwargs
        )


class CustomSpeakerPipeline(BasePipeline):
    """Pipeline with custom speaker labeling using pyannote integration"""

    def __init__(self, config: PipelineConfig, pyannote_token: Optional[str] = None):
        super().__init__(config)
        self.pyannote_token = pyannote_token

    def process_with_custom_speakers(
        self,
        source: str,
        speaker_labels: Dict[str, str],
        source_type: AudioSource = AudioSource.LOCAL_FILE,
        features: Optional[TranscriptionFeatures] = None,
        **kwargs,
    ) -> PipelineResult:
        """Process audio with custom speaker labels"""

        if features is None:
            features = TranscriptionFeatures(
                speaker_diarization=True, word_timestamps=True
            )

        # First, get standard transcription
        config = self._build_transcription_config(features)
        audio_url = self._prepare_audio_source(source, source_type, **kwargs)

        try:
            transcript = self.transcriber.transcribe(audio_url, config)

            source_info = {
                "source": source,
                "source_type": source_type.value,
                "custom_speakers": True,
                "speaker_labels": speaker_labels,
            }

            result = self._format_result(transcript, source_info, features)

            # Apply custom speaker labels
            result = self._apply_custom_speaker_labels(result, speaker_labels)

            return result

        except Exception as e:
            logger.error(f"Custom speaker transcription failed: {str(e)}")
            raise

    def _apply_custom_speaker_labels(
        self, result: PipelineResult, speaker_labels: Dict[str, str]
    ) -> PipelineResult:
        """Apply custom speaker labels to the result"""
        if result.speakers:
            for speaker in result.speakers:
                speaker_id = speaker["speaker"]
                if speaker_id in speaker_labels:
                    speaker["custom_label"] = speaker_labels[speaker_id]

        if result.words:
            for word in result.words:
                if word.get("speaker") and word["speaker"] in speaker_labels:
                    word["custom_speaker"] = speaker_labels[word["speaker"]]

        result.custom_speaker_labels = speaker_labels
        return result

    def _prepare_audio_source(
        self, source: str, source_type: AudioSource, **kwargs
    ) -> str:
        """Prepare audio source (same as general pipeline)"""
        return GeneralTranscriptionPipeline._prepare_audio_source(
            self, source, source_type, **kwargs
        )


class RedactionModerationPipeline(BasePipeline):
    """Specialized pipeline for content safety, redaction, and moderation"""

    def process(
        self,
        source: str,
        source_type: AudioSource = AudioSource.LOCAL_FILE,
        redaction_policies: Optional[List[str]] = None,
        save_files: bool = True,
        **kwargs,
    ) -> PipelineResult:
        """Process audio with comprehensive content safety and redaction features"""

        features = TranscriptionFeatures(
            speaker_diarization=True,
            word_timestamps=True,
            pii_redaction=True,
            entity_redaction=True,
            content_moderation=True,
            hate_speech_detection=True,
        )

        config = self._build_transcription_config(features)
        audio_url = self._prepare_audio_source(source, source_type, **kwargs)

        try:
            transcript = self.transcriber.transcribe(audio_url, config)

            # Store both original and redacted audio URLs
            self.original_audio_url = audio_url
            self.redacted_audio_url = transcript.get_redacted_audio_url()

            source_info = {
                "source": source,
                "source_type": source_type.value,
                "content_type": "redaction_moderation",
                "pipeline_type": "redaction_moderation",
                "original_audio_url": self.original_audio_url,
                "redacted_audio_url": self.redacted_audio_url,
            }

            result = self._format_result(transcript, source_info, features)

            # Add redaction-specific enhancements
            result = self._enhance_redaction_result(result)

            # Save files if requested and output config is available
            if save_files and self.config.output_config:
                saved_files = self._save_outputs(result, "redaction_moderation")
                result.source_info["saved_files"] = saved_files

                # Log redacted audio status
                if "redacted_audio" in saved_files:
                    logger.info(
                        f"Redacted audio saved successfully: {saved_files['redacted_audio']}"
                    )
                else:
                    logger.warning(
                        "Redacted audio not available - check if PII was detected and redaction policies are configured"
                    )

            return result

        except Exception as e:
            logger.error(f"Redaction/Moderation transcription failed: {str(e)}")
            raise

    def _enhance_redaction_result(self, result: PipelineResult) -> PipelineResult:
        """Add redaction-specific enhancements and analysis"""

        # Create redaction summary
        redaction_summary = {
            "total_pii_instances": (
                len(result.pii_detected) if result.pii_detected else 0
            ),
            "pii_types_found": (
                list(set(item["label"] for item in result.pii_detected))
                if result.pii_detected
                else []
            ),
            "content_safety_issues": (
                len(result.content_safety.get("results", []))
                if result.content_safety
                else 0
            ),
            "redaction_applied": result.pii_detected is not None
            and len(result.pii_detected) > 0,
        }

        result.source_info["redaction_summary"] = redaction_summary
        return result

    def _prepare_audio_source(
        self, source: str, source_type: AudioSource, **kwargs
    ) -> str:
        """Prepare audio source (same as general pipeline)"""
        return GeneralTranscriptionPipeline._prepare_audio_source(
            self, source, source_type, **kwargs
        )


class ContentAnalysisPipeline(BasePipeline):
    """Adaptive pipeline for content analysis based on audio type"""

    def process(
        self,
        source: str,
        content_type: ContentType = ContentType.GENERAL,
        source_type: AudioSource = AudioSource.LOCAL_FILE,
        context_metadata: Optional[Dict[str, Any]] = None,
        save_files: bool = True,
        **kwargs,
    ) -> PipelineResult:
        """Process audio with adaptive content analysis features"""

        # Configure features based on content type
        features = self._get_content_specific_features(content_type)
        config = self._build_transcription_config(features)

        # Content-type specific configuration
        if content_type == ContentType.MEETING:
            config.summary_model = aai.SummarizationModel.informative
            config.summary_type = aai.SummarizationType.bullets

        audio_url = self._prepare_audio_source(source, source_type, **kwargs)

        try:
            transcript = self.transcriber.transcribe(audio_url, config)

            source_info = {
                "source": source,
                "source_type": source_type.value,
                "content_type": content_type.value,
                "context_metadata": context_metadata or {},
                "pipeline_type": "content_analysis",
            }

            result = self._format_result(transcript, source_info, features)

            # Add content-type specific enhancements
            result = self._enhance_content_analysis_result(
                result, content_type, context_metadata
            )

            # Save files if requested and output config is available
            if save_files and self.config.output_config:
                saved_files = self._save_outputs(
                    result, "content_analysis", content_type.value
                )
                result.source_info["saved_files"] = saved_files

            return result

        except Exception as e:
            logger.error(f"Content analysis transcription failed: {str(e)}")
            raise

    def _get_content_specific_features(
        self, content_type: ContentType
    ) -> TranscriptionFeatures:
        """Get optimized features for specific content types"""

        if content_type == ContentType.PODCAST:
            return TranscriptionFeatures(
                speaker_diarization=True,
                word_timestamps=True,
                auto_highlights=True,
                auto_chapters=True,
                sentiment_analysis=True,
                topic_detection=True,
                iab_categories=True,
            )

        elif content_type == ContentType.MEETING:
            return TranscriptionFeatures(
                speaker_diarization=True,
                word_timestamps=True,
                auto_highlights=True,
                summarization=True,
                sentiment_analysis=True,
                topic_detection=True,
            )

        elif content_type == ContentType.INTERVIEW:
            return TranscriptionFeatures(
                speaker_diarization=True,
                word_timestamps=True,
                auto_highlights=True,
                summarization=True,
                sentiment_analysis=True,
                topic_detection=True,
            )

        else:  # GENERAL, LECTURE, etc.
            return TranscriptionFeatures(
                speaker_diarization=True,
                word_timestamps=True,
                auto_highlights=True,
                summarization=True,
                sentiment_analysis=True,
            )

    def _enhance_content_analysis_result(
        self,
        result: PipelineResult,
        content_type: ContentType,
        metadata: Optional[Dict[str, Any]],
    ) -> PipelineResult:
        """Add content-type specific enhancements"""

        content_insights = {}

        if content_type == ContentType.PODCAST:
            content_insights = self._generate_podcast_insights(result, metadata)
        elif content_type == ContentType.MEETING:
            content_insights = self._generate_meeting_insights(result, metadata)
        elif content_type == ContentType.INTERVIEW:
            content_insights = self._generate_interview_insights(result, metadata)

        result.source_info["content_insights"] = content_insights
        return result

    def _generate_podcast_insights(
        self, result: PipelineResult, metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate podcast-specific insights"""
        insights = {}

        if result.chapters:
            insights["episode_structure"] = {
                "total_chapters": len(result.chapters),
                "chapter_titles": [c["headline"] for c in result.chapters],
                "average_chapter_length": sum(
                    c["end"] - c["start"] for c in result.chapters
                )
                / len(result.chapters),
            }

            # Generate show notes
            insights["show_notes"] = {
                "timestamps": [
                    f"{c['start']:.0f}s - {c['headline']}" for c in result.chapters
                ],
                "chapter_summaries": [
                    f"{c['headline']}: {c['gist']}" for c in result.chapters
                ],
            }

        if result.highlights:
            insights["key_moments"] = [
                {"text": h["text"], "rank": h["rank"], "timestamps": h["timestamps"]}
                for h in sorted(result.highlights, key=lambda x: x["rank"])[:10]
            ]

        if result.iab_categories:
            insights["content_categories"] = result.iab_categories["summary"]

        return insights

    def _generate_meeting_insights(
        self, result: PipelineResult, metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate meeting-specific insights"""
        insights = {}

        if result.highlights:
            insights["key_decisions"] = [h["text"] for h in result.highlights[:5]]
            insights["action_items"] = [
                h["text"]
                for h in result.highlights
                if any(
                    keyword in h["text"].lower()
                    for keyword in ["action", "todo", "follow up", "next steps"]
                )
            ]

        if result.speakers:
            # Participation analysis
            speaker_stats = {}
            for speaker in result.speakers:
                speaker_id = speaker["speaker"]
                if speaker_id not in speaker_stats:
                    speaker_stats[speaker_id] = {
                        "word_count": 0,
                        "speaking_time": 0,
                        "segments": 0,
                    }

                speaker_stats[speaker_id]["word_count"] += len(speaker["text"].split())
                speaker_stats[speaker_id]["speaking_time"] += (
                    speaker["end"] - speaker["start"]
                )
                speaker_stats[speaker_id]["segments"] += 1

            insights["participation_analysis"] = speaker_stats

        if result.summary:
            insights["meeting_summary"] = result.summary

        return insights

    def _generate_interview_insights(
        self, result: PipelineResult, metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate interview-specific insights"""
        insights = {}

        if result.highlights:
            insights["key_quotes"] = [h["text"] for h in result.highlights[:8]]

        if result.sentiment:
            # Analyze sentiment flow
            sentiment_timeline = []
            for sentiment_item in result.sentiment:
                sentiment_timeline.append(
                    {
                        "time": sentiment_item["start"],
                        "sentiment": sentiment_item["sentiment"],
                        "confidence": sentiment_item["confidence"],
                    }
                )
            insights["sentiment_flow"] = sentiment_timeline

        if result.speakers and len(result.speakers) >= 2:
            # Interview dynamic analysis
            speakers = list(set(s["speaker"] for s in result.speakers))
            if len(speakers) == 2:
                insights["interview_dynamic"] = {
                    "interviewer": speakers[0],  # Typically first speaker
                    "interviewee": speakers[1],
                    "question_answer_ratio": self._analyze_qa_ratio(result.speakers),
                }

        return insights

    def _analyze_qa_ratio(self, speakers: List[Dict[str, Any]]) -> float:
        """Analyze question to answer ratio in interview"""
        question_indicators = ["?", "what", "how", "why", "when", "where", "who"]

        questions = 0
        total_segments = len(speakers)

        for speaker in speakers:
            text_lower = speaker["text"].lower()
            if any(indicator in text_lower for indicator in question_indicators):
                questions += 1

        return questions / total_segments if total_segments > 0 else 0

    def _prepare_audio_source(
        self, source: str, source_type: AudioSource, **kwargs
    ) -> str:
        """Prepare audio source (same as general pipeline)"""
        return GeneralTranscriptionPipeline._prepare_audio_source(
            self, source, source_type, **kwargs
        )


class BatchTranscriptionPipeline:
    """Pipeline for batch processing multiple audio files"""

    def __init__(self, base_pipeline: BasePipeline, max_concurrent: int = 5):
        self.base_pipeline = base_pipeline
        self.max_concurrent = max_concurrent

    async def process_batch(
        self,
        sources: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[PipelineResult]:
        """Process multiple audio sources concurrently"""

        semaphore = asyncio.Semaphore(self.max_concurrent)
        completed = 0

        async def process_single(source_config: Dict[str, Any]) -> PipelineResult:
            nonlocal completed
            async with semaphore:
                loop = asyncio.get_event_loop()

                # Extract parameters
                source = source_config["source"]
                source_type = source_config.get("source_type", AudioSource.LOCAL_FILE)
                features = source_config.get("features")
                kwargs = source_config.get("kwargs", {})

                # Run transcription in thread pool
                result = await loop.run_in_executor(
                    None,
                    self.base_pipeline.process,
                    source,
                    source_type,
                    features,
                    **kwargs,
                )

                completed += 1
                if progress_callback:
                    progress_callback(completed, len(sources))

                return result

        tasks = [process_single(source_config) for source_config in sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return results


# Pipeline Factory
class PipelineFactory:
    """Factory for creating appropriate pipelines based on content type and pipeline purpose"""

    @staticmethod
    def create_pipeline(
        content_type: ContentType, config: PipelineConfig, **kwargs
    ) -> BasePipeline:
        """Create appropriate pipeline for content type"""

        if content_type == ContentType.MEETING:
            return MeetingTranscriptionPipeline(config)
        elif content_type == ContentType.PODCAST:
            return PodcastTranscriptionPipeline(config)
        elif content_type in [ContentType.LECTURE, ContentType.INTERVIEW]:
            return GeneralTranscriptionPipeline(config)
        else:
            return GeneralTranscriptionPipeline(config)

    @staticmethod
    def create_redaction_pipeline(
        config: PipelineConfig,
    ) -> RedactionModerationPipeline:
        """Create specialized redaction and moderation pipeline"""
        return RedactionModerationPipeline(config)

    @staticmethod
    def create_content_analysis_pipeline(
        config: PipelineConfig,
    ) -> ContentAnalysisPipeline:
        """Create adaptive content analysis pipeline"""
        return ContentAnalysisPipeline(config)

    @staticmethod
    def create_custom_speaker_pipeline(
        config: PipelineConfig, pyannote_token: Optional[str] = None
    ) -> CustomSpeakerPipeline:
        """Create custom speaker labeling pipeline"""
        return CustomSpeakerPipeline(config, pyannote_token)

    @staticmethod
    def create_batch_pipeline(
        base_pipeline: BasePipeline, max_concurrent: int = 5
    ) -> BatchTranscriptionPipeline:
        """Create batch processing pipeline"""
        return BatchTranscriptionPipeline(base_pipeline, max_concurrent)


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
