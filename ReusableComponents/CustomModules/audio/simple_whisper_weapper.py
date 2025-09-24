"""
Audio Processing Pipeline with OpenAI Whisper + AWS S3

This module provides functionality to:
- Transcribe audio files using OpenAI Whisper
- Save transcripts locally
- Optionally upload transcripts to AWS S3
- Process multiple files concurrently

Usage:
    python process_audio_whisper.py
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from typing import Union, List, Optional
import boto3
import openai


class WhisperAudioProcessor:
    """
    Handles audio transcription using OpenAI Whisper with optional S3 integration.

    Attributes:
        api_key (str): OpenAI API key loaded from environment.
        s3_client (boto3.client): S3 client for uploads.
    """

    def __init__(self) -> None:
        """Initialize environment, logging, and clients."""
        # Load environment variables
        load_dotenv(override=True)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        # OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            logging.error("OPENAI_API_KEY not found in environment.")
            sys.exit(1)

        openai.api_key = api_key
        self.s3_client = boto3.client("s3")

    @staticmethod
    def build_output_path(base_dir: str, client: str, provider: str, call_id: str, extension: str) -> Path:
        """
        Build formatted save path.

        Args:
            base_dir: Base directory for output files
            client: Client name
            provider: Provider name
            call_id: Unique call identifier
            extension: File extension (.txt)

        Returns:
            Path: Generated output file path
        """
        today = datetime.now(timezone.utc)
        return Path(
            f"{base_dir}/{client}/{provider}/{today.year}/{today.month:02}/{today.day:02}/{provider}_{call_id}{extension}"
        )

    def upload_to_s3(self, local_path: Path, bucket: str, s3_key: str) -> None:
        """
        Upload file to S3 bucket.

        Args:
            local_path: Local file path
            bucket: S3 bucket name
            s3_key: Destination key in bucket
        """
        try:
            self.s3_client.upload_file(str(local_path), bucket, s3_key)
            logging.info(f"Uploaded {local_path} to s3://{bucket}/{s3_key}")
        except Exception as e:
            logging.error(f"Failed to upload {local_path} to S3: {e}", exc_info=True)

    def process_audio(
        self,
        audio_source: Union[str, Path],
        client: str,
        provider: str,
        call_id: str,
        output_base: str = "tf-call-recordings-raw",
        save_to_s3: bool = False,
        bucket: Optional[str] = None,
    ) -> None:
        """
        Process audio using OpenAI Whisper to generate transcript only.

        Args:
            audio_source: Local path to audio file
            client: Client name
            provider: Provider name
            call_id: Unique call ID
            output_base: Base directory for saving transcripts
            save_to_s3: Whether to upload results to S3
            bucket: Target S3 bucket if save_to_s3=True
        """
        try:
            audio_path = Path(audio_source)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            logging.info(f"Transcribing {audio_path} using Whisper...")

            # Call Whisper transcription
            with open(audio_path, "rb") as f:
                transcript = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="verbose_json",
                    timestamp_granularities=["word"],
                )

            with open("transcript_debug.json", "w", encoding="utf-8") as debug_file:
                import json
                json.dump(transcript.to_dict(), debug_file, ensure_ascii=False, indent=4)
            
            # Save transcript locally
            transcript_path = self.build_output_path(output_base, client, provider, call_id, ".txt")
            transcript_path.parent.mkdir(parents=True, exist_ok=True)

            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript.text or "")

            logging.info(f"Transcript saved to {transcript_path}")

            # Optionally upload transcript to S3
            if save_to_s3 and bucket:
                self.upload_to_s3(transcript_path, bucket, str(transcript_path))

        except Exception as e:
            logging.error(f"Failed to process {audio_source}: {e}", exc_info=True)

    def process_multiple(
        self,
        audio_files: List[Union[str, Path]],
        client: str,
        provider: str,
        save_to_s3: bool = False,
        bucket: Optional[str] = None,
    ) -> None:
        """
        Run transcription concurrently for multiple files.

        Args:
            audio_files: List of audio file paths
            client: Client name
            provider: Provider name
            save_to_s3: Whether to upload results to S3
            bucket: Target S3 bucket if save_to_s3=True
        """
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    self.process_audio,
                    f,
                    client,
                    provider,
                    f"call{i+1}",
                    save_to_s3=save_to_s3,
                    bucket=bucket,
                )
                for i, f in enumerate(audio_files)
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Task failed: {e}", exc_info=True)


if __name__ == "__main__":
    processor = WhisperAudioProcessor()
    files = [
        "C:\\Users\\OMEN\\Downloads\\personal information.mp3",
        "C:\\Users\\OMEN\\Downloads\\LEARN ENGLISH WITH CONVERSATION - PERSONAL INFORMATION.mp3",
    ]
    processor.process_multiple(files, client="clientA", provider="whisper", save_to_s3=False)
