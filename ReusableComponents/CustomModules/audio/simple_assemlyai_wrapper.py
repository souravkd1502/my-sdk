"""
Audio Processing Pipeline with AssemblyAI + AWS S3

This module provides functionality to:
- Transcribe audio using AssemblyAI
- Redact PII in both transcript and audio
- Save results locally (and optionally upload to AWS S3)
- Support concurrent processing of multiple audio files

Usage:
    python process_audio.py
"""

import os
import sys
import logging
import boto3
import requests
import assemblyai as aai
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from typing import Optional, Union, List


class AudioProcessor:
    """
    AudioProcessor handles transcription and redaction of audio files
    using AssemblyAI, with optional S3 integration.

    Attributes:
        api_key (str): AssemblyAI API key loaded from environment.
        s3_client (boto3.client): S3 client for uploads/downloads.
    """

    def __init__(self) -> None:
        """Initialize environment, logging, and clients."""
        # Load environment variables
        load_dotenv()

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        # AssemblyAI API key
        api_key = os.getenv("ASSEMBLYAI_API_KEY", "")
        if not api_key:
            logging.error("ASSEMBLYAI_API_KEY not found in environment.")
            sys.exit(1)

        aai.settings.api_key = api_key
        self.s3_client = boto3.client("s3")

    @staticmethod
    def build_output_path(
        base_dir: str, client: str, provider: str, call_id: str, job: str, extension: str
    ) -> Path:
        """
        Build formatted save path for transcript or audio.

        Args:
            base_dir: Base directory for output
            client: Client name
            provider: Provider name
            call_id: Unique call identifier
            job: Job name (e.g., "transcript", "redacted_audio")
            extension: File extension (.txt, .mp3, etc.)

        Returns:
            Path: Generated output file path
        """
        today = datetime.now(timezone.utc)
        return Path(
            f"{base_dir}/{client}/{provider}/{today.year}/{today.month:02}/{today.day:02}/{job}/{provider}_{call_id}.{extension}"
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

    def download_from_s3(self, bucket: str, key: str, local_path: Path) -> Path:
        """
        Download file from S3 to local path.

        Args:
            bucket: S3 bucket name
            key: Object key in bucket
            local_path: Local path where file will be saved

        Returns:
            Path: Path to downloaded file

        Raises:
            Exception: If download fails
        """
        try:
            self.s3_client.download_file(bucket, key, str(local_path))
            logging.info(f"Downloaded s3://{bucket}/{key} to {local_path}")
            return local_path
        except Exception as e:
            logging.error(f"Failed to download from S3: {e}", exc_info=True)
            raise

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
        Process audio with AssemblyAI: generate transcript + redacted audio.

        Args:
            audio_source: Local path or S3 URL to audio file
            client: Client name
            provider: Provider name
            call_id: Unique call ID
            output_base: Base directory for saving files
            save_to_s3: Whether to upload results to S3
            bucket: Target S3 bucket if save_to_s3=True
        """
        try:
            # Handle S3 input
            if str(audio_source).startswith("s3://"):
                _, _, bucket_name, *key_parts = str(audio_source).split("/", 3)
                key = key_parts[-1]
                local_audio_path = Path(f"/tmp/{Path(key).name}")
                self.download_from_s3(bucket_name, key, local_audio_path)
            else:
                local_audio_path = Path(audio_source)

            if not local_audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {local_audio_path}")

            # Configure transcription with PII redaction
            config = aai.TranscriptionConfig(
                redact_pii=True,
                redact_pii_audio=True,
                redact_pii_policies=[
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
                ],
                redact_pii_sub=aai.PIISubstitutionPolicy.hash,
            )

            logging.info(f"Transcribing {local_audio_path}...")
            transcript = aai.Transcriber().transcribe(str(local_audio_path), config)

            # Save transcript locally
            job = "transcript"
            transcript_path = self.build_output_path(
                output_base, client, provider, call_id, job, ".txt"
            )
            transcript_path.parent.mkdir(parents=True, exist_ok=True)
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript.text or "")
            logging.info(f"Transcript saved to {transcript_path}")

            # Download redacted audio
            audio_path = None
            redacted_audio_url = transcript.get_redacted_audio_url()
            if redacted_audio_url:
                job = "redacted_audio_ready"
                audio_path = self.build_output_path(
                    output_base, client, provider, call_id, job, ".mp3"
                )
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    response = requests.get(redacted_audio_url, stream=True)
                    response.raise_for_status()
                    with open(audio_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    logging.info(f"Redacted audio saved to {audio_path}")
                except Exception as e:
                    logging.error(
                        f"Failed to download redacted audio: {e}", exc_info=True
                    )

            # Upload to S3 if enabled
            if save_to_s3 and bucket:
                if transcript_path.exists():
                    self.upload_to_s3(transcript_path, bucket, str(transcript_path))
                if audio_path and audio_path.exists():
                    self.upload_to_s3(audio_path, bucket, str(audio_path))

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
            audio_files: List of audio file paths or S3 URIs
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
    processor = AudioProcessor()
    files = [
        "C:\\Users\\OMEN\\Downloads\\personal information.mp3",
        "C:\\Users\\OMEN\\Downloads\\LEARN ENGLISH WITH CONVERSATION - PERSONAL INFORMATION.mp3",
    ]
    processor.process_multiple(
        files, client="clientA", provider="assemblyai", save_to_s3=False
    )
