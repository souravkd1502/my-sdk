#!/usr/bin/env python3
"""
Email Reader Module

This module provides functionality to connect to an IMAP-based email server,
fetch the latest emails, extract content, and monitor for new messages.

Features:
- Connect to an IMAP server (SSL)
- Fetch the latest N emails from a specified mailbox
- Decode subject and body content (supports plain text and basic HTML)
- Monitor mailbox for new emails in a loop
- Gracefully handle errors and interruptions

- IMAP server connection example
    | Provider | IMAP Server             | Port |
    -------- | ----------------------- | ---- |
    | Gmail    | `imap.gmail.com`        | 993  |
    | Outlook  | `outlook.office365.com` | 993  |
    | Yahoo    | `imap.mail.yahoo.com`   | 993  |
"""

import imaplib
import email
from email.header import decode_header
from typing import List, Optional
from dataclasses import dataclass
import logging

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@dataclass
class EmailData:
    """Simple data class for email information."""

    id: str
    subject: str
    sender: str
    date: str
    body: str


class EmailReader:
    """
    Simple email reader with basic filtering capabilities.
    """

    def __init__(self, email_address: str, password: str, imap_server: str):
        """
        Initialize the EmailReader with connection details.

        Args:
            email_address: Email address for authentication
            password: Password or app-specific password for authentication
            imap_server: IMAP server hostname (e.g., 'imap.gmail.com')
        """
        self.email_address = email_address
        self.password = password
        self.imap_server = imap_server
        self.mail: Optional[imaplib.IMAP4_SSL] = None

        logger.info(f"EmailReader initialized for {email_address}")

    def connect(self) -> bool:
        """
        Establish IMAP connection.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.mail = imaplib.IMAP4_SSL(self.imap_server)
            self.mail.login(self.email_address, self.password)
            logger.info(f"Connection established to {self.email_address}")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from IMAP server."""
        if self.mail:
            try:
                self.mail.logout()
                logger.info("Disconnected")
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")

    def get_latest_emails(
        self,
        mailbox: str = "INBOX",
        count: int = 5,
        unread_only: bool = False,
        sender_filter: Optional[str] = None,
        subject_filter: Optional[str] = None,
        since_date: Optional[str] = None,
    ) -> List[EmailData]:
        """
        Retrieve latest emails with filtering.

        Args:
            mailbox: Mailbox name to search in
            count: Maximum number of emails to retrieve (default: 5)
            unread_only: Only retrieve unread emails
            sender_filter: Filter by sender email/name (case-insensitive)
            subject_filter: Filter by subject content (case-insensitive)
            since_date: Retrieve emails since date (format: "01-Jul-2025")

        Returns:
            List of EmailData objects matching the criteria
        """
        try:
            if not self.mail:
                raise ConnectionError("No active IMAP connection")

            # Select mailbox
            status, _ = self.mail.select(mailbox)
            if status != "OK":
                raise ValueError(f"Failed to select mailbox: {mailbox}")

            # Build search criteria
            criteria = []
            if unread_only:
                criteria.append("UNSEEN")
            if since_date:
                criteria.append(f'SINCE "{since_date}"')

            if not criteria:
                criteria = ["ALL"]

            # Execute search
            status, messages = self.mail.search(None, *criteria)
            if status != "OK":
                raise ValueError("Failed to execute email search")

            email_ids = messages[0].split()

            logger.info(f"Found {len(email_ids)} emails. Processing latest {count}...")

            if not email_ids:
                logger.info("No emails found matching criteria")
                return []

            # Process emails (newest first)
            filtered_emails = []

            for eid in reversed(
                email_ids[-count * 2 :]
            ):  # Get more than needed for filtering
                if len(filtered_emails) >= count:
                    break

                email_data = self._process_single_email(eid)
                if not email_data:
                    continue

                # Apply filters
                if self._email_matches_filters(
                    email_data, sender_filter, subject_filter
                ):
                    filtered_emails.append(email_data)

            logger.info(f"Retrieved {len(filtered_emails)} emails after filtering")
            return filtered_emails[:count]  # Ensure we don't exceed count

        except Exception as e:
            logger.error(f"Error fetching emails: {e}")
            return []

    def _email_matches_filters(
        self,
        email_data: EmailData,
        sender_filter: Optional[str],
        subject_filter: Optional[str],
    ) -> bool:
        """Check if email matches specified filters."""
        return (
            not sender_filter or sender_filter.lower() in email_data.sender.lower()
        ) and (
            not subject_filter or subject_filter.lower() in email_data.subject.lower()
        )

    def _process_single_email(self, email_id: bytes) -> Optional[EmailData]:
        """
        Process a single email and extract data.

        Args:
            email_id: Email ID bytes

        Returns:
            EmailData if successful, None otherwise
        """
        try:
            # Fetch email
            status, msg_data = self.mail.fetch(email_id, "(RFC822)")
            if status != "OK":
                return None

            email_message = email.message_from_bytes(msg_data[0][1])

            return EmailData(
                id=email_id.decode(),
                subject=self._decode_header(email_message.get("Subject")),
                sender=self._decode_header(email_message.get("From", "")),
                date=email_message.get("Date", ""),
                body=self._extract_body(email_message),
            )

        except Exception as e:
            logger.warning(f"Failed to process email {email_id.decode()}: {e}")
            return None

    def _decode_header(self, header_value: Optional[str]) -> str:
        """
        Decode email header.

        Args:
            header_value: Raw header value

        Returns:
            Decoded header string
        """
        if not header_value:
            return ""

        try:
            decoded = decode_header(header_value)
            result = ""

            for part, encoding in decoded:
                if isinstance(part, bytes):
                    try:
                        result += part.decode(encoding or "utf-8")
                    except (UnicodeDecodeError, LookupError):
                        result += part.decode("utf-8", errors="replace")
                else:
                    result += str(part)

            return result.strip()

        except Exception as e:
            logger.warning(f"Header decoding failed: {e}")
            return str(header_value)

    def _extract_body(self, email_message) -> str:
        """
        Extract email body text.

        Args:
            email_message: Email message object

        Returns:
            Extracted body text (limited to 1000 chars for preview)
        """
        body = ""

        try:
            if email_message.is_multipart():
                for part in email_message.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition") or "")

                    if "attachment" in content_disposition.lower():
                        continue

                    if content_type == "text/plain" and (
                        payload := part.get_payload(decode=True)
                    ):
                        body += payload.decode("utf-8", errors="replace")
                        break  # Use first text/plain part
            elif payload := email_message.get_payload(decode=True):
                body = payload.decode("utf-8", errors="replace")

        except Exception as e:
            logger.warning(f"Error extracting body: {e}")
            body = "[Error reading email body]"

        return body.strip()


# Example usage
if __name__ == "__main__":
    import os
    import dotenv

    # Load environment variables from .env file
    dotenv.load_dotenv(override=True)
    EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")  # Email address to connect
    PASSWORD = os.getenv("EMAIL_PASSWORD")  # Password or app-specific password
    IMAP_SERVER = os.getenv("IMAP_SERVER")  # IMAP server address (e.g., imap.gmail.com)

    # Create reader instance
    reader = EmailReader(EMAIL_ADDRESS, PASSWORD, IMAP_SERVER)

    try:
        # Connect
        if reader.connect():

            # Example with filters
            print("\n--- Filtered Results ---")
            filtered_emails = reader.get_latest_emails(
                count=3,
                unread_only=False,
                sender_filter="sourav.das@thefunctionary.com",
            )

            for email_data in filtered_emails:
                print(f"Filtered: {email_data.subject} from {email_data.sender}")

    finally:
        reader.disconnect()
