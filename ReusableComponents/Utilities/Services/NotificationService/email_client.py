"""
email_client.py
----------------

This module contains the EmailSender class which is used to send an email with an optional file attachment using SendGrid.

Required Environment Variables:
--------------------------------
    - SENDGRID_API_KEY: SendGrid API key.
    
Requirements:
-------------
    - sendgrid==6.11.0
    - tenacity==9.0.0
    
Author:
-------
Sourav Das

Date:
-----
2024-12-27
"""

# Adding directories to system path to allow importing custom modules
import sys

sys.path.append("./")
sys.path.append("../")

# Import dependencies
import os
import base64
import logging
from sendgrid import SendGridAPIClient
from tenacity import retry, stop_after_attempt, wait_exponential
from sendgrid.helpers.mail import (
    Mail,
    Attachment,
    FileContent,
    FileName,
    FileType,
    Disposition,
)

from typing import List, Optional, Tuple, Dict


# Set up logging
_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - line: %(lineno)d",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class EmailSender:
    """
    A class to send an email with an optional file attachment using SendGrid.

    Example usage:

    ``` python
    # Create an instance of EmailSender
    sender = EmailSender(
        from_email='from_email@example.com',
        to_email=['to@example.com'],
        subject='Sending with Twilio SendGrid is Fun',
        html_content='<strong>and easy to do anywhere, even with Python</strong>',
        cc_emails=['cc@example.com'],
        bcc_emails=['bcc@example.com'],
        file_path='data.csv',
        file_type='text/csv'
    )

    # Send the email with the attachment
    status_code, body, headers = sender.send_mail()
    ```
    """

    def __init__(
        self,
        from_email: str,
        to_email: List[str],
        subject: str,
        html_content: str,
        cc_emails: Optional[List[str]] = None,
        bcc_emails: Optional[List[str]] = None,
        file_path: Optional[str] = None,
        file_type: Optional[str] = None,
    ):
        """
        Initialize the EmailSender with email details and SendGrid API key.

        Args:
            from_email (str): Sender's email address.
            to_email (List[str]): List of recipient email addresses.
            subject (str): Subject of the email.
            html_content (str): HTML content of the email.
            cc_emails (Optional[List[str]]): List of CC email addresses.
            bcc_emails (Optional[List[str]]): List of BCC email addresses.
            file_path (Optional[str]): Path to the file to be attached.
            file_type (Optional[str]): MIME type of the file.
        """
        self.from_email = from_email
        self.to_emails = to_email
        self.subject = subject
        self.html_content = html_content
        self.cc_emails = cc_emails
        self.bcc_emails = bcc_emails
        self.file_path = file_path
        self.file_type = file_type

    def create_attachment(
        self, file_path: str, file_type: str, disposition: str = "attachment"
    ) -> Attachment:
        """
        Create a SendGrid Attachment object from a file.

        Args:
            file_path (str): Path to the file to be attached.
            file_type (str): MIME type of the file.
            disposition (str): Disposition of the file (default is 'attachment').

        Returns:
            Attachment: SendGrid Attachment object.
        """
        try:
            with open(file_path, "rb") as f:
                file_content = base64.b64encode(f.read()).decode("utf-8")
            file_name = os.path.basename(file_path)
            attachment = Attachment(
                FileContent(file_content),
                FileName(file_name),
                FileType(file_type),
                Disposition(disposition),
            )
            _logger.info(f"Attachment created successfully from file: {file_path}")
            return attachment
        except Exception as e:
            _logger.error(f"Error creating attachment from file: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def send_mail(self) -> Tuple[int, str, Dict[str, str]]:
        """
        Send an email with an optional file attachment using SendGrid.

        Returns:
            Tuple[int, str, Dict[str, str]]: Status code, response body, and response headers.
        """
        if not self.to_emails:
            raise ValueError("Recipient email addresses (to_emails) must be provided.")
        """
        Send an email with an optional file attachment using SendGrid.

        Returns:
            Tuple[int, str, Dict[str, str]]: Status code, response body, and response headers.
        """
        try:
            # Create a SendGrid API client
            api_key = os.getenv("SENDGRID_API_KEY")
            if not api_key:
                raise ValueError("SENDGRID_API_KEY environment variable is not set.")
            sg = SendGridAPIClient(api_key=api_key)

            # Create a Mail object
            message = Mail(
                from_email=self.from_email,
                to_emails=self.to_emails,
                subject=self.subject,
                html_content=self.html_content,
            )

            # Add CC and BCC emails if provided
            if self.cc_emails:
                message.cc = self.cc_emails
            if self.bcc_emails:
                message.bcc = self.bcc_emails

            # Add the attachment to the email if provided
            if self.file_path and self.file_type:
                if not os.path.isfile(self.file_path):
                    raise FileNotFoundError(f"File not found: {self.file_path}")
                attachment = self.create_attachment(
                    file_path=self.file_path,
                    file_type=self.file_type,
                )
                message.attachment = attachment

            # Send the email
            response = sg.send(message)
            if response.status_code >= 200 and response.status_code < 300:
                _logger.info(f"Email sent successfully to: {self.to_emails}")
            else:
                _logger.error(
                    f"Failed to send email. Status code: {response.status_code}, Body: {response.body}"
                )
            return (response.status_code, response.body, response.headers)

        except Exception as e:
            _logger.error(f"Error sending email: {e}")
            raise
