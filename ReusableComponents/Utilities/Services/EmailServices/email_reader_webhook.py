#!/usr/bin/env python3
"""
Flask webhook to read the latest email using EmailReader.
"""

from datetime import datetime
from typing import Dict, List
from flask import Flask, jsonify, request
from email_reader import EmailReader
import os
import dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load env variables
dotenv.load_dotenv(override=True)

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
PASSWORD = os.getenv("EMAIL_PASSWORD")
IMAP_SERVER = os.getenv("IMAP_SERVER")

# Flask app
app = Flask(__name__)

@app.route("/read-email", methods=["POST"])
def read_email():
    """
    Webhook to read emails with optional filters.
    Request body:
    {
        "count": 5,
        "subject": "Invoice",
        "sender": "noreply@example.com",
        "after": "2025-07-20",        # optional ISO date
        "unread_only": true           # optional flag
    }
    """
    try:
        req = request.get_json(force=True)
        count = int(req.get("count", 5))
        subject_filter = req.get("subject", "").lower()
        sender_filter = req.get("sender", "").lower()
        after_date_str = req.get("after")
        unread_only = req.get("unread_only", False)

        # Convert after_date to datetime
        after_date = None
        if after_date_str:
            try:
                after_date = datetime.strptime(after_date_str, "%Y-%m-%d")
            except ValueError:
                return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400

        # Initialize and connect reader
        reader = EmailReader(EMAIL_ADDRESS, PASSWORD, IMAP_SERVER)
        if not reader.connect():
            return jsonify({"error": "Failed to connect to email server"}), 500

        # Get emails with filters
        emails = reader.get_latest_emails(
            count=count,
            subject_filter=subject_filter,
            sender_filter=sender_filter,
            since_date=after_date,
            unread_only=unread_only
        )

        return jsonify({"emails": emails}), 200
    
    except Exception as e:
        logger.error(f"Error reading emails: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)