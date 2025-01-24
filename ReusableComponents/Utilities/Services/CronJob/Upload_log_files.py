"""
upload_log_files.py
-------------------

Description:
------------
This module provides a class for uploading log files to Azure Blob Storage or AWS S3.
It uses Flask Scheduler to schedule the upload of log files at regular intervals.

Usage:
------
1. Create an instance of the UploadLogFiles class.
2. Start the upload process by calling the `start_upload` method.

Requirements:
-------------
- boto3==1.36.2
- azure-core==1.32.0
- azure-storage-blob==12.24.0
- Flask==3.1.0
- Flask-APScheduler==1.13.1
"""

# Adding directories to system path to allow importing custom modules
import sys

sys.path.append("./")
sys.path.append("../")

# Import dependencies
import os
import logging
from datetime import datetime
from azure.storage.blob import BlobServiceClient

from flask import Flask
from flask_apscheduler import APScheduler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_logger = logging.getLogger(__name__)

# Set up flask App instance
app = Flask(__name__)

# Set up Scheduler instance
scheduler = APScheduler()


def upload_to_azure_blob(
    log_file: str,
    connection_string: str,
    container_name: str,
    user_id: str,
    service: str,
    session_id: str,
):
    """
    Upload the log file to Azure Blob Storage with partitioning.

    Args:
    -----
        - log_file (str): Path to the log file to be uploaded.
        - connection_string (str): Azure Blob Storage connection string.
        - container_name (str): Name of the container in Azure Blob Storage.
        - user_id (str): User ID of the user who generated the log file.
        - service (str): Name of the service that generated the log file.
        - session_id (str): Session ID of the session during which the log file was generated.
    """
    if not log_file or not os.path.exists(log_file):
        raise FileNotFoundError("Log file not found. Please check the log file path.")

    current_time = datetime.now()
    blob_path = f"{service}/{user_id}/{session_id}/{current_time.year}/{current_time.month:02}/{current_time.day:02}/{current_time.timestamp()}.log"

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(
        container=container_name, blob=blob_path
    )

    with open(log_file, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

    _logger.info(f"Log file uploaded to Azure Blob Storage: {blob_path}")

    # remove the log file after uploading
    os.remove(log_file)


@scheduler.task("interval", id="upload_log_files", seconds=36)
def upload_log_files():
    """
    Upload log files to Azure Blob Storage or AWS S3.
    """
    # Add your code here to upload log files
    log_file = "logs.csv"
    service = "service-name"
    connection_string="<connection-string>"
    container_name="test-logs"
    user_id="user123"
    session_id="service456"
    
    upload_to_azure_blob(
        log_file=log_file,
        connection_string=connection_string,
        container_name=container_name,
        user_id=user_id,
        service=service,
        session_id=session_id
    )
    
if __name__ == "__main__":
    scheduler.init_app(app)
    scheduler.start()
    app.run(debug=False)
