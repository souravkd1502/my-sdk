"""
logger.py
---------

Description:
------------
This module provides a class for logging and metric tracking. It supports logging to console and file, 
exporting logs to CSV, uploading logs to Azure Blob Storage or AWS S3, recording metrics, 
and visualizing metrics using matplotlib.

Usage:
------
1. Create an instance of EnhancedLogger with a name and an optional log file path.
2. Log messages using the log method with a log level ('info', 'warning', 'error').
3. Record metrics using the record_metric method.
4. Visualize metrics using the visualize_metrics method.
5. Export logs to CSV using the export_logs_to_csv method.
6. Upload logs to Azure Blob Storage using the upload_to_azure_blob method.
7. Upload logs to AWS S3 using the upload_to_s3 method.

Requirements:
-------------
- boto3==1.36.2
- matplotlib==3.10.0
- azure-core==1.32.0
- azure-storage-blob==12.24.0

Author:
-------
Sourav Das

Version:
--------
1.0

Date Created:
-------------
21-01-2025
"""

# Adding directories to system path to allow importing custom modules
import sys

sys.path.append("./")
sys.path.append("../")

# Import dependencies
import os
import csv
import boto3
import psutil
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from azure.storage.blob import BlobServiceClient

from typing import Dict, Any, Optional

class EnhancedLogger:
    FILE_NOT_FOUND_ERROR_MSG = "Log file not found or not specified."

    def __init__(self, name: str, log_file: Optional[str] = None, service: Optional[str] = None):
        """
        Initialize the EnhancedLogger.

        Args:
            name (str): Name of the logger.
            log_file (Optional[str]): Path to the log file.
            service (Optional[str]): Name of the service.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler (optional)
        self.log_file = log_file
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        self.metrics = {}
        self.service = service if service else "DefaultService"

    def log(self, level: str, message: str):
        """
        Log a message.

        Args:
            level (str): Logging level (e.g., 'info', 'warning', 'error').
            message (str): Message to log.
        """
        log_message = f"[{self.service}] {message}"
        log_method = getattr(self.logger, level.lower(), None)
        if callable(log_method):
            log_method(log_message)
        else:
            raise ValueError(f"Invalid log level: {level}")

    def export_logs_to_csv(self, csv_file: str):
        """
        Export logs to a CSV file.

        Args:
            csv_file (str): Path to the CSV file.
        """
        if not self.log_file or not os.path.exists(self.log_file):
            raise FileNotFoundError(self.FILE_NOT_FOUND_ERROR_MSG)

        with open(self.log_file, 'r') as log_file, open(csv_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Timestamp', 'Level', 'Message'])

            for line in log_file:
                parts = line.strip().split(' - ', 2)
                if len(parts) == 3:
                    csv_writer.writerow(parts)

    def upload_to_azure_blob(self, connection_string: str, container_name: str, user_id: str, service_id: str):
        """
        Upload the log file to Azure Blob Storage with partitioning.

        Args:
            connection_string (str): Azure Blob Storage connection string.
            container_name (str): Name of the container.
            user_id (str): User ID.
            service_id (str): Service ID.
        """
        if not self.log_file or not os.path.exists(self.log_file):
            raise FileNotFoundError(self.FILE_NOT_FOUND_ERROR_MSG)

        current_time = datetime.now()
        blob_path = f"{self.service}/{user_id}/{service_id}/{current_time.year}/{current_time.month:02}/{current_time.day:02}/{current_time.timestamp()}.log"

        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_path)

        with open(self.log_file, 'rb') as data:
            blob_client.upload_blob(data, overwrite=True)

    def upload_to_s3(self, bucket_name: str, user_id: str, service_id: str, aws_access_key: str, aws_secret_key: str, region_name: str):
        """
        Upload the log file to AWS S3 with partitioning.

        Args:
            bucket_name (str): Name of the S3 bucket.
            user_id (str): User ID.
            service_id (str): Service ID.
            aws_access_key (str): AWS access key ID.
            aws_secret_key (str): AWS secret access key.
            region_name (str): AWS region.
        """
        if not self.log_file or not os.path.exists(self.log_file):
            raise FileNotFoundError(self.FILE_NOT_FOUND_ERROR_MSG)

        current_time = datetime.now()
        object_name = f"{self.service}/{user_id}/{service_id}/{current_time.year}/{current_time.month:02}/{current_time.day:02}/{current_time.timestamp()}.log"

        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region_name
        )

        s3_client.upload_file(self.log_file, bucket_name, object_name)

    def record_metric(self, metric_name: str, value: Any):
        """
        Record a metric.

        Args:
            metric_name (str): Name of the metric.
            value (Any): Value of the metric.
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)

    def record_resource_usage(self):
        """
        Record system resource usage.
        """
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent

        self.record_metric("CPU_Usage", cpu_usage)
        self.record_metric("Memory_Usage", memory_usage)

    def save_metrics_to_csv(self, csv_file: str):
        """
        Save recorded metrics to a CSV file.

        Args:
            csv_file (str): Path to the CSV file.
        """
        with open(csv_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Metric Name', 'Values'])

            for metric_name, values in self.metrics.items():
                csv_writer.writerow([metric_name, values])
                

    def visualize_metrics(self, csv_file: str ,save_as_png: Optional[str] = None):
        """
        Visualize recorded metrics using matplotlib.

        Args:
            csv_file (str): Path to the CSV file containing metrics.
            save_as_png (Optional[str]): Path to save the visualization as a PNG file.
        """
        try:
            with open(csv_file, "r") as csvfile:
                reader = csv.reader(csvfile)
                headers = next(reader)

                data = {header: [] for header in headers}
                for row in reader:
                    for header, value in zip(headers, row):
                        data[header].append(value)

                for metric_name in data.keys():
                    if metric_name != "Metric Name":
                        plt.plot(data[metric_name], label=metric_name)

            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.title('Metrics Visualization')
            plt.legend()

            if save_as_png:
                plt.savefig(save_as_png)
            else:
                plt.show()
        except FileNotFoundError:
            raise FileNotFoundError("Metrics CSV file not found.")
        except Exception as e:
            raise RuntimeError(f"Error visualizing metrics: {e}")

# Example usage
if __name__ == "__main__":
    logger = EnhancedLogger("MyLogger", "app.log", service="MyService")
    logger.log("info", "Application started.")
    logger.log("error", "An error occurred.")

    # Export logs to CSV
    logger.export_logs_to_csv("logs.csv")
    
    # Upload logs to Azure Blob Storage or AWS S3 (replace with actual credentials)
    logger.upload_to_azure_blob(
        connection_string="<connection-string>",
        container_name="test-logs",
        user_id="user123",
        service_id="service456"
    )
    # logger.upload_to_s3("<bucket_name>", "user123", "service456", "<aws_access_key>", "<aws_secret_key>", "<region_name>")
