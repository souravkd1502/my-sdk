"""
Usage:
------

# Initialize Flask and MetricsCollector
app = Flask(__name__)
Metrics = MetricsCollector()
Metrics.init_app(app)


@app.route("/data")
def data():
    '''Test API Endpoint'''
    return jsonify({"message": "Metrics collected!"})


@app.route("/metrics")
def get_metrics():
    '''Endpoint to view collected metrics'''
    return jsonify(Metrics.get_metrics(metric_endpoint=["/view", "/metrics"]))


@app.route("/reset")
def reset_metrics():
    '''Endpoint to reset collected metrics'''
    Metrics.reset_metrics()
    return jsonify({"message": "Metrics reset!"})


@app.route("/save")
def save_metrics():
    '''Endpoint to save collected metrics to a file'''
    Metrics.save_metrics_to_file()
    return jsonify({"message": "Metrics saved!"})


@app.route("/view")
def index():
    '''Renders the dashboard'''
    return Metrics.visualize_metrics()


if __name__ == "__main__":
    app.run(debug=True, port=8080)

"""

# Adding directories to system path to allow importing custom modules
import sys

sys.path.append("./")
sys.path.append("../")

# Importing necessary libraries and modules
import io
import csv
import time
import json
import redis
import psutil
import base64
import logging
import pymongo
import psycopg2
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from flask import Flask, render_template_string, request, jsonify
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")

# Set up logging
_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - line: %(lineno)d",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Load Environment variables
load_dotenv(override=True)


class MetricsCollector:
    """
    A wrapper to collect system usage metrics for API calls without using OpenTelemetry.
    Works on both Windows and Linux.

    Purpose:
    --------
        self._metrics_data = {}
    and execution time for each API endpoint in a Flask application. It helps in monitoring
    the performance and resource consumption of the API.

    Usage:
    ------
    1. Initialize the Flask app and MetricsCollector:
        app = Flask(__name__)
        metrics = MetricsCollector()
        metrics.init_app(app)

    2. Define your API endpoints as usual:
        @app.route("/data")
        def data():
            return jsonify({"message": "Metrics collected!"})

        @app.route("/metrics")
        def get_metrics():
            return jsonify(metrics.get_metrics())

    3. Run the Flask app:
        if __name__ == "__main__":
        if endpoint not in self._metrics_data:
            self._metrics_data[endpoint] = {"count": 0, "cpu": [], "memory": [], "time": []}
    """

    def __init__(self):
        """
        Initialize the MetricsCollector.

        This class stores metrics of each API call in a dictionary. The keys are the
        endpoint paths, and the values are dictionaries containing the count of calls,
        CPU usage, memory usage, and execution time.

        This class also provides methods to retrieve, reset, and save the collected metrics.

        Returns
        -------
        None
        """
        self.metrics_data = {}  # Store metrics for each endpoint
        self.process = (
            psutil.Process()
        )  # Store the process object to measure CPU and memory usage

    def start_timer(self) -> None:
        """
        Store the request start time.

        This method is registered as a before_request handler in the init_app method.
        It stores the current time in the request object under the attribute "start_time".
        """
        _logger.info("MetricsCollector.start_timer called")
        if not hasattr(request, "start_time"):
            request.start_time = time.time()
            _logger.info(
                f"MetricsCollector.start_timer: request.start_time = {request.start_time}"
            )

    def init_app(self, app: Flask) -> None:
        """
        Initialize the Flask app with metrics tracking.

        This method registers the start_timer and collect_metrics methods as before_request and
        after_request handlers, respectively. This allows the metrics to be collected automatically
        for each API call without requiring any additional code.

        Parameters
        ----------
        app : Flask
            The Flask application instance to be initialized with metrics tracking.

        Returns
        -------
        None
        """
        _logger.info("MetricsCollector.init_app called")
        app.before_request(self.start_timer)
        _logger.info("MetricsCollector.init_app: before_request handler set")
        app.after_request(self.collect_metrics)

    def collect_metrics(self, response):
        """
        Collect metrics for the current API call.

        This method collects metrics for the current API call and stores them in the
        metrics_data dictionary. The metrics collected are:

        - count: The number of times the API call was made.
        - cpu: The CPU usage as a percentage.
        - memory: The memory usage in megabytes.
        - time: The execution time in milliseconds.
        - timestamp: The timestamp of when the API call was made.
        - request_size: The size of the request in bytes.
        - response_size: The size of the response in bytes.
        - error_count: The number of times the API call resulted in an error.

        Parameters
        ----------
        response : flask.Response
            The response object of the current API call.

        Returns
        -------
        flask.Response
            The response object of the current API call.
        """
        # Get the endpoint and request method
        endpoint = getattr(request, "path", "unknown")
        method = getattr(request, "method", "unknown")
        _logger.info(f"Collecting metrics for {method} {endpoint}")

        # Measure CPU and memory usage
        cpu_usage = psutil.cpu_percent(percpu=True)
        cpu_usage = sum(cpu_usage) / len(cpu_usage)
        memory_usage = self.process.memory_info().rss / (1024 * 1024)
        _logger.info(f"CPU Usage: {cpu_usage}%")
        _logger.info(f"Memory Usage: {memory_usage} MB")

        # Measure execution time in milliseconds
        execution_time = (time.time() - request.start_time) * 1000
        _logger.info(f"Execution Time: {execution_time} ms")

        request_size = request.content_length or 0
        response_size = len(response.data) if response.data else 0
        status_code = response.status_code

        # Store metrics
        if endpoint not in self.metrics_data:
            _logger.info(f"Initializing metrics storage for {endpoint}")
            self.metrics_data[endpoint] = {
                "count": 0,
                "cpu": [],
                "memory": [],
                "time": [],
                "timestamp": [],
                "request_size": [],
                "response_size": [],
                "error_count": 0,
            }

        self.metrics_data[endpoint]["request_size"].append(request_size)
        self.metrics_data[endpoint]["response_size"].append(response_size)
        self.metrics_data[endpoint]["count"] += 1
        self.metrics_data[endpoint]["cpu"].append(cpu_usage)
        self.metrics_data[endpoint]["memory"].append(memory_usage)
        self.metrics_data[endpoint]["time"].append(execution_time)
        self.metrics_data[endpoint]["timestamp"].append(
            time.strftime("%Y-%m-%d %H:%M:%S")
        )
        self.metrics_data[endpoint]["method"] = method

        if status_code >= 400:
            self.metrics_data[endpoint]["error_count"] += 1

        # Calculate average, maximum, minimum CPU and memory usage
        self.metrics_data[endpoint]["avg_cpu"] = sum(
            self.metrics_data[endpoint]["cpu"]
        ) / len(self.metrics_data[endpoint]["cpu"])
        self.metrics_data[endpoint]["avg_memory"] = sum(
            self.metrics_data[endpoint]["memory"]
        ) / len(self.metrics_data[endpoint]["memory"])
        self.metrics_data[endpoint]["avg_time"] = sum(
            self.metrics_data[endpoint]["time"]
        ) / len(self.metrics_data[endpoint]["time"])
        self.metrics_data[endpoint]["max_cpu"] = max(self.metrics_data[endpoint]["cpu"])
        self.metrics_data[endpoint]["max_memory"] = max(
            self.metrics_data[endpoint]["memory"]
        )
        self.metrics_data[endpoint]["max_time"] = max(
            self.metrics_data[endpoint]["time"]
        )
        self.metrics_data[endpoint]["min_cpu"] = min(self.metrics_data[endpoint]["cpu"])
        self.metrics_data[endpoint]["min_memory"] = min(
            self.metrics_data[endpoint]["memory"]
        )
        self.metrics_data[endpoint]["min_time"] = min(
            self.metrics_data[endpoint]["time"]
        )

        # Calculate percentiles for request time (P50, P90, P99)
        times = sorted(self.metrics_data[endpoint]["time"])
        self.metrics_data[endpoint]["p50"] = (
            times[int(len(times) * 0.50)] if times else 0
        )
        self.metrics_data[endpoint]["p90"] = (
            times[int(len(times) * 0.90)] if times else 0
        )
        self.metrics_data[endpoint]["p99"] = (
            times[int(len(times) * 0.99)] if times else 0
        )

        _logger.info(
            f"Metrics collected for {method} {endpoint}: {self.metrics_data[endpoint]}"
        )

        return response

    def get_metrics(self, metric_endpoint: List[str] = None) -> Dict[str, Any]:
        """
        Retrieve the collected metrics.

        Args:
        -----
        metric_endpoint : str, optional
            The endpoint to retrieve metrics for. If not provided, all metrics will be returned.

        This method returns a dictionary where each key is an endpoint and the value is another dictionary containing the following metrics:

        - count (int): Number of times the endpoint was called.
        - cpu (list): List of CPU usage percentages for each call.
        - memory (list): List of memory usage in MB for each call.
        - time (list): List of execution times for each call.
        - method (str): HTTP method used for the endpoint.
        - avg_cpu (float): Average CPU usage.
        - avg_memory (float): Average memory usage.
        - avg_time (float): Average execution time.
        - max_cpu (float): Maximum CPU usage.
        - max_memory (float): Maximum memory usage.
        - max_time (float): Maximum execution time.
        - min_cpu (float): Minimum CPU usage.
        - min_memory (float): Minimum memory usage.
        - min_time (float): Minimum execution time.
        - requested_at (str): Timestamp of the last request in 'YYYY-MM-DD HH:MM:SS' format.

        Returns
        -------
        dict:
            A dictionary containing the collected metrics.
        """
        _logger.info("Retrieving collected metrics.")
        if metric_endpoint:
            _logger.info("No endpoint provided. Returning all metrics.")
            for endpoint in metric_endpoint:
                if endpoint and endpoint in self.metrics_data.keys():
                    del self.metrics_data[endpoint]

        metrics_data = self.metrics_data
        _logger.info(f"Metrics data: {metrics_data}")
        return metrics_data

    def reset_metrics(self) -> None:
        """
        Reset the collected metrics.

        This method resets the collected metrics by clearing the metrics_data dictionary.
        """
        _logger.info("Resetting collected metrics.")
        self.metrics_data = {}

    def save_metrics_to_file(self, filename="metrics.json"):
        """
        Save the collected metrics to a file.

        This method saves the collected metrics to a file in JSON format.

        Args:
            filename (str): The name of the file to save the metrics to. Defaults to "metrics.json".
        """
        with open(filename, "w") as f:
            json.dump(self.metrics_data, f, indent=4)
            _logger.info(f"Metrics saved to {filename}")

    def visualize_metrics(self):
        """Renders the dashboard with performance metrics and error statistics."""
        data = self.get_metrics(metric_endpoint=["/view", "/metrics", "/favicon.ico"])

        tables = {}
        plots = {}

        for endpoint, metrics in data.items():
            stats_df = pd.DataFrame(
                {
                    "Metric": ["Average", "Max", "Min"],
                    "CPU Usage (%)": [
                        metrics.get("avg_cpu", 0),
                        metrics.get("max_cpu", 0),
                        metrics.get("min_cpu", 0),
                    ],
                    "Memory Usage (MB)": [
                        metrics.get("avg_memory", 0),
                        metrics.get("max_memory", 0),
                        metrics.get("min_memory", 0),
                    ],
                    "Response Time (MS)": [
                        metrics.get("avg_time", 0),
                        metrics.get("max_time", 0),
                        metrics.get("min_time", 0),
                    ],
                }
            )

            summary_df = pd.DataFrame(
                {
                    "Total Requests": [metrics.get("count", 0)],
                    "Errors": [metrics.get("error_count", 0)],
                    "p50 (MS)": [metrics.get("p50", 0)],
                    "p90 (MS)": [metrics.get("p90", 0)],
                    "p99 (MS)": [metrics.get("p99", 0)],
                }
            )

            tables[endpoint] = {
                "stats": stats_df.to_html(classes="table table-striped", index=False),
                "summary": summary_df.to_html(
                    classes="table table-striped", index=False
                ),
            }
            plots[endpoint] = self.generate_plot(metrics)

        HTML = """
        <!DOCTYPE html>
        <html lang="en">
        <style>
            .table th {
                background-color: #f8f9fa;
                font-weight: bold;
                text-align: center;
            }
            .table td {
                text-align: center;
                vertical-align: middle;
            }
        </style>

        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Dashboard</title>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
        </head>
        <body>
            <div class="container mt-4">
                <h1 class="text-center">Metrics Dashboard</h1>
                
                {% for endpoint, tables in tables.items() %}
                <div class="mt-5">
                    <h3>Endpoint: {{ endpoint }}</h3>
                    
                    <div class="table-responsive">
                        <h5>Performance Metrics</h5>
                        {{ tables.stats | safe }}
                    </div>

                    <div class="table-responsive mt-3">
                        <h5>Summary Statistics</h5>
                        {{ tables.summary | safe }}
                    </div>

                    <div class="text-center mt-3">
                        <img src="data:image/png;base64,{{ plots[endpoint] }}" alt="Plot for {{ endpoint }}" class="img-fluid">
                    </div>
                </div>
                {% endfor %}
            </div>

            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        </body>
        </html>
        """

        return render_template_string(HTML, tables=tables, plots=plots)

    @staticmethod
    def generate_plot(data):
        """Generates base64-encoded plot images."""
        time_series_df = pd.DataFrame(
            {
                "Timestamp": pd.to_datetime(data["timestamp"]),
                "CPU": data["cpu"],
                "Memory": data["memory"],
                "Response Time": data["time"],
                "Request Size": data["request_size"],
                "Response Size": data["response_size"],
            }
        )

        _, ax = plt.subplots(5, 1, figsize=(10, 8))

        ax[0].plot(
            time_series_df["Timestamp"],
            time_series_df["CPU"],
            marker="o",
            linestyle="-",
            color="b",
        )
        ax[0].set_title("CPU Usage Over Time")
        ax[0].set_ylabel("CPU (%)")
        ax[0].grid(True)

        ax[1].plot(
            time_series_df["Timestamp"],
            time_series_df["Memory"],
            marker="o",
            linestyle="-",
            color="g",
        )
        ax[1].set_title("Memory Usage Over Time")
        ax[1].set_ylabel("Memory (MB)")
        ax[1].grid(True)

        ax[2].plot(
            time_series_df["Timestamp"],
            time_series_df["Response Time"],
            marker="o",
            linestyle="-",
            color="r",
        )
        ax[2].set_title("Response Time Over Time")
        ax[2].set_ylabel("Response Time (ms)")
        ax[2].grid(True)

        ax[3].plot(
            time_series_df["Timestamp"],
            time_series_df["Request Size"],
            marker="o",
            linestyle="-",
            color="y",
        )
        ax[3].set_title("Request Size Over Time")
        ax[3].set_ylabel("Request Size (bytes)")
        ax[3].grid(True)

        ax[4].plot(
            time_series_df["Timestamp"],
            time_series_df["Response Size"],
            marker="o",
            linestyle="-",
            color="c",
        )
        ax[4].set_title("Response Size Over Time")
        ax[4].set_ylabel("Response Size (bytes)")
        ax[4].grid(True)

        plt.xticks(rotation=45)
        plt.tight_layout()

        img = io.BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return plot_url


# Initialize Flask and MetricsCollector
app = Flask(__name__)
Metrics = MetricsCollector()
Metrics.init_app(app)


@app.route("/data")
def data():
    """Test API Endpoint"""
    return jsonify({"message": "Metrics collected!"})


@app.route("/data2")
def data2():
    """Test API Endpoint"""
    import random

    int_value = random.randint(1, 100)
    return jsonify({"message": "Metrics collected!", "random_int": int_value})


@app.route("/metrics")
def get_metrics():
    """Endpoint to view collected metrics"""
    return jsonify(Metrics.get_metrics(metric_endpoint=["/view", "/metrics"]))


@app.route("/reset")
def reset_metrics():
    """Endpoint to reset collected metrics"""
    Metrics.reset_metrics()
    return jsonify({"message": "Metrics reset!"})


@app.route("/save")
def save_metrics():
    """Endpoint to save collected metrics to a file"""
    Metrics.save_metrics_to_file()
    return jsonify({"message": "Metrics saved!"})


@app.route("/view")
def index():
    """Renders the dashboard"""
    return Metrics.visualize_metrics()


if __name__ == "__main__":
    app.run(debug=False, port=8080)
