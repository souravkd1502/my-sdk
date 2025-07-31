"""
metric_service.py
------------------
This module provides a MetricsCollector class to collect and store system usage metrics
for API calls in a Flask application. It supports various storage options including file,
memory, and database storage.

It also includes a MetricStorage class to handle the storage and retrieval of metrics data.
The MetricsCollector class collects metrics such as CPU usage, memory usage, execution time,
and request counts.

It provides methods to initialize the Flask app, collect metrics, reset metrics, save metrics to a file,
and visualize metrics in a dashboard format. The collected metrics can be viewed through a web interface
rendered by Flask, which displays performance metrics and error statistics for each API endpoint.

This module is designed to work without OpenTelemetry, making it suitable for both Windows and Linux environments.
It uses the psutil library to gather system metrics and matplotlib for visualizations.

This module is part of a larger application and is intended to be used as a reusable component for monitoring and 
performance tracking. It is structured to allow easy integration into existing Flask applications and provides a 
simple interface for collecting and storing metrics.

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
        
3. Run the Flask app:
    if __name__ == "__main__":
        app.run(debug=True)
        
4. Create `/view` endpoint to visualize metrics:
    @app.route("/view")
    def view_metrics():
        return metrics.visualize_metrics()

5. Access the metrics dashboard at `/view` to see the collected metrics and visualizations.

Dependencies:
-------------
- Flask: For creating the web application and handling requests.
- psutil: For collecting system metrics such as CPU and memory usage.
"""

# Adding directories to system path to allow importing custom modules
import os
import sys

sys.path.append("./")
sys.path.append("../")

# Importing necessary libraries and modules
import io
import time
import json
import redis
import psutil
import base64
import logging
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from flask import Flask, render_template_string, request, jsonify

from typing import Any, Dict, List, Literal

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


class MetricStorage:
    """
    A class to handle the storage of metrics data.

    This class provides methods to save metrics data to a file and load it from a file.
    It is used to persist the collected metrics data across application restarts.
    """

    def __init__(
        self,
        storage_type: Literal["file", "memory", "database", "instance"] = "instance",
    ):
        """
        Initialize the MetricStorage class.

        This class initializes an empty dictionary to store metrics data.
        """
        self.storage_type = storage_type

        if storage_type == "file":
            self.storage_path = "metrics_data.json"
        elif storage_type == "instance":
            self.metrics_data = {}

    def store_metrics(self, metrics_data: Dict[str, Any]) -> None:
        """
        Store the collected metrics data.

        Args:
        -----
        metrics_data : dict
            The metrics data to be stored.

        This method stores the collected metrics data in the specified storage type.
        If the storage type is 'file', it saves the data to a JSON file.
        If the storage type is 'memory', it stores the data in memory.
        If the storage type is 'database', it should implement database storage logic.
        If the storage type is 'instance', it stores the data in an instance variable.
        """
        if self.storage_type == "file":
            with open(self.storage_path, "w") as f:
                json.dump(metrics_data, f, indent=4)
                _logger.info(f"Metrics saved to {self.storage_path}")
        elif self.storage_type == "memory":
            self._save_to_redis(metrics_data)
        elif self.storage_type == "database":
            self._save_to_database(metrics_data)
        elif self.storage_type == "instance":
            self.metrics_data = metrics_data
        else:
            raise ValueError(
                "Invalid storage type. Must be one of: 'file', 'memory', 'database', 'instance'."
            )

    def get_metrics(self, metric_endpoint: List[str] = None) -> Dict[str, Any]:
        """
        Retrieve the collected metrics.

        Args:
        -----
        metric_endpoint : List[str], optional
            The endpoints to exclude from the returned metrics. If not provided, all metrics will be returned.

        Returns
        -------
        dict:
            A dictionary containing the collected metrics.
        """
        _logger.info("Retrieving collected metrics.")
        raw_metrics_data = self._retrieve_metrics_data()
        metrics_data = {}

        for endpoint, value in raw_metrics_data.items():
            if metric_endpoint and endpoint in metric_endpoint:
                continue  # Skip excluded endpoints

            if isinstance(value, (bytes, str)):
                try:
                    decoded = value.decode("utf-8") if isinstance(value, bytes) else value
                    metrics_data[endpoint] = json.loads(decoded)
                except Exception as e:
                    _logger.error(f"Failed to decode/parse metrics for {endpoint}: {e}")
                    metrics_data[endpoint] = {}
            else:
                metrics_data[endpoint] = value

        _logger.info(f"Metrics data: {metrics_data}")
        return metrics_data

    
    def _retrieve_metrics_data(self) -> Dict[str, Any]:
        """
        Retrieve metrics data based on the storage type.

        This method retrieves the metrics data from the specified storage type.
        If the storage type is 'file', it loads the data from a JSON file.
        If the storage type is 'memory', it retrieves the data from memory.
        If the storage type is 'database', it should implement database retrieval logic.
        If the storage type is 'instance', it returns the instance variable.

        Returns
        -------
        dict:
            The retrieved metrics data.
        """
        if self.storage_type == "file":
            with open(self.storage_path, "r") as f:
                return json.load(f)
        elif self.storage_type == "memory":
            return self._retrieve_from_redis()
        elif self.storage_type == "database":
            return self._retrieve_from_database()
        elif self.storage_type == "instance":
            return self.metrics_data
        else:
            raise ValueError(
                "Invalid storage type. Must be one of: 'file', 'memory', 'database', 'instance'."
            )

    def reset_metrics(self) -> None:
        """
        Reset the collected metrics.

        This method resets the collected metrics by clearing the metrics_data dictionary.
        """
        _logger.info("Resetting collected metrics.")
        self.metrics_data = {}

    def _save_to_redis(self, metrics_data: Dict[str, Any]) -> None:
        """
        Save metrics data to Redis.

        This method is a placeholder for saving metrics data to a Redis database.
        It should implement the logic to connect to Redis and store the metrics data.

        Args:
            metrics_data (dict): The metrics data to be saved.
        """
        redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=0,
        )
        for endpoint, data in metrics_data.items():
            redis_client.set(endpoint, json.dumps(data))
        _logger.info("Metrics data saved to Redis.")

    def _save_to_database(self, metrics_data: Dict[str, Any]) -> None:
        """
        Save metrics data to a database.

        This method is a placeholder for saving metrics data to a database.
        It should implement the logic to connect to the database and store the metrics data.

        Args:
            metrics_data (dict): The metrics data to be saved.
        """
        pass

    def _retrieve_from_redis(self) -> Dict[str, Any]:
        """
        Retrieve metrics data from Redis.

        This method is a placeholder for retrieving metrics data from a Redis database.
        It should implement the logic to connect to Redis and retrieve the metrics data.

        Returns:
            dict: The retrieved metrics data.
        """
        redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=0,
        )
        keys = redis_client.keys()
        metrics_data = {key.decode("utf-8"): redis_client.get(key) for key in keys}
        _logger.info("Metrics data retrieved from Redis.")
        return metrics_data

    def _retrieve_from_database(self) -> Dict[str, Any]:
        """
        Retrieve metrics data from a database.

        This method is a placeholder for retrieving metrics data from a database.
        It should implement the logic to connect to the database and retrieve the metrics data.

        Returns:
            dict: The retrieved metrics data.
        """
        return {}


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

    def __init__(
        self,
        storage_type: Literal["file", "memory", "database", "instance"] = "instance",
    ):
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
        self.storage = MetricStorage(storage_type=storage_type)
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
            self.start_time = time.time()
            _logger.info(
                f"MetricsCollector.start_timer: request.start_time = {self.start_time}"
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

    def get_metrics(self, metric_endpoint: List[str] = None) -> Dict[str, Any]:
        """
        Retrieve the collected metrics.

        Args:
        -----
        metric_endpoint : str, optional
            The endpoint to retrieve metrics for. If not provided, all metrics will be returned.

        Returns
        -------
        dict:
            A dictionary containing the collected metrics.
        """
        return self.storage.get_metrics(metric_endpoint)

    def reset_metrics(self) -> None:
        """
        Reset the collected metrics.

        This method resets the collected metrics by clearing the metrics_data dictionary.
        """
        _logger.info("Resetting collected metrics.")
        self.metrics_data = {}
        self.storage.reset_metrics()

    def save_metrics_to_file(self, filename: str = "metrics_data.json") -> None:
        """
        Save the collected metrics to a file.

        Args:
        -----
        filename : str, optional
            The filename to save the metrics to. Default is "metrics_data.json".
        """
        with open(filename, "w") as f:
            json.dump(self.metrics_data, f, indent=4)
        _logger.info(f"Metrics saved to {filename}")

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
        execution_time = (time.time() - self.start_time) * 1000
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

        # Store metrics in the specified storage type
        self.storage.store_metrics(self.metrics_data)

        return response

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
Metrics = MetricsCollector(storage_type="memory")
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
    app.run(debug=True, port=8080)
