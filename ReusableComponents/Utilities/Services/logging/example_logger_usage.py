"""
example_logger_usage.py

Demonstrates usage of EnhancedLogger, LogExporter, and LogDashboardViewer
from logger.py (excluding cloud upload features).

Run this script directly to:
- Log messages to file and SQLite DB
- Export logs to CSV and JSON
- Display logs in CLI

Requirements:
    pip install rich

Author: Example (2025)
"""

import os
import time
from logger import EnhancedLogger, LogExporter, LogDashboardViewer

LOG_FILE = "logs/example_app.log"
DB_PATH = "logs/example_app.db"
CSV_EXPORT = "logs/exported_logs.csv"
JSON_EXPORT = "logs/exported_logs.json"

def main():
    # Ensure log directory exists
    os.makedirs("logs", exist_ok=True)

    # 1. Initialize EnhancedLogger
    logger = EnhancedLogger(
        name="example_app",
        log_file=LOG_FILE,
        db_path=DB_PATH,
        service="ExampleService"
    )

    # 2. Log messages at various levels with/without metadata
    logger.log("info", "Application started")
    logger.log("debug", "Debugging details", user_id="user42", session_id="sess1")
    logger.log("warning", "Low disk space", organization="OrgA")
    logger.log("error", "An error occurred", user_id="user42", extra={"error_code": 123})
    logger.log("critical", "Critical failure!", session_id="sess1", organization="OrgA")
    logger.log("info", "User login", user_id="user99", session_id="sess2", organization="OrgB", extra={"ip": "127.0.0.1"})

    # Allow async DB logging to complete
    time.sleep(1)

    # 3. Export logs to CSV and JSON (from DB)
    exporter = LogExporter(
        source=DB_PATH,
        output_path=CSV_EXPORT,
        use_db=True
    )
    exporter.export_to_csv(filters={})  # Export all logs

    exporter = LogExporter(
        source=DB_PATH,
        output_path=JSON_EXPORT,
        use_db=True
    )
    exporter.export_to_json(filters={"level": "error"})  # Export only error logs

    print(f"Logs exported to {CSV_EXPORT} (all logs) and {JSON_EXPORT} (error logs)")

    # 4. Display logs in CLI using LogDashboardViewer
    print("\nDisplaying first page of logs from DB in CLI:")
    viewer = LogDashboardViewer(
        source=DB_PATH,
        source_type="db",
        page_size=10
    )
    viewer.display_cli(page=1)

    print("\nExample complete. Check the logs/ directory for output files.")
    
    # 5. Display logs in Web page using LogDashboardViewer
    viewer.start_html_dashboard(
        port=5000,
    )

if __name__ == "__main__":
    main()
