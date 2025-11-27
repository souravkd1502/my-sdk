from flask import Flask, render_template
import sqlite3

app = Flask(__name__)
DB_PATH = "mcp_info.db"

def fetch_data(table_name):
    """Fetch all rows from a table."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute(f"SELECT * FROM {table_name}")
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]

@app.route("/")
def index():
    tools = fetch_data("tools")
    resources = fetch_data("resources")
    resource_templates = fetch_data("resource_templates")
    prompts = fetch_data("prompts")
    return render_template("index.html", tools=tools, resources=resources, resource_templates=resource_templates, prompts=prompts)

if __name__ == "__main__":
    app.run(debug=True)
