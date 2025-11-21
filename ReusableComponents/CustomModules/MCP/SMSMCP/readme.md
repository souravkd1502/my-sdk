# SMS MCP Server

A **Modular Communication Protocol (MCP) server** for sending SMS messages via templates or raw messages, designed 
for integration with **LLMs, automation agents, and messaging workflows**.

---

## Features

* **Template-Based SMS**: Send messages with dynamic variables using predefined templates.
* **Raw SMS**: Send custom messages without templates.
* **Template Management**: List available templates and preview them before sending.
* **LLM-Friendly**: All tool calls return JSON-compatible objects with deterministic structure.
* **Environment Safety**: Validates required environment variables before sending.
* **Logging & Error Handling**: Tracks all activity and raises informative exceptions.

---

## Requirements

* Python 3.10+
* Twilio account (for SMS delivery)
* MCP framework (e.g., FastMCP)
* Packages:

  ```bash
  pip install twilio python-dotenv
  ```

---

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/your-org/sms-mcp-server.git
   cd sms-mcp-server
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your Twilio credentials:

   ```text
   TWILIO_ACCOUNT_SID=your_account_sid
   TWILIO_AUTH_TOKEN=your_auth_token
   TWILIO_FROM_NUMBER=+1234567890
   ```

4. Add SMS templates in `templates/` directory:

   ```text
   templates/
   ├── otp.txt
   ├── welcome.txt
   └── receipt.txt
   ```

   Example template:

   ```text
   Hello {{name}}, your OTP is {{code}}. It expires in {{expires_in}}.
   ```

---

## Usage

### Starting the MCP Server

```python
from mcp.server.fastmcp import FastMCP
from tools.sms_tools import server  # Contains all tool definitions

server.run(port=8000)
```

### Tools

#### `send_sms_with_template`

Send an SMS using a template.

```python
{
  "tool": "send_sms_with_template",
  "arguments": {
    "phone_number": "+1234567890",
    "template_id": "otp",
    "variables": {
      "name": "Alice",
      "code": "123456",
      "expires_in": "5 minutes"
    }
  }
}
```

#### `send_sms_raw`

Send a raw SMS message.

```python
{
  "tool": "send_sms_raw",
  "arguments": {
    "phone_number": "+1234567890",
    "message": "Your order has shipped!"
  }
}
```

#### `list_templates`

List all available templates.

```python
{
  "tool": "list_templates",
  "arguments": {}
}
```

#### `preview_template`

Preview a template without sending.

```python
{
  "tool": "preview_template",
  "arguments": {
    "template_id": "receipt",
    "variables": {"name": "Bob", "amount": "$24.90"}
  }
}
```

---

## Error Handling

* Raises **`FileNotFoundError`** if a template is missing.
* Raises **`EnvironmentError`** if Twilio credentials are not set.
* Raises **`TwilioRestException`** for Twilio API issues.
* Missing template variables are replaced with empty strings (logged for debugging).

---

## Logging

* Uses Python’s built-in `logging` module.
* Logs all tool calls, errors, and missing template keys.

---

## LLM Integration

* Designed for **LLM agents** and **tool-driven workflows**.
* All tools return JSON objects with predictable fields (`status`, `delivery_id`, `message`).
* Supports dynamic variable replacement and template discovery for autonomous systems.

---

## License

MIT License
