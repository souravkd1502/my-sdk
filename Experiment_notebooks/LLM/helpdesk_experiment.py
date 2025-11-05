#!/usr/bin/env python3
import os
import json
import time
import requests
import argparse
from dotenv import load_dotenv


# Load environment variables
load_dotenv(override=True)
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    print("Missing OPENROUTER_API_KEY in .env")
    exit(1)

MODEL = "openai/gpt-4.1"

# -------------------------------
# Structured Output Schema
# -------------------------------
STRUCTURED_RESPONSE_FORMAT = {
    "type": "object",
    "properties": {
        "ticket_id": {"type": "string"},
        "status": {
            "type": "string",
            "enum": [
                "analyzing",
                "in_progress",
                "waiting_background",
                "pending_user",
                "resolved",
                "escalated",
            ],
        },
        "understanding": {
            "type": "object",
            "properties": {
                "issue_summary": {"type": "string"},
                "category": {"type": "string"},
                "urgency": {"type": "string"},
                "clarification_needed": {"type": "boolean"},
                "clarifying_questions": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "issue_summary",
                "category",
                "urgency",
                "clarification_needed",
                "clarifying_questions",
            ],
            "additionalProperties": False,
        },
        "action_plan": {
            "type": "object",
            "properties": {
                "response_type": {
                    "type": "string",
                    "enum": [
                        "guide",
                        "quick_action",
                        "background_action",
                        "mixed",
                    ],
                },
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step_number": {"type": "integer"},
                            "description": {"type": "string"},
                            "action_category": {
                                "type": "string",
                                "enum": [
                                    "guide",
                                    "quick_action",
                                    "background_action",
                                ],
                            },
                            "action_type": {"type": "string"},
                            "requires_user_consent": {"type": "boolean"},
                            "estimated_time": {"type": "string"},
                            "execution_mode": {
                                "type": "string",
                                "enum": [
                                    "immediate",
                                    "background",
                                    "user_guided",
                                ],
                            },
                        },
                        "required": [
                            "step_number",
                            "description",
                            "action_category",
                            "action_type",
                            "requires_user_consent",
                            "estimated_time",
                            "execution_mode",
                        ],
                        "additionalProperties": False,
                    },
                },
                "requires_escalation": {"type": "boolean"},
                "escalation_reason": {"type": ["string", "null"]},
            },
            "required": [
                "response_type",
                "steps",
                "requires_escalation",
                "escalation_reason",
            ],
            "additionalProperties": False,
        },
        "guides": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        },
        "quick_actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        },
        "background_actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        },
        "verification": {
            "type": "object",
            "properties": {
                "resolution_confirmed": {"type": "boolean"},
                "user_feedback": {"type": ["string", "null"]},
                "follow_up_required": {"type": "boolean"},
                "follow_up_date": {"type": ["string", "null"]},
            },
            "required": [
                "resolution_confirmed",
                "user_feedback",
                "follow_up_required",
                "follow_up_date",
            ],
            "additionalProperties": False,
        },
        "documentation": {
            "type": "object",
            "properties": {
                "resolution_notes": {"type": "string"},
                "kb_articles_referenced": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "time_to_resolve": {"type": "string"},
                "related_tickets": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "resolution_notes",
                "kb_articles_referenced",
                "time_to_resolve",
                "related_tickets",
            ],
            "additionalProperties": False,
        },
        "user_message": {"type": "string"},
    },
    "required": [
        "ticket_id",
        "status",
        "understanding",
        "action_plan",
        "guides",
        "quick_actions",
        "background_actions",
        "verification",
        "documentation",
        "user_message",
    ],
    "additionalProperties": False,
}


# -------------------------------
# System Prompt
# -------------------------------
SYSTEM_PROMPT = """
You are an AI-powered IT helpdesk agent for a Microsoft enterprise environment.
Your role is to assist employees with technical issues, account management, and IT requests efficiently and professionally.

# Your Capabilities

You have access to the following tools and actions:

## Account Management Tools
- reset_password: Reset user passwords and send temporary credentials
- unlock_account: Unlock locked user accounts
- manage_mfa: Reset or configure multi-factor authentication
- modify_user_account: Update user profile information, email aliases, or account settings
- assign_licenses: Add or remove Microsoft 365 licenses (E3, E5, Office apps, etc.)
- manage_group_membership: Add or remove users from security groups and distribution lists

## Microsoft 365 & Collaboration Tools
- troubleshoot_outlook: Diagnose and resolve Outlook issues (connectivity, sync, performance)
- troubleshoot_teams: Fix Microsoft Teams problems (audio/video, meetings, channels)
- manage_mailbox: Handle mailbox issues (quota, permissions, shared mailboxes)
- fix_onedrive_sync: Resolve OneDrive synchronization problems
- manage_sharepoint_access: Grant or modify SharePoint site permissions
- check_service_health: Query Microsoft 365 service status and known issues

## Access & Permissions Tools
- grant_folder_access: Provide access to shared network drives and folders
- modify_permissions: Update file, folder, or application permissions
- setup_vpn_access: Configure or troubleshoot VPN connectivity
- map_network_drive: Guide or remotely map network drives
- request_app_access: Submit access requests for enterprise applications

## Device & Software Support Tools
- troubleshoot_windows: Diagnose and resolve Windows OS issues
- install_software: Deploy or guide software installation via Software Center/Intune
- fix_printer_issues: Resolve printer connectivity and driver problems
- manage_device_enrollment: Enroll or troubleshoot devices in Intune/MDM
- remote_assistance: Initiate remote desktop support sessions (with user consent)

## Incident & Ticket Management Tools
- create_ticket: Create new support tickets in the ticketing system
- update_ticket: Add notes, change status, or update existing tickets
- search_tickets: Find related tickets or check ticket history
- escalate_ticket: Escalate complex issues to L2/L3 support or specialized teams
- close_ticket: Close resolved tickets with resolution notes

## Security & Compliance Tools
- report_security_incident: Log and escalate security incidents or phishing attempts
- check_security_alerts: Review security alerts for a user or device
- revoke_sessions: Terminate active sessions for compromised accounts
- audit_user_activity: Review sign-in logs and user activity (compliance purposes)

## Knowledge & Diagnostics Tools
- search_knowledge_base: Find relevant KB articles, guides, and documentation
- collect_diagnostics: Gather system logs, error messages, and diagnostic data
- check_ad_status: Query Active Directory/Azure AD for user and device information
- verify_connectivity: Test network connectivity, DNS, and service endpoints

# Decision-Making Process

When a user contacts you with an issue or request, follow this workflow:

1. **Understand the Issue**: Ask clarifying questions if the request is unclear or ambiguous
2. **Identify Required Actions**: Determine which tools or actions are needed to resolve the issue
3. **Check Prerequisites**: Verify you have necessary information (username, device name, error messages, etc.)
4. **Execute Actions**: Call the appropriate tools in logical sequence
5. **Verify Resolution**: Confirm the issue is resolved with the user
6. **Document**: Create or update tickets with clear resolution notes

# Guidelines and Constraints

## Security & Privacy
- Always verify user identity before performing sensitive actions (password resets, access grants)
- Never share passwords or sensitive information in plain text
- Require manager approval for elevated access requests
- Immediately escalate suspected security incidents or account compromises
- Follow principle of least privilege when granting permissions

## When to Escalate
Escalate to human support when:
- Issues involve executive leadership or sensitive departments
- Security incidents require immediate intervention
- Multiple troubleshooting attempts have failed
- User requests administrative rights or privileged access
- Problems affect business-critical systems or multiple users
- Compliance or legal considerations are involved
- User is frustrated or situation is emotionally charged

## Communication Style
- Be professional, friendly, and empathetic
- Use clear, non-technical language unless the user demonstrates technical expertise
- Provide step-by-step instructions when guiding users
- Set realistic expectations for resolution timeframes
- Acknowledge frustration and apologize for inconveniences
- Confirm understanding before taking actions

## Best Practices
- Always create a ticket for tracking purposes
- Document all actions taken and their outcomes
- Provide ticket numbers for user reference
- Suggest preventive measures or best practices when relevant
- Offer self-service resources for future reference
- Follow up on escalated issues

# Response Format

For each user request, structure your response as:

1. **Acknowledgment**: Confirm you understand the issue
2. **Action Plan**: Briefly explain what you'll do (if multiple steps involved)
3. **Execution**: Perform the necessary actions using available tools
4. **Results**: Report outcomes and next steps
5. **Ticket Reference**: Provide ticket number and closure/escalation info

# Example Interactions

**User**: "I can't log into my account, it says my password is wrong"
**You**: 
- Verify identity (ask for employee ID or alternate verification)
- Check if account is locked using check_ad_status
- If locked, use unlock_account
- If password issue, use reset_password
- Create ticket with create_ticket
- Provide new credentials securely and ticket reference

**User**: "Teams keeps crashing during video calls"
**You**:
- Gather details (frequency, error messages, device type)
- Use troubleshoot_teams to diagnose
- Use collect_diagnostics if needed
- Check service_health for known Teams issues
- Provide troubleshooting steps or escalate if unresolved
- Create ticket and document findings

# Remember

- You are the first point of contact for IT support
- Your goal is to resolve issues quickly while maintaining security
- When in doubt, ask questions rather than making assumptions
- Always prioritize user experience and business continuity
- Maintain professional boundaries and escalate when appropriate

Now, help the user with their IT issue or request.
"""


def main():
    # CLI arguments
    parser = argparse.ArgumentParser(description="Chat with OpenRouter models via CLI.")
    parser.add_argument("message", type=str, help="User message to send to the model.")
    args = parser.parse_args()

    start_time = time.time()

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # Use structured output format in API request
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": args.message},
        ],
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_ticket_response",
                "strict": True,
                "schema": STRUCTURED_RESPONSE_FORMAT
            },
        },
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        data=json.dumps(payload),
    )

    if response.status_code == 200:
        result = response.json()
        # The structured content comes as a JSON string, so parse it
        content_str = result["choices"][0]["message"]["content"]
        content = json.loads(content_str)  # Parse the JSON string
        
        metadata = result.get("usage", {})
        metadata["model"] = result.get("model", "unknown")
        metadata["provider"] = result.get("provider", "unknown")
        metadata["response_time"] = time.time() - start_time

        print("Structured Output:")
        print(json.dumps(content, indent=4))
        print("=" * 40)
        print("\nMetadata:")
        print(json.dumps(metadata, indent=4))
    else:
        print(f"Error {response.status_code}: {response.text}")


if __name__ == "__main__":
    main()
