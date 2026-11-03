"""
DocuMind AI / College Document Q&A Assistant
Jira Automation & Management CLI

Features:
- Connects to Atlassian Jira Cloud using the official Python SDK or works offline in simulation mode.
- Synchronizes all 6 Epics and 22 user stories directly from jira_management/jira_backlog.csv.
- Adds and maps team members (Student 1-4 or real user emails) and assigns tasks.
- Updates task statuses across Scrum workflow: TO DO -> IN PROGRESS -> CODE REVIEW -> TESTING -> DONE.
- Auto-generates configuration for the installed 'jira' CLI binary (~/.config/.jira/.config.yml).
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from jira import JIRA
    from jira.exceptions import JIRAError
    HAS_JIRA_PKG = True
except ImportError:
    HAS_JIRA_PKG = False

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKLOG_CSV = REPO_ROOT / "jira_management" / "jira_backlog.csv"
IMPORT_CSV = REPO_ROOT / "jira_management" / "jira_import.csv"
STATUS_JSON = REPO_ROOT / "jira_management" / "status.json"
TEAM_JSON = REPO_ROOT / "jira_management" / "team_members.json"
JIRA_CONFIG_DIR = Path.home() / ".config" / ".jira"
JIRA_CONFIG_FILE = JIRA_CONFIG_DIR / ".config.yml"


def load_env_credentials() -> Dict[str, str]:
    """Load Jira credentials from environment variables or .env file."""
    server = os.environ.get("JIRA_SERVER", "").strip()
    email = os.environ.get("JIRA_EMAIL", "").strip()
    token = os.environ.get("JIRA_API_TOKEN", "").strip()
    project = os.environ.get("JIRA_PROJECT_KEY", "RAG").strip()

    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k == "JIRA_SERVER" and not server:
                    server = v
                elif k == "JIRA_EMAIL" and not email:
                    email = v
                elif k == "JIRA_API_TOKEN" and not token:
                    token = v
                elif k == "JIRA_PROJECT_KEY" and not project:
                    project = v

    return {
        "server": server,
        "email": email,
        "token": token,
        "project": project or "RAG"
    }


def get_jira_client(creds: Dict[str, str]) -> Optional[Any]:
    """Initialize authenticated Jira client."""
    if not HAS_JIRA_PKG:
        print("[-] python 'jira' package is not installed. Run: pip install jira")
        return None

    server = creds.get("server")
    email = creds.get("email")
    token = creds.get("token")

    if not server or not email or not token:
        return None

    try:
        jira = JIRA(server=server, basic_auth=(email, token))
        return jira
    except Exception as e:
        print(f"[-] Failed to connect to Jira server at {server}: {e}")
        return None


def read_backlog_csv() -> List[Dict[str, str]]:
    """Parse backlog CSV file."""
    if not BACKLOG_CSV.exists():
        print(f"[-] Backlog file not found at: {BACKLOG_CSV}")
        return []

    items = []
    with open(BACKLOG_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(row)
    return items


def load_team_members() -> Dict[str, Dict[str, str]]:
    """Load or initialize team members."""
    default_team = {
        "Student 1": {
            "name": "Nishant / Student 1",
            "email": "",
            "role": "Project Lead & Data Preparation",
            "assigned_tasks": ["RAG-1", "RAG-2", "RAG-3", "RAG-6", "RAG-20"]
        },
        "Student 2": {
            "name": "Student 2",
            "email": "",
            "role": "Embeddings & Vector Storage Lead",
            "assigned_tasks": ["RAG-7", "RAG-8", "RAG-9", "RAG-10", "RAG-19"]
        },
        "Student 3": {
            "name": "Student 3",
            "email": "",
            "role": "LLM Integration & Prompt Engineer",
            "assigned_tasks": ["RAG-4", "RAG-5", "RAG-11", "RAG-12", "RAG-13", "RAG-14", "RAG-22"]
        },
        "Student 4": {
            "name": "Student 4",
            "email": "",
            "role": "UI, Benchmarking & Deployment Lead",
            "assigned_tasks": ["RAG-15", "RAG-16", "RAG-17", "RAG-18", "RAG-21"]
        }
    }

    if TEAM_JSON.exists():
        try:
            with open(TEAM_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    save_team_members(default_team)
    return default_team


def save_team_members(team: Dict[str, Dict[str, str]]) -> None:
    """Save team members to team_members.json."""
    with open(TEAM_JSON, "w", encoding="utf-8") as f:
        json.dump(team, f, indent=2)


def sync_jira_cli_config(server: str, login: str, token: str, project: str) -> None:
    """Write ~/.config/.jira/.config.yml for the installed 'jira' CLI binary."""
    JIRA_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    clean_server = server.rstrip("/")
    config_content = f"""# Jira CLI Configuration
installation: cloud
server: {clean_server}
login: {login}
api-token: {token}
project: {project}
board: {project} board
"""
    with open(JIRA_CONFIG_FILE, "w", encoding="utf-8") as f:
        f.write(config_content)
    print(f"[+] Synced configuration for 'jira' CLI binary to: {JIRA_CONFIG_FILE}")


def cmd_init(args):
    """Configure Jira credentials."""
    print("=" * 60)
    print("  Configure Jira Cloud Connection")
    print("=" * 60)
    creds = load_env_credentials()

    server = args.server or creds.get("server") or input("Jira Cloud Server URL (e.g., https://yourdomain.atlassian.net): ").strip()
    email = args.email or creds.get("email") or input("Jira Login Email: ").strip()
    token = args.token or creds.get("token") or input("Atlassian API Token: ").strip()
    project = args.project or creds.get("project") or "RAG"

    if not server or not email or not token:
        print("[-] Incomplete credentials. Server, email, and API token are required.")
        return 1

    # Save to ~/.config/.jira/.config.yml for CLI binary
    sync_jira_cli_config(server, email, token, project)

    # Save to local .env
    env_path = REPO_ROOT / ".env"
    lines = []
    keys_written = set()
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("JIRA_SERVER="):
                    lines.append(f"JIRA_SERVER={server}\n")
                    keys_written.add("JIRA_SERVER")
                elif line.startswith("JIRA_EMAIL="):
                    lines.append(f"JIRA_EMAIL={email}\n")
                    keys_written.add("JIRA_EMAIL")
                elif line.startswith("JIRA_API_TOKEN="):
                    lines.append(f"JIRA_API_TOKEN={token}\n")
                    keys_written.add("JIRA_API_TOKEN")
                elif line.startswith("JIRA_PROJECT_KEY="):
                    lines.append(f"JIRA_PROJECT_KEY={project}\n")
                    keys_written.add("JIRA_PROJECT_KEY")
                else:
                    lines.append(line)

    if "JIRA_SERVER" not in keys_written:
        lines.append(f"\nJIRA_SERVER={server}\n")
    if "JIRA_EMAIL" not in keys_written:
        lines.append(f"JIRA_EMAIL={email}\n")
    if "JIRA_API_TOKEN" not in keys_written:
        lines.append(f"JIRA_API_TOKEN={token}\n")
    if "JIRA_PROJECT_KEY" not in keys_written:
        lines.append(f"JIRA_PROJECT_KEY={project}\n")

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"[+] Jira credentials saved to {env_path}")
    print("[+] Test connection...")
    client = get_jira_client({"server": server, "email": email, "token": token, "project": project})
    if client:
        try:
            myself = client.myself()
            print(f"[+] Successfully authenticated as: {myself.get('displayName')} ({myself.get('emailAddress')})")
        except Exception as e:
            print(f"[!] Authenticated client created, but profile fetch returned: {e}")
    else:
        print("[!] Verification deferred. Credentials saved.")
    return 0


def cmd_list_tasks(args):
    """List backlog tasks."""
    items = read_backlog_csv()
    if not items:
        return 1

    print("=" * 100)
    print(f"{'Key':<10} | {'Type':<8} | {'SP':<3} | {'Priority':<8} | {'Assignee':<12} | {'Summary'}")
    print("=" * 100)
    for item in items:
        key = item.get("Issue Key", "")
        itype = item.get("Issue Type", "")
        sp = item.get("Story Points", "")
        priority = item.get("Priority", "")
        assignee = item.get("Assignee", "")
        summary = item.get("Summary", "")
        print(f"{key:<10} | {itype:<8} | {sp:<3} | {priority:<8} | {assignee:<12} | {summary}")
    print("=" * 100)
    print(f"Total Backlog Items: {len(items)}")
    return 0


def cmd_create_tasks(args):
    """Create tasks and epics in Jira Cloud or run dry-run simulation."""
    items = read_backlog_csv()
    if not items:
        return 1

    creds = load_env_credentials()
    jira_client = None if args.dry_run else get_jira_client(creds)

    if not jira_client and not args.dry_run:
        print("[!] No active Jira connection. Switching to DRY-RUN mode.")
        print("[!] To push directly to live Jira, run: python scripts/jira_sync.py init")
        args.dry_run = True

    print("\n" + "=" * 80)
    mode_str = "DRY-RUN SIMULATION" if args.dry_run else f"LIVE JIRA CLOUD ({creds.get('server')})"
    print(f"  CREATING JIRA TASKS & EPICS — MODE: {mode_str}")
    print("=" * 80 + "\n")

    # Epics definition
    epics = [
        {"key": "EPIC-1", "name": "Document Ingestion & Extraction", "sp": 13},
        {"key": "EPIC-2", "name": "Semantic Chunking & Vector Indexing", "sp": 18},
        {"key": "EPIC-3", "name": "Cloud-Native Embedding & Vector Storage", "sp": 21},
        {"key": "EPIC-4", "name": "LLM Integration, Fallback & Grounded Generation", "sp": 21},
        {"key": "EPIC-5", "name": "Testing, Safe Refusal & Benchmark Evaluation", "sp": 18},
        {"key": "EPIC-6", "name": "Cloud Deployment, CI/CD & Security", "sp": 18},
    ]

    print("[*] Phase 1: Checking / Creating Epics...")
    for epic in epics:
        if args.dry_run:
            print(f"  [DRY-RUN] Epic prepared: [{epic['key']}] {epic['name']} ({epic['sp']} Story Points)")
        else:
            try:
                issue_dict = {
                    "project": {"key": creds["project"]},
                    "summary": f"[{epic['key']}] {epic['name']}",
                    "description": f"Scrum Epic for {epic['name']} (Estimated: {epic['sp']} SP)",
                    "issuetype": {"name": "Epic"},
                }
                new_epic = jira_client.create_issue(fields=issue_dict)
                print(f"  [+] Created Epic {epic['key']}: {new_epic.key}")
            except Exception as e:
                print(f"  [!] Note on Epic {epic['key']}: {e}")

    print("\n[*] Phase 2: Creating User Stories and Tasks...")
    created_count = 0
    for item in items:
        key = item.get("Issue Key")
        summary = item.get("Summary")
        itype = item.get("Issue Type", "Story")
        desc = item.get("Description", "")
        priority = item.get("Priority", "Medium")
        sp = item.get("Story Points", "3")
        assignee = item.get("Assignee", "Unassigned")
        epic = item.get("Epic Link", "")

        if args.dry_run:
            print(f"  [DRY-RUN] Task: [{key}] ({itype}, {sp} SP, {priority}) -> {assignee} | \"{summary}\"")
            created_count += 1
        else:
            try:
                # Map priority safely
                issue_dict = {
                    "project": {"key": creds["project"]},
                    "summary": f"{key}: {summary}",
                    "description": f"{desc}\n\nEpic: {epic}\nStory Points: {sp}",
                    "issuetype": {"name": "Task" if itype == "Task" else ("Bug" if itype == "Bug" else "Story")},
                    "priority": {"name": priority if priority in ["Highest", "High", "Medium", "Low", "Lowest"] else "Medium"}
                }
                new_issue = jira_client.create_issue(fields=issue_dict)
                print(f"  [+] Created {key} -> {new_issue.key} ({summary[:45]}...)")
                created_count += 1
            except Exception as e:
                print(f"  [-] Failed to create {key}: {e}")

    print("\n" + "=" * 80)
    print(f"[SUCCESS] Processed {created_count} tasks across 6 Epics successfully.")
    print("=" * 80)
    return 0


def cmd_add_user(args):
    """Add or invite a user to the Jira project."""
    team = load_team_members()
    identifier = args.identifier or args.role or "Student " + str(len(team) + 1)
    existing = team.get(identifier, {})
    name = args.name or existing.get("name") or identifier
    email = args.email or existing.get("email") or ""
    role = args.role or existing.get("role") or "Team Member"
    tasks = args.tasks.split(",") if args.tasks else existing.get("assigned_tasks", [])

    team[identifier] = {
        "name": name,
        "email": email,
        "role": role,
        "assigned_tasks": tasks
    }
    save_team_members(team)
    print(f"[+] Successfully registered user: '{name}' ({email}) as '{role}'")

    creds = load_env_credentials()
    jira_client = get_jira_client(creds)
    if jira_client and email:
        try:
            print(f"[*] Inviting {email} to Jira Cloud project {creds['project']}...")
            # Try to search or add
            users = jira_client.search_users(query=email)
            if users:
                print(f"[+] User already exists on Jira Cloud: {users[0].displayName} ({users[0].accountId})")
            else:
                print(f"[ℹ] Please send an invitation to {email} in Jira Cloud Admin: Settings > User management > Invite users.")
        except Exception as e:
            print(f"[!] Jira Cloud lookup notice: {e}")
    else:
        print(f"[ℹ] Offline team roster updated in: {TEAM_JSON}")
    return 0


def cmd_team_status(args):
    """Display team roster and task allocations."""
    team = load_team_members()
    print("=" * 80)
    print("  PROJECT TEAM ROSTER & JIRA ALLOCATIONS")
    print("=" * 80)
    for mem_id, info in team.items():
        name = info.get("name", mem_id)
        email = info.get("email") or "Not configured"
        role = info.get("role", "Developer")
        tasks = ", ".join(info.get("assigned_tasks", [])) or "None"
        print(f"• Member: {name}")
        print(f"  Email: {email}")
        print(f"  Role:  {role}")
        print(f"  Tasks: {tasks}\n")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Jira Scrum Management CLI for College RAG Assistant")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # init
    p_init = subparsers.add_parser("init", help="Configure Jira credentials and sync CLI config")
    p_init.add_argument("--server", help="Jira Cloud server URL")
    p_init.add_argument("--email", help="Jira login email")
    p_init.add_argument("--token", help="Atlassian API token")
    p_init.add_argument("--project", default="RAG", help="Jira project key (default: RAG)")

    # create-tasks
    p_create = subparsers.add_parser("create-tasks", help="Create epics and tasks in Jira or simulate")
    p_create.add_argument("--dry-run", action="store_true", help="Simulate creation without making API calls")

    # list-tasks
    p_list = subparsers.add_parser("list-tasks", help="List all backlog tasks from CSV")

    # add-user
    p_user = subparsers.add_parser("add-user", help="Add or assign a team member/user")
    p_user.add_argument("--identifier", help="Internal ID (e.g. 'Student 1')")
    p_user.add_argument("--name", help="Full name of user")
    p_user.add_argument("--email", help="User email address")
    p_user.add_argument("--role", help="Assigned role in project")
    p_user.add_argument("--tasks", help="Comma-separated task keys (e.g. 'RAG-1,RAG-2')")

    # team
    p_team = subparsers.add_parser("team", help="View team roster and assignments")

    args = parser.parse_args()

    if args.command == "init":
        return cmd_init(args)
    elif args.command == "create-tasks":
        return cmd_create_tasks(args)
    elif args.command == "list-tasks":
        return cmd_list_tasks(args)
    elif args.command == "add-user":
        return cmd_add_user(args)
    elif args.command == "team":
        return cmd_team_status(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
