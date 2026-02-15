"""CLI interface for agentic-tmux.

The CLI provides debugging and monitoring commands. For planning and execution,
use the MCP server via `agentic-tmux mcp` which integrates with GitHub Copilot CLI,
Claude Code, and other MCP clients.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from agentic import __version__
from agentic.config import (
    clear_current_session,
    get_current_session_id,
    get_pid_file,
    get_storage_client,
)
from agentic.models import (
    SessionStatus,
    Task,
)
from agentic.orchestrator import stop_orchestrator
from agentic.tmux_manager import TmuxManager

console = Console()


@click.group()
@click.version_option(version=__version__)
def main():
    """Agentic TMUX - Multi-agent orchestration for CLI coding assistants.
    
    Primary interface is the MCP server. Start it with: agentic-tmux mcp
    
    These CLI commands are for debugging and monitoring.
    """
    pass


@main.command()
@click.option("--transport", "-t", default="stdio", help="Transport type (stdio, sse, streamable-http)")
def mcp(transport: str):
    """Start the MCP server for integration with MCP clients.
    
    This is the primary interface for orchestrating agents.
    """
    from agentic.mcp_server import mcp as mcp_server
    
    console.print(f"[cyan]Starting MCP server with {transport} transport...[/cyan]")
    mcp_server.run(transport=transport)


@main.command()
@click.option("--watch", "-w", is_flag=True, help="Watch mode (updates every 2s)")
def status(watch: bool):
    """Show status of all agents and tasks."""
    working_dir = os.getcwd()
    session_id = get_current_session_id(working_dir)
    if not session_id:
        console.print("[yellow]No active session[/yellow]")
        return
    
    redis = get_storage_client(working_dir)
    
    if watch:
        with Live(console=console, refresh_per_second=0.5) as live:
            while True:
                try:
                    live.update(render_status(session_id, redis))
                    time.sleep(2)
                except KeyboardInterrupt:
                    break
    else:
        console.print(render_status(session_id, redis))


def render_status(session_id: str, storage: Any) -> Panel:
    """Render status panel."""
    session = storage.get_session(session_id)
    if not session:
        return Panel("[red]Session not found[/red]")
    
    # Agents table
    agent_table = Table(title="")
    agent_table.add_column("Pane", style="dim")
    agent_table.add_column("Role")
    agent_table.add_column("Status")
    agent_table.add_column("Current Task")
    agent_table.add_column("Queue")
    agent_table.add_column("❤️", justify="right")
    
    agents = storage.get_all_agents(session_id)
    for agent in agents:
        status_str = agent.status.value
        if agent.status.value == "working":
            status_str = f"[green]{status_str}[/green]"
        elif agent.status.value == "idle":
            status_str = f"[dim]{status_str}[/dim]"
        elif agent.status.value.startswith("waiting"):
            status_str = f"[yellow]{status_str}[/yellow]"
        
        heartbeat_age = int(time.time() - agent.last_heartbeat)
        heartbeat_str = f"{heartbeat_age}s" if heartbeat_age < 120 else f"[red]{heartbeat_age}s[/red]"
        
        agent_table.add_row(
            agent.id,
            agent.role,
            status_str,
            agent.current_task_id or "-",
            str(agent.task_queue_length),
            heartbeat_str,
        )
    
    # Progress
    dag = storage.get_dag(session_id)
    if dag:
        completed, total = dag.get_completion_progress()
        progress_bar = "█" * completed + "░" * (total - completed)
        progress_str = f"{progress_bar} {completed}/{total} tasks complete"
    else:
        progress_str = "No tasks"
    
    from rich.console import Group
    from rich.text import Text
    content = Group(agent_table, Text(), Text(f"Progress: {progress_str}"))
    
    return Panel(
        content,
        title=f"AGENTIC SESSION: {session_id}",
        subtitle=f"Status: {session.status.value}",
        border_style="blue",
    )


@main.command()
@click.argument("agent_id")
@click.option("--follow", "-f", is_flag=True, help="Follow log output")
@click.option("--lines", "-n", default=50, help="Number of lines to show")
def logs(agent_id: str, follow: bool, lines: int):
    """View logs for a specific agent."""
    working_dir = os.getcwd()
    session_id = get_current_session_id(working_dir)
    if not session_id:
        console.print("[yellow]No active session[/yellow]")
        return
    
    redis = get_storage_client(working_dir)
    
    # Get initial logs
    log_entries = redis.get_agent_logs(session_id, agent_id, count=lines)
    
    for log in log_entries:
        ts = time.strftime("%H:%M:%S", time.localtime(log.timestamp))
        file_str = f" ({log.file})" if log.file else ""
        console.print(f"[dim]{ts}[/dim] [{log.agent_id}] {log.action}{file_str}")
    
    if follow:
        last_id = "0" if not log_entries else str(int(log_entries[-1].timestamp * 1000))
        console.print("[dim]Waiting for new logs... (Ctrl+C to stop)[/dim]")
        
        while True:
            try:
                new_logs = redis.get_agent_logs(session_id, agent_id, count=10, last_id=last_id)
                for log in new_logs:
                    ts = time.strftime("%H:%M:%S", time.localtime(log.timestamp))
                    file_str = f" ({log.file})" if log.file else ""
                    console.print(f"[dim]{ts}[/dim] [{log.agent_id}] {log.action}{file_str}")
                if new_logs:
                    last_id = str(int(new_logs[-1].timestamp * 1000))
                time.sleep(1)
            except KeyboardInterrupt:
                break


@main.command()
@click.argument("agent_id")
@click.argument("task_description")
def send(agent_id: str, task_description: str):
    """Send a task to a specific agent."""
    working_dir = os.getcwd()
    session_id = get_current_session_id(working_dir)
    if not session_id:
        console.print("[red]Error:[/red] No active session")
        sys.exit(1)
    
    redis = get_storage_client(working_dir)
    
    task = Task(
        title=task_description[:50],
        description=task_description,
        from_agent="cli",
    )
    
    redis.push_task(session_id, agent_id, task)
    console.print(f"[green]✓[/green] Task sent to {agent_id}")


@main.command()
def stop():
    """Stop the current agentic session."""
    working_dir = os.getcwd()
    session_id = get_current_session_id(working_dir)
    if not session_id:
        console.print("[yellow]No active session[/yellow]")
        return
    
    redis = get_storage_client(working_dir)
    session = redis.get_session(session_id)
    if session:
        working_dir = session.working_directory
    
    # Send done signal to all agents
    redis.push_done_to_all(session_id)
    
    # Stop orchestrator
    pid_file = get_pid_file(working_dir)
    if pid_file.exists():
        stop_orchestrator(str(pid_file))
        console.print("[green]✓[/green] Stopped orchestrator")
    
    # Kill tmux session
    tmux = TmuxManager()
    if tmux.session_exists():
        if Confirm.ask("Kill tmux session?"):
            tmux.kill_session()
            console.print("[green]✓[/green] Killed tmux session")
    
    # Update session status
    redis.update_session_status(session_id, SessionStatus.COMPLETED)
    clear_current_session(working_dir)
    
    console.print(f"[green]✓[/green] Session [cyan]{session_id}[/cyan] stopped")


@main.command()
@click.option("--force", "-f", is_flag=True, help="Force clear without confirmation")
def clear(force: bool):
    """Clear all workers but keep the session."""
    working_dir = os.getcwd()
    session_id = get_current_session_id(working_dir)
    if not session_id:
        console.print("[yellow]No active session[/yellow]")
        return
    
    if not force and not Confirm.ask("Kill all worker panes?"):
        return
    
    redis = get_storage_client(working_dir)
    tmux = TmuxManager()
    
    # Send done signal
    redis.push_done_to_all(session_id)
    
    # Kill worker panes
    killed = tmux.kill_all_workers()
    console.print(f"[green]✓[/green] Killed {killed} worker panes")
    
    # Clear agents from Redis
    agents = redis.get_all_agents(session_id)
    for agent in agents:
        redis.delete_agent(session_id, agent.id)
    
    console.print("[green]✓[/green] Session cleared")


@main.command()
@click.option("--output", "-o", default="session_export.json", help="Output file")
def export(output: str):
    """Export session transcript and state."""
    working_dir = os.getcwd()
    session_id = get_current_session_id(working_dir)
    if not session_id:
        console.print("[yellow]No active session[/yellow]")
        return
    
    import json
    
    redis = get_storage_client(working_dir)
    session = redis.get_session(session_id)
    
    if not session:
        console.print("[red]Error:[/red] Session not found")
        return
    
    # Collect all data
    export_data = {
        "session": session.to_config(),
        "agents": [],
        "dag": None,
        "logs": {},
    }
    
    agents = redis.get_all_agents(session_id)
    for agent in agents:
        export_data["agents"].append(agent.to_config())
        log_entries = redis.get_agent_logs(session_id, agent.id, count=1000)
        export_data["logs"][agent.id] = [
            {
                "timestamp": log.timestamp,
                "action": log.action,
                "file": log.file,
                "tool": log.tool,
            }
            for log in log_entries
        ]
    
    dag = redis.get_dag(session_id)
    if dag:
        export_data["dag"] = dag.to_dict()
    
    # Write to file
    with open(output, "w") as f:
        json.dump(export_data, f, indent=2)
    
    console.print(f"[green]✓[/green] Exported to {output}")


# =============================================================================
# Inter-agent messaging commands (for use by workers)
# =============================================================================


@main.command(name="msg-send")
@click.argument("to_agent")
@click.argument("message")
def msg_send(to_agent: str, message: str):
    """Send a message to another agent.
    
    Workers can use this to communicate with other agents.
    Requires AGENTIC_SESSION_ID and AGENTIC_AGENT_ID environment variables.
    """
    # For workers, get working_dir from env var
    working_dir = os.environ.get("AGENTIC_WORKING_DIR") or os.getcwd()
    session_id = os.environ.get("AGENTIC_SESSION_ID") or get_current_session_id(working_dir)
    agent_id = os.environ.get("AGENTIC_AGENT_ID")
    
    if not session_id:
        console.print("[red]Error:[/red] No active session (set AGENTIC_SESSION_ID)")
        sys.exit(1)
    if not agent_id:
        console.print("[red]Error:[/red] Agent ID not set (set AGENTIC_AGENT_ID)")
        sys.exit(1)
    
    storage = get_storage_client(working_dir)
    
    # Verify target exists
    target = storage.get_agent(session_id, to_agent)
    if not target:
        console.print(f"[red]Error:[/red] Agent {to_agent} not found")
        sys.exit(1)
    
    # Send message
    msg_id = storage.send_agent_message(session_id, agent_id, to_agent, message)
    console.print(f"[green]✓[/green] Message sent to {to_agent} (id: {msg_id})")


@main.command(name="msg-recv")
@click.option("--timeout", "-t", default=30, help="Seconds to wait for message")
@click.option("--raw", is_flag=True, help="Output raw JSON")
def msg_recv(timeout: int, raw: bool):
    """Receive the next message from the queue.
    
    Workers use this to receive messages from other agents.
    Blocks until a message arrives or timeout is reached.
    """
    import json
    
    # For workers, get working_dir from env var
    working_dir = os.environ.get("AGENTIC_WORKING_DIR") or os.getcwd()
    session_id = os.environ.get("AGENTIC_SESSION_ID") or get_current_session_id(working_dir)
    agent_id = os.environ.get("AGENTIC_AGENT_ID")
    
    if not session_id:
        console.print("[red]Error:[/red] No active session (set AGENTIC_SESSION_ID)")
        sys.exit(1)
    if not agent_id:
        console.print("[red]Error:[/red] Agent ID not set (set AGENTIC_AGENT_ID)")
        sys.exit(1)
    
    storage = get_storage_client(working_dir)
    msg = storage.receive_agent_message(session_id, agent_id, timeout=timeout)
    
    if not msg:
        if raw:
            console.print("{}")
        else:
            console.print("[yellow]No message received[/yellow]")
        sys.exit(0)
    
    if raw:
        console.print(json.dumps(msg))
    else:
        console.print(f"[cyan]From:[/cyan] {msg.get('from')}")
        console.print(f"[cyan]Message:[/cyan] {msg.get('message')}")


@main.command(name="msg-list")
@click.option("--raw", is_flag=True, help="Output raw JSON")
def msg_list(raw: bool):
    """List available agents and pending message count.
    
    Shows all agents in the session and how many messages are waiting.
    """
    import json
    
    # For workers, get working_dir from env var
    working_dir = os.environ.get("AGENTIC_WORKING_DIR") or os.getcwd()
    session_id = os.environ.get("AGENTIC_SESSION_ID") or get_current_session_id(working_dir)
    agent_id = os.environ.get("AGENTIC_AGENT_ID")
    
    if not session_id:
        console.print("[red]Error:[/red] No active session")
        sys.exit(1)
    
    storage = get_storage_client(working_dir)
    agents = storage.get_all_agents(session_id)
    
    if raw:
        data = {
            "my_id": agent_id,
            "agents": [
                {
                    "id": a.id,
                    "role": a.role,
                    "pending_messages": storage.get_message_count(session_id, a.id),
                }
                for a in agents
            ],
        }
        console.print(json.dumps(data))
    else:
        console.print(f"[cyan]Your ID:[/cyan] {agent_id or 'not set'}")
        console.print(f"[cyan]Session:[/cyan] {session_id}")
        console.print()
        
        table = Table(title="Agents")
        table.add_column("ID")
        table.add_column("Role")
        table.add_column("Pending Messages", justify="right")
        
        for agent in agents:
            msg_count = storage.get_message_count(session_id, agent.id)
            is_me = " (me)" if agent.id == agent_id else ""
            table.add_row(
                f"{agent.id}{is_me}",
                agent.role[:40],
                str(msg_count),
            )
        
        console.print(table)


@main.command()
@click.option("--refresh", "-r", default=1.5, help="Refresh rate in seconds")
def monitor(refresh: float):
    """Launch the real-time monitoring dashboard.
    
    Shows agent status, message activity, heartbeats, and session health.
    Automatically waits for a session if none is active.
    """
    from agentic.monitor import run_monitor
    run_monitor(refresh_rate=refresh)


@main.command(name="worker-mcp")
@click.option("--transport", "-t", default="stdio", help="Transport type")
def worker_mcp_cmd(transport: str):
    """Start the worker MCP server for inter-agent communication.
    
    Workers can use this MCP server to send/receive messages to other agents.
    Requires AGENTIC_SESSION_ID and AGENTIC_AGENT_ID environment variables.
    """
    from agentic.worker_mcp import worker_mcp
    
    session_id = os.environ.get("AGENTIC_SESSION_ID")
    agent_id = os.environ.get("AGENTIC_AGENT_ID")
    
    if not session_id or not agent_id:
        console.print("[yellow]Warning:[/yellow] AGENTIC_SESSION_ID or AGENTIC_AGENT_ID not set")
        console.print("Make sure these are set for full functionality")
    
    console.print(f"[cyan]Starting Worker MCP server ({transport})...[/cyan]")
    worker_mcp.run(transport=transport)


@main.command()
@click.option("--cli", "-c", type=click.Choice(["copilot", "claude", "both"]), help="Which CLI to configure")
@click.option("--hooks", is_flag=True, help="Install debug hooks for Copilot CLI")
def setup(cli: str | None, hooks: bool):
    """Interactive setup wizard for agentic-tmux.
    
    Configures MCP integration for GitHub Copilot CLI and/or Claude Code.
    """
    import json
    import shutil
    import subprocess
    
    from rich.panel import Panel
    from rich.prompt import Prompt
    
    console.print(Panel.fit(
        "[bold cyan]Agentic TMUX Setup Wizard[/bold cyan]\n"
        "Configure multi-agent orchestration for your AI coding tools",
        border_style="cyan",
    ))
    
    # Check prerequisites
    console.print("\n[bold]Checking prerequisites...[/bold]\n")
    
    prereqs_ok = True
    
    # tmux
    if shutil.which("tmux"):
        console.print("  [green]✓[/green] tmux")
    else:
        console.print("  [red]✗[/red] tmux - install with: brew install tmux / apt install tmux")
        prereqs_ok = False
    
    # AI CLIs
    has_copilot = bool(shutil.which("copilot"))
    has_claude = bool(shutil.which("claude"))
    
    if has_copilot:
        console.print("  [green]✓[/green] GitHub Copilot CLI")
    else:
        console.print("  [yellow]○[/yellow] GitHub Copilot CLI - npm install -g @githubnext/github-copilot-cli")
    
    if has_claude:
        console.print("  [green]✓[/green] Claude Code")
    else:
        console.print("  [yellow]○[/yellow] Claude Code - npm install -g @anthropic-ai/claude-code")
    
    if not has_copilot and not has_claude:
        console.print("\n[yellow]Warning:[/yellow] No AI CLI found. Install at least one to use agents.")
    
    if not prereqs_ok:
        console.print("\n[red]Please install missing prerequisites and run again.[/red]")
        return
    
    # Determine which CLI to configure
    if not cli:
        if has_copilot and has_claude:
            cli = Prompt.ask(
                "\nWhich CLI would you like to configure?",
                choices=["copilot", "claude", "both"],
                default="both",
            )
        elif has_copilot:
            cli = "copilot"
        elif has_claude:
            cli = "claude"
        else:
            cli = Prompt.ask(
                "\nWhich CLI will you use?",
                choices=["copilot", "claude", "both"],
                default="copilot",
            )
    
    # Setup MCP configuration
    console.print("\n[bold]Configuring MCP integration...[/bold]\n")
    
    if cli in ("copilot", "both"):
        # Copilot CLI global MCP config
        copilot_config_dir = Path.home() / ".copilot"
        copilot_config_dir.mkdir(parents=True, exist_ok=True)
        copilot_config_path = copilot_config_dir / "mcp-config.json"
        
        copilot_config: dict = {}
        if copilot_config_path.exists():
            try:
                with open(copilot_config_path) as f:
                    copilot_config = json.load(f)
            except json.JSONDecodeError:
                pass
        
        copilot_config.setdefault("mcpServers", {})["agentic"] = {
            "type": "local",
            "tools": ["*"],
            "command": "agentic-tmux",
            "args": ["mcp"],
        }
        
        with open(copilot_config_path, "w") as f:
            json.dump(copilot_config, f, indent=2)
        
        console.print(f"  [green]✓[/green] Copilot CLI: {copilot_config_path}")
    
    if cli in ("claude", "both"):
        # Claude Code reads MCP config from ~/.claude.json (global) or .mcp.json (project)
        claude_config_path = Path.home() / ".claude.json"
        
        claude_config: dict = {}
        if claude_config_path.exists():
            try:
                with open(claude_config_path) as f:
                    claude_config = json.load(f)
            except json.JSONDecodeError:
                pass
        
        claude_config.setdefault("mcpServers", {})["agentic"] = {
            "command": "agentic-tmux",
            "args": ["mcp"],
        }
        
        with open(claude_config_path, "w") as f:
            json.dump(claude_config, f, indent=2)
        
        console.print(f"  [green]✓[/green] Claude Code: {claude_config_path}")
    
    # Install debug hooks
    if hooks or (not hooks and Confirm.ask("\nInstall debug hooks for Copilot CLI?", default=False)):
        hooks_src = Path(__file__).parent.parent / ".github" / "hooks"
        hooks_dst = Path.home() / ".github" / "hooks"
        
        if hooks_src.exists():
            hooks_dst.mkdir(parents=True, exist_ok=True)
            for hook_file in hooks_src.glob("*"):
                shutil.copy(hook_file, hooks_dst)
                if hook_file.suffix == ".sh":
                    (hooks_dst / hook_file.name).chmod(0o755)
            console.print(f"  [green]✓[/green] Debug hooks installed: {hooks_dst}")
        else:
            console.print("  [yellow]![/yellow] Debug hooks not found in package")
    
    # Print success message
    console.print(Panel.fit(
        "[bold green]Setup Complete![/bold green]\n\n"
        "[cyan]Quick Start:[/cyan]\n"
        "1. Start tmux: [yellow]tmux new -s work[/yellow]\n"
        "2. Open Copilot CLI/Claude Code in your project\n"
        "3. Ask your AI to use [yellow]agentic[/yellow] MCP tools\n\n"
        "And monitor from CLI: [yellow]agentic-tmux monitor[/yellow]",
        border_style="green",
    ))


@main.command()
def doctor():
    """Check system configuration and diagnose issues.
    
    Verifies prerequisites, configuration, and connection to services.
    """
    import shutil
    
    console.print("[bold]Agentic TMUX Health Check[/bold]\n")
    
    all_ok = True
    
    # Check tmux
    console.print("[cyan]System Requirements:[/cyan]")
    if shutil.which("tmux"):
        result = subprocess.run(["tmux", "-V"], capture_output=True, text=True)
        console.print(f"  [green]✓[/green] tmux: {result.stdout.strip()}")
    else:
        console.print("  [red]✗[/red] tmux not found")
        all_ok = False
    
    # Check Python
    import sys
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 11):
        console.print(f"  [green]✓[/green] Python {py_version}")
    else:
        console.print(f"  [red]✗[/red] Python {py_version} (need 3.11+)")
        all_ok = False
    
    # Check AI CLIs
    console.print("\n[cyan]AI CLI Tools:[/cyan]")
    has_cli = False
    if shutil.which("copilot"):
        console.print("  [green]✓[/green] GitHub Copilot CLI")
        has_cli = True
    else:
        console.print("  [dim]○[/dim] GitHub Copilot CLI not installed")
    
    if shutil.which("claude"):
        console.print("  [green]✓[/green] Claude Code")
        has_cli = True
    else:
        console.print("  [dim]○[/dim] Claude Code not installed")
    
    if not has_cli:
        console.print("  [yellow]![/yellow] No AI CLI found - install at least one")
    
    # Check agentic commands
    console.print("\n[cyan]Agentic Commands:[/cyan]")
    for cmd in ["agentic-tmux", "agentic-worker-mcp"]:
        if shutil.which(cmd):
            console.print(f"  [green]✓[/green] {cmd}")
        else:
            console.print(f"  [yellow]![/yellow] {cmd} not in PATH")
    
    # Check Redis (optional)
    console.print("\n[cyan]Storage Backend:[/cyan]")
    try:
        import redis
        r = redis.Redis(
            host=os.environ.get("AGENTIC_REDIS_HOST", "localhost"),
            port=int(os.environ.get("AGENTIC_REDIS_PORT", "6379")),
        )
        r.ping()
        console.print("  [green]✓[/green] Redis available (will use Redis)")
    except Exception:
        console.print("  [dim]○[/dim] Redis not available (will use SQLite)")
    
    # Check current session
    console.print("\n[cyan]Session Status:[/cyan]")
    working_dir = os.getcwd()
    session_id = get_current_session_id(working_dir)
    if session_id:
        console.print(f"  [green]●[/green] Active session: {session_id}")
    else:
        console.print("  [dim]○[/dim] No active session")
    
    # Check tmux session
    if shutil.which("tmux"):
        result = subprocess.run(
            ["tmux", "list-sessions"],
            capture_output=True, text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            sessions = result.stdout.strip().split("\n")
            for s in sessions[:3]:
                console.print(f"  [dim]  tmux: {s.split(':')[0]}[/dim]")
    
    # Check MCP configuration
    console.print("\n[cyan]MCP Configuration:[/cyan]")
    
    # Copilot CLI: reads from ~/.copilot/mcp-config.json
    copilot_mcp = Path.home() / ".copilot" / "mcp-config.json"
    if copilot_mcp.exists():
        console.print(f"  [green]✓[/green] Copilot CLI: {copilot_mcp}")
    else:
        console.print("  [dim]○[/dim] Copilot CLI: ~/.copilot/mcp-config.json not configured (run agentic-tmux setup)")
    
    # Claude Code: reads from ~/.claude.json (global) or .mcp.json (project)
    claude_config = Path.home() / ".claude.json"
    if claude_config.exists():
        console.print(f"  [green]✓[/green] Claude Code: {claude_config}")
    else:
        console.print("  [dim]○[/dim] Claude Code: ~/.claude.json not configured (run agentic-tmux setup)")
    
    # Summary
    console.print()
    if all_ok:
        console.print("[green]All critical checks passed![/green]")
    else:
        console.print("[yellow]Some issues found. Run 'agentic-tmux setup' to fix.[/yellow]")


if __name__ == "__main__":
    main()
