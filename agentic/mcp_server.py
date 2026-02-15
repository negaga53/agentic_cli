"""MCP Server interface for agentic-tmux orchestration.

This module provides a Model Context Protocol server that exposes
agentic-tmux functionality as MCP tools, resources, and prompts.
The server can be used by any MCP-compatible client (GitHub Copilot CLI, Claude Code, etc).

Usage:
    # Via CLI
    agentic-tmux mcp
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from agentic.config import (
    clear_current_session,
    ensure_config_dir,
    cleanup_session_data,
    get_current_session_id,
    get_pid_file,
    get_storage_client as _get_storage_client,
    save_current_session_id,
    WORKING_DIR_ENV_VAR,
)
from agentic.models import (
    Agent,
    AgenticSession,
    FileScope,
    SessionStatus,
)
from agentic.monitor import log_activity
from agentic.orchestrator import start_orchestrator_background, stop_orchestrator
from agentic.redis_client import reset_sqlite_client
from agentic.tmux_manager import TmuxManager, check_tmux_available


def get_redis_client(working_dir: str | None = None):
    """Get storage client (Redis preferred, SQLite fallback).
    
    Args:
        working_dir: Working directory for per-repo storage. If None, uses CWD.
    """
    return _get_storage_client(working_dir)


# Create the MCP server
mcp = FastMCP("Agentic TMUX")


# =============================================================================
# MCP Tools
# =============================================================================


def _detect_tmux_session() -> str | None:
    """Detect the tmux session this process belongs to.
    
    Uses three strategies in order of reliability:
    1. Walk the process tree to find an ancestor that's a tmux pane shell
       (works regardless of env var inheritance)
    2. Use TMUX_PANE env var with display-message
    3. Use TMUX env var with display-message
    
    Returns session name or None if detection fails.
    """
    import subprocess as sp
    
    # Strategy 1: Walk process tree to find the tmux pane ancestor
    try:
        result = sp.run(
            ["tmux", "list-panes", "-a", "-F", "#{pane_pid} #{session_name}"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            # Build map: pane shell PID → session name
            pane_sessions: dict[str, str] = {}
            for line in result.stdout.strip().split('\n'):
                parts = line.split(None, 1)
                if len(parts) >= 2:
                    pane_sessions[parts[0]] = parts[1]
            
            # Walk up the process tree from current PID
            pid = os.getpid()
            while pid > 1:
                if str(pid) in pane_sessions:
                    return pane_sessions[str(pid)]
                try:
                    with open(f'/proc/{pid}/stat') as f:
                        ppid = int(f.read().split()[3])
                    if ppid == pid:
                        break
                    pid = ppid
                except (FileNotFoundError, ValueError, PermissionError):
                    break
    except Exception:
        pass
    
    # Strategy 2: Use TMUX_PANE env var
    tmux_pane = os.environ.get("TMUX_PANE")
    if tmux_pane:
        try:
            result = sp.run(
                ["tmux", "display-message", "-t", tmux_pane, "-p", "#S"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass
    
    # Strategy 3: Use TMUX env var context
    if os.environ.get("TMUX"):
        try:
            result = sp.run(
                ["tmux", "display-message", "-p", "#S"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass
    
    return None


def _ensure_admin_pane(tmux: TmuxManager, working_dir: str) -> None:
    """Ensure the admin window exists and has the monitor running.
    
    If the admin window was closed or the monitor exited, recreate it.
    """
    try:
        admin_exists = False
        for window in tmux.session.windows:
            if window.name == "admin":
                # Check if the monitor process is running in the pane
                pane = window.active_pane
                if pane and pane.current_command and "monitor" in (pane.current_command or ""):
                    admin_exists = True
                break
        
        if not admin_exists:
            tmux.create_admin_pane(working_dir)
    except Exception:
        pass  # Best-effort — don't block session start


def _start_session_internal(
    working_dir: str,
    cli_command: str = "copilot -i",
    tmux_session: str | None = None,
) -> dict[str, Any]:
    """Internal function to start a session."""
    # Validate prerequisites
    if not check_tmux_available():
        return {"error": "tmux is not installed or not in PATH"}
    
    # Normalize working_dir early
    working_dir = os.path.abspath(working_dir)
    
    storage = get_redis_client(working_dir)
    
    # Check for existing session
    existing_id = get_current_session_id(working_dir)
    if existing_id:
        session = storage.get_session(existing_id)
        if session and session.status not in (SessionStatus.COMPLETED, SessionStatus.FAILED):
            # Verify the orchestrator process is still alive
            pid_file = get_pid_file(working_dir)
            orchestrator_alive = False
            if pid_file.exists():
                try:
                    pid = int(pid_file.read_text().strip())
                    os.kill(pid, 0)  # signal 0 = existence check
                    orchestrator_alive = True
                except (ProcessLookupError, ValueError, PermissionError, OSError):
                    pass
            
            if orchestrator_alive:
                # Session is genuinely running — ensure admin pane exists
                tmux_session_name = session.config.get("tmux_session", "agentic")
                tmux = TmuxManager(session_name=tmux_session_name, use_current_session=False)
                _ensure_admin_pane(tmux, session.working_directory)
                
                return {
                    "session_id": existing_id,
                    "status": "existing",
                    "working_directory": session.working_directory,
                }
            
            # Orchestrator is dead → session is stale, fall through to cleanup
    
    # Clean up old session data (logs, database) for a fresh start
    cleanup_result = cleanup_session_data(working_dir)
    
    # Reset the SQLite singleton to ensure fresh connection after cleanup
    reset_sqlite_client(working_dir)
    
    # Detect current tmux session if not provided
    if not tmux_session:
        tmux_session = _detect_tmux_session()
        if not tmux_session:
            tmux_session = "agentic"  # fallback
    
    # Re-get storage after cleanup (singleton was cleared)
    storage = get_redis_client(working_dir)
    
    # Create new session
    session = AgenticSession(working_directory=working_dir)
    session.config["cli_command"] = cli_command
    session.config["tmux_session"] = tmux_session
    
    storage.create_session(session)
    save_current_session_id(session.id, working_dir)

    # Create tmux session with admin pane (use configured session name)
    tmux = TmuxManager(session_name=tmux_session, use_current_session=False)
    admin_pane_id = tmux.create_admin_pane(working_dir)
    session.admin_pane_id = admin_pane_id
    storage.update_session_status(session.id, SessionStatus.RUNNING)
    
    # Start orchestrator daemon
    config_dir = ensure_config_dir(working_dir)
    log_file = config_dir / f"orchestrator_{session.id}.log"
    pid = start_orchestrator_background(
        session_id=session.id,
        log_file=str(log_file),
        working_dir=working_dir,
    )
    
    pid_file = get_pid_file(working_dir)
    if pid:
        pid_file.write_text(str(pid))
    
    # Log session start
    log_activity("session_start", {
        "working_dir": working_dir,
        "cli_command": cli_command,
        "orchestrator_pid": pid,
        "cleanup": cleanup_result,
    }, session_id=session.id, working_dir=working_dir)
    
    return {
        "session_id": session.id,
        "status": "started",
        "working_directory": working_dir,
        "cli_command": cli_command,
        "orchestrator_pid": pid,
        "cleanup": cleanup_result,
    }


@mcp.tool()
def start_session(
    working_dir: str = Field(description="Working directory for the agents"),
    cli_command: str = Field(
        default="copilot -i",
        description="CLI command to use (e.g., 'copilot -i', 'claude', 'aider')",
    ),
    tmux_session: str = Field(
        default="",
        description="Tmux session name to use. If empty, creates new 'agentic' session. Set this to your current session name to add workers to your existing session.",
    ),
) -> dict[str, Any]:
    """
    Start a new agentic session.
    
    Creates agents in a tmux session. Specify tmux_session to add workers
    to your current session, or leave empty to create a separate 'agentic' session.
    """
    return _start_session_internal(
        working_dir, 
        cli_command, 
        tmux_session=tmux_session if tmux_session else None
    )


@mcp.tool()
def stop_session(
    kill_panes: bool = Field(default=False, description="Kill worker tmux panes (default: keep them for review)"),
    clear_data: bool = Field(default=True, description="Clear session data from database"),
) -> dict[str, Any]:
    """
    Stop the current agentic session completely.
    
    By default, keeps worker panes alive so you can review the conversations.
    Set kill_panes=True to clean them up.
    """
    # Try to find the session from CWD first
    working_dir = os.getcwd()
    session_id = get_current_session_id(working_dir)
    if not session_id:
        return {"status": "no_session", "message": "No active session"}
    
    redis = get_redis_client(working_dir)
    session = redis.get_session(session_id)
    
    # Get actual working dir from session if available
    if session:
        working_dir = session.working_directory
    
    killed_panes = []
    
    if session and kill_panes:
        # Kill all worker panes
        agents = redis.get_all_agents(session_id)
        tmux_session = session.config.get("tmux_session", "agentic")
        tmux = TmuxManager(session_name=tmux_session, use_current_session=False)
        
        for agent in agents:
            if agent.pane_id:
                try:
                    tmux.kill_pane(agent.pane_id)
                    killed_panes.append(agent.pane_id)
                except Exception:
                    pass  # Pane may already be gone
    
    # Send done signal to all agents (for any still running)
    redis.push_done_to_all(session_id)
    
    # Stop orchestrator
    pid_file = get_pid_file(working_dir)
    if pid_file.exists():
        stop_orchestrator(str(pid_file))
    
    if clear_data:
        # Delete all agents and session data
        # Note: This clears agents from the session but keeps session record
        for agent in redis.get_all_agents(session_id):
            redis.delete_agent(session_id, agent.id)
        redis.update_session_status(session_id, SessionStatus.COMPLETED)
    else:
        redis.update_session_status(session_id, SessionStatus.COMPLETED)
    
    clear_current_session(working_dir)
    
    # Log session stop
    log_activity("session_stop", {
        "killed_panes": killed_panes,
        "data_cleared": clear_data,
    }, session_id=session_id, working_dir=working_dir)
    
    return {
        "session_id": session_id,
        "status": "stopped",
        "killed_panes": killed_panes,
        "data_cleared": clear_data,
    }


def _build_communication_instructions(agent_id: str, other_agents: list[str]) -> str:
    """Build inter-agent communication instructions to include in agent prompts.
    
    Uses imperative language and consequence framing based on research showing
    this improves LLM tool-calling reliability.
    """
    agents_list = ", ".join(other_agents) if other_agents else "unknown (call list_agents() to discover)"
    return f"""
## ⚠️ CRITICAL: Communication Protocol (READ FIRST)

You are agent **{agent_id}**. Known agents: {agents_list}

**Your text responses are NOT delivered to other agents or the orchestrator.**
You MUST use MCP tools for ALL communication. If you don't call the tools below,
your work will be LOST and the task will FAIL.

---

## REQUIRED MCP Tools (from "agentic-worker" server)

### 1. list_agents() - CALL THIS FIRST
Discovers all agents in this multi-agent session. Returns agent IDs and roles.
You MUST call this before doing any work so you know who to coordinate with.
Without this call, you cannot properly address messages to other agents.

### 2. send_to_agent(agent_id, message) - REQUIRED FOR RESULTS
Sends a message to another agent or the orchestrator. This is your ONLY method
to communicate. 
- To reply to another agent → use their agent_id (e.g., "W1", "W2")
- To report final results → use agent_id="orchestrator"
⚠️ If you don't call this, your output goes NOWHERE. Text responses are not delivered.

### 3. receive_message(timeout=300) - REQUIRED FOR COORDINATION  
Waits for incoming messages. Returns status: "received", "no_message", or
"session_terminated". You MUST call this in a loop after reporting results.
Exit ONLY when status is "session_terminated" or message contains "TERMINATE".

### 4. check_messages() - Optional
Non-blocking check for pending messages. Use before blocking operations.

### 5. broadcast_message(message) - Optional
Sends to all agents at once. Use for session-wide announcements.

---

## MANDATORY 4-Phase Workflow

### Phase 1: DISCOVERY (DO THIS IMMEDIATELY)
```
FIRST ACTION: Call list_agents() now.
```
Do NOT proceed to your task until you have discovered other agents.

### Phase 2: EXECUTE YOUR TASK
Complete your assigned work. Stay focused on your specific task.

### Phase 3: REPORT RESULTS (CRITICAL - NEVER SKIP)
When communicating with other agents, reply to them directly:
```
send_to_agent(agent_id="W1", message="<your reply>")  # reply to W1
```
When your task is FULLY complete, report to orchestrator:
```
send_to_agent(agent_id="orchestrator", message="{{status: 'complete', agent_id: '{agent_id}', result: <summary>}}")
```
⚠️ Skipping send_to_agent = your output is LOST. Text responses go nowhere.

### Phase 4: POLLING LOOP (DO NOT EXIT EARLY)
```
ENTER THIS LOOP and stay in it:
while True:
    msg = receive_message(timeout=300)
    if msg.status == "session_terminated": break
    if msg.status == "received" and "TERMINATE" in msg.message: break
    if msg.status == "received": process and respond
    # "no_message" status = keep polling, DO NOT EXIT
```
⚠️ Exiting without TERMINATE signal will cause session failure.

---

## Pre-Completion Checklist
Before ending your turn, verify you have:
✓ Called list_agents() 
✓ Completed your task
✓ Called send_to_agent("orchestrator", ...) with results
✓ Entered polling loop OR received TERMINATE

Missing any step = STOP and complete it NOW.
"""


@mcp.tool()
def spawn_agent(
    role: str = Field(description="Full task description for the agent (used as initial prompt)"),
    scope_patterns: list[str] = Field(
        default=["**/*"],
        description="File patterns the agent can access",
    ),
    read_only: bool = Field(
        default=False,
        description="Whether the agent has read-only access",
    ),
    initial_task: str = Field(
        default="",
        description="Optional initial task (if empty, role is used as the initial prompt)",
    ),
    enable_communication: bool = Field(
        default=True,
        description="Add inter-agent communication instructions to the prompt (default: True)",
    ),
    wait_ready: bool = Field(
        default=True,
        description="Wait for agent to initialize before returning (checks for CLI readiness)",
    ),
    wait_timeout: int = Field(
        default=60,
        description="Seconds to wait for agent to be ready",
    ),
) -> dict[str, Any]:
    """
    Spawn a single new agent in a tmux pane with its initial task.
    
    The 'role' parameter should include the FULL task description - not just a role name.
    The agent will begin executing this task immediately upon startup.
    
    WORKFLOW: After spawning all agents, you MUST call receive_message_from_agents()
    to collect their results. Workers send results via send_to_agent("orchestrator", ...).
    
    Example role: "Analyze all Python files in src/ for security vulnerabilities,
    then send a JSON report to the orchestrator via send_to_agent."
    """
    working_dir = os.getcwd()
    session_id = get_current_session_id(working_dir)
    
    # Auto-start session if none exists
    if not session_id:
        start_result = _start_session_internal(working_dir, "copilot -i")
        if "error" in start_result:
            return {"error": f"Failed to auto-start session: {start_result['error']}"}
        session_id = start_result["session_id"]
    
    redis = get_redis_client(working_dir)
    session = redis.get_session(session_id)
    if not session:
        return {"error": "Session not found"}
    
    # Use session's working directory
    working_dir = session.working_directory
    
    # Get existing agents to determine next ID
    existing_agents = redis.get_all_agents(session_id)
    next_id = f"W{len(existing_agents) + 1}"
    other_agent_ids = [a.id for a in existing_agents]
    
    # Build prompt with optional communication instructions
    agent_prompt = initial_task if initial_task else role
    if enable_communication:
        comm_instructions = _build_communication_instructions(next_id, other_agent_ids)
        agent_prompt = agent_prompt + comm_instructions
    
    # Create agent
    agent = Agent(
        id=next_id,
        role=role,
        scope=FileScope(patterns=scope_patterns, read_only=read_only),
    )
    
    # Spawn pane using stored tmux session
    tmux_session = session.config.get("tmux_session", "agentic")
    tmux = TmuxManager(session_name=tmux_session, use_current_session=False)
    cli_command = session.config.get("cli_command", "copilot -i")
    
    pane_id = tmux.spawn_worker_pane(
        agent=agent,
        working_dir=session.working_directory,
        cli_command=cli_command,
        session_id=session_id,
        initial_task=agent_prompt,
    )
    
    agent.pane_id = pane_id
    redis.register_agent(session_id, agent)
    
    # Log agent spawn
    log_activity("agent_spawn", {
        "agent_id": agent.id,
        "role": role[:50],
        "pane_id": pane_id,
    }, session_id=session_id, working_dir=working_dir)
    
    result = {
        "agent_id": agent.id,
        "role": role[:100] + "..." if len(role) > 100 else role,  # Truncate for readability
        "pane_id": pane_id,
        "scope": {
            "patterns": agent.scope.patterns,
            "read_only": agent.scope.read_only,
        },
        "next_step": "Call receive_message_from_agents() to collect results, then terminate_all_agents() when done.",
    }
    
    # Wait for agent to be ready if requested
    if wait_ready:
        ready_status = _wait_for_agent_ready(tmux, pane_id, timeout=wait_timeout)
        result["ready"] = ready_status["ready"]
        result["wait_time"] = ready_status.get("elapsed_seconds", 0)
        if not ready_status["ready"]:
            result["ready_warning"] = "Agent may not be fully initialized - check pane output"
    
    return result


def _wait_for_agent_ready(
    tmux: TmuxManager,
    pane_id: str,
    timeout: int = 60,
    poll_interval: float = 2.0,
) -> dict[str, Any]:
    """
    Internal function to wait for an agent to be ready.
    
    Looks for signs the CLI has loaded (prompt indicator or output activity).
    """
    import hashlib
    
    start_time = time.time()
    initial_output = tmux.capture_pane_output(pane_id, lines=50)
    initial_hash = hashlib.md5(initial_output.encode()).hexdigest()
    
    # Wait for output to change (indicating CLI has started processing)
    elapsed = 0.0
    stable_count = 0
    last_hash = initial_hash
    
    while elapsed < timeout:
        time.sleep(poll_interval)
        elapsed = time.time() - start_time
        
        current_output = tmux.capture_pane_output(pane_id, lines=50)
        current_hash = hashlib.md5(current_output.encode()).hexdigest()
        
        # Check for readiness indicators in output
        output_lower = current_output.lower()
        
        # Look for common CLI readiness indicators
        ready_indicators = [
            "ready",
            "listening",
            "started",
            "initialized",
            "waiting for input",
            ">",  # ASCII prompt character
            "❯",  # Unicode prompt (Copilot CLI uses this)
            "copilot",  # CLI name
            "mcp",  # MCP servers indicator
            "configured",  # "Configured MCP servers"
        ]
        
        # If output changed and has any indicator, consider ready
        if current_hash != initial_hash:
            for indicator in ready_indicators:
                if indicator in output_lower:
                    return {
                        "ready": True,
                        "elapsed_seconds": round(elapsed, 1),
                        "indicator": indicator,
                    }
            
            # If output has stabilized (same for 2 polls), consider ready
            if current_hash == last_hash:
                stable_count += 1
                if stable_count >= 2:
                    return {
                        "ready": True,
                        "elapsed_seconds": round(elapsed, 1),
                        "reason": "output_stabilized",
                    }
            else:
                stable_count = 0
        
        last_hash = current_hash
    
    return {
        "ready": False,
        "elapsed_seconds": round(elapsed, 1),
        "reason": "timeout",
    }


@mcp.tool()
def get_status() -> dict[str, Any]:
    """
    Get the current status of all agents and tasks.
    """
    working_dir = os.getcwd()
    session_id = get_current_session_id(working_dir)
    if not session_id:
        return {"error": "No active session"}
    
    redis = get_redis_client(working_dir)
    session = redis.get_session(session_id)
    if not session:
        return {"error": "Session not found"}
    
    agents = redis.get_all_agents(session_id)
    current_time = time.time()
    
    agent_status = []
    for agent in agents:
        heartbeat_age = int(current_time - agent.last_heartbeat)
        role_short = agent.role[:60] + "..." if len(agent.role) > 60 else agent.role
        agent_status.append({
            "id": agent.id,
            "role": role_short,
            "status": agent.status.value,
            "current_task": agent.current_task_id,
            "queue_length": agent.task_queue_length,
            "heartbeat_age_seconds": heartbeat_age,
            "healthy": heartbeat_age < 120,
        })
    
    return {
        "session_id": session_id,
        "session_status": session.status.value,
        "working_directory": session.working_directory,
        "agents": agent_status,
    }


# =============================================================================
# Inter-Agent Communication Tools
# =============================================================================


@mcp.tool()
def send_message(
    agent_id: str = Field(description="ID of the agent to send the message to"),
    message: str = Field(description="Message to send to the agent"),
    from_agent: str = Field(default="orchestrator", description="Sender ID (defaults to 'orchestrator')"),
) -> dict[str, Any]:
    """
    Send a follow-up message to an agent's message queue.
    
    NOTE: Agents already start with their initial task from spawn_agent.
    Use this only for follow-up instructions AFTER the agent has completed
    its initial task and is polling for messages.
    
    The message is queued and the agent will receive it when they call
    receive_message().
    """
    working_dir = os.getcwd()
    session_id = get_current_session_id(working_dir)
    if not session_id:
        return {"error": "No active session"}
    
    redis = get_redis_client(working_dir)
    session = redis.get_session(session_id)
    if not session:
        return {"error": "Session not found"}
    
    agent = redis.get_agent(session_id, agent_id)
    if not agent:
        return {"error": f"Agent {agent_id} not found"}
    
    # Send message via the queue (same as worker MCP send_to_agent)
    msg_id = redis.send_agent_message(
        session_id=session_id,
        from_agent=from_agent,
        to_agent=agent_id,
        message=message,
    )
    
    # Log message sent
    log_activity("message_sent", {
        "from": from_agent,
        "to": agent_id,
        "message_preview": message[:500],
    }, session_id=session_id, working_dir=session.working_directory)
    
    return {
        "status": "queued",
        "message_id": msg_id,
        "from": from_agent,
        "to": agent_id,
        "message_preview": message[:100] + "..." if len(message) > 100 else message,
    }


@mcp.tool()
def receive_message_from_agents(
    timeout: int = Field(default=60, description="Seconds to wait for a message"),
) -> dict[str, Any]:
    """
    Receive a message sent to the orchestrator from any agent.
    
    Agents can send messages to 'orchestrator' using send_to_agent.
    Use this to receive those messages. Call this in a loop to collect
    results from all workers.
    
    IMPORTANT: After collecting all results, call `terminate_all_agents()` to
    signal workers to exit their polling loops. Otherwise they will hang forever.
    
    Returns the first message in the queue, or indicates no message after timeout.
    """
    working_dir = os.getcwd()
    session_id = get_current_session_id(working_dir)
    if not session_id:
        return {"error": "No active session"}
    
    redis = get_redis_client(working_dir)
    session = redis.get_session(session_id)
    
    # Receive message addressed to 'orchestrator'
    msg = redis.receive_agent_message(session_id, "orchestrator", timeout=timeout)
    
    if not msg:
        return {
            "status": "no_message",
            "waited_seconds": timeout,
            "hint": "If all expected agents reported, call terminate_all_agents() to end session cleanly.",
        }
    
    # Log message received
    log_activity("message_received", {
        "agent_id": "orchestrator",
        "from": msg.get("from"),
        "message_preview": msg.get("message", "")[:50],
    }, session_id=session_id, working_dir=session.working_directory if session else working_dir)
    
    return {
        "status": "received",
        "message_id": msg.get("id"),
        "from": msg.get("from"),
        "message": msg.get("message"),
        "timestamp": msg.get("timestamp"),
        "next_step": "Collect more results or call terminate_all_agents() when done.",
    }


@mcp.tool()
def check_orchestrator_messages() -> dict[str, Any]:
    """
    Check how many messages are waiting for the orchestrator (non-blocking).
    
    Use this to see if any agents have sent messages without blocking.
    """
    working_dir = os.getcwd()
    session_id = get_current_session_id(working_dir)
    if not session_id:
        return {"error": "No active session"}
    
    redis = get_redis_client(working_dir)
    count = redis.get_message_count(session_id, "orchestrator")
    
    # Also peek at the messages
    messages = redis.peek_agent_messages(session_id, "orchestrator", count=5)
    
    return {
        "pending_messages": count,
        "has_messages": count > 0,
        "preview": [
            {
                "from": m.get("from"),
                "message_preview": m.get("message", "")[:100],
            }
            for m in messages
        ],
    }


@mcp.tool()
def terminate_agent(
    agent_id: str = Field(description="ID of the agent to terminate (e.g., 'W1', 'W2')"),
) -> dict[str, Any]:
    """
    Send a TERMINATE message to a specific agent, signaling it to exit its polling loop.
    
    IMPORTANT: Workers wait in a polling loop after completing their task. You MUST
    call this (or terminate_all_agents) after collecting results, otherwise workers
    will hang indefinitely waiting for more instructions.
    
    The agent will receive a message containing "TERMINATE" and should exit gracefully.
    """
    working_dir = os.getcwd()
    session_id = get_current_session_id(working_dir)
    if not session_id:
        return {"error": "No active session"}
    
    redis = get_redis_client(working_dir)
    session = redis.get_session(session_id)
    agent = redis.get_agent(session_id, agent_id)
    if not agent:
        return {"error": f"Agent {agent_id} not found"}
    
    # Send TERMINATE message
    msg_id = redis.send_agent_message(
        session_id=session_id,
        from_agent="orchestrator",
        to_agent=agent_id,
        message="TERMINATE",
    )
    
    # Log agent terminate
    log_activity("agent_terminate", {
        "agent_id": agent_id,
    }, session_id=session_id, working_dir=session.working_directory if session else working_dir)
    
    return {
        "status": "terminate_sent",
        "agent_id": agent_id,
        "message_id": msg_id,
    }


@mcp.tool()
def terminate_all_agents() -> dict[str, Any]:
    """
    Send TERMINATE to all agents and mark the session as done.
    
    IMPORTANT: Call this after you have collected all results from workers.
    This ensures all agents exit their polling loops gracefully. Workers that
    are still in their polling loop will receive the termination signal.
    
    This is the recommended way to end a multi-agent session cleanly.
    """
    working_dir = os.getcwd()
    session_id = get_current_session_id(working_dir)
    if not session_id:
        return {"error": "No active session"}
    
    redis = get_redis_client(working_dir)
    agents = redis.get_all_agents(session_id)
    
    terminated = []
    for agent in agents:
        redis.send_agent_message(
            session_id=session_id,
            from_agent="orchestrator",
            to_agent=agent.id,
            message="TERMINATE",
        )
        terminated.append(agent.id)
    
    # Also mark session as done (backup termination via is_session_done check)
    redis.push_done_to_all(session_id)
    
    return {
        "status": "all_terminated",
        "agents_notified": terminated,
        "count": len(terminated),
        "session_marked_done": True,
    }


@mcp.tool()
def read_pane_output(
    agent_id: str = Field(description="ID of the agent to read output from"),
    lines: int = Field(default=50, description="Number of lines to capture"),
) -> dict[str, Any]:
    """
    Read recent output from an agent's tmux pane (DEBUGGING ONLY).
    
    WARNING: Do NOT use this for regular workflow. Agents send results via
    send_to_agent("orchestrator", ...). Use receive_message_from_agents() instead.
    
    This tool is for debugging only - to inspect what an agent is doing when
    message-based communication isn't working.
    """
    working_dir = os.getcwd()
    session_id = get_current_session_id(working_dir)
    if not session_id:
        return {"error": "No active session"}
    
    redis = get_redis_client(working_dir)
    agent = redis.get_agent(session_id, agent_id)
    if not agent:
        return {"error": f"Agent {agent_id} not found"}
    
    if not agent.pane_id:
        return {"error": f"Agent {agent_id} has no pane"}
    
    session = redis.get_session(session_id)
    tmux_session = session.config.get("tmux_session", "agentic") if session else "agentic"
    tmux = TmuxManager(session_name=tmux_session, use_current_session=False)
    
    output = tmux.capture_pane_output(agent.pane_id, lines=lines)
    
    return {
        "agent_id": agent_id,
        "output": output,
        "lines_captured": len(output.split('\n')),
    }


# =============================================================================
# MCP Resources
# =============================================================================


@mcp.resource("session://status")
def get_session_status_resource() -> str:
    """Current session status as JSON."""
    working_dir = os.getcwd()
    session_id = get_current_session_id(working_dir)
    if not session_id:
        return json.dumps({"error": "No active session"})
    
    redis = get_redis_client(working_dir)
    session = redis.get_session(session_id)
    if not session:
        return json.dumps({"error": "Session not found"})
    
    return json.dumps({
        "session_id": session_id,
        "status": session.status.value,
        "working_directory": session.working_directory,
        "created_at": session.created_at,
    }, indent=2)


@mcp.resource("agents://list")
def get_agents_resource() -> str:
    """List of all agents in the current session."""
    working_dir = os.getcwd()
    session_id = get_current_session_id(working_dir)
    if not session_id:
        return json.dumps({"error": "No active session"})
    
    redis = get_redis_client(working_dir)
    agents = redis.get_all_agents(session_id)
    
    return json.dumps([
        {
            "id": a.id,
            "role": a.role,
            "status": a.status.value,
            "scope": {
                "patterns": a.scope.patterns,
                "read_only": a.scope.read_only,
            },
        }
        for a in agents
    ], indent=2)


# =============================================================================
# MCP Prompts
# =============================================================================


# =============================================================================
# Entry Point
@mcp.prompt()
def simple_multi_agent(
    task: str = Field(description="The multi-agent task to perform"),
) -> str:
    """
    Generate a prompt for simple multi-agent workflows using tmux-agents.
    """
    return f"""Run this multi-agent task: {task}

**WORKFLOW**:

1. `start_session()` - Create tmux session

2. `spawn_agent(role="<full task description>")` for each worker
   - Include COMPLETE task in role (not just a title)
   - Tell agent to send results via `send_to_agent("orchestrator", "<results>")`

3. `receive_message_from_agents()` - CRITICAL: Wait for worker results
   - Keep calling this until all workers report back
   - DO NOT skip this step!

4. `terminate_all_agents()` - CRITICAL: Signal workers to exit
   - Workers are waiting in a polling loop for more instructions
   - If you don't call this, they will hang indefinitely!

5. `stop_session()` - End session when done

**IMPORTANT**: After collecting all results, you MUST call `terminate_all_agents()` 
to signal workers to exit. Skipping this leaves workers stuck in their polling loop.
"""


# =============================================================================


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
