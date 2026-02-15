"""Tmux pane management for agentic-tmux."""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import libtmux
from libtmux.constants import PaneDirection
from libtmux.exc import LibTmuxException

from agentic.models import Agent

# Maximum number of side-by-side columns before splitting rows
MAX_WORKER_COLUMNS = 3


# AGENTS.md content - loaded into system context by Copilot CLI
# NOTE: This is agent-agnostic; each agent discovers its own ID via the
# get_my_info() MCP tool or the AGENTIC_AGENT_ID environment variable.
AGENTS_MD_CONTENT = '''# Multi-Agent Communication Protocol

You are a worker agent in a coordinated multi-agent system. You MUST follow
this protocol EXACTLY. Failure to comply will cause task failure.

## CRITICAL: Your Role

Your agent ID is set in the AGENTIC_AGENT_ID environment variable.
Your results are ONLY delivered via MCP tools.
Text responses are NOT visible to other agents or the orchestrator.

## Available MCP Tools (USE THESE)

You have access to these MCP tools from the "agentic-worker" server:

1. **list_agents()** - Discover other agents in this session
   - Call this FIRST before any other work
   - Returns all agent IDs and their roles
   - Without this, you don't know who else is working

2. **send_to_agent(agent_id, message)** - Send message to agent or "orchestrator"
   - This is your ONLY way to deliver results
   - If you don't call this, your work is LOST
   - Use agent_id="orchestrator" to report final results

3. **receive_message(timeout=300)** - Wait for incoming messages
   - Call this in a loop after completing your task
   - Returns status: "received", "no_message", or "session_terminated"
   - NEVER exit without checking for messages

4. **check_messages()** - Non-blocking message check
   - Use to see if messages are waiting without blocking

5. **broadcast_message(message)** - Send to all agents
   - Use for announcements that everyone needs

## MANDATORY Workflow (4 Phases)

### Phase 1: DISCOVERY (REQUIRED)
```
result = list_agents()
# Note your ID and other agent IDs
```
⚠️ Do NOT proceed until you have called list_agents().

### Phase 2: EXECUTE
Perform your assigned task. Stay within your file scope.

### Phase 3: REPORT (CRITICAL - DO NOT SKIP)
```
send_to_agent(
    agent_id="orchestrator",
    message=json.dumps({
        "status": "complete",
        "agent_id": "<your agent ID>",
        "result": "<your results here>"
    })
)
```
⚠️ If you skip this step, your work is LOST. The orchestrator will never
receive your results because text output is not delivered to it.

### Phase 4: POLL (CRITICAL - DO NOT EXIT)
```
while True:
    msg = receive_message(timeout=300)
    
    if msg["status"] == "session_terminated":
        break  # Exit gracefully
    
    if msg["status"] == "received":
        if "TERMINATE" in msg["message"]:
            break  # Exit gracefully
        # Process message, respond, continue
    
    # status == "no_message" -> continue polling (DO NOT EXIT)
```
⚠️ NEVER exit without explicit TERMINATE signal.

## Self-Check Before Ending Turn

Before ending your turn, verify ALL are true:
- [ ] Called list_agents() at start
- [ ] Completed assigned task  
- [ ] Called send_to_agent("orchestrator", ...) with results
- [ ] Currently in Phase 4 polling loop OR received TERMINATE

If ANY box is unchecked, you are NOT done. Complete the missing steps NOW.

## Common Mistakes (AVOID)

❌ Exiting after completing task without calling send_to_agent()
❌ Reporting results but not entering the polling loop
❌ Assuming "no_message" means session ended (it doesn't - keep polling!)
❌ Using print/text output instead of send_to_agent() for results
❌ Forgetting to call list_agents() first
'''


@dataclass
class PaneInfo:
    """Information about a tmux pane."""

    pane_id: str
    window_id: str
    session_name: str
    width: int
    height: int
    active: bool
    current_command: str | None = None


class TmuxManager:
    """Manages tmux sessions, windows, and panes for agentic orchestration."""

    def __init__(self, session_name: str = "agentic", use_current_session: bool = True):
        self.server = libtmux.Server()
        self._session: libtmux.Session | None = None
        # Track bottom-most pane ID in each column for grid layout
        self._worker_column_tips: list[str] = []
        
        # If inside tmux and use_current_session is enabled, use the current session
        if use_current_session and self._is_inside_tmux():
            self.session_name = self._get_current_session_name()
        else:
            self.session_name = session_name

    def _is_inside_tmux(self) -> bool:
        """Check if we're running inside a tmux session (internal helper)."""
        return "TMUX" in os.environ

    def _get_current_session_name(self) -> str:
        """Get the name of the current tmux session we're inside."""
        try:
            result = subprocess.run(
                ["tmux", "display-message", "-p", "#S"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass
        return "agentic"  # fallback

    @property
    def session(self) -> libtmux.Session:
        """Get or create the tmux session."""
        if self._session is None:
            self._session = self._get_or_create_session()
        return self._session

    def _get_or_create_session(self) -> libtmux.Session:
        """Get existing session or create a new one."""
        try:
            session = self.server.sessions.get(session_name=self.session_name, default=None)
            if session:
                return session
        except LibTmuxException:
            pass

        # Create new session (suppress the "already exists" error gracefully)
        try:
            return self.server.new_session(
                session_name=self.session_name,
                window_name="admin",
                attach=False,
            )
        except LibTmuxException:
            # Session was created by another process in the meantime
            session = self.server.sessions.get(session_name=self.session_name, default=None)
            if session:
                return session
            raise

    def is_inside_tmux(self) -> bool:
        """Check if we're running inside a tmux session."""
        return self._is_inside_tmux()

    def get_current_pane_id(self) -> str | None:
        """Get the ID of the current pane (if inside tmux)."""
        if not self.is_inside_tmux():
            return None
        return os.environ.get("TMUX_PANE")

    def session_exists(self) -> bool:
        """Check if the agentic session exists."""
        try:
            return self.server.sessions.get(session_name=self.session_name, default=None) is not None
        except LibTmuxException:
            return False

    def create_admin_pane(self, working_dir: str = ".") -> str:
        """
        Create or get the admin pane and launch the monitoring dashboard.
        
        Returns:
            Pane ID of the admin pane.
        """
        window = self.session.active_window
        if window.name != "admin":
            window = self.session.new_window(window_name="admin")
        
        pane = window.active_pane
        
        # Wait for shell to be ready
        time.sleep(0.3)
        
        # Set environment variables (mirrors worker pane setup)
        pane.send_keys(f'export AGENTIC_WORKING_DIR="{working_dir}"', enter=True)
        time.sleep(0.1)
        
        if working_dir != ".":
            pane.send_keys(f"cd {working_dir}", enter=True)
            time.sleep(0.2)
        
        # Launch the monitoring dashboard
        pane.send_keys("agentic-tmux monitor", enter=True)
        
        return pane.id

    def _write_agents_md(self, working_dir: str) -> None:
        """
        Write AGENTS.md file to working directory (once).
        
        This file is automatically loaded into system context by Copilot CLI,
        providing mandatory communication protocol instructions.  The content
        is agent-agnostic; each agent discovers its own ID via the
        AGENTIC_AGENT_ID env var or the get_my_info() MCP tool.
        """
        agents_md_path = Path(working_dir) / "AGENTS.md"
        
        try:
            # Only write if it doesn't already exist
            if not agents_md_path.exists():
                agents_md_path.write_text(AGENTS_MD_CONTENT)
        except Exception as e:
            # Log but don't fail - the inline instructions are backup
            debug_dir = Path(working_dir) / ".agentic"
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_file = debug_dir / "agents_md_errors.log"
            with open(debug_file, "a") as f:
                f.write(f"{time.time()}: Failed to write AGENTS.md: {e}\n")

    def spawn_worker_pane(
        self,
        agent: Agent,
        working_dir: str = ".",
        cli_command: str = "copilot -i",
        session_id: str = "",
        initial_task: str | None = None,
    ) -> str:
        """
        Spawn a new pane for a worker agent.
        
        Args:
            agent: The agent configuration
            working_dir: Working directory for the pane
            cli_command: The CLI command to run (e.g., "copilot -i", "claude")
            session_id: The agentic session ID
            initial_task: Optional initial task to send to the agent
        
        Returns:
            Pane ID of the new worker pane.
        """
        # Write AGENTS.md with protocol instructions (loaded into system context)
        self._write_agents_md(working_dir)
        
        # Find the admin window or create worker window
        worker_window = None
        for window in self.session.windows:
            if window.name == "workers":
                worker_window = window
                break
        
        if worker_window is None:
            worker_window = self.session.new_window(window_name="workers")
            # The new window creates a pane, use it for the first worker
            pane = worker_window.active_pane
            self._worker_column_tips = [pane.id]
        else:
            # Reconstruct column state if not tracked
            if not self._worker_column_tips:
                self._worker_column_tips = self._reconstruct_column_tips(worker_window)
            
            num_columns = len(self._worker_column_tips)
            existing_panes = list(worker_window.panes)
            n = len(existing_panes)
            
            if num_columns < MAX_WORKER_COLUMNS:
                # Phase 1: Add columns (split rightmost tip to the right)
                target = self._get_pane_by_id(self._worker_column_tips[-1])
                pane = target.split(direction=PaneDirection.Right)
                self._worker_column_tips.append(pane.id)
            else:
                # Phase 2: Round-robin split columns below
                col_idx = (n - MAX_WORKER_COLUMNS) % MAX_WORKER_COLUMNS
                target = self._get_pane_by_id(self._worker_column_tips[col_idx])
                pane = target.split(direction=PaneDirection.Below)
                # Update tip to the new bottom pane
                self._worker_column_tips[col_idx] = pane.id
        
        # Wait for shell to be ready
        time.sleep(0.3)
        
        # Set up environment variables one at a time with small delays
        pane.send_keys(f'export AGENTIC_SESSION_ID="{session_id}"', enter=True)
        time.sleep(0.1)
        pane.send_keys(f'export AGENTIC_AGENT_ID="{agent.id}"', enter=True)
        time.sleep(0.1)
        # Export working directory for per-repo storage
        pane.send_keys(f'export AGENTIC_WORKING_DIR="{working_dir}"', enter=True)
        time.sleep(0.1)
        # Don't export full role (too long, contains special chars)
        # Store a shortened version for reference
        short_role = agent.role[:50].replace('"', "'").replace('\n', ' ')
        pane.send_keys(f'export AGENTIC_AGENT_ROLE="{short_role}"', enter=True)
        time.sleep(0.1)
        pane.send_keys(f'export AGENTIC_PANE_ID="{pane.id}"', enter=True)
        time.sleep(0.1)
        pane.send_keys(f"cd {working_dir}", enter=True)
        time.sleep(0.2)  # Longer delay for cd
        
        # Start the CLI first, then send the initial task separately
        # Extract base command without -i flag arguments
        # For copilot: "copilot -i" => "copilot", "copilot" => "copilot"
        # For other CLIs like claude/aider, just use as-is
        base_cli = cli_command
        if " -i" in cli_command:
            # Remove the -i flag as we'll send the prompt separately
            base_cli = cli_command.replace(" -i", "")
        
        # Start the CLI with the initial task
        # For Copilot, use -i flag with properly escaped prompt
        # For other CLIs, start them and send prompt separately (may not work for all)
        base_cli = cli_command
        
        if initial_task:
            # Clean the task text
            clean_task = initial_task.replace('\n', ' ').strip()
            # Limit length to avoid shell issues (shells typically handle 100K+ chars)
            if len(clean_task) > 50000:
                clean_task = clean_task[:50000] + "..."
            
            # Debug: Log what we're sending
            from pathlib import Path
            debug_dir = Path(working_dir) / ".agentic"
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_file = debug_dir / "spawn_debug.log"
            with open(debug_file, "a") as f:
                f.write(f"\n=== {time.time()} ===\n")
                f.write(f"Pane: {pane.id}\n")
                f.write(f"Agent: {agent.id}\n")
                f.write(f"Task length: {len(clean_task)}\n")
                f.write(f"Task preview: {clean_task[:200]}...\n")
            
            # Check if this is a copilot command (handles "copilot", "copilot -i", etc.)
            if "copilot" in cli_command.lower():
                # Use copilot -i with the prompt
                # Escape single quotes in the prompt for shell
                escaped_task = clean_task.replace("'", "'\"'\"'")
                
                # Build MCP config for worker MCP tools
                # The worker MCP server needs AGENTIC_SESSION_ID and AGENTIC_AGENT_ID env vars
                # Format matches ~/.copilot/mcp-config.json structure
                import json
                mcp_config = {
                    "mcpServers": {
                        "agentic-worker": {
                            "type": "local",
                            "tools": ["*"],
                            "command": "agentic-worker-mcp",
                            "args": [],
                            "env": {
                                "AGENTIC_SESSION_ID": session_id,
                                "AGENTIC_AGENT_ID": agent.id,
                                "AGENTIC_WORKING_DIR": working_dir,
                            },
                            "timeout": 600000,  # 10 minutes - for long polling operations
                        }
                    }
                }
                # Escape the JSON for shell (single quotes escape, double quotes are fine inside)
                mcp_json = json.dumps(mcp_config)
                escaped_mcp = mcp_json.replace("'", "'\"'\"'")
                
                # Build full command with MCP config and allow-all for non-interactive
                full_cmd = f"copilot --additional-mcp-config '{escaped_mcp}' --allow-all -i '{escaped_task}'"
                
                with open(debug_file, "a") as f:
                    f.write(f"Using copilot -i with escaped prompt and MCP config\n")
                    f.write(f"Full cmd length: {len(full_cmd)}\n")
                    f.write(f"MCP config: {mcp_json}\n")
                pane.send_keys(full_cmd, enter=True)
            else:
                # For other CLIs, start them then send prompt separately
                # (This may not work for all TUI-based CLIs)
                if " -i" in base_cli:
                    base_cli = base_cli.replace(" -i", "")
                pane.send_keys(base_cli, enter=True)
                time.sleep(5.0)  # Wait for CLI to start
                # Try to send the prompt (may not work for all CLIs)
                import subprocess
                subprocess.run(
                    ["tmux", "send-keys", "-t", pane.id, "-l", clean_task],
                    capture_output=True,
                    check=False,
                )
                time.sleep(0.3)
                subprocess.run(
                    ["tmux", "send-keys", "-t", pane.id, "C-m"],
                    capture_output=True,
                    check=False,
                )
        else:
            # No initial task, just start the CLI
            if " -i" in base_cli:
                base_cli = base_cli.replace(" -i", "")
            pane.send_keys(base_cli, enter=True)
        
        time.sleep(0.5)
        return pane.id

    def spawn_multiple_workers(
        self,
        agents: list[Agent],
        working_dir: str = ".",
        cli_command: str = "copilot -i",
        session_id: str = "",
        layout: str = "tiled",
    ) -> dict[str, str]:
        """
        Spawn multiple worker panes and arrange them.
        
        Args:
            agents: List of agent configurations
            working_dir: Working directory for panes
            cli_command: The CLI command to run
            session_id: The agentic session ID
            layout: Tmux layout (tiled, even-horizontal, even-vertical, main-horizontal, main-vertical)
        
        Returns:
            Dict mapping agent_id to pane_id.
        """
        pane_mapping: dict[str, str] = {}
        
        for agent in agents:
            pane_id = self.spawn_worker_pane(
                agent=agent,
                working_dir=working_dir,
                cli_command=cli_command,
                session_id=session_id,
            )
            pane_mapping[agent.id] = pane_id
            agent.pane_id = pane_id
        
        # Only apply layout if explicitly requested (grid layout is managed
        # by spawn_worker_pane's column/row splitting logic)
        if layout != "tiled":
            self._apply_layout(layout)
        
        return pane_mapping

    def _reconstruct_column_tips(self, worker_window: libtmux.Window) -> list[str]:
        """Reconstruct column tip pane IDs from existing window layout.
        
        Groups panes by their left x-position to identify columns,
        then picks the bottom-most pane in each column as the tip.
        """
        panes = list(worker_window.panes)
        if not panes:
            return []
        
        # Group panes by x-position (column)
        columns: dict[int, list[libtmux.Pane]] = {}
        for pane in panes:
            left = int(pane.pane_left)
            columns.setdefault(left, []).append(pane)
        
        # Sort by x-position (left to right) and pick bottom-most pane per column
        tips = []
        for col_x in sorted(columns):
            bottom = max(columns[col_x], key=lambda p: int(p.pane_top))
            tips.append(bottom.id)
        
        return tips

    def _apply_layout(self, layout: str | None = "tiled") -> None:
        """Apply a layout to the workers window."""
        if layout is None:
            return
        for window in self.session.windows:
            if window.name == "workers":
                try:
                    window.select_layout(layout)
                except LibTmuxException:
                    pass
                break

    def send_keys_to_pane(self, pane_id: str, keys: str, enter: bool = True) -> bool:
        """
        Send keys to a specific pane.
        
        Args:
            pane_id: Target pane ID
            keys: Keys/text to send
            enter: Whether to press Enter after
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            pane = self._get_pane_by_id(pane_id)
            if pane:
                pane.send_keys(keys, enter=enter)
                return True
        except LibTmuxException:
            pass
        return False

    def send_prompt_to_worker(
        self,
        pane_id: str,
        prompt: str,
        context: dict[str, Any] | None = None,
    ) -> bool:
        """
        Send a prompt to a worker pane's CLI.
        
        Args:
            pane_id: Target pane ID
            prompt: The prompt to send
            context: Additional context to include
        
        Returns:
            True if successful.
        """
        # Build the full prompt with context if provided
        full_prompt = prompt
        if context:
            ctx_str = " ".join(f"[{k}: {v}]" for k, v in context.items())
            full_prompt = f"{ctx_str} {prompt}"
        
        return self.send_keys_to_pane(pane_id, full_prompt, enter=True)

    def capture_pane_output(self, pane_id: str, lines: int = 100) -> str:
        """
        Capture recent output from a pane.
        
        Args:
            pane_id: Target pane ID
            lines: Number of lines to capture
        
        Returns:
            Captured text.
        """
        try:
            pane = self._get_pane_by_id(pane_id)
            if pane:
                return "\n".join(pane.capture_pane(start=-lines))
        except LibTmuxException:
            pass
        return ""

    def get_pane_info(self, pane_id: str) -> PaneInfo | None:
        """Get information about a pane."""
        try:
            pane = self._get_pane_by_id(pane_id)
            if pane:
                return PaneInfo(
                    pane_id=pane.id,
                    window_id=pane.window.id,
                    session_name=self.session_name,
                    width=int(pane.width),
                    height=int(pane.height),
                    active=pane == pane.window.active_pane,
                    current_command=pane.current_command,
                )
        except (LibTmuxException, AttributeError):
            pass
        return None

    def kill_pane(self, pane_id: str) -> bool:
        """Kill a specific pane."""
        try:
            pane = self._get_pane_by_id(pane_id)
            if pane:
                pane.kill()
                return True
        except LibTmuxException:
            pass
        return False

    def respawn_pane(self, pane_id: str, command: str | None = None) -> bool:
        """
        Respawn a pane (kill and restart).
        
        Args:
            pane_id: Target pane ID
            command: Command to run in respawned pane (optional)
        
        Returns:
            True if successful.
        """
        try:
            # Use tmux command directly for respawn-pane
            cmd = ["tmux", "respawn-pane", "-k", "-t", pane_id]
            if command:
                cmd.append(command)
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False

    def kill_all_workers(self) -> int:
        """
        Kill all worker panes.
        
        Returns:
            Number of panes killed.
        """
        killed = 0
        for window in self.session.windows:
            if window.name == "workers":
                for pane in window.panes:
                    try:
                        pane.kill()
                        killed += 1
                    except LibTmuxException:
                        pass
        return killed

    def list_all_panes(self) -> list[PaneInfo]:
        """List all panes in the session."""
        panes = []
        try:
            for window in self.session.windows:
                for pane in window.panes:
                    panes.append(
                        PaneInfo(
                            pane_id=pane.id,
                            window_id=window.id,
                            session_name=self.session_name,
                            width=int(pane.width),
                            height=int(pane.height),
                            active=pane == window.active_pane,
                            current_command=pane.current_command,
                        )
                    )
        except LibTmuxException:
            pass
        return panes

    def focus_pane(self, pane_id: str) -> bool:
        """Focus (select) a specific pane."""
        try:
            pane = self._get_pane_by_id(pane_id)
            if pane:
                pane.select()
                return True
        except LibTmuxException:
            pass
        return False

    def attach_session(self) -> None:
        """Attach to the agentic session (blocks until detached)."""
        self.session.attach()

    def kill_session(self) -> bool:
        """Kill the entire agentic session."""
        try:
            self.session.kill()
            self._session = None
            return True
        except LibTmuxException:
            return False

    def _get_pane_by_id(self, pane_id: str) -> libtmux.Pane | None:
        """Find a pane by its ID."""
        try:
            for window in self.session.windows:
                for pane in window.panes:
                    if pane.id == pane_id:
                        return pane
        except LibTmuxException:
            pass
        return None

    def set_pane_title(self, pane_id: str, title: str) -> bool:
        """Set a pane's title for easier identification."""
        try:
            cmd = ["tmux", "select-pane", "-t", pane_id, "-T", title]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False


def get_tmux_manager(session_name: str = "agentic") -> TmuxManager:
    """Factory function to get a TmuxManager instance."""
    return TmuxManager(session_name=session_name)


def check_tmux_available() -> bool:
    """Check if tmux is available on the system."""
    try:
        result = subprocess.run(
            ["tmux", "-V"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False
