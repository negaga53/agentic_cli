# Agentic TMUX

> Multi-agent orchestration for GitHub Copilot CLI and Claude Code via tmux panes.

Spawn multiple AI coding agents that work in parallel, communicate with each other, and report back to you—all running in separate tmux panes you can watch in real-time.

## Quick Start

### Prerequisites

| Requirement | How to Install |
|-------------|---------------|
| **tmux** | `brew install tmux` or `apt install tmux` |
| **Python 3.11+** | [python.org](https://python.org) or your package manager |
| **GitHub Copilot CLI** | `npm install -g @githubnext/github-copilot-cli` |
| *or* **Claude Code** | `npm install -g @anthropic-ai/claude-code` |

### Install

```bash
pip install agentic-tmux
```

Or run the setup script (checks prerequisites + configures everything):

```bash
curl -fsSL https://raw.githubusercontent.com/negaga53/agentic-tmux/main/setup.sh | bash
```

Or from source:

```bash
git clone https://github.com/negaga53/agentic-tmux
cd agentic-tmux
pip install -e .
agentic-tmux setup  # Interactive configuration wizard
```

### Configure MCP

Run `agentic-tmux setup` to configure automatically, or add manually:

**For Copilot CLI**, add to `~/.copilot/mcp-config.json`:

```json
{
  "mcpServers": {
    "agentic": {
      "type": "local",
      "tools": ["*"],
      "command": "agentic-tmux",
      "args": ["mcp"]
    }
  }
}
```

**For Claude Code**, add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "agentic": {
      "command": "agentic-tmux",
      "args": ["mcp"]
    }
  }
}
```

### Use It

1. **Start tmux** (agents run in tmux panes):
   ```bash
   tmux new -s work
   ```

2. **Start Copilot CLI or Claude Code in your project**

3. **Ask your AI assistant to use the agentic tools**:
   ```
   "Use agentic to spawn 2 agents: one to refactor the auth module, 
    another to write tests. Have them report back when done."
   ```

4. **Watch agents work** in tmux panes, or use the monitor:
   ```bash
   agentic-tmux status --watch
   OR
   agentic-tmux monitor
   ```

---

## How It Works

```
┌──────────────────────────────────────────────────────────────────────────┐
│               Your Terminal (Copilot CLI / Claude Code)                   │
│           "Spawn 3 agents to refactor, test, and review..."              │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │ MCP Protocol
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         Agentic MCP Server                                │
│                 Coordinates agents, manages message queues               │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
│  TMUX Pane: W1    │   │  TMUX Pane: W2    │   │  TMUX Pane: W3    │
│                   │ < │                   │ < │                   │
│  copilot -i       │ > │  copilot -i       │ > │  copilot -i       │
│  "Refactor auth"  │   │  "Write tests"    │   │  "Review code"    │
└─────────┬─────────┘   └─────────┬─────────┘   └─────────┬─────────┘
          │                       │                       │
          └───────────────────────┴───────────────────────┘
                                  │
                    Message Queue (SQLite/Redis)
                    Agents can talk to each other!
```

**Key Features:**
- ✅ Agents run in visible tmux panes—watch then work in real-time
- ✅ Agents communicate via message queues (no shared state issues)
- ✅ Works offline with SQLite (Redis optional for persistence)
- ✅ Auto-cleanup on new sessions
- ✅ Compatible with GitHub Copilot CLI and Claude Code

---

## MCP Tools Reference

When you configure MCP, your AI assistant gets these tools:

### Essential Tools (Simple Workflow)

| Tool | What It Does |
|------|--------------|
| `start_session()` | Start a new multi-agent session |
| `spawn_agent(role="...")` | Spawn an agent with a task |
| `receive_message_from_agents()` | Get results from agents |
| `terminate_all_agents()` | Clean up when done |
| `stop_session()` | End the session |

### Example: Simple Two-Agent Task

Tell your AI assistant:
```
Use agentic to:
1. Start a session
2. Spawn agent W1 to refactor src/auth.py with better error handling
3. Spawn agent W2 to write tests for src/auth.py  
4. Wait for both to finish and report their results
5. Stop the session
```

### Advanced Tools

| Tool | What It Does |
|------|--------------|
| `get_status()` | Monitor all agents and tasks |
| `get_agent_logs(agent_id)` | View detailed logs for an agent |

---

## CLI Commands

The CLI is for monitoring and debugging. Use MCP tools for orchestration.

```bash
# Run the setup wizard
agentic-tmux setup

# Check system configuration
agentic-tmux doctor

# Start MCP server (usually auto-started by your CLI tool)
agentic-tmux mcp

# Monitor agents in real-time
agentic-tmux status --watch

# View agent logs
agentic-tmux logs W1 -f

# Interactive monitoring dashboard
agentic-tmux monitor

# Stop current session
agentic-tmux stop

# Export session data
agentic-tmux export
```

---

## Configuration

### Environment Variables (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENTIC_REDIS_HOST` | localhost | Redis host (if using Redis) |
| `AGENTIC_REDIS_PORT` | 6379 | Redis port |

Without Redis, data is stored in `.agentic/agentic.db` in your project directory—automatically cleaned up on new sessions.

### Per-Project Configuration

For Claude Code, add `.mcp.json` to a project root. For Copilot CLI, the global `~/.copilot/mcp-config.json` applies to all projects.

---

## Troubleshooting

### "agentic-tmux: command not found"

Your Python scripts directory isn't in PATH:
```bash
export PATH="$HOME/.local/bin:$PATH"
# Add to ~/.bashrc or ~/.zshrc to make permanent
```

### Agents not spawning

1. Make sure tmux is running: `tmux new -s work`
2. Check you're inside tmux when running commands
3. Run `agentic-tmux doctor` to diagnose issues

### Agents not communicating

1. Check storage is working: `agentic-tmux doctor`
2. If using Redis, make sure it's running: `redis-server`
3. Check agent logs: `agentic-tmux logs W1`

### MCP tools not appearing

1. Restart your CLI tool after adding MCP configuration
2. Verify config syntax with `agentic-tmux doctor`
3. Check MCP server starts: `agentic-tmux mcp` (should show "Starting MCP server...")

### Debug Hooks (Advanced)

For detailed agent behavior logging with GitHub Copilot CLI:

```bash
# Install hooks
agentic-tmux setup --hooks

# After running agents, analyze logs
~/.github/hooks/analyze-agent-logs.sh
```

Logs are written to `.agentic/logs/` in your project directory.

---

## How Agents Communicate

Each spawned agent automatically gets instructions (via `AGENTS.md`) to:

1. **Discover** other agents using `list_agents()`
2. **Execute** their assigned task
3. **Report** results with `send_to_agent("orchestrator", ...)`
4. **Poll** for follow-up instructions until terminated

This ensures reliable communication—agents report back via message queues, not just text output.

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type check
mypy agentic/

# Format code
black agentic/
ruff agentic/
```

---

## License

MIT

## Links

- [GitHub Repository](https://github.com/negaga53/agentic-tmux)
- [Issue Tracker](https://github.com/negaga53/agentic-tmux/issues)
- [GitHub Copilot CLI](https://githubnext.com/projects/copilot-cli/)
- [Claude Code](https://www.anthropic.com/claude-code)
