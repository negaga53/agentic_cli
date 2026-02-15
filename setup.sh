#!/usr/bin/env bash
# Agentic TMUX - Easy Setup Script
# Supports: GitHub Copilot CLI, Claude Code
# Usage: curl -fsSL https://raw.githubusercontent.com/negaga53/agentic-tmux/main/setup.sh | bash

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║              AGENTIC TMUX - Multi-Agent Setup                 ║"
    echo "║        Orchestrate AI coding agents in tmux panes             ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

check_prereq() {
    local name="$1"
    local cmd="$2"
    local install_hint="$3"
    
    if command -v "$cmd" &> /dev/null; then
        echo -e "  ${GREEN}✓${NC} $name"
        return 0
    else
        echo -e "  ${RED}✗${NC} $name - ${YELLOW}$install_hint${NC}"
        return 1
    fi
}

print_header

echo -e "${BLUE}Checking prerequisites...${NC}\n"

PREREQS_OK=true

# Required
check_prereq "Python 3.11+" "python3" "Install from python.org or your package manager" || PREREQS_OK=false
check_prereq "tmux" "tmux" "brew install tmux / apt install tmux" || PREREQS_OK=false
check_prereq "pip" "pip3" "Usually comes with Python" || PREREQS_OK=false

# Check Python version
if command -v python3 &> /dev/null; then
    PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
    if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]; }; then
        echo -e "  ${RED}✗${NC} Python version $PY_VERSION (need 3.11+)"
        PREREQS_OK=false
    fi
fi

# Optional (at least one needed)
echo ""
echo -e "${BLUE}AI CLI tools (at least one required):${NC}"
HAS_CLI=false
if command -v copilot &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} GitHub Copilot CLI"
    HAS_CLI=true
else
    echo -e "  ${YELLOW}○${NC} GitHub Copilot CLI - npm install -g @githubnext/github-copilot-cli"
fi

if command -v claude &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} Claude Code"
    HAS_CLI=true
else
    echo -e "  ${YELLOW}○${NC} Claude Code - npm install -g @anthropic-ai/claude-code"
fi

if [ "$HAS_CLI" = false ]; then
    echo -e "\n  ${RED}Warning:${NC} No AI CLI found. Install at least one:"
    echo -e "    ${CYAN}npm install -g @githubnext/github-copilot-cli${NC}"
    echo -e "    ${CYAN}npm install -g @anthropic-ai/claude-code${NC}"
fi

if [ "$PREREQS_OK" = false ]; then
    echo -e "\n${RED}Please install missing prerequisites and run again.${NC}"
    exit 1
fi

echo -e "\n${GREEN}Prerequisites OK!${NC}\n"

# Install agentic-tmux
echo -e "${BLUE}Installing agentic-tmux...${NC}"

# Check if we're in the repo directory
if [ -f "pyproject.toml" ] && grep -q "agentic-tmux" pyproject.toml 2>/dev/null; then
    echo "  Installing from local source..."
    pip3 install -e . --quiet
else
    echo "  Installing from PyPI..."
    pip3 install agentic-tmux --quiet 2>/dev/null || {
        echo "  PyPI not available, cloning from GitHub..."
        TEMP_DIR=$(mktemp -d)
        git clone --depth 1 https://github.com/negaga53/agentic-tmux.git "$TEMP_DIR" 2>/dev/null
        pip3 install -e "$TEMP_DIR" --quiet
    }
fi

echo -e "  ${GREEN}✓${NC} agentic-tmux installed"

# Verify installation
if ! command -v agentic-tmux &> /dev/null; then
    echo -e "${YELLOW}Note:${NC} 'agentic-tmux' not in PATH. You may need to:"
    echo -e "  ${CYAN}export PATH=\"\$HOME/.local/bin:\$PATH\"${NC}"
    echo -e "  Add this to your ~/.bashrc or ~/.zshrc"
fi

# Setup MCP configuration
echo -e "\n${BLUE}Would you like to configure MCP integration?${NC}"
echo "  1) Copilot CLI (~/.copilot/mcp-config.json)"
echo "  2) Claude Code (~/.claude.json)"
echo "  3) Both"
echo "  4) Skip (configure manually later)"
echo ""
read -p "Choose [1-4]: " MCP_CHOICE

setup_copilot_mcp() {
    COPILOT_CONFIG_DIR="$HOME/.copilot"
    COPILOT_CONFIG_FILE="$COPILOT_CONFIG_DIR/mcp-config.json"
    
    mkdir -p "$COPILOT_CONFIG_DIR"
    
    if [ -f "$COPILOT_CONFIG_FILE" ]; then
        echo "  Existing Copilot CLI MCP config found, backing up..."
        cp "$COPILOT_CONFIG_FILE" "$COPILOT_CONFIG_FILE.bak"
        echo -e "  ${YELLOW}Note:${NC} Please manually add agentic to your existing config"
    else
        cat > "$COPILOT_CONFIG_FILE" << 'EOF'
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
EOF
        echo -e "  ${GREEN}✓${NC} Copilot CLI configured at $COPILOT_CONFIG_FILE"
    fi
}

setup_claude_mcp() {
    CLAUDE_CONFIG_FILE="$HOME/.claude.json"
    
    if [ -f "$CLAUDE_CONFIG_FILE" ]; then
        echo "  Existing Claude Code config found, backing up..."
        cp "$CLAUDE_CONFIG_FILE" "$CLAUDE_CONFIG_FILE.bak"
        # Try to merge (simple approach - just notify user)
        echo -e "  ${YELLOW}Note:${NC} Please manually add agentic to your existing config"
    else
        cat > "$CLAUDE_CONFIG_FILE" << 'EOF'
{
  "mcpServers": {
    "agentic": {
      "command": "agentic-tmux",
      "args": ["mcp"]
    }
  }
}
EOF
        echo -e "  ${GREEN}✓${NC} Claude Code configured at $CLAUDE_CONFIG_FILE"
    fi
}

case $MCP_CHOICE in
    1)
        setup_copilot_mcp
        ;;
    2)
        setup_claude_mcp
        ;;
    3)
        setup_copilot_mcp
        setup_claude_mcp
        ;;
    4|*)
        echo "  Skipping MCP configuration"
        ;;
esac

# Install debug hooks (optional)
echo -e "\n${BLUE}Would you like to install debug hooks for Copilot CLI?${NC}"
echo "  These capture detailed logs of agent behavior for debugging"
read -p "Install debug hooks? [y/N]: " INSTALL_HOOKS

if [[ "$INSTALL_HOOKS" =~ ^[Yy]$ ]]; then
    HOOKS_DIR="$HOME/.github/hooks"
    mkdir -p "$HOOKS_DIR"
    
    # Copy hooks from package or download
    if [ -d ".github/hooks" ]; then
        cp -r .github/hooks/* "$HOOKS_DIR/"
    else
        # Download from GitHub
        HOOKS_URL="https://raw.githubusercontent.com/negaga53/agentic-tmux/main/.github/hooks"
        for hook in hooks.json log-session-start.sh log-session-end.sh log-prompt.sh log-pre-tool.sh log-post-tool.sh log-error.sh analyze-agent-logs.sh; do
            curl -fsSL "$HOOKS_URL/$hook" -o "$HOOKS_DIR/$hook" 2>/dev/null || true
        done
    fi
    chmod +x "$HOOKS_DIR"/*.sh 2>/dev/null || true
    echo -e "  ${GREEN}✓${NC} Debug hooks installed to $HOOKS_DIR"
fi

# Final instructions
echo -e "\n${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}                    Setup Complete!                             ${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${CYAN}Quick Start:${NC}"
echo ""
echo "  1. Start tmux:"
echo -e "     ${YELLOW}tmux new -s work${NC}"
echo ""
echo "  2. Start Copilot CLI or Claude Code in your project and use MCP tools:"
echo -e "     Ask your AI to use ${YELLOW}agentic${NC} MCP tools"
echo ""
echo "  3. Or use the CLI for manual control:"
echo -e "     ${YELLOW}agentic-tmux status --watch${NC}"
echo ""
echo -e "${CYAN}Documentation:${NC} https://github.com/negaga53/agentic-tmux"
echo ""
