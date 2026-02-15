*This is a submission for the [GitHub Copilot CLI Challenge](https://dev.to/challenges/github-2026-01-21)*

## What I Built

**Agentic TMUX** — a multi-agent orchestration system for GitHub Copilot CLI that runs multiple AI agents in parallel tmux panes.

The idea is simple: instead of one AI doing everything sequentially, you describe the work and agents split it up. Each agent runs in its own tmux pane, so you can literally watch them think, code, and talk to each other in real-time.

What makes this different from other multi-agent setups:

- **One prompt is all it takes.** You tell Copilot CLI what you want done. It spawns the agents, assigns tasks, collects results. You just watch.
- **Agents talk directly to each other.** No round-tripping through a coordinator. Agent W1 can message W2 directly via message queues. This keeps context windows small and eliminates the "middle man" bottleneck where a coordinator has to relay everything.
- **Full transparency.** Every agent runs in a visible tmux pane. You see exactly what each agent is doing — their thought process, tool calls, file edits, everything. No black box.

### How It Works

```
You: "Spawn 2 agents: a coder and a reviewer.
      Build an HTML snake game together."
         │
         ▼
┌─────────────────────────────────────────┐
│          Agentic MCP Server             │
│     Spawns panes · Routes messages      │
└─────────┬───────────────────┬───────────┘
          ▼                   ▼
   ┌─────────────┐    ┌─────────────┐
   │  Pane: W1   │◄──►│  Pane: W2   │
   │  Coder      │    │  Reviewer   │
   │  copilot -i │    │  copilot -i │
   └─────────────┘    └─────────────┘
         ▲  Direct messaging  ▲
         └────────────────────┘
```

The agents communicate through lightweight message queues (SQLite by default — no external services needed). Each agent has MCP tools to `send_to_agent()`, `receive_message()`, and `list_agents()`. They coordinate autonomously.

### Why Direct Communication Matters

Most multi-agent frameworks funnel everything through a central orchestrator. Agent A finishes → sends results to coordinator → coordinator summarizes → forwards to Agent B. Every hop burns context tokens and loses details.

With Agentic TMUX, Agent A sends directly to Agent B. The full message arrives without summarization or token overhead. This matters when agents need to iterate — a code reviewer and coder going back and forth 5 times would require 10 coordinator round-trips in a traditional setup. Here, they just talk.

## Demo

### Demo 1: Guess the Number

Two agents play a number-guessing game. One picks a secret number between 1–100, the other guesses. They communicate back and forth with "higher/lower" hints until the number is found.

![Guess the Number demo](demos/guess_number.gif)

**The prompt:**
> Spawn 2 agents. Agent 1 picks a secret number between 1 and 100. Agent 2 guesses it. They exchange messages with higher/lower hints until the number is found.

This demo shows direct agent-to-agent messaging in action — no orchestrator in the loop during the guessing rounds.

---

### Demo 2: Snake Game (Coder + Reviewer)

A coder agent builds an HTML/JS snake game from scratch. A reviewer agent inspects each iteration, suggests improvements, and approves the final version.

![Snake implementation](demos/snake_impl.gif)

**The prompt:**
> Spawn 2 agents. Agent 1 is a frontend developer who builds an HTML/JS snake game. Agent 2 is an expert code reviewer who reviews each iteration and suggests improvements. They go back and forth until the reviewer approves.

**What actually happened (from the activity log):**

> **W2 → W1:** Here are the review criteria: modern visuals with smooth animations, responsive controls, smooth interpolated movement, cross-device support, and well-commented single-file code.
>
> **W1 → W2:** Confirmed. Beginning implementation.
>
> *— W1 codes —*
>
> **W1 → W2:** Initial implementation submitted. Ready for review.
>
> **W2 → W1:** Six items to fix — add food/snake animations and board grid, pause button visual state, interpolated movement instead of grid-jumping, reduce shadow blur on mobile, add minimum playable area with dynamic scaling, group code with section headers.
>
> **W1 → W2:** All 6 items acknowledged. Updating now.
>
> *— W1 updates —*
>
> **W2 → W1:** Re-review complete. Visually polished, smooth animations, excellent UX. **No further improvements required. Approved.**

Two agents, one review cycle. The coder and reviewer talked directly — no orchestrator relaying messages between them.

**The final game:**

![Snake game output](demos/snake_demo.gif)

---

**Repo:** [github.com/negaga53/agentic-tmux](https://github.com/negaga53/agentic-tmux)

## My Experience with GitHub Copilot CLI

Building Agentic TMUX was a case of using Copilot CLI to build a tool *for* Copilot CLI. The entire system is an MCP server that extends what Copilot CLI can do — giving it the ability to spawn copies of itself in tmux panes and coordinate them.

Copilot CLI was the primary development tool throughout. It helped scaffold the MCP server, implement the SQLite storage layer, build the Rich-based monitoring dashboard, and debug the tricky parts of tmux pane management (getting environment variables to propagate correctly through shell commands sent to panes was surprisingly finicky).

The most interesting part was discovering how well Copilot CLI works as an *agent* in multi-agent scenarios. When given clear instructions via the `AGENTS.md` protocol file and MCP tools for messaging, it reliably follows the discover → execute → report → poll workflow. The agents don't need hand-holding — they call `list_agents()`, do their work, send results, and wait for further instructions.

One insight: keeping agent prompts focused on *what* to do (not *how* to communicate) produces better results. The communication protocol is baked into the `AGENTS.md` file that Copilot CLI loads automatically, so the spawn prompt can focus entirely on the task. This separation of concerns — task prompt vs. protocol instructions — was key to making the system reliable.
