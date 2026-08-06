# github/

Human-browsable automation suite backing `.github/workflows/agent_sync.yml`,
kept separate from the dot-prefixed `.github/` directory (which GitHub itself
reads) so the actual logic is easy to find, read, and edit.

Ported from Visual-Graph-Programming's `github/` automation suite.

| Directory | Purpose |
| --- | --- |
| `config/` | `automation_rules.yaml` (policy DSL) and `project_labels.json` (label taxonomy) |
| `scripts/` | `agent_tools.py` (ProjectV2 GraphQL client), `sync_backlog.py` (roadmap→board reconciler), `check_commit_ref.py` (commit-message ticket linker) |
| `hooks/` | Local git hooks (`pre-commit`, `post-commit`) plus `install.sh` to symlink them into `.git/hooks/` |

## Setup

```bash
bash github/hooks/install.sh
export PROJECT_ID="PVT_..."      # ProjectV2 node ID, see `gh project view <n> --owner <o> --format json`
export GITHUB_TOKEN="..."        # token with repo + project scopes
```

## CI

`.github/workflows/agent_sync.yml` runs `github/scripts/sync_backlog.py` on
every push to `moon/ROADMAP.md` or `moon/CHANGELOG.md`, or on demand via
`workflow_dispatch`. It needs two repository secrets
(`PROJECT_AUTOMATION_TOKEN`, `GEMINI_API_KEY`) and one repository variable
(`PROJECT_ID`) configured before it can mutate a live board — until then,
runs will fail fast rather than silently no-op.

> **TODO (template adoption):** update `github/config/automation_rules.yaml`'s
> `project.owner`/`project.number` to point at your own board.
