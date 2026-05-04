---
description: Operate the tokenq local proxy (start, stop, status, reset, install, test, logs, dashboard, env)
argument-hint: [start|stop|status|reset|install|test|logs|dashboard|env|help]
allowed-tools: Bash, Read
---

You are operating the **tokenq** local proxy from inside this project. The user invoked `/tokenq` with arguments: `$ARGUMENTS`.

**Project context:**
- Working directory: this repo's root
- Virtualenv: `.venv/` at the repo root
- CLI entry point: `.venv/bin/tokenq`
- Default ports: proxy `8089`, dashboard `8090`
- DB: `~/.tokenq/tokenq.db`

**Dispatch rules — pick the matching subcommand from `$ARGUMENTS` and run only that path. If no argument is given, show the help block.**

---

### `start`
Start the proxy + dashboard in the background and confirm both are responding.

```bash
# fail fast if .venv is missing
test -x .venv/bin/tokenq || { echo "Run /tokenq install first"; exit 1; }
# don't double-start
if lsof -ti:8089 >/dev/null 2>&1; then
  echo "tokenq already running on :8089"
else
  nohup .venv/bin/tokenq start > /tmp/tokenq.log 2>&1 &
  sleep 2
fi
# verify
curl -fsS http://127.0.0.1:8089/healthz && echo
curl -fsS -o /dev/null -w "dashboard HTTP %{http_code}\n" http://127.0.0.1:8090/
```

After it's up, tell the user to run `export ANTHROPIC_BASE_URL=http://127.0.0.1:8089` in their shell and restart Claude Code / Cursor to route through the proxy. Also remind them the dashboard is at <http://127.0.0.1:8090>.

### `stop`
Stop any running tokenq processes cleanly.

```bash
pkill -f 'tokenq start|tokenq.proxy.app|tokenq.dashboard.app' 2>/dev/null || true
sleep 1
if lsof -ti:8089,8090 >/dev/null 2>&1; then
  lsof -ti:8089,8090 | xargs kill -9 2>/dev/null || true
fi
echo "stopped"
```

### `status`
Print last-24h usage from the local SQLite DB and check liveness.

```bash
.venv/bin/tokenq status 2>/dev/null || echo "(no .venv yet — run /tokenq install)"
echo
if curl -fsS http://127.0.0.1:8089/healthz >/dev/null 2>&1; then
  echo "proxy: running on :8089"
else
  echo "proxy: not running"
fi
if curl -fsS http://127.0.0.1:8090/ >/dev/null 2>&1; then
  echo "dashboard: running on :8090"
else
  echo "dashboard: not running"
fi
```

### `reset`
Wipe the local database. **Confirm with the user before running** unless they passed `--yes` or `force` in `$ARGUMENTS`.

```bash
.venv/bin/tokenq reset --yes
```

### `install`
Set up the venv and install the package in dev mode. Idempotent.

```bash
if [ ! -d .venv ]; then python3.13 -m venv .venv || python3.11 -m venv .venv; fi
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -e ".[dev]"
.venv/bin/python -c "import tokenq; print('tokenq', tokenq.__version__, 'ready')"
```

### `test`
Run the test suite.

```bash
.venv/bin/pytest -q
```

### `logs`
Show the last 20 intercepted requests from the DB (model, tokens, latency, status). If the server isn't running, this still works against the persisted DB.

```bash
DB="$HOME/.tokenq/tokenq.db"
if [ ! -f "$DB" ]; then echo "no DB yet — start the proxy and make a request"; exit 0; fi
sqlite3 -header -column "$DB" "SELECT datetime(ts,'unixepoch','localtime') AS time, model, input_tokens AS in_t, output_tokens AS out_t, latency_ms AS lat_ms, status_code AS status, cached_locally AS cache FROM requests ORDER BY id DESC LIMIT 20;"
```

### `dashboard`
Print the dashboard URL and check if it's up. On macOS, open it in the default browser.

```bash
URL="http://127.0.0.1:8090"
echo "$URL"
curl -fsS -o /dev/null "$URL" && echo "(up)" || echo "(not running — try /tokenq start)"
[ "$(uname)" = "Darwin" ] && open "$URL" 2>/dev/null || true
```

### `env`
Print the exact shell snippet the user needs to route their client through tokenq.

```bash
cat <<'EOF'
# Add to your shell or set per-session:
export ANTHROPIC_BASE_URL=http://127.0.0.1:8089
# Then restart Claude Code / Cursor so they pick up the new env.
EOF
```

### `help` (or no argument)
Print this menu — do **not** run any other command.

```
/tokenq start       — start proxy (8089) + dashboard (8090) in the background
/tokenq stop        — stop both
/tokenq status      — show last-24h stats + liveness
/tokenq logs        — last 20 intercepted requests
/tokenq dashboard   — open dashboard in browser
/tokenq env         — print ANTHROPIC_BASE_URL snippet
/tokenq install     — create .venv and install deps
/tokenq test        — run pytest
/tokenq reset       — wipe local DB (asks first)
```

---

**Important:**
- Run **only** the subcommand the user specified — don't chain unrelated steps.
- If `$ARGUMENTS` is empty or `help`, just print the menu above and stop.
- If a command fails because `.venv` doesn't exist, suggest `/tokenq install`.
- If a command fails because the proxy isn't running, suggest `/tokenq start`.
- Keep the final reply concise — show the command output, then a one-line summary.
