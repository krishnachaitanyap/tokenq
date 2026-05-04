"""Tests for the codeburn-style activity classifier.

Pinning the classifier behavior is important: the dashboard's activity-mix
panel groups requests by these labels, and a regression that flips even one
turn from 'debugging' to 'coding' would silently distort historical pivots.
"""
from __future__ import annotations


def _user_msg(text):
    return {"role": "user", "content": text}


def _assistant_tool(uid, name, **inp):
    return {"role": "assistant", "content": [
        {"type": "tool_use", "id": uid, "name": name, "input": inp},
    ]}


# ---------- direct classify_turn() unit tests ----------

def test_no_tools_no_keywords_is_conversation():
    from tokenq.proxy.classify import classify_turn
    assert classify_turn(tools_used=[], user_text="thanks!") == "conversation"


def test_no_tools_brainstorm_keyword():
    from tokenq.proxy.classify import classify_turn
    assert classify_turn(
        tools_used=[], user_text="what if we tried a different approach?"
    ) == "brainstorming"


def test_no_tools_explore_keyword():
    from tokenq.proxy.classify import classify_turn
    assert classify_turn(
        tools_used=[], user_text="can you explain how the cache works?"
    ) == "exploration"


def test_no_tools_debug_keyword():
    from tokenq.proxy.classify import classify_turn
    assert classify_turn(
        tools_used=[], user_text="getting a 500 error from the API"
    ) == "debugging"


def test_edit_with_debug_keyword_is_debugging():
    from tokenq.proxy.classify import classify_turn
    assert classify_turn(
        tools_used=["Edit"], user_text="fix the bug in auth.py",
    ) == "debugging"


def test_edit_with_refactor_keyword_is_refactoring():
    from tokenq.proxy.classify import classify_turn
    assert classify_turn(
        tools_used=["Edit", "Read"],
        user_text="please rename this and clean up the helpers",
    ) == "refactoring"


def test_edit_with_feature_keyword_is_feature_dev():
    from tokenq.proxy.classify import classify_turn
    assert classify_turn(
        tools_used=["Write"],
        user_text="add a new endpoint that returns user stats",
    ) == "feature_dev"


def test_edit_without_intent_keywords_is_plain_coding():
    from tokenq.proxy.classify import classify_turn
    assert classify_turn(
        tools_used=["Edit"], user_text="ok",
    ) == "coding"


def test_bash_pytest_is_testing_even_with_edits():
    """Codeburn priority: testing wins over coding when both apply."""
    from tokenq.proxy.classify import classify_turn
    assert classify_turn(
        tools_used=["Bash", "Edit"],
        bash_text="pytest tests/test_auth.py",
        user_text="run the tests",
    ) == "testing"


def test_bash_git_commit_is_git_ops():
    from tokenq.proxy.classify import classify_turn
    assert classify_turn(
        tools_used=["Bash"],
        bash_text="git commit -m 'fix bug'",
    ) == "git_ops"


def test_bash_npm_build_is_build_deploy():
    from tokenq.proxy.classify import classify_turn
    assert classify_turn(
        tools_used=["Bash"],
        bash_text="npm run build && npm publish",
    ) == "build_deploy"


def test_bash_pip_install_is_build_deploy():
    from tokenq.proxy.classify import classify_turn
    assert classify_turn(
        tools_used=["Bash"], bash_text="pip install fastembed",
    ) == "build_deploy"


def test_only_reads_is_exploration():
    from tokenq.proxy.classify import classify_turn
    assert classify_turn(
        tools_used=["Read", "Grep"], user_text="where is auth defined?",
    ) == "exploration"


def test_task_tool_is_planning():
    from tokenq.proxy.classify import classify_turn
    assert classify_turn(tools_used=["TaskCreate"]) == "planning"
    assert classify_turn(tools_used=["TodoWrite", "Read"]) == "planning"


def test_agent_tool_is_delegation():
    from tokenq.proxy.classify import classify_turn
    assert classify_turn(tools_used=["Agent"]) == "delegation"


def test_search_tools_are_exploration():
    from tokenq.proxy.classify import classify_turn
    assert classify_turn(tools_used=["WebSearch"]) == "exploration"
    assert classify_turn(tools_used=["WebFetch"]) == "exploration"


def test_unknown_tool_only_is_general():
    from tokenq.proxy.classify import classify_turn
    assert classify_turn(tools_used=["WeirdTool"]) == "general"


# ---------- helpers (collect_bash_text / collect_edit_files) ----------

def test_collect_bash_text_concatenates_commands():
    from tokenq.proxy.classify import collect_bash_text
    body = {"messages": [
        _assistant_tool("u1", "Bash", command="ls"),
        _assistant_tool("u2", "Bash", command="pytest -x"),
    ]}
    assert "ls" in collect_bash_text(body)
    assert "pytest" in collect_bash_text(body)


def test_collect_bash_verbs_extracts_first_token():
    from tokenq.proxy.classify import collect_bash_verbs
    body = {"messages": [
        _assistant_tool("u1", "Bash", command="git push origin main"),
        _assistant_tool("u2", "Bash", command="npm install fastembed"),
        _assistant_tool("u3", "Bash", command="ls -la"),
    ]}
    assert collect_bash_verbs(body) == ["git", "npm", "ls"]


def test_collect_bash_verbs_strips_path_and_dedupes():
    from tokenq.proxy.classify import collect_bash_verbs
    body = {"messages": [
        _assistant_tool("u1", "Bash", command="/usr/bin/git status"),
        _assistant_tool("u2", "Bash", command="git diff"),  # already seen
    ]}
    assert collect_bash_verbs(body) == ["git"]


def test_collect_bash_verbs_strips_shell_prelude():
    """`sudo` / `time` etc. are prelude — the actual verb is what matters."""
    from tokenq.proxy.classify import collect_bash_verbs
    body = {"messages": [
        _assistant_tool("u1", "Bash", command="sudo apt install foo"),
        _assistant_tool("u2", "Bash", command="time pytest -x"),
        _assistant_tool("u3", "Bash", command="ENV=prod python deploy.py"),
    ]}
    assert collect_bash_verbs(body) == ["apt", "pytest", "python"]


def test_collect_bash_verbs_takes_first_segment_of_pipeline():
    """`git log | head -20` should attribute to git, not also to head — the
    pipeline tail is shell plumbing, not the user's intended verb."""
    from tokenq.proxy.classify import collect_bash_verbs
    body = {"messages": [
        _assistant_tool("u1", "Bash", command="git log --oneline | head -20"),
    ]}
    assert collect_bash_verbs(body) == ["git"]


def test_collect_bash_verbs_ignores_non_bash_tools():
    from tokenq.proxy.classify import collect_bash_verbs
    body = {"messages": [
        _assistant_tool("u1", "Read", file_path="/x.py"),
        _assistant_tool("u2", "Edit", file_path="/y.py"),
    ]}
    assert collect_bash_verbs(body) == []


def test_collect_edit_files_dedupes_and_orders():
    from tokenq.proxy.classify import collect_edit_files
    body = {"messages": [
        _assistant_tool("u1", "Edit", file_path="/a.py"),
        _assistant_tool("u2", "Write", path="/b.py"),
        _assistant_tool("u3", "Edit", file_path="/a.py"),  # dup, skipped
    ]}
    assert collect_edit_files(body) == ["/a.py", "/b.py"]


def test_latest_user_text_walks_backwards():
    from tokenq.proxy.classify import latest_user_text
    body = {"messages": [
        _user_msg("first message"),
        {"role": "assistant", "content": "reply"},
        _user_msg("most recent message"),
    ]}
    assert latest_user_text(body) == "most recent message"


# ---------- integration through observe.extract() ----------

def test_extract_includes_activity_label():
    from tokenq.proxy.observe import extract
    body = {
        "system": "Primary working directory: /repo/widgets",
        "messages": [
            _user_msg("fix the failing test in payment.py"),
            _assistant_tool("u1", "Bash", command="pytest tests/test_payment.py"),
        ],
    }
    obs = extract(body)
    assert obs["activity"] == "testing"
    assert obs["edit_files"] == ""  # nothing edited yet


def test_extract_records_edited_files():
    import json as _json
    from tokenq.proxy.observe import extract
    body = {
        "messages": [
            _user_msg("fix the off-by-one"),
            _assistant_tool("u1", "Edit", file_path="/repo/auth.py"),
        ],
    }
    obs = extract(body)
    assert obs["activity"] == "debugging"
    assert _json.loads(obs["edit_files"]) == ["/repo/auth.py"]


def test_extract_records_bash_verbs():
    import json as _json
    from tokenq.proxy.observe import extract
    body = {
        "messages": [
            _user_msg("commit the changes"),
            _assistant_tool("u1", "Bash", command="git commit -m wip"),
            _assistant_tool("u2", "Bash", command="git push"),
        ],
    }
    obs = extract(body)
    assert _json.loads(obs["bash_verbs"]) == ["git"]
