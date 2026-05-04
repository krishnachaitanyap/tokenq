"""Tests for proxy.observe — extracts session/project/tool dimensions from
the Anthropic Messages request body, used to power the codeburn-style
dashboard panels (sessions, by_project, by_tool)."""
from __future__ import annotations

import json


def test_session_id_is_stable_across_identical_inputs():
    from tokenq.proxy.observe import session_id
    body = {
        "system": "you are helpful",
        "messages": [{"role": "user", "content": "hi"}],
    }
    assert session_id(body) == session_id(body)


def test_session_id_changes_with_first_user_message():
    from tokenq.proxy.observe import session_id
    a = session_id({"system": "x", "messages": [{"role": "user", "content": "a"}]})
    b = session_id({"system": "x", "messages": [{"role": "user", "content": "b"}]})
    assert a != b


def test_session_id_unaffected_by_later_messages():
    """Adding follow-up turns must not change the session id — that's the
    whole point: we want one id per conversation."""
    from tokenq.proxy.observe import session_id
    base = {"system": "x", "messages": [{"role": "user", "content": "first"}]}
    extended = {
        "system": "x",
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "follow up"},
        ],
    }
    assert session_id(base) == session_id(extended)


def test_project_label_extracts_claude_code_cwd():
    from tokenq.proxy.observe import project_label
    body = {
        "system": (
            "Some preamble.\n"
            "Primary working directory: /Users/x/Documents/AI-casestudies/tokenq\n"
            "More instructions."
        ),
        "messages": [],
    }
    assert project_label(body) == "tokenq"


def test_project_label_handles_list_form_system():
    from tokenq.proxy.observe import project_label
    body = {
        "system": [
            {"type": "text", "text": "first block"},
            {"type": "text", "text": "Working directory: /home/foo/bar"},
        ],
        "messages": [],
    }
    assert project_label(body) == "bar"


def test_project_label_returns_empty_when_missing():
    from tokenq.proxy.observe import project_label
    assert project_label({"system": "no cwd here", "messages": []}) == ""
    assert project_label({"messages": []}) == ""


def test_tools_used_dedupes_and_preserves_order():
    from tokenq.proxy.observe import tools_used
    body = {
        "messages": [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "u1", "name": "Read", "input": {}},
                {"type": "tool_use", "id": "u2", "name": "Edit", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "u1", "content": "x"},
            ]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "u3", "name": "Read", "input": {}},
                {"type": "tool_use", "id": "u4", "name": "Bash", "input": {}},
            ]},
        ],
    }
    assert tools_used(body) == ["Read", "Edit", "Bash"]


def test_tools_used_empty_when_no_tool_use():
    from tokenq.proxy.observe import tools_used
    assert tools_used({"messages": [{"role": "user", "content": "hello"}]}) == []


def test_turn_index_counts_user_messages():
    from tokenq.proxy.observe import turn_index
    body = {"messages": [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "x"},
        {"role": "user", "content": "b"},
        {"role": "assistant", "content": "y"},
        {"role": "user", "content": "c"},
    ]}
    assert turn_index(body) == 3


def test_extract_returns_all_dimensions():
    """extract() bundles the dimensions log_request consumes — keep this test
    in sync as fields are added so the SQL schema and the writer don't drift."""
    from tokenq.proxy.observe import extract
    body = {
        "system": "Primary working directory: /repo/widgetshop",
        "messages": [
            {"role": "user", "content": "find the bug"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "u1", "name": "Read", "input": {}},
            ]},
        ],
    }
    obs = extract(body)
    assert set(obs) == {
        "session_id", "project", "tools_used", "turn_index",
        "activity", "edit_files", "bash_verbs",
    }
    assert obs["project"] == "widgetshop"
    assert json.loads(obs["tools_used"]) == ["Read"]
    assert obs["turn_index"] == 1
    assert len(obs["session_id"]) == 16
    # Bug-keyword + Read-only → debugging? No — refine_coding only fires when
    # an Edit happened. Just Read with debug intent stays at exploration.
    assert obs["activity"] in ("exploration", "debugging")
    assert obs["edit_files"] == ""


def test_extract_resilient_to_malformed_input():
    from tokenq.proxy.observe import extract
    # No exceptions; sensible defaults.
    obs = extract({})
    assert obs["project"] == ""
    assert obs["tools_used"] == ""
    assert obs["turn_index"] == 0
    assert isinstance(obs["session_id"], str)
    assert obs["activity"] in (
        "coding", "debugging", "feature_dev", "refactoring", "testing",
        "exploration", "planning", "delegation", "git_ops", "build_deploy",
        "brainstorming", "conversation", "general",
    )
