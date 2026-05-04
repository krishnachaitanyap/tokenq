"""Tests for SkillLoader."""
from __future__ import annotations


def _skill_list(n: int) -> str:
    """Build a system prompt with N skills, deterministic descriptions."""
    header = "The following skills are available for use with the Skill tool:\n\n"
    rows = [
        f"- skill-{i:02d}: Use this skill when the user wants to {topic}."
        for i, topic in enumerate([
            "configure settings or hooks for the harness",
            "rebind keyboard shortcuts and chord bindings",
            "review code for reuse, quality, and efficiency",
            "scan logs and reduce permission prompts",
            "loop a prompt on a recurring schedule",
            "schedule a remote agent on a cron",
            "build claude api anthropic sdk apps",
            "operate the tokenq local proxy",
            "initialize a new claude.md doc file",
            "review a github pull request branch",
            "complete a security review of pending changes",
            "render diagrams from mermaid sources",
        ][:n])
    ]
    return header + "\n".join(rows)


async def test_trims_long_skill_list_to_top_k():
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.skills import SkillLoader

    body = {
        "system": _skill_list(12),
        "messages": [{"role": "user", "content": "review my pull request"}],
    }
    req = PipelineRequest(body=body, headers={})
    await SkillLoader(top_k=5, min_list=8).run(req)

    sys_text = body["system"]
    # 5 kept + the placeholder row.
    skill_lines = [ln for ln in sys_text.split("\n") if ln.startswith("- skill-")]
    assert len(skill_lines) == 5
    assert "additional skills hidden by tokenq" in sys_text
    assert req.metadata["saved_by_skills"] > 0


async def test_keyword_overlap_chooses_relevant_skills():
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.skills import SkillLoader

    body = {
        "system": _skill_list(12),
        "messages": [{"role": "user", "content": "review my pull request"}],
    }
    req = PipelineRequest(body=body, headers={})
    await SkillLoader(top_k=3, min_list=8).run(req)

    sys_text = body["system"]
    # "review" appears in skills 02, 09, 10 — all three should survive.
    assert "skill-02" in sys_text
    assert "skill-09" in sys_text
    assert "skill-10" in sys_text


async def test_explicit_slash_invocation_always_kept():
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.skills import SkillLoader

    body = {
        "system": _skill_list(12),
        # The user invokes a skill that has zero token overlap with
        # mainstream English query terms.
        "messages": [{"role": "user", "content": "/skill-11 please"}],
    }
    req = PipelineRequest(body=body, headers={})
    await SkillLoader(top_k=2, min_list=8).run(req)

    assert "skill-11" in body["system"]


async def test_short_lists_left_alone():
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.skills import SkillLoader

    original = _skill_list(4)
    body = {
        "system": original,
        "messages": [{"role": "user", "content": "anything"}],
    }
    req = PipelineRequest(body=body, headers={})
    await SkillLoader(top_k=5, min_list=8).run(req)

    assert body["system"] == original
    assert req.metadata.get("saved_by_skills", 0) == 0


async def test_no_skill_list_is_noop():
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.skills import SkillLoader

    body = {
        "system": "You are a helpful assistant. Be concise.",
        "messages": [{"role": "user", "content": "hi"}],
    }
    req = PipelineRequest(body=body, headers={})
    await SkillLoader().run(req)

    assert body["system"] == "You are a helpful assistant. Be concise."
    assert req.metadata.get("saved_by_skills", 0) == 0


async def test_handles_list_form_system_prompt():
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.skills import SkillLoader

    body = {
        "system": [
            {"type": "text", "text": "You are Claude."},
            {"type": "text", "text": _skill_list(12)},
        ],
        "messages": [{"role": "user", "content": "review code please"}],
    }
    req = PipelineRequest(body=body, headers={})
    await SkillLoader(top_k=4, min_list=8).run(req)

    block_text = body["system"][1]["text"]
    skill_lines = [ln for ln in block_text.split("\n") if ln.startswith("- skill-")]
    assert len(skill_lines) == 4
    assert "additional skills hidden" in block_text


async def test_multi_line_skill_descriptions_parsed_as_continuations():
    """Real harnesses (e.g. Claude Code) emit skills whose description spans
    multiple lines (TRIGGER:/SKIP: continuation lines). The parser must treat
    those as continuations of the previous skill, not as block terminators."""
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.skills import SkillLoader

    system = (
        "The following skills are available for use with the Skill tool:\n\n"
        "- alpha: First skill, single line.\n"
        "- beta: Second skill spans multiple lines.\n"
        "TRIGGER when: code imports beta library or asks about beta features.\n"
        "SKIP: anything unrelated to beta.\n"
        "- gamma: Third skill, also multi-line.\n"
        "  Continued description for gamma here.\n"
        "- delta: Fourth single-line skill.\n"
        "- epsilon: Fifth single-line skill.\n"
        "- zeta: Sixth single-line skill.\n"
    )
    body = {
        "system": system,
        "messages": [{"role": "user", "content": "do something with alpha"}],
    }
    req = PipelineRequest(body=body, headers={})
    await SkillLoader(top_k=2, min_list=4).run(req)

    sys_text = body["system"]
    # Parser must have seen all 6 skills (not bailed at TRIGGER line).
    assert "additional skills hidden" in sys_text
    # Alpha should be kept (matched on "alpha" in query).
    assert "- alpha:" in sys_text
    # When beta is kept, its continuation lines must travel with it.
    if "- beta:" in sys_text:
        assert "TRIGGER when:" in sys_text
        assert "SKIP:" in sys_text
    assert req.metadata["saved_by_skills"] > 0


async def test_block_ends_at_markdown_heading():
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.skills import SkillLoader

    system = (
        "- a01: first skill description.\n"
        "- a02: second skill description.\n"
        "- a03: third skill description.\n"
        "- a04: fourth skill description.\n"
        "\n"
        "## Auto Mode Active\n"
        "Some narrative paragraph after the list that should not be treated\n"
        "as a continuation of skill a04.\n"
    )
    body = {"system": system, "messages": [{"role": "user", "content": "x"}]}
    req = PipelineRequest(body=body, headers={})
    # min_list=3 so the trim engages; top_k=2 so something is dropped.
    await SkillLoader(top_k=2, min_list=3).run(req)
    # Heading and trailing narrative must still be present unchanged.
    assert "## Auto Mode Active" in body["system"]
    assert "narrative paragraph" in body["system"]


async def test_uses_last_user_message_not_assistant():
    """The query for scoring should come from the user, not the assistant."""
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.skills import SkillLoader

    body = {
        "system": _skill_list(12),
        "messages": [
            {"role": "user", "content": "review my pull request"},
            {"role": "assistant", "content": "noise about diagrams and configuration"},
        ],
    }
    req = PipelineRequest(body=body, headers={})
    await SkillLoader(top_k=3, min_list=8).run(req)

    sys_text = body["system"]
    # Should still pick up review-related skills, not configuration ones.
    assert "skill-02" in sys_text or "skill-09" in sys_text or "skill-10" in sys_text
