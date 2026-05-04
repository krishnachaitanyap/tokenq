"""Tests for BanditRouter."""
from __future__ import annotations

import random


async def test_default_is_shadow_and_does_not_mutate(tmp_home, monkeypatch):
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.bandit import MODE_SHADOW, BanditRouter
    from tokenq.storage import init_db

    monkeypatch.delenv("TOKENQ_BANDIT_MODE", raising=False)
    monkeypatch.delenv("TOKENQ_BANDIT_ENABLED", raising=False)
    await init_db()
    stage = BanditRouter()

    assert stage.mode == MODE_SHADOW

    body = {"model": "claude-opus-4-7", "messages": [{"role": "user", "content": "hi"}]}
    req = PipelineRequest(body=body, headers={})
    out = await stage.run(req)

    # Shadow never mutates the request body — only logs counterfactuals.
    assert out is req
    assert body["model"] == "claude-opus-4-7"
    assert "bandit_arm" not in req.metadata
    # But it does record what the router *would* have picked.
    assert "bandit_shadow_arm" in req.metadata


async def test_explicit_off_disables_everything(tmp_home, monkeypatch):
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.bandit import MODE_OFF, BanditRouter
    from tokenq.storage import init_db

    monkeypatch.setenv("TOKENQ_BANDIT_MODE", "off")
    await init_db()
    stage = BanditRouter()
    assert stage.mode == MODE_OFF

    body = {"model": "claude-opus-4-7", "messages": [{"role": "user", "content": "hi"}]}
    req = PipelineRequest(body=body, headers={})
    await stage.run(req)
    assert "bandit_shadow_arm" not in req.metadata
    assert "bandit_arm" not in req.metadata


async def test_haiku_request_skips_routing(tmp_home):
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.bandit import BanditRouter
    from tokenq.storage import init_db

    await init_db()
    stage = BanditRouter(enabled=True)

    body = {"model": "claude-haiku-4-5", "messages": [{"role": "user", "content": "hi"}]}
    req = PipelineRequest(body=body, headers={})
    await stage.run(req)

    # Haiku is the floor — nothing to route.
    assert body["model"] == "claude-haiku-4-5"
    assert "bandit_arm" not in req.metadata


async def test_opus_request_routes_within_tier(tmp_home):
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.bandit import BanditRouter
    from tokenq.storage import init_db

    await init_db()
    stage = BanditRouter(enabled=True)

    random.seed(0)
    body = {"model": "claude-opus-4-7", "messages": [{"role": "user", "content": "hi"}]}
    req = PipelineRequest(body=body, headers={})
    await stage.run(req)

    assert body["model"] in {
        "claude-haiku-4-5",
        "claude-sonnet-4-6",
        "claude-opus-4-7",
    }
    assert req.metadata["bandit_arm"] == body["model"]
    assert req.metadata["bandit_original_model"] == "claude-opus-4-7"
    assert "bandit_bucket" in req.metadata


async def test_sonnet_request_never_routes_to_opus(tmp_home):
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.bandit import BanditRouter
    from tokenq.storage import init_db

    await init_db()
    stage = BanditRouter(enabled=True)

    # Many trials — opus must never appear because it's above the ceiling.
    seen = set()
    for i in range(50):
        random.seed(i)
        body = {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "x"}]}
        req = PipelineRequest(body=body, headers={})
        await stage.run(req)
        seen.add(body["model"])

    assert "claude-opus-4-7" not in seen
    assert seen.issubset({"claude-haiku-4-5", "claude-sonnet-4-6"})


def test_reward_function_branches():
    from tokenq.pipeline.bandit import reward_from

    # end_turn with zero cost → near 1.0
    r_good = reward_from("end_turn", None, 0.0)
    # max_tokens → low
    r_trunc = reward_from("max_tokens", None, 0.0)
    # error → zero
    r_err = reward_from(None, "boom", 0.0)
    # high cost dampens reward
    r_costly = reward_from("end_turn", None, 0.05)

    assert r_good > 0.9
    assert r_trunc < 0.5
    assert r_err == 0.0
    assert r_costly < r_good


async def test_after_updates_state_with_reward(tmp_home):
    import aiosqlite

    from tokenq.config import DB_PATH
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.bandit import BanditRouter, _key
    from tokenq.storage import init_db

    await init_db()
    random.seed(1)
    stage = BanditRouter(enabled=True)

    body = {"model": "claude-opus-4-7", "messages": [{"role": "user", "content": "hi"}]}
    req = PipelineRequest(body=body, headers={})
    await stage.run(req)
    arm = req.metadata["bandit_arm"]
    bucket = req.metadata["bandit_bucket"]

    # Simulate a clean end_turn response.
    response = {
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    await stage.after(req, response)

    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "SELECT params FROM bandit_state WHERE arm = ?", (_key(bucket, arm),)
        )
        row = await cur.fetchone()
    assert row is not None
    import json

    data = json.loads(row[0])
    # alpha should have grown above the prior of 1.0.
    assert data["alpha"] > 1.0
    assert data["n"] == 1
    assert data["total_reward"] > 0.0


async def test_context_bucket_separates_state(tmp_home):
    from tokenq.pipeline.bandit import context_bucket

    small = {"messages": [{"role": "user", "content": "hi"}]}
    large = {"messages": [{"role": "user", "content": "x" * 10000}]}
    with_tools = {"messages": [{"role": "user", "content": "hi"}], "tools": [{"name": "t"}]}

    assert context_bucket(small) != context_bucket(large)
    assert context_bucket(small) != context_bucket(with_tools)


async def test_thompson_prefers_arm_with_higher_alpha(tmp_home):
    """After enough successful pulls of one arm, it should dominate selections."""
    import aiosqlite

    from tokenq.config import DB_PATH
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.bandit import BanditRouter, _key, context_bucket
    from tokenq.storage import init_db

    await init_db()
    stage = BanditRouter(enabled=True)

    body = {"model": "claude-opus-4-7", "messages": [{"role": "user", "content": "x"}]}
    bucket = context_bucket(body)

    # Hand-seed a strong, graduated belief that haiku is best. The graduated
    # flag is required under live-mode routing — strong Beta priors alone don't
    # qualify an arm; it has to have proven itself with live observations.
    import json
    import time

    async with aiosqlite.connect(DB_PATH) as db:
        for arm, alpha, beta, graduated in [
            ("claude-haiku-4-5", 50.0, 1.0, True),
            ("claude-sonnet-4-6", 1.0, 50.0, True),
            ("claude-opus-4-7", 1.0, 50.0, False),  # opus is the original tier, always eligible
        ]:
            params = json.dumps(
                {
                    "alpha": alpha,
                    "beta": beta,
                    "n": 51,
                    "live_n": 51,
                    "total_reward": alpha - 1.0,
                    "graduated": graduated,
                    "recent_rewards": [],
                }
            )
            await db.execute(
                "INSERT OR REPLACE INTO bandit_state (arm, params, updated_at) VALUES (?, ?, ?)",
                (_key(bucket, arm), params.encode("utf-8"), time.time()),
            )
        await db.commit()

    random.seed(42)
    picks = {"haiku": 0, "sonnet": 0, "opus": 0}
    for _ in range(100):
        b = dict(body)
        req = PipelineRequest(body=b, headers={})
        await stage.run(req)
        if "haiku" in b["model"]:
            picks["haiku"] += 1
        elif "sonnet" in b["model"]:
            picks["sonnet"] += 1
        else:
            picks["opus"] += 1

    assert picks["haiku"] > 80, picks


async def test_after_stream_updates_state(tmp_home):
    import aiosqlite

    from tokenq.config import DB_PATH
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.bandit import BanditRouter, _key
    from tokenq.storage import init_db

    await init_db()
    random.seed(2)
    stage = BanditRouter(enabled=True)

    body = {"model": "claude-opus-4-7", "messages": [{"role": "user", "content": "hi"}]}
    req = PipelineRequest(body=body, headers={})
    await stage.run(req)

    captured = {
        "input_tokens": 10,
        "output_tokens": 5,
        "stop_reason": "end_turn",
        "error": None,
    }
    await stage.after_stream(req, b"", captured)

    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "SELECT params FROM bandit_state WHERE arm = ?",
            (_key(req.metadata["bandit_bucket"], req.metadata["bandit_arm"]),),
        )
        row = await cur.fetchone()
    assert row is not None
    import json

    data = json.loads(row[0])
    assert data["n"] == 1
    assert data["alpha"] > 1.0


# ---------------------------------------------------------------------------
# Safe-by-default: shadow mode, graduation, demotion, backward compat.
# ---------------------------------------------------------------------------


async def test_legacy_enabled_env_maps_to_live_mode(tmp_home, monkeypatch):
    from tokenq.pipeline.bandit import MODE_LIVE, BanditRouter

    monkeypatch.setenv("TOKENQ_BANDIT_ENABLED", "1")
    monkeypatch.delenv("TOKENQ_BANDIT_MODE", raising=False)
    stage = BanditRouter()
    assert stage.mode == MODE_LIVE
    assert stage.enabled is True


async def test_explicit_mode_env_overrides_legacy(tmp_home, monkeypatch):
    from tokenq.pipeline.bandit import MODE_SHADOW, BanditRouter

    monkeypatch.setenv("TOKENQ_BANDIT_ENABLED", "1")
    monkeypatch.setenv("TOKENQ_BANDIT_MODE", "shadow")
    stage = BanditRouter()
    assert stage.mode == MODE_SHADOW


async def test_shadow_mode_does_not_mutate_body(tmp_home):
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.bandit import MODE_SHADOW, BanditRouter
    from tokenq.storage import init_db

    await init_db()
    stage = BanditRouter(mode=MODE_SHADOW)
    body = {"model": "claude-opus-4-7", "messages": [{"role": "user", "content": "x"}]}
    req = PipelineRequest(body=body, headers={})
    await stage.run(req)

    assert body["model"] == "claude-opus-4-7"
    assert "bandit_arm" not in req.metadata
    assert "bandit_shadow_arm" in req.metadata
    assert req.metadata["bandit_shadow_arm"] in (
        "claude-haiku-4-5",
        "claude-sonnet-4-6",
        "claude-opus-4-7",
    )


async def test_shadow_mode_logs_decision_with_honest_cost_delta(tmp_home):
    import aiosqlite

    from tokenq.config import DB_PATH
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.bandit import MODE_SHADOW, BanditRouter
    from tokenq.storage import init_db

    await init_db()
    stage = BanditRouter(mode=MODE_SHADOW)
    body = {"model": "claude-opus-4-7", "messages": [{"role": "user", "content": "hi"}]}
    req = PipelineRequest(body=body, headers={})
    # Force shadow recommendation to haiku so the cost delta is non-trivial.
    await stage.run(req)
    req.metadata["bandit_shadow_arm"] = "claude-haiku-4-5"

    response = {
        "usage": {"input_tokens": 1000, "output_tokens": 200},
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "ok"}],
    }
    await stage.after(req, response)

    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "SELECT original_arm, recommended_arm, est_cost_full_usd, "
            "est_cost_routed_usd, est_cost_saved_usd, est_success "
            "FROM bandit_shadow_decisions"
        )
        rows = await cur.fetchall()
    assert len(rows) == 1
    orig, rec, full, routed, saved, est_succ = rows[0]
    assert orig == "claude-opus-4-7"
    assert rec == "claude-haiku-4-5"
    assert full > routed > 0
    assert abs(saved - (full - routed)) < 1e-9
    assert est_succ == 1.0  # short end_turn → full credit


async def test_shadow_mode_does_not_update_bandit_state(tmp_home):
    import aiosqlite

    from tokenq.config import DB_PATH
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.bandit import MODE_SHADOW, BanditRouter
    from tokenq.storage import init_db

    await init_db()
    stage = BanditRouter(mode=MODE_SHADOW)
    body = {"model": "claude-opus-4-7", "messages": [{"role": "user", "content": "hi"}]}
    req = PipelineRequest(body=body, headers={})
    await stage.run(req)

    response = {
        "usage": {"input_tokens": 100, "output_tokens": 50},
        "stop_reason": "end_turn",
        "content": [],
    }
    await stage.after(req, response)

    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("SELECT COUNT(*) FROM bandit_state")
        (count,) = await cur.fetchone()
    assert count == 0


async def test_live_mode_only_routes_to_graduated_or_original(tmp_home):
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.bandit import MODE_LIVE, BanditRouter
    from tokenq.storage import init_db

    await init_db()
    # explore_rate=0 disables ε-exploration so the gate is the only filter.
    stage = BanditRouter(mode=MODE_LIVE, explore_rate=0.0)

    random.seed(7)
    picks: dict[str, int] = {}
    body = {"model": "claude-opus-4-7", "messages": [{"role": "user", "content": "x"}]}
    for _ in range(50):
        b = dict(body)
        req = PipelineRequest(body=b, headers={})
        await stage.run(req)
        picks[b["model"]] = picks.get(b["model"], 0) + 1

    # No arms graduated, no exploration → must always pick original opus.
    assert picks == {"claude-opus-4-7": 50}, picks


async def test_live_mode_explores_ungraduated_arms_under_epsilon(tmp_home):
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.bandit import MODE_LIVE, BanditRouter
    from tokenq.storage import init_db

    await init_db()
    # explore_rate=1.0 → always include an ungraduated arm in the candidate set,
    # so we should occasionally pick something other than opus.
    stage = BanditRouter(mode=MODE_LIVE, explore_rate=1.0)

    random.seed(11)
    picks: dict[str, int] = {}
    body = {"model": "claude-opus-4-7", "messages": [{"role": "user", "content": "x"}]}
    for _ in range(100):
        b = dict(body)
        req = PipelineRequest(body=b, headers={})
        await stage.run(req)
        picks[b["model"]] = picks.get(b["model"], 0) + 1

    non_opus = sum(v for k, v in picks.items() if "opus" not in k)
    assert non_opus > 10, picks


async def test_arm_graduates_after_enough_successful_observations(tmp_home):
    import aiosqlite

    from tokenq.config import DB_PATH
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.bandit import (
        MODE_LIVE,
        ArmState,
        BanditRouter,
        _key,
        context_bucket,
    )
    from tokenq.storage import init_db

    await init_db()
    stage = BanditRouter(
        mode=MODE_LIVE, graduate_min_n=5, graduate_min_reward=0.5, explore_rate=0.0
    )

    body = {"model": "claude-opus-4-7", "messages": [{"role": "user", "content": "x"}]}
    bucket = context_bucket(body)
    arm = "claude-haiku-4-5"

    # Simulate 5 successful live observations on haiku.
    for _ in range(5):
        req = PipelineRequest(
            body=dict(body),
            headers={},
            metadata={
                "bandit_bucket": bucket,
                "bandit_arm": arm,
                "bandit_original_model": "claude-opus-4-7",
                "bandit_mode": MODE_LIVE,
            },
        )
        await stage._update(req, reward=0.9)

    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "SELECT params FROM bandit_state WHERE arm = ?", (_key(bucket, arm),)
        )
        row = await cur.fetchone()
    state = ArmState.from_blob(row[0])
    assert state.live_n == 5
    assert state.graduated is True


async def test_arm_demoted_when_rolling_reward_drops(tmp_home):
    import aiosqlite

    from tokenq.config import DB_PATH
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.bandit import (
        MODE_LIVE,
        ArmState,
        BanditRouter,
        _key,
        context_bucket,
    )
    from tokenq.storage import init_db

    await init_db()
    stage = BanditRouter(
        mode=MODE_LIVE,
        graduate_min_n=3,
        graduate_min_reward=0.5,
        demote_window=5,
        demote_reward=0.3,
        explore_rate=0.0,
    )

    body = {"model": "claude-opus-4-7", "messages": [{"role": "user", "content": "x"}]}
    bucket = context_bucket(body)
    arm = "claude-haiku-4-5"

    def mk_req():
        return PipelineRequest(
            body=dict(body),
            headers={},
            metadata={
                "bandit_bucket": bucket,
                "bandit_arm": arm,
                "bandit_original_model": "claude-opus-4-7",
                "bandit_mode": MODE_LIVE,
            },
        )

    # Graduate it with strong rewards.
    for _ in range(5):
        await stage._update(mk_req(), reward=0.9)

    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "SELECT params FROM bandit_state WHERE arm = ?", (_key(bucket, arm),)
        )
        row = await cur.fetchone()
    assert ArmState.from_blob(row[0]).graduated is True

    # Now bombard with bad rewards until the rolling window flips it.
    for _ in range(5):
        await stage._update(mk_req(), reward=0.0)

    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "SELECT params FROM bandit_state WHERE arm = ?", (_key(bucket, arm),)
        )
        row = await cur.fetchone()
    state = ArmState.from_blob(row[0])
    assert state.graduated is False
    # Window was cleared on demotion; subsequent updates rebuild it from scratch,
    # so it must be strictly smaller than the demote window.
    assert len(state.recent_rewards) < stage.demote_window


async def test_estimate_shadow_reward_is_conservative_for_long_outputs():
    from tokenq.pipeline.bandit import estimate_shadow_reward

    assert estimate_shadow_reward("end_turn", None, output_tokens=100) == 1.0
    # Long output → cheaper arm assumed riskier → only partial credit.
    assert estimate_shadow_reward("end_turn", None, output_tokens=5000) == 0.4
    # Tool use is risky for cheaper arms.
    assert estimate_shadow_reward("tool_use", None, output_tokens=100) == 0.6
    assert estimate_shadow_reward(None, None, output_tokens=100) == 0.0
    assert estimate_shadow_reward("error", None, output_tokens=100) == 0.0
    assert estimate_shadow_reward("end_turn", "boom", output_tokens=100) == 0.0


async def test_arm_state_backward_compat_treats_old_n_as_live_n(tmp_home):
    from tokenq.pipeline.bandit import ArmState

    legacy = b'{"alpha": 10.0, "beta": 2.0, "n": 11, "total_reward": 9.0}'
    s = ArmState.from_blob(legacy)
    assert s.alpha == 10.0
    assert s.beta == 2.0
    assert s.n == 11
    assert s.live_n == 11  # legacy n promoted to live_n
    assert s.graduated is False  # not promoted automatically; needs new evidence
    assert s.recent_rewards == []
    # New shadow fields default to zero on legacy state.
    assert s.shadow_n == 0
    assert s.shadow_total_success == 0.0
    assert s.promoted_from_shadow is False


async def test_auto_graduate_from_shadow_after_enough_evidence(tmp_home):
    """A shadow-rewarded arm with high success + meaningful cost win graduates."""
    from tokenq.pipeline import PipelineRequest
    from tokenq.pipeline.bandit import (
        MODE_SHADOW,
        BanditRouter,
        _load_state,
    )
    from tokenq.storage import init_db

    await init_db()
    # Low graduate threshold (n=5) so the test is fast.
    stage = BanditRouter(
        mode=MODE_SHADOW,
        graduate_min_n=5,
        graduate_min_reward=0.7,
    )
    bucket = "s|t=0|sys=0|img=0|t0=1"
    arm = "claude-haiku-4-5"
    for _ in range(6):
        await stage._record_shadow_observation(
            bucket=bucket,
            arm=arm,
            est_success=1.0,
            cost_full=0.10,   # original
            cost_routed=0.01, # 90% saving → easily over the 30% gate
        )
    state = (await _load_state(bucket, [arm]))[arm]
    assert state.graduated is True
    assert state.promoted_from_shadow is True
    assert state.shadow_n >= 5


async def test_no_auto_graduate_when_cost_savings_too_small(tmp_home):
    from tokenq.pipeline.bandit import (
        MODE_SHADOW,
        BanditRouter,
        _load_state,
    )
    from tokenq.storage import init_db

    await init_db()
    stage = BanditRouter(
        mode=MODE_SHADOW,
        graduate_min_n=5,
        graduate_min_reward=0.7,
    )
    bucket = "s|t=0|sys=0|img=0|t0=1"
    arm = "claude-sonnet-4-6"
    for _ in range(6):
        # Same cost in shadow vs. original → 0% savings, below the 30% gate.
        await stage._record_shadow_observation(
            bucket=bucket,
            arm=arm,
            est_success=1.0,
            cost_full=0.05,
            cost_routed=0.05,
        )
    state = (await _load_state(bucket, [arm]))[arm]
    assert state.graduated is False
    assert state.promoted_from_shadow is False


async def test_promoted_from_shadow_uses_tighter_demote_floor(tmp_home):
    from tokenq.pipeline.bandit import (
        DEFAULT_AUTO_DEMOTE_REWARD,
        DEFAULT_DEMOTE_REWARD,
        ArmState,
        BanditRouter,
    )

    stage = BanditRouter()
    promoted = ArmState(graduated=True, promoted_from_shadow=True)
    user_flipped = ArmState(graduated=True, promoted_from_shadow=False)

    assert stage._demote_floor_for(promoted) == DEFAULT_AUTO_DEMOTE_REWARD
    assert stage._demote_floor_for(user_flipped) == DEFAULT_DEMOTE_REWARD
    assert DEFAULT_AUTO_DEMOTE_REWARD > DEFAULT_DEMOTE_REWARD
