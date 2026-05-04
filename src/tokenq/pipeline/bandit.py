"""Thompson-sampling model router with shadow/live modes and per-arm graduation.

Three modes via TOKENQ_BANDIT_MODE:
  off    — no-op. Set explicitly to disable counterfactual logging.
  shadow — pick an arm but never mutate the request; log the would-be decision
           and an estimated counterfactual reward to bandit_shadow_decisions.
           Zero correctness risk: the user's chosen model is always called.
           This is the default — every install collects evidence on day 1
           without changing any request bytes.
  live   — route for real, but only to arms graduated for this (bucket, arm)
           cell. The user's original tier is always eligible. With probability
           TOKENQ_BANDIT_EXPLORE_RATE an ungraduated arm is included so it can
           earn graduation. Demotion fires when rolling reward drops.

Backward compat: legacy TOKENQ_BANDIT_ENABLED=1/true/yes/on → mode=live.

Maintains Beta(alpha, beta) parameters per (context_bucket, arm) in the
bandit_state table. After a live response, alpha/beta are updated using a
composite reward (stop_reason + cost penalty); shadow responses do not update
Beta (no real reward observed).

Never picks a model above the user's requested tier — never an upgrade. If the
user requested haiku, no routing happens.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any

import aiosqlite

from ..config import DB_PATH
from . import PipelineRequest, Stage

# Cheapest first. The user's chosen model is the ceiling — we never route up.
TIER_ORDER: list[tuple[str, str]] = [
    ("haiku", "claude-haiku-4-5"),
    ("sonnet", "claude-sonnet-4-6"),
    ("opus", "claude-opus-4-7"),
]

DEFAULT_REWARD_COST_CAP_USD = 0.05
DEFAULT_SUCCESS_WEIGHT = 0.7
DEFAULT_COST_WEIGHT = 0.3

MODE_OFF = "off"
MODE_SHADOW = "shadow"
MODE_LIVE = "live"
VALID_MODES = (MODE_OFF, MODE_SHADOW, MODE_LIVE)

DEFAULT_GRADUATE_MIN_N = 30
DEFAULT_GRADUATE_MIN_REWARD = 0.6
DEFAULT_DEMOTE_REWARD = 0.4
# Tighter floor for cells auto-promoted from shadow rewards. A user-flipped
# live mode is a deliberate choice; a shadow→live promotion was decided by
# tokenq, so we want to back off faster if quality drops.
DEFAULT_AUTO_DEMOTE_REWARD = 0.55
DEFAULT_DEMOTE_WINDOW = 20
DEFAULT_EXPLORE_RATE = 0.05
SHADOW_SHORT_OUTPUT_TOKENS = 1500
# Minimum fractional cost savings required to auto-graduate a shadow arm.
# Prevents promoting an arm whose cost is comparable to the original — there
# is real correctness risk on the routed call, so we require a real upside.
AUTO_GRADUATE_MIN_COST_SAVINGS = 0.30


def mode_from_env() -> str:
    """Resolve mode from env. Honors legacy TOKENQ_BANDIT_ENABLED for backward compat.

    Default is shadow: counterfactual decisions are logged on every request but
    the user's chosen model is always called. Set TOKENQ_BANDIT_MODE=off to
    disable entirely.
    """
    explicit = os.getenv("TOKENQ_BANDIT_MODE")
    if explicit:
        m = explicit.strip().lower()
        if m in VALID_MODES:
            return m
    if os.getenv("TOKENQ_BANDIT_ENABLED", "0") in ("1", "true", "yes", "on"):
        return MODE_LIVE
    return MODE_SHADOW


def is_enabled() -> bool:
    """Legacy: True iff bandit will do anything (shadow or live)."""
    return mode_from_env() != MODE_OFF


def _tier_of(model: str | None) -> int | None:
    if not model:
        return None
    for i, (name, _m) in enumerate(TIER_ORDER):
        if name in model:
            return i
    return None


def _msg_chars(messages: Any) -> int:
    if not isinstance(messages, list):
        return 0
    total = 0
    for m in messages:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, str):
            total += len(c)
        elif isinstance(c, list):
            for blk in c:
                if not isinstance(blk, dict):
                    continue
                if isinstance(blk.get("text"), str):
                    total += len(blk["text"])
                elif isinstance(blk.get("content"), str):
                    total += len(blk["content"])
    return total


def _has_images(messages: Any) -> bool:
    if not isinstance(messages, list):
        return False
    for m in messages:
        c = m.get("content") if isinstance(m, dict) else None
        if not isinstance(c, list):
            continue
        for blk in c:
            if isinstance(blk, dict) and blk.get("type") == "image":
                return True
    return False


def context_bucket(body: dict[str, Any]) -> str:
    chars = _msg_chars(body.get("messages"))
    if chars < 500:
        size = "s"
    elif chars < 2000:
        size = "m"
    elif chars < 8000:
        size = "l"
    else:
        size = "xl"
    has_tools = bool(body.get("tools"))
    has_system = bool(body.get("system"))
    has_imgs = _has_images(body.get("messages"))
    temp = body.get("temperature")
    temp_zero = temp is None or temp == 0
    return f"{size}|t={int(has_tools)}|sys={int(has_system)}|img={int(has_imgs)}|t0={int(temp_zero)}"


@dataclass
class ArmState:
    alpha: float = 1.0
    beta: float = 1.0
    n: int = 0
    total_reward: float = 0.0
    graduated: bool = False
    live_n: int = 0
    recent_rewards: list[float] = None  # type: ignore[assignment]
    shadow_n: int = 0
    shadow_total_success: float = 0.0
    shadow_total_cost_full_usd: float = 0.0
    shadow_total_cost_routed_usd: float = 0.0
    promoted_from_shadow: bool = False

    def __post_init__(self) -> None:
        if self.recent_rewards is None:
            self.recent_rewards = []

    def to_json(self) -> str:
        return json.dumps(
            {
                "alpha": self.alpha,
                "beta": self.beta,
                "n": self.n,
                "total_reward": self.total_reward,
                "graduated": self.graduated,
                "live_n": self.live_n,
                "recent_rewards": list(self.recent_rewards or []),
                "shadow_n": self.shadow_n,
                "shadow_total_success": self.shadow_total_success,
                "shadow_total_cost_full_usd": self.shadow_total_cost_full_usd,
                "shadow_total_cost_routed_usd": self.shadow_total_cost_routed_usd,
                "promoted_from_shadow": self.promoted_from_shadow,
            }
        )

    @classmethod
    def from_blob(cls, blob: bytes | str | None) -> "ArmState":
        if not blob:
            return cls()
        try:
            text = blob.decode("utf-8") if isinstance(blob, (bytes, bytearray)) else blob
            data = json.loads(text)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return cls()
        n = int(data.get("n", 0))
        # Legacy state (pre-shadow) had no live_n; treat all observations as live
        # so existing graduated-by-experience arms aren't reset to zero.
        live_n = int(data.get("live_n", n))
        recent = data.get("recent_rewards") or []
        if isinstance(recent, list):
            recent = [float(x) for x in recent]
        else:
            recent = []
        return cls(
            alpha=float(data.get("alpha", 1.0)),
            beta=float(data.get("beta", 1.0)),
            n=n,
            total_reward=float(data.get("total_reward", 0.0)),
            graduated=bool(data.get("graduated", False)),
            live_n=live_n,
            recent_rewards=recent,
            shadow_n=int(data.get("shadow_n", 0)),
            shadow_total_success=float(data.get("shadow_total_success", 0.0)),
            shadow_total_cost_full_usd=float(
                data.get("shadow_total_cost_full_usd", 0.0)
            ),
            shadow_total_cost_routed_usd=float(
                data.get("shadow_total_cost_routed_usd", 0.0)
            ),
            promoted_from_shadow=bool(data.get("promoted_from_shadow", False)),
        )

    def mean_shadow_success(self) -> float:
        if self.shadow_n <= 0:
            return 0.0
        return self.shadow_total_success / self.shadow_n

    def shadow_cost_savings_ratio(self) -> float:
        if self.shadow_total_cost_full_usd <= 0:
            return 0.0
        delta = self.shadow_total_cost_full_usd - self.shadow_total_cost_routed_usd
        return delta / self.shadow_total_cost_full_usd

    def mean_reward(self) -> float:
        if self.live_n <= 0:
            return 0.0
        return self.total_reward / self.live_n

    def rolling_mean(self) -> float:
        if not self.recent_rewards:
            return 0.0
        return sum(self.recent_rewards) / len(self.recent_rewards)


def _key(bucket: str, arm: str) -> str:
    return f"{bucket}::{arm}"


async def _load_state(bucket: str, arms: list[str]) -> dict[str, ArmState]:
    out: dict[str, ArmState] = {a: ArmState() for a in arms}
    keys = [_key(bucket, a) for a in arms]
    placeholders = ",".join("?" * len(keys))
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            f"SELECT arm, params FROM bandit_state WHERE arm IN ({placeholders})",
            keys,
        )
        rows = await cur.fetchall()
    for arm_key, blob in rows:
        _, _, arm_name = arm_key.partition("::")
        if arm_name in out:
            out[arm_name] = ArmState.from_blob(blob)
    return out


async def _save_state(bucket: str, arm: str, state: ArmState) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO bandit_state (arm, params, updated_at) VALUES (?, ?, ?)",
            (_key(bucket, arm), state.to_json().encode("utf-8"), time.time()),
        )
        await db.commit()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def is_graduated_arm(state: ArmState, min_n: int, min_reward: float) -> bool:
    if state.graduated:
        return True
    return state.live_n >= min_n and state.mean_reward() >= min_reward


def should_demote(state: ArmState, demote_window: int, demote_floor: float) -> bool:
    if not state.graduated:
        return False
    if len(state.recent_rewards) < demote_window:
        return False
    return state.rolling_mean() < demote_floor


def estimate_shadow_reward(
    stop_reason: str | None,
    error: str | bool | None,
    output_tokens: int,
    has_tool_use: bool = False,
) -> float:
    """Heuristic counterfactual: would a cheaper arm have succeeded on this turn?

    Conservative: only credit short, single-shot turns. Tool use is risky for
    cheaper arms (correctness), so down-weight even when it succeeded.
    """
    if error or stop_reason in (None, "error"):
        return 0.0
    if stop_reason == "max_tokens":
        return 0.1
    if output_tokens > SHADOW_SHORT_OUTPUT_TOKENS:
        return 0.4  # long outputs are riskier for cheaper arms
    if stop_reason == "tool_use" or has_tool_use:
        return 0.6
    if stop_reason == "end_turn":
        return 1.0
    return 0.5


async def _log_shadow_decision(
    *,
    bucket: str,
    original_arm: str,
    recommended_arm: str,
    est_success: float,
    est_cost_full_usd: float,
    est_cost_routed_usd: float,
    stop_reason: str | None,
) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO bandit_shadow_decisions (
                ts, bucket, original_arm, recommended_arm,
                est_success, est_cost_full_usd, est_cost_routed_usd,
                est_cost_saved_usd, stop_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                time.time(),
                bucket,
                original_arm,
                recommended_arm,
                est_success,
                est_cost_full_usd,
                est_cost_routed_usd,
                max(0.0, est_cost_full_usd - est_cost_routed_usd),
                stop_reason,
            ),
        )
        await db.commit()


def reward_from(
    stop_reason: str | None,
    error: str | bool | None,
    cost_usd: float,
    cost_cap: float = DEFAULT_REWARD_COST_CAP_USD,
    success_w: float = DEFAULT_SUCCESS_WEIGHT,
    cost_w: float = DEFAULT_COST_WEIGHT,
) -> float:
    # Hard errors short-circuit to zero — the arm failed to deliver, so it
    # should not get credit for "failing cheaply."
    if error or stop_reason in (None, "error"):
        return 0.0
    if stop_reason == "end_turn":
        success = 1.0
    elif stop_reason == "tool_use":
        success = 0.9
    elif stop_reason == "max_tokens":
        success = 0.2
    else:
        success = 0.5
    cost_norm = min(max(cost_usd, 0.0) / max(cost_cap, 1e-9), 1.0)
    cost_score = 1.0 - cost_norm
    return max(0.0, min(1.0, success_w * success + cost_w * cost_score))


class BanditRouter(Stage):
    name = "bandit"

    def __init__(
        self,
        enabled: bool | None = None,
        mode: str | None = None,
        cost_cap: float = DEFAULT_REWARD_COST_CAP_USD,
        graduate_min_n: int | None = None,
        graduate_min_reward: float | None = None,
        demote_reward: float | None = None,
        auto_demote_reward: float | None = None,
        demote_window: int | None = None,
        explore_rate: float | None = None,
    ) -> None:
        # Backward compat: enabled=True maps to mode=live, enabled=False to off.
        if mode is not None:
            self._mode_override: str | None = mode
        elif enabled is True:
            self._mode_override = MODE_LIVE
        elif enabled is False:
            self._mode_override = MODE_OFF
        else:
            self._mode_override = None
        self.cost_cap = cost_cap
        self._graduate_min_n = graduate_min_n
        self._graduate_min_reward = graduate_min_reward
        self._demote_reward = demote_reward
        self._auto_demote_reward = auto_demote_reward
        self._demote_window = demote_window
        self._explore_rate = explore_rate

    @property
    def mode(self) -> str:
        return self._mode_override if self._mode_override is not None else mode_from_env()

    @property
    def enabled(self) -> bool:
        """Legacy alias: True iff mode is shadow or live."""
        return self.mode != MODE_OFF

    @property
    def graduate_min_n(self) -> int:
        return self._graduate_min_n if self._graduate_min_n is not None else _env_int(
            "TOKENQ_BANDIT_GRADUATE_MIN_N", DEFAULT_GRADUATE_MIN_N
        )

    @property
    def graduate_min_reward(self) -> float:
        return self._graduate_min_reward if self._graduate_min_reward is not None else _env_float(
            "TOKENQ_BANDIT_GRADUATE_MIN_REWARD", DEFAULT_GRADUATE_MIN_REWARD
        )

    @property
    def demote_reward(self) -> float:
        return self._demote_reward if self._demote_reward is not None else _env_float(
            "TOKENQ_BANDIT_DEMOTE_REWARD", DEFAULT_DEMOTE_REWARD
        )

    @property
    def auto_demote_reward(self) -> float:
        """Tighter floor for cells auto-promoted from shadow data."""
        return self._auto_demote_reward if self._auto_demote_reward is not None else _env_float(
            "TOKENQ_BANDIT_AUTO_DEMOTE_FLOOR", DEFAULT_AUTO_DEMOTE_REWARD
        )

    def _demote_floor_for(self, state: ArmState) -> float:
        return self.auto_demote_reward if state.promoted_from_shadow else self.demote_reward

    @property
    def demote_window(self) -> int:
        return self._demote_window if self._demote_window is not None else _env_int(
            "TOKENQ_BANDIT_DEMOTE_WINDOW", DEFAULT_DEMOTE_WINDOW
        )

    @property
    def explore_rate(self) -> float:
        return self._explore_rate if self._explore_rate is not None else _env_float(
            "TOKENQ_BANDIT_EXPLORE_RATE", DEFAULT_EXPLORE_RATE
        )

    def _select_arm(
        self, arms: list[str], original: str, state_by_arm: dict[str, ArmState], live_gate: bool
    ) -> str:
        """Sample Thompson over candidate arms.

        live_gate=True → restrict to (original tier) + graduated arms; with prob
        explore_rate include one ungraduated arm for exploration so it can earn
        graduation. live_gate=False → all arms eligible (used by shadow mode,
        which logs only and never mutates).
        """
        if live_gate:
            min_n = self.graduate_min_n
            min_r = self.graduate_min_reward
            candidates = [
                a
                for a in arms
                if a == original or is_graduated_arm(state_by_arm[a], min_n, min_r)
            ]
            ungraduated = [a for a in arms if a not in candidates]
            if ungraduated and random.random() < self.explore_rate:
                candidates.append(random.choice(ungraduated))
            if not candidates:
                candidates = [original]
        else:
            candidates = list(arms)

        best_arm = candidates[0]
        best_theta = -1.0
        for arm in candidates:
            s = state_by_arm[arm]
            theta = random.betavariate(s.alpha, s.beta)
            if theta > best_theta:
                best_theta = theta
                best_arm = arm
        return best_arm

    async def run(self, req: PipelineRequest):
        mode = self.mode
        if mode == MODE_OFF:
            return req
        body = req.body
        if not isinstance(body, dict):
            return req
        original = body.get("model")
        if not isinstance(original, str):
            return req
        ceiling = _tier_of(original)
        if ceiling is None or ceiling == 0:
            return req

        arms = [TIER_ORDER[i][1] for i in range(ceiling + 1)]
        bucket = context_bucket(body)
        state_by_arm = await _load_state(bucket, arms)

        best_arm = self._select_arm(
            arms, original, state_by_arm, live_gate=(mode == MODE_LIVE)
        )

        req.metadata["bandit_bucket"] = bucket
        req.metadata["bandit_original_model"] = original
        req.metadata["bandit_mode"] = mode
        if mode == MODE_LIVE:
            if best_arm != original:
                body["model"] = best_arm
            req.metadata["bandit_arm"] = best_arm
        else:
            # shadow: never mutate body; record recommendation for after-hook.
            req.metadata["bandit_shadow_arm"] = best_arm
        return req

    async def _update(self, req: PipelineRequest, reward: float) -> None:
        bucket = req.metadata.get("bandit_bucket")
        arm = req.metadata.get("bandit_arm")
        if not bucket or not arm:
            return
        state = (await _load_state(bucket, [arm]))[arm]
        state.alpha += reward
        state.beta += 1.0 - reward
        state.n += 1
        state.live_n += 1
        state.total_reward += reward
        window = self.demote_window
        state.recent_rewards = (state.recent_rewards + [reward])[-window:]
        # Graduation check (only for non-original arms; original is always eligible).
        original = req.metadata.get("bandit_original_model")
        if arm != original and not state.graduated:
            if state.live_n >= self.graduate_min_n and state.mean_reward() >= self.graduate_min_reward:
                state.graduated = True
        # Demotion check. Auto-promoted cells use a tighter floor.
        if should_demote(state, window, self._demote_floor_for(state)):
            state.graduated = False
            state.promoted_from_shadow = False
            state.recent_rewards = []
        await _save_state(bucket, arm, state)

    @staticmethod
    def _response_has_tool_use(response: dict[str, Any]) -> bool:
        content = response.get("content")
        if not isinstance(content, list):
            return False
        return any(isinstance(b, dict) and b.get("type") == "tool_use" for b in content)

    async def _handle_observation(
        self,
        req: PipelineRequest,
        *,
        in_tok: int,
        out_tok: int,
        stop: str | None,
        error: str | bool | None,
        has_tool_use: bool,
    ) -> None:
        from ..pricing import estimate_cost

        mode = req.metadata.get("bandit_mode") or self.mode
        if mode == MODE_OFF:
            return
        if mode == MODE_LIVE:
            routed = req.body.get("model")
            cost = estimate_cost(routed, in_tok, out_tok)
            original = req.metadata.get("bandit_original_model")
            if original and routed and original != routed:
                cost_full = estimate_cost(original, in_tok, out_tok)
                req.metadata["saved_by_bandit_usd"] = max(0.0, cost_full - cost)
            reward = reward_from(stop, error, cost, self.cost_cap)
            await self._update(req, reward)
            return
        # shadow: log the would-be decision with honest cost delta + heuristic success.
        bucket = req.metadata.get("bandit_bucket")
        original = req.metadata.get("bandit_original_model")
        recommended = req.metadata.get("bandit_shadow_arm")
        if not bucket or not original or not recommended:
            return
        cost_full = estimate_cost(original, in_tok, out_tok)
        cost_routed = estimate_cost(recommended, in_tok, out_tok)
        est_success = estimate_shadow_reward(stop, error, out_tok, has_tool_use)
        await _log_shadow_decision(
            bucket=bucket,
            original_arm=original,
            recommended_arm=recommended,
            est_success=est_success,
            est_cost_full_usd=cost_full,
            est_cost_routed_usd=cost_routed,
            stop_reason=stop,
        )
        # Accumulate per-arm shadow evidence and auto-graduate once we have
        # enough volume + a clear cost win + acceptable success. Routed-arm
        # only — the user's chosen tier never needs graduation.
        if recommended != original:
            await self._record_shadow_observation(
                bucket=bucket,
                arm=recommended,
                est_success=est_success,
                cost_full=cost_full,
                cost_routed=cost_routed,
            )

    async def _record_shadow_observation(
        self,
        *,
        bucket: str,
        arm: str,
        est_success: float,
        cost_full: float,
        cost_routed: float,
    ) -> None:
        state = (await _load_state(bucket, [arm]))[arm]
        state.shadow_n += 1
        state.shadow_total_success += est_success
        state.shadow_total_cost_full_usd += cost_full
        state.shadow_total_cost_routed_usd += cost_routed
        # Auto-graduate gates: enough samples, success at or above bar, and a
        # real cost win. Cost-savings gate prevents promoting an arm that's the
        # same price as the original (no value, only risk).
        if (
            not state.graduated
            and state.shadow_n >= self.graduate_min_n
            and state.mean_shadow_success() >= self.graduate_min_reward
            and state.shadow_cost_savings_ratio() >= AUTO_GRADUATE_MIN_COST_SAVINGS
        ):
            state.graduated = True
            state.promoted_from_shadow = True
        await _save_state(bucket, arm, state)

    async def after(self, req: PipelineRequest, response: dict[str, Any]) -> None:
        if self.mode == MODE_OFF:
            return
        usage = response.get("usage") or {}
        in_tok = int(usage.get("input_tokens", 0) or 0)
        out_tok = int(usage.get("output_tokens", 0) or 0)
        stop = response.get("stop_reason")
        error: str | None = None
        if isinstance(response.get("error"), dict):
            error = response["error"].get("message")
        has_tool_use = self._response_has_tool_use(response)
        await self._handle_observation(
            req,
            in_tok=in_tok,
            out_tok=out_tok,
            stop=stop,
            error=error,
            has_tool_use=has_tool_use,
        )

    async def after_stream(
        self, req: PipelineRequest, raw: bytes, captured: dict[str, Any]
    ) -> None:
        if self.mode == MODE_OFF:
            return
        in_tok = int(captured.get("input_tokens", 0) or 0)
        out_tok = int(captured.get("output_tokens", 0) or 0)
        stop = captured.get("stop_reason")
        error = captured.get("error")
        has_tool_use = bool(captured.get("tool_use"))
        await self._handle_observation(
            req,
            in_tok=in_tok,
            out_tok=out_tok,
            stop=stop,
            error=error,
            has_tool_use=has_tool_use,
        )
