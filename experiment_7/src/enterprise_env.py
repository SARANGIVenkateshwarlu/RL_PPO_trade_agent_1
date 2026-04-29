"""
Experiment 7: Inside Bar Trend-Following Trading Environment

Strategy (long-only):
  1. SPX > 50/100/200 EMA
  2. Stock within 25% of 52-week high
  3. 21-day relative volatility < 20% of normal
  4. Strong prior uptrend (>25%/1m or >30%/3-12m)
  5. Weekly + daily inside bars
  6. Current low > previous low
  7. Entry on cross above previous day high, within 2% of prev close
  8. SL = current day low
  9. At 1:2 RR → SL to breakeven
  10. Close below 10 EMA → book 50%
  11. Close below 21 EMA → book remaining 50%
  12. Risk per trade = 10% of capital
  13. Multiple positions allowed (capital-permitting)

Agent action space: HOLD / BUY (entry timing decision only)
All exits are rule-based — agent does not decide when to exit.
"""
from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYMNASIUM = True
except ImportError:
    import gym
    from gym import spaces
    _GYMNASIUM = False


class InsideBarTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self, df, window_size: int = 30, feature_columns=None,
        risk_per_trade_pct: float = 0.10,
        reward_scale: float = 1.0,
        random_start: bool = True, min_episode_steps: int = 200,
        episode_max_steps: int | None = None,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
        use_action_masking: bool = True,
        require_entry_gate: bool = True,
    ):
        super().__init__()
        df = df.copy()
        if "Date" in df.columns:
            df = df.sort_values("Date").reset_index(drop=True)
        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)

        self.feature_columns = [c for c in (feature_columns or []) if c in self.df.columns]
        if not self.feature_columns:
            raise ValueError("No valid feature columns")

        self.window_size = int(window_size)
        self.risk_per_trade_pct = float(risk_per_trade_pct)
        self.reward_scale = float(reward_scale)
        self.use_action_masking = use_action_masking
        self.require_entry_gate = require_entry_gate

        self.random_start = bool(random_start)
        self.min_episode_steps = int(min_episode_steps)
        self.episode_max_steps = episode_max_steps
        self.feature_mean = feature_mean
        self.feature_std = feature_std

        # 2 actions: HOLD, BUY (long-only)
        self.action_space = spaces.Discrete(2)

        self.base_num_features = len(self.feature_columns)
        self.state_num_features = 5
        self.num_features = self.base_num_features + self.state_num_features

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, self.num_features), dtype=np.float32,
        )

        self._reset_state()
        print(f"[InsideBarEnv] 2 actions (HOLD/BUY) | "
              f"Risk={self.risk_per_trade_pct*100:.0f}% | "
              f"EntryGate={'ON' if require_entry_gate else 'OFF'} | "
              f"Masking={'ON' if use_action_masking else 'OFF'}")

    # ─── state management ──────────────────────────────────────────

    def _reset_state(self):
        self.current_step = 0
        self.steps_in_episode = 0
        self.terminated = False
        self.truncated = False
        self.initial_equity = 100000.0
        self.equity = self.initial_equity
        self.cash = self.initial_equity
        self.peak_equity = self.initial_equity
        self.positions = []  # list of dicts
        self.equity_curve = []
        self.trade_history = []
        self.daily_returns = []

    @property
    def _row(self):
        return self.df.iloc[self.current_step]

    @property
    def _position_count(self):
        return len(self.positions)

    def _equity_in_positions(self):
        return sum(p["shares"] * self._row["Close"] for p in self.positions)

    # ─── entry conditions ──────────────────────────────────────────

    def _entry_gate_passed(self) -> bool:
        if not self.require_entry_gate:
            return True
        row = self._row
        if "entry_gate" in row.index:
            return int(row["entry_gate"]) == 1

        # Fallback: check individual conditions
        r = row.to_dict()
        ok = (
            r.get("spx_bull", 1) == 1
            and r.get("near_52w_high", 1) == 1
            and r.get("low_vol", 1) == 1
            and r.get("strong_uptrend", 1) == 1
            and r.get("weekly_inside_bar", 1) == 1
            and r.get("daily_inside_bar", 1) == 1
            and r.get("low_holding", 1) == 1
            and r.get("cross_above_prev_high", 1) == 1
            and r.get("entry_near_prev_close", 1) == 1
        )
        return ok

    def action_masks(self) -> np.ndarray:
        """Mask: BUY valid only when entry gate passes + capital available."""
        can_buy = False
        if self._entry_gate_passed():
            risk_amount = self.equity * self.risk_per_trade_pct
            entry_price = self._row["Close"]
            if entry_price > 0 and self.cash >= risk_amount:
                can_buy = True
        return np.array([True, can_buy], dtype=bool)

    # ─── observation ───────────────────────────────────────────────

    def _get_state_features(self):
        gate = int(self._entry_gate_passed())
        pos_count_norm = min(self._position_count / 5.0, 1.0)
        dd_norm = max(0.0, (self.peak_equity - self.equity) / (self.peak_equity + 1e-8))
        invested_pct = self._equity_in_positions() / (self.equity + 1e-8)
        return np.array([
            float(gate),
            float(pos_count_norm),
            float(dd_norm),
            float(invested_pct),
            float(self.cash / self.initial_equity),
        ], dtype=np.float32)

    def _get_observation(self):
        start = max(0, self.current_step - self.window_size)
        od = self.df.iloc[start:self.current_step][self.feature_columns].values.astype(np.float32)
        if len(od) == 0:
            base = np.tile(self.df.iloc[0][self.feature_columns].values.astype(np.float32), (self.window_size, 1))
        elif od.shape[0] < self.window_size:
            base = np.vstack([np.tile(od[0], (self.window_size - od.shape[0], 1)), od])
        else:
            base = od
        state = np.tile(self._get_state_features(), (self.window_size, 1))
        obs = np.hstack([base, state]).astype(np.float32)
        if self.feature_mean is not None and self.feature_std is not None:
            nf = self.base_num_features
            m = self.feature_mean.reshape(1, -1)
            s = self.feature_std.reshape(1, -1)
            s = np.where(s == 0, 1.0, s)
            obs[:, :nf] = (obs[:, :nf] - m) / s
        return obs

    # ─── position management ───────────────────────────────────────

    def _open_position(self):
        """Open a new long position with SL = current day low."""
        row = self._row
        entry_price = float(row["Close"])
        sl_price = float(row["Low"])

        stop_dist = entry_price - sl_price
        if stop_dist <= 0:
            return

        risk_amount = self.equity * self.risk_per_trade_pct
        shares = max(1, int(risk_amount / stop_dist))
        cost = shares * entry_price
        if cost > self.cash:
            shares = max(1, int(self.cash / entry_price))
        if shares <= 0:
            return

        self.positions.append({
            "shares": shares,
            "entry_price": entry_price,
            "sl_price": sl_price,
            "breakeven": False,
            "ema10_exited": False,
            "ema21_exited": False,
            "entry_step": self.current_step,
            "remaining_shares": shares,
        })
        self.cash -= shares * entry_price

    def _check_exits(self) -> list:
        """Check all positions for exit conditions. Returns list of (pnl, reason)."""
        row = self._row
        close = float(row["Close"])
        high = float(row["High"])
        low = float(row["Low"])
        ema10 = float(row.get("ema_10", close))
        ema21 = float(row.get("ema_21", close))

        exit_results = []
        remaining_positions = []

        for pos in self.positions:
            entry = pos["entry_price"]
            sl = pos["sl_price"]
            remaining = pos["remaining_shares"]
            if remaining <= 0:
                continue

            # Step 1: price hits 1:2 RR → move SL to breakeven
            if not pos["breakeven"]:
                risk = entry - sl
                tp_1_2_rr = entry + risk * 2  # 1:2 reward-risk
                if high >= tp_1_2_rr:
                    pos["breakeven"] = True
                    pos["sl_price"] = entry  # move SL to breakeven

            # Step 2: check stop-loss
            effective_sl = pos["sl_price"]
            if low <= effective_sl:
                exit_price = effective_sl
                total_pnl = pos["shares"] * (exit_price - entry)
                self.cash += pos["shares"] * exit_price
                exit_results.append((total_pnl, "SL_HIT" if pos["breakeven"] else "SL"))
                continue  # position fully closed

            # Step 3: close below 10 EMA (first time) → book 50%
            if close < ema10 and not pos["ema10_exited"]:
                pos["ema10_exited"] = True
                exit_shares = pos["shares"] // 2
                if exit_shares > 0:
                    pnl = exit_shares * (close - entry)
                    self.cash += exit_shares * close
                    pos["remaining_shares"] -= exit_shares
                    exit_results.append((pnl, "EMA10_50%"))
                if pos["remaining_shares"] <= 0:
                    continue

            # Step 4: close below 21 EMA (first time) → book remaining 50%
            if close < ema21 and not pos["ema21_exited"]:
                pos["ema21_exited"] = True
                exit_shares = pos["remaining_shares"]
                if exit_shares > 0:
                    pnl = exit_shares * (close - entry)
                    self.cash += exit_shares * close
                    exit_results.append((pnl, "EMA21_REMAINING"))
                continue  # position fully closed

            remaining_positions.append(pos)

        self.positions = remaining_positions
        return exit_results

    # ─── gym API ───────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        if self.random_start:
            mx = self.n_steps - max(self.min_episode_steps, self.window_size) - 2
            self.current_step = (
                int(np.random.randint(self.window_size, mx))
                if mx > self.window_size
                else self.window_size
            )
        else:
            self.current_step = self.window_size
        return (self._get_observation(), {}) if _GYMNASIUM else self._get_observation()

    def step(self, action: int):
        if self.terminated or self.truncated:
            obs = self._get_observation()
            return (
                (obs, 0.0, True, False, {"action_mask": self.action_masks()})
                if _GYMNASIUM
                else (obs, 0.0, True, {"action_mask": self.action_masks()})
            )

        self.steps_in_episode += 1
        raw_action = int(action)

        # Apply action masking
        if self.use_action_masking:
            mask = self.action_masks()
            if not mask[raw_action]:
                raw_action = 0

        # Execute entry
        if raw_action == 1 and self._entry_gate_passed():
            self._open_position()

        # Check exits
        exit_results = self._check_exits()

        # Update equity
        mtm = self._equity_in_positions()
        prev_equity = self.equity
        self.equity = self.cash + mtm
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        # Record trade history
        for pnl, reason in exit_results:
            self.trade_history.append({
                "reason": reason,
                "pnl_usd": float(pnl),
                "equity": float(self.equity),
                "step": self.current_step,
            })

        # Reward = PnL from closed trades scaled
        reward = sum(pnl / self.initial_equity for pnl, _ in exit_results)

        # Small hold bonus when out of positions and gate is not met
        if self._position_count == 0 and not self._entry_gate_passed():
            reward += 0.0001

        self.current_step += 1
        self.equity_curve.append(float(self.equity))

        # Termination conditions
        if self.current_step >= self.n_steps - 1:
            self._liquidate_all()
            self.terminated = True
        if self.episode_max_steps and self.steps_in_episode >= self.episode_max_steps:
            self._liquidate_all()
            self.truncated = True
        if self.peak_equity > 0 and self.equity / self.peak_equity < 0.75:
            self._liquidate_all()
            self.terminated = True
            reward -= 0.05

        reward *= self.reward_scale

        obs = self._get_observation()
        info = {
            "equity": float(self.equity),
            "position_count": self._position_count,
            "action_mask": self.action_masks(),
            "gate_passed": int(self._entry_gate_passed()),
        }

        return (
            (obs, float(reward), self.terminated, self.truncated, info)
            if _GYMNASIUM
            else (obs, float(reward), bool(self.terminated or self.truncated), info)
        )

    def _liquidate_all(self):
        """Close all positions at current close price."""
        close = float(self._row["Close"])
        for pos in self.positions:
            total_pnl = pos["remaining_shares"] * (close - pos["entry_price"])
            self.cash += pos["remaining_shares"] * close
            self.trade_history.append({
                "reason": "EPISODE_END",
                "pnl_usd": float(total_pnl),
                "equity": float(self.cash),
                "step": self.current_step,
            })
        self.positions = []
        self.equity = self.cash

    def render(self):
        g = "GATE" if self._entry_gate_passed() else "----"
        print(f"Step={self.current_step} | ${self.equity:,.0f} | "
              f"Pos={self._position_count} | Cash=${self.cash:,.0f} | {g}")
