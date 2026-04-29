"""
Experiment 6: Enterprise-Grade Production Trading Environment

Action Masking : Invalid actions are masked in the policy distribution
(not post-filtered), so the agent never learns to take impossible actions.

All constraints from previous experiments preserved:
  - Breakout confirmation (High > Prev High / Low < Prev Low)
  - BB Squeeze detection (sigma < 3% SMA, width < 0.12, volume < 70%)
  - 9 EMA trailing exit
  - Candle-based stop loss
  - 20% cash, 1% risk, 3% reward (1:3 RR)
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


class EnterpriseTradingEnv(gym.Env):
    """
    Production-ready trading environment with action masking.

    action_mask tells the agent which actions are valid:
      - BUY (1): valid only when breakout_up + squeeze
      - SELL (2): valid only when breakout_down + squeeze
      - HOLD (0): always valid
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self, df, window_size: int = 30, feature_columns=None,
        cash_fraction: float = 0.20, risk_per_trade_pct: float = 0.01,
        reward_target_pct: float = 0.03,
        commission_pct: float = 0.001, slippage_pct: float = 0.001,
        reward_scale: float = 1.0,
        random_start: bool = True, min_episode_steps: int = 200,
        episode_max_steps: int | None = None,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
        require_squeeze: bool = True, squeeze_min_level: int = 1,
        use_action_masking: bool = True,
    ):
        super().__init__()
        # Ensure chronological order: oldest first (idx 0), newest last (idx -1)
        # This guarantees x-axis flows left→right = past→present
        df = df.copy()
        if "Date" in df.columns:
            df = df.sort_values("Date").reset_index(drop=True)
        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)
        self.feature_columns = [c for c in (feature_columns or []) if c in self.df.columns]
        if not self.feature_columns: raise ValueError("No valid feature columns")

        self.window_size = int(window_size)
        self.cash_fraction = float(cash_fraction)
        self.risk_per_trade_pct = float(risk_per_trade_pct)
        self.reward_target_pct = float(reward_target_pct)
        self.commission_pct = float(commission_pct)
        self.slippage_pct = float(slippage_pct)
        self.reward_scale = float(reward_scale)
        self.require_squeeze = require_squeeze
        self.squeeze_min_level = squeeze_min_level
        self.use_action_masking = use_action_masking

        self.random_start = bool(random_start)
        self.min_episode_steps = int(min_episode_steps)
        self.episode_max_steps = episode_max_steps
        self.feature_mean = feature_mean
        self.feature_std = feature_std

        # 3 actions: HOLD, BUY, SELL
        self.action_space = spaces.Discrete(3)

        self.base_num_features = len(self.feature_columns)
        self.state_num_features = 5
        self.num_features = self.base_num_features + self.state_num_features

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, self.num_features), dtype=np.float32,
        )

        self._reset_state()
        print(f"[EnterpriseEnv] 3 actions | Squeeze Lv{self.squeeze_min_level}+ | "
              f"Action Masking={'ON' if use_action_masking else 'OFF'}")

    def _reset_state(self):
        self.current_step = 0; self.steps_in_episode = 0
        self.terminated = False; self.truncated = False
        self.position = 0; self.entry_price = 0.0
        self.entry_candle_low = 0.0; self.entry_candle_high = 0.0
        self.shares = 0; self.time_in_trade = 0
        self.initial_equity = 100000.0; self.equity = self.initial_equity
        self.cash = self.initial_equity; self.peak_equity = self.initial_equity
        self.equity_curve = []; self.trade_history = []; self.daily_returns = []

    @property
    def _get_step_data(self):
        idx = self.current_step
        keys = ["Open", "High", "Low", "Close", "ema_20_50_spread", "squeeze_signal",
                "breakout_up", "breakout_down", "rsi_14", "adx_14", "ema_9", "atr_14"]
        return {k: self.df.loc[idx, k] if k in self.df.columns else 0.0 for k in keys}

    def _get_breakout_flags(self) -> tuple:
        idx = self.current_step
        if idx < 1: return 0, 0
        return (int(self.df.loc[idx, "High"] > self.df.loc[idx - 1, "High"]),
                int(self.df.loc[idx, "Low"] < self.df.loc[idx - 1, "Low"]))

    def _check_squeeze(self) -> bool:
        if not self.require_squeeze: return True
        sq = int(self.df.loc[self.current_step, "squeeze_signal"]) if "squeeze_signal" in self.df.columns else 0
        return sq >= self.squeeze_min_level

    def action_masks(self) -> np.ndarray:
        """Return boolean action mask: [HOLD_valid, BUY_valid, SELL_valid]."""
        bu, bd = self._get_breakout_flags()
        sq = self._check_squeeze()
        return np.array([True, bu == 1 and sq, bd == 1 and sq], dtype=bool)

    def _get_state_features(self):
        bu, bd = self._get_breakout_flags()
        sq = int(self.df.loc[self.current_step, "squeeze_signal"]) if "squeeze_signal" in self.df.columns else 0
        return np.array([float(self.position), float(bu), float(bd),
                         float(sq) / 2.0, float(self.time_in_trade) / 252.0], dtype=np.float32)

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
            m = self.feature_mean.reshape(1, -1); s = self.feature_std.reshape(1, -1)
            s = np.where(s == 0, 1.0, s)
            obs[:, :nf] = (obs[:, :nf] - m) / s
        return obs

    def _open_trade(self, direction: int):
        price = self.df.loc[self.current_step, "Close"]
        lo, hi = self.df.loc[self.current_step, "Low"], self.df.loc[self.current_step, "High"]
        atr = self.df.loc[self.current_step, "atr_14"] if "atr_14" in self.df.columns else price * 0.02

        max_cash = self.equity * self.cash_fraction
        risk_amt = self.equity * self.risk_per_trade_pct

        slip = np.random.uniform(0, self.slippage_pct) * price
        if direction == 1:
            entry = price + slip; sl = lo
            tp = entry + (entry - sl) * (self.reward_target_pct / self.risk_per_trade_pct)
        else:
            entry = price - slip; sl = hi
            tp = entry - (sl - entry) * (self.reward_target_pct / self.risk_per_trade_pct)

        stop_dist = abs(entry - sl)
        if stop_dist <= 0: return

        shares = int(risk_amt / stop_dist)
        cost = shares * entry * (1 + self.commission_pct)
        if cost > max_cash: shares = int(max_cash / (entry * (1 + self.commission_pct)))
        if shares <= 0: return
        if shares * entry * (1 + self.commission_pct) > self.cash:
            shares = int(self.cash / (entry * (1 + self.commission_pct)))
        if shares <= 0: return

        self.position = direction; self.entry_price = entry
        self.entry_candle_low = lo; self.entry_candle_high = hi
        self.shares = shares; self.sl_price = sl; self.tp_price = tp
        self.cash -= shares * entry * (1 + self.commission_pct); self.time_in_trade = 0

    def _check_exit(self) -> float | None:
        if self.position == 0: return None
        close = self.df.loc[self.current_step, "Close"]
        lo, hi = self.df.loc[self.current_step, "Low"], self.df.loc[self.current_step, "High"]
        ema9 = self.df.loc[self.current_step, "ema_9"] if "ema_9" in self.df.columns else close

        if self.position == 1 and lo <= self.sl_price: return self._close("SL_HIT", self.sl_price)
        if self.position == -1 and hi >= self.sl_price: return self._close("SL_HIT", self.sl_price)
        if self.position == 1 and hi >= self.tp_price: return self._close("TP_HIT", self.tp_price)
        if self.position == -1 and lo <= self.tp_price: return self._close("TP_HIT", self.tp_price)
        if self.position == 1 and close < ema9: return self._close("EMA9_EXIT", close)
        if self.position == -1 and close > ema9: return self._close("EMA9_EXIT", close)
        return None

    def _close(self, reason: str, exit_price: float) -> float:
        if self.position == 0 or self.shares == 0: return 0.0
        gross = self.shares * exit_price * (1 - self.commission_pct)
        cost_basis = self.shares * self.entry_price * (1 + self.commission_pct)
        pnl = (gross - cost_basis) * self.position
        pnl_pct = pnl / cost_basis * 100 if cost_basis > 0 else 0
        self.cash += gross; self.equity = self.cash
        if self.equity > self.peak_equity: self.peak_equity = self.equity
        self.trade_history.append({
            "reason": reason, "direction": self.position, "entry": self.entry_price,
            "exit": exit_price, "shares": self.shares, "pnl_pct": float(pnl_pct),
            "equity": float(self.equity), "time_in_trade": self.time_in_trade,
        })
        self.position = 0; self.entry_price = 0.0; self.shares = 0
        self.sl_price = 0.0; self.tp_price = 0.0; self.time_in_trade = 0
        return pnl_pct

    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self._reset_state()
        if self.random_start:
            mx = self.n_steps - max(self.min_episode_steps, self.window_size) - 2
            self.current_step = int(np.random.randint(self.window_size, mx)) if mx > self.window_size else self.window_size
        else:
            self.current_step = self.window_size
        # Verify chronological order at init
        if "Date" in self.df.columns and self.current_step > 0:
            d_now = self.df.loc[self.current_step, "Date"]
            d_prev = self.df.loc[self.current_step - 1, "Date"]
            assert d_now >= d_prev, f"Data NOT chronological at step {self.current_step}: {d_prev} → {d_now}"
        return (self._get_observation(), {}) if _GYMNASIUM else self._get_observation()

    def step(self, action: int):
        if self.terminated or self.truncated:
            return (self._get_observation(), 0.0, True, False, {"action_mask": self.action_masks()}) if _GYMNASIUM else (self._get_observation(), 0.0, True, {"action_mask": self.action_masks()})

        self.steps_in_episode += 1
        raw_action = int(action)

        # Action masking: if action is invalid, force HOLD
        if self.use_action_masking:
            mask = self.action_masks()
            if not mask[raw_action]:
                raw_action = 0

        # Execute entry
        if raw_action == 1 and self.position == 0: self._open_trade(1)
        elif raw_action == 2 and self.position == 0: self._open_trade(-1)

        exit_pnl = self._check_exit()

        if self.position != 0:
            self.time_in_trade += 1
            close = self.df.loc[self.current_step, "Close"]
            mtm = self.shares * (close - self.entry_price) * self.position
            self.equity = self.cash + mtm

        reward = 0.0
        if exit_pnl is not None: reward += exit_pnl / 100.0  # % → reward units
        if self.position == 0 and raw_action == 0: reward += 0.0001

        self.current_step += 1
        if self.current_step >= self.n_steps - 1:
            if self.position != 0: self._close("END", self.df.loc[self.current_step - 1, "Close"])
            self.terminated = True
        if self.episode_max_steps and self.steps_in_episode >= self.episode_max_steps:
            self.truncated = True
        if self.peak_equity > 0 and self.equity / self.peak_equity < 0.75:
            if self.position != 0: self._close("DD_LIMIT", self.df.loc[self.current_step - 1, "Close"])
            self.terminated = True; reward -= 0.05

        self.equity_curve.append(float(self.equity))
        obs = self._get_observation()
        reward *= self.reward_scale

        info = {"equity": float(self.equity), "position": int(self.position),
                "action_mask": self.action_masks(),
                "filtered": int(raw_action != int(action)),
                "time_in_trade": int(self.time_in_trade)}

        return (obs, float(reward), self.terminated, self.truncated, info) if _GYMNASIUM else (
            obs, float(reward), bool(self.terminated or self.truncated), info)

    def render(self):
        ps = {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(self.position, "?")
        sq = "SQZ" if self._check_squeeze() else "---"
        print(f"Step={self.current_step} | ${self.equity:,.0f} | {ps} | {sq}")
