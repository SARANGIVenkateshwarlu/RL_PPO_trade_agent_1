"""
Experiment 5: Squeeze-Breakout Constrained RL Trading Environment

Entry Constraints (ALL must be true):
  1. Breakout: High_t > High_{t-1} (BUY) / Low_t < Low_{t-1} (SELL)
  2. BB Squeeze: sigma_20 < 1.5% SMA AND BB_Width < 0.10
  3. Model signal: BUY or SELL

Exit Conditions:
  - Stop Loss: Entry candle's Low (BUY) / Entry candle's High (SELL)
  - Exit signal: Close crosses below 9 EMA (BUY) / above 9 EMA (SELL)

Position Sizing: 20% cash, 1% ATR risk, 3% reward target (1:3 RR)
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


class SqueezeBreakoutEnv(gym.Env):
    """
    BB Squeeze + Breakout + EMA exit constrained trading environment.

    Entry gate:
      A_entry_t = 1(Breakout_t) × 1(Squeeze_t) × Model_Signal_t

    Exit:
      SL = Entry candle low (long) / Entry candle high (short)
      Trailing exit = Close < 9 EMA (long) / Close > 9 EMA (short)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        window_size: int = 30,
        feature_columns=None,
        cash_fraction: float = 0.20,
        risk_per_trade_pct: float = 0.01,
        reward_target_pct: float = 0.03,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.001,
        reward_scale: float = 1.0,
        random_start: bool = True,
        min_episode_steps: int = 200,
        episode_max_steps: int | None = None,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
        require_squeeze: bool = True,        # MANDATORY squeeze gate
        squeeze_min_level: int = 1,           # 1=moderate, 2=full
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)

        self.feature_columns = [c for c in (feature_columns or []) if c in self.df.columns]
        if not self.feature_columns:
            raise ValueError("No valid feature columns.")

        self.window_size = int(window_size)
        self.cash_fraction = float(cash_fraction)
        self.risk_per_trade_pct = float(risk_per_trade_pct)
        self.reward_target_pct = float(reward_target_pct)
        self.commission_pct = float(commission_pct)
        self.slippage_pct = float(slippage_pct)
        self.reward_scale = float(reward_scale)
        self.require_squeeze = require_squeeze
        self.squeeze_min_level = squeeze_min_level

        self.random_start = bool(random_start)
        self.min_episode_steps = int(min_episode_steps)
        self.episode_max_steps = episode_max_steps

        self.feature_mean = feature_mean
        self.feature_std = feature_std

        # Action: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)

        self.base_num_features = len(self.feature_columns)
        self.state_num_features = 5  # position, breakout_up, breakout_down, squeeze_level, time_in_trade
        self.num_features = self.base_num_features + self.state_num_features

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, self.num_features),
            dtype=np.float32,
        )

        self._reset_state()
        print(f"[SqueezeEnv] 3 actions (HOLD/BUY/SELL) | "
              f"Squeeze gate: Lv{self.squeeze_min_level}+ | "
              f"SL: Entry candle | Exit: 9 EMA cross | 1:3 RR")

    def _reset_state(self):
        self.current_step = 0
        self.steps_in_episode = 0
        self.terminated = False
        self.truncated = False

        self.position = 0
        self.entry_price = 0.0
        self.entry_candle_low = 0.0   # SL for long
        self.entry_candle_high = 0.0  # SL for short
        self.shares = 0
        self.time_in_trade = 0
        self.trailing_exit_triggered = False

        self.initial_equity = 100000.0
        self.equity = self.initial_equity
        self.cash = self.initial_equity
        self.peak_equity = self.initial_equity

        self.equity_curve = []
        self.trade_history = []
        self.daily_returns = []

    @property
    def _get_step_data(self):
        """Get data for current step."""
        idx = self.current_step
        return {
            "high": float(self.df.loc[idx, "High"]),
            "low": float(self.df.loc[idx, "Low"]),
            "close": float(self.df.loc[idx, "Close"]),
            "ema_9": float(self.df.loc[idx, "ema_9"]) if "ema_9" in self.df.columns else float(self.df.loc[idx, "Close"]),
            "squeeze_signal": int(self.df.loc[idx, "squeeze_signal"]) if "squeeze_signal" in self.df.columns else 0,
        }

    def _get_breakout_flags(self) -> tuple:
        idx = self.current_step
        if idx < 1:
            return 0, 0
        h_t = float(self.df.loc[idx, "High"])
        l_t = float(self.df.loc[idx, "Low"])
        h_tm1 = float(self.df.loc[idx - 1, "High"])
        l_tm1 = float(self.df.loc[idx - 1, "Low"])
        return int(h_t > h_tm1), int(l_t < l_tm1)

    def _check_squeeze(self) -> bool:
        """Check if BB squeeze condition is met."""
        if not self.require_squeeze:
            return True
        squeeze = int(self.df.loc[self.current_step, "squeeze_signal"]) if "squeeze_signal" in self.df.columns else 0
        return squeeze >= self.squeeze_min_level

    def _apply_entry_gate(self, model_action: int) -> int:
        """
        Entry gate: breakout AND squeeze AND model signal.

        A_entry_t = 1(Breakout_t) × 1(Squeeze_t) × Model_Signal_t
        """
        breakout_up, breakout_down = self._get_breakout_flags()
        has_squeeze = self._check_squeeze()

        a_buy = breakout_up * int(has_squeeze) * (1 if model_action == 1 else 0)
        a_sell = breakout_down * int(has_squeeze) * (1 if model_action == 2 else 0)

        if a_buy == 1:
            return 1
        elif a_sell == 1:
            return 2
        return 0

    def _get_state_features(self):
        bu, bd = self._get_breakout_flags()
        squeeze = int(self.df.loc[self.current_step, "squeeze_signal"]) if "squeeze_signal" in self.df.columns else 0
        t_norm = float(self.time_in_trade) / 252.0
        pos = float(self.position)
        return np.array([pos, float(bu), float(bd), float(squeeze) / 2.0, t_norm], dtype=np.float32)

    def _get_observation(self):
        start = max(0, self.current_step - self.window_size)
        obs_data = self.df.iloc[start:self.current_step][self.feature_columns].values.astype(np.float32)

        if len(obs_data) == 0:
            base = np.tile(self.df.iloc[0][self.feature_columns].values.astype(np.float32), (self.window_size, 1))
        elif obs_data.shape[0] < self.window_size:
            pad = np.tile(obs_data[0], (self.window_size - obs_data.shape[0], 1))
            base = np.vstack([pad, obs_data])
        else:
            base = obs_data

        state = np.tile(self._get_state_features(), (self.window_size, 1))
        obs = np.hstack([base, state]).astype(np.float32)

        if self.feature_mean is not None and self.feature_std is not None:
            nf = self.base_num_features
            m = self.feature_mean.reshape(1, -1)
            s = self.feature_std.reshape(1, -1)
            s = np.where(s == 0, 1.0, s)
            obs[:, :nf] = (obs[:, :nf] - m) / s

        return obs

    def _open_trade(self, direction: int):
        price = float(self.df.loc[self.current_step, "Close"])
        low_candle = float(self.df.loc[self.current_step, "Low"])
        high_candle = float(self.df.loc[self.current_step, "High"])
        atr = float(self.df.loc[self.current_step, "atr_14"]) if "atr_14" in self.df.columns else price * 0.02

        max_cash = self.equity * self.cash_fraction
        risk_amount = self.equity * self.risk_per_trade_pct

        # SL = entry candle low (long) / entry candle high (short)
        # Target = 3 × SL distance (1:3 RR)
        if direction == 1:  # Long
            slip = np.random.uniform(0, self.slippage_pct) * price
            entry = price + slip
            sl = low_candle
            tp = entry + (entry - sl) * (self.reward_target_pct / self.risk_per_trade_pct)
        else:  # Short
            slip = np.random.uniform(0, self.slippage_pct) * price
            entry = price - slip
            sl = high_candle
            tp = entry - (sl - entry) * (self.reward_target_pct / self.risk_per_trade_pct)

        stop_distance = abs(entry - sl)
        if stop_distance <= 0:
            return

        shares = int(risk_amount / stop_distance)
        cost = shares * entry * (1 + self.commission_pct)

        if cost > max_cash:
            shares = int(max_cash / (entry * (1 + self.commission_pct)))
            if shares <= 0:
                return
            cost = shares * entry * (1 + self.commission_pct)

        if cost > self.cash:
            shares = int(self.cash / (entry * (1 + self.commission_pct)))
            if shares <= 0:
                return

        self.position = direction
        self.entry_price = entry
        self.entry_candle_low = low_candle
        self.entry_candle_high = high_candle
        self.shares = shares
        self.sl_price = sl
        self.tp_price = tp
        self.cash -= shares * entry * (1 + self.commission_pct)
        self.time_in_trade = 0
        self.trailing_exit_triggered = False

    def _check_exit_conditions(self) -> float | None:
        """Check exit: candle SL/TP + 9 EMA trailing exit."""
        if self.position == 0:
            return None

        close = float(self.df.loc[self.current_step, "Close"])
        ema9 = float(self.df.loc[self.current_step, "ema_9"]) if "ema_9" in self.df.columns else close
        low = float(self.df.loc[self.current_step, "Low"])
        high = float(self.df.loc[self.current_step, "High"])

        # 1) Hard SL (entry candle low/high intra-bar)
        if self.position == 1 and low <= self.sl_price:
            return self._close_trade("SL_HIT_CANDLE_LOW", self.sl_price)

        if self.position == -1 and high >= self.sl_price:
            return self._close_trade("SL_HIT_CANDLE_HIGH", self.sl_price)

        # 2) TP (1:3 target)
        if self.position == 1 and high >= self.tp_price:
            return self._close_trade("TP_HIT", self.tp_price)

        if self.position == -1 and low <= self.tp_price:
            return self._close_trade("TP_HIT", self.tp_price)

        # 3) 9 EMA trailing exit
        if self.position == 1 and close < ema9:
            return self._close_trade("EMA9_EXIT_LONG", close)

        if self.position == -1 and close > ema9:
            return self._close_trade("EMA9_EXIT_SHORT", close)

        return None

    def _close_trade(self, reason: str, exit_price: float) -> float:
        if self.position == 0 or self.shares == 0:
            return 0.0

        gross = self.shares * exit_price * (1 - self.commission_pct)
        cost_basis = self.shares * self.entry_price * (1 + self.commission_pct)
        pnl = (gross - cost_basis) * self.position
        pnl_pct = pnl / cost_basis * 100 if cost_basis > 0 else 0.0

        self.cash += gross
        self.equity = self.cash

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        trade = {
            "event": "CLOSE", "reason": reason,
            "direction": self.position,
            "entry": self.entry_price, "exit": exit_price,
            "shares": self.shares,
            "pnl": float(pnl), "pnl_pct": float(pnl_pct),
            "equity": float(self.equity),
            "time_in_trade": self.time_in_trade,
            "step": self.current_step,
        }
        self.trade_history.append(trade)

        self.position = 0; self.entry_price = 0.0; self.shares = 0
        self.sl_price = 0.0; self.tp_price = 0.0
        self.time_in_trade = 0; self.trailing_exit_triggered = False

        return pnl_pct

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        if self.random_start:
            mx = self.n_steps - max(self.min_episode_steps, self.window_size) - 2
            self.current_step = int(np.random.randint(self.window_size, mx)) if mx > self.window_size else self.window_size
        else:
            self.current_step = self.window_size
        obs = self._get_observation()
        return (obs, {}) if _GYMNASIUM else obs

    def step(self, action: int):
        if self.terminated or self.truncated:
            obs = self._get_observation()
            return (obs, 0.0, True, False, {}) if _GYMNASIUM else (obs, 0.0, True, {})

        self.steps_in_episode += 1
        raw_action = int(action)
        filtered_action = self._apply_entry_gate(raw_action)

        # Execute filtered entry
        if filtered_action == 1 and self.position == 0:
            self._open_trade(1)
        elif filtered_action == 2 and self.position == 0:
            self._open_trade(-1)

        # Check exits
        exit_pnl = self._check_exit_conditions()

        # Update MTM
        if self.position != 0:
            self.time_in_trade += 1
            close = float(self.df.loc[self.current_step, "Close"])
            mtm = self.shares * (close - self.entry_price) * self.position
            self.equity = self.cash + mtm

        reward = 0.0
        if exit_pnl is not None:
            reward += exit_pnl / 100.0  # Convert % pnl to reward units

        if self.position == 0 and filtered_action == 0:
            reward += 0.0001  # Discipline bonus

        self.current_step += 1

        # Termination
        if self.current_step >= self.n_steps - 1:
            if self.position != 0:
                self._close_trade("END", float(self.df.loc[self.current_step - 1, "Close"]))
            self.terminated = True

        if self.episode_max_steps and self.steps_in_episode >= self.episode_max_steps:
            self.truncated = True

        if self.peak_equity > 0 and self.equity / self.peak_equity < 0.75:
            if self.position != 0:
                self._close_trade("DD_LIMIT", float(self.df.loc[self.current_step - 1, "Close"]))
            self.terminated = True
            reward -= 0.05

        self.equity_curve.append(float(self.equity))
        obs = self._get_observation()
        reward *= self.reward_scale

        bu, bd = self._get_breakout_flags()
        info = {
            "equity": float(self.equity),
            "position": int(self.position),
            "raw_action": raw_action,
            "filtered_action": filtered_action,
            "breakout_up": bu, "breakout_down": bd,
            "has_squeeze": int(self._check_squeeze()),
            "time_in_trade": int(self.time_in_trade),
        }

        return (obs, float(reward), self.terminated, self.truncated, info) if _GYMNASIUM else (
            obs, float(reward), bool(self.terminated or self.truncated), info
        )

    def render(self):
        ps = {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(self.position, "?")
        sq = "SQZ" if self._check_squeeze() else "---"
        print(f"Step={self.current_step} | Eq=${self.equity:,.0f} | {ps} | {sq} | "
              f"SL={self.sl_price:.2f} TP={self.tp_price:.2f}")
