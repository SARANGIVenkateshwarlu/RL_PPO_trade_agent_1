"""
Experiment 4: Breakout-Constrained RL Trading Environment

Core Trading Rules (MANDATORY):
  - BUY:  Only when Current Day HIGH > Previous Day HIGH AND model signals BUY
  - SELL: Only when Current Day LOW < Previous Day LOW AND model signals SELL
  - NO TRADE: If breakout condition fails → Force HOLD regardless of model signal

Mathematical Constraints:
  A_buy_t  = 1(High_t > High_{t-1}) × Model_Buy_t
  A_sell_t = 1(Low_t < Low_{t-1}) × Model_Sell_t
  Final_Action_t = argmax(A_buy_t, A_sell_t, 0)

Position Sizing:
  - 20% of cash allocated per trade attempt
  - 1% risk per trade (ATR-based stop)
  - 3% reward target (TP = 3× risk, 1:3 RR)

Reward Function:
  R_t = Sharpe_Return_t × Hold_Duration_Penalty - Slippage_Cost
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


class BreakoutTradingEnv(gym.Env):
    """
    Breakout-Constrained RL Trading Environment.

    State space S_t = [OHLC_t, RSI_14_t, SMA_20_dist_t, Pivot_R1/S1_dist_t,
                       Volume_ratio_t, Breakout_Flag_t, Position_t, ...]

    Action: 0=HOLD, 1=BUY, 2=SELL
    Breakout filter applied post-model: BUY requires breakout_up, SELL requires breakout_down
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        window_size: int = 30,
        feature_columns=None,
        # Position sizing
        cash_fraction: float = 0.20,     # 20% of cash per trade
        risk_per_trade_pct: float = 0.01, # 1% risk (ATR-based stop)
        reward_target_pct: float = 0.03,  # 3% reward target (TP = 3× risk → 1:3 RR)
        # Costs
        commission_pct: float = 0.001,    # 0.1% per side
        slippage_pct: float = 0.001,      # 0.1% slippage
        # Reward
        reward_scale: float = 1.0,
        hold_duration_penalty: float = 0.001,
        # Episode
        random_start: bool = True,
        min_episode_steps: int = 200,
        episode_max_steps: int | None = None,
        # Normalization
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
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
        self.hold_duration_penalty = float(hold_duration_penalty)

        self.random_start = bool(random_start)
        self.min_episode_steps = int(min_episode_steps)
        self.episode_max_steps = episode_max_steps

        self.feature_mean = feature_mean
        self.feature_std = feature_std

        # Action space: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)

        self.base_num_features = len(self.feature_columns)
        self.state_num_features = 3  # position, breakout_up, breakout_down
        self.num_features = self.base_num_features + self.state_num_features

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, self.num_features),
            dtype=np.float32,
        )

        self._reset_state()
        print(f"[BreakoutEnv] Actions: 3 (HOLD/BUY/SELL) | "
              f"Size: {self.cash_fraction*100:.0f}% cash, {self.risk_per_trade_pct*100:.0f}% risk, "
              f"{self.reward_target_pct*100:.0f}% reward (1:3 RR)")

    def _reset_state(self):
        self.current_step = 0
        self.steps_in_episode = 0
        self.terminated = False
        self.truncated = False

        self.position = 0         # 0=flat, 1=long, -1=short
        self.entry_price = 0.0
        self.shares = 0
        self.time_in_trade = 0
        self.cumulative_return = 0.0

        self.initial_equity = 100000.0
        self.equity = self.initial_equity
        self.cash = self.initial_equity
        self.peak_equity = self.initial_equity

        self.equity_curve = []
        self.trade_history = []
        self.daily_returns = []

    def _get_breakout_flags(self) -> tuple:
        """Return (breakout_up, breakout_down) at current step using raw OHLC data."""
        idx = self.current_step
        if idx < 1:
            return 0, 0
        high_t = float(self.df.loc[idx, "High"])
        low_t = float(self.df.loc[idx, "Low"])
        high_tm1 = float(self.df.loc[idx - 1, "High"])
        low_tm1 = float(self.df.loc[idx - 1, "Low"])
        return int(high_t > high_tm1), int(low_t < low_tm1)

    def _apply_breakout_filter(self, model_action: int) -> int:
        """
        Apply the MANDATORY breakout constraint.

        A_buy_t  = 1(High_t > High_{t-1}) × Model_Buy_t
        A_sell_t = 1(Low_t < Low_{t-1}) × Model_Sell_t
        Final_Action_t = argmax(A_buy_t, A_sell_t, 0)
        """
        breakout_up, breakout_down = self._get_breakout_flags()

        a_buy = breakout_up * (1 if model_action == 1 else 0)
        a_sell = breakout_down * (1 if model_action == 2 else 0)

        if a_buy == 1:
            return 1  # BUY
        elif a_sell == 1:
            return 2  # SELL
        return 0  # HOLD

    def _get_state_features(self):
        breakout_up, breakout_down = self._get_breakout_flags()
        pos = float(self.position)
        return np.array([pos, float(breakout_up), float(breakout_down)], dtype=np.float32)

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

    # ------------------------------------------------------------------
    # Position Sizing
    # ------------------------------------------------------------------
    def _compute_position_size(self, price: float, atr: float) -> tuple:
        """
        Position sizing: 20% of cash, 1% risk (ATR-based stop), 3% reward.

        risk_amount = equity × risk_per_trade_pct
        stop_distance = atr (or atr-based)
        shares = risk_amount / stop_distance
        cost = shares × price
        if cost > equity × cash_fraction: scale down
        """
        if atr <= 0 or price <= 0:
            return 0, 0.0, 0.0

        max_cash = self.equity * self.cash_fraction
        risk_amount = self.equity * self.risk_per_trade_pct

        # Stop distance based on ATR
        stop_distance = atr
        target_distance = atr * (self.reward_target_pct / self.risk_per_trade_pct)

        if stop_distance <= 0:
            return 0, 0.0, 0.0

        shares = risk_amount / stop_distance
        cost = shares * price

        if cost > max_cash:
            shares = max_cash / price
            cost = max_cash

        return int(shares), float(stop_distance), float(target_distance)

    def _open_trade(self, direction: int):
        price = float(self.df.loc[self.current_step, "Close"])
        atr = float(self.df.loc[self.current_step, "atr_14"]) if "atr_14" in self.df.columns else price * 0.02

        shares, stop_dist, target_dist = self._compute_position_size(price, atr)
        if shares <= 0:
            return

        slip = np.random.uniform(0, self.slippage_pct) * price
        if direction == 1:
            entry = price + slip
            sl_price = entry - stop_dist
            tp_price = entry + target_dist
        else:
            entry = price - slip
            sl_price = entry + stop_dist
            tp_price = entry - target_dist

        cost = shares * entry * (1 + self.commission_pct)
        if cost > self.cash:
            shares = int(self.cash / (entry * (1 + self.commission_pct)))
            if shares <= 0:
                return
            cost = shares * entry * (1 + self.commission_pct)

        self.position = direction
        self.entry_price = entry
        self.shares = shares
        self.sl_price = sl_price
        self.tp_price = tp_price
        self.cash -= cost
        self.time_in_trade = 0

        self._last_trade_open = {
            "event": "OPEN", "direction": direction, "shares": shares,
            "entry": entry, "sl": sl_price, "tp": tp_price,
            "cost": cost, "step": self.current_step,
        }

    def _close_trade(self, reason: str, exit_price: float):
        if self.position == 0 or self.shares == 0:
            return 0.0

        direction = self.position
        gross = self.shares * exit_price * (1 - self.commission_pct)
        cost_basis = self.shares * self.entry_price * (1 + self.commission_pct)
        pnl = (gross - cost_basis) * direction  # positive for profit
        pnl_pct = pnl / cost_basis * 100.0 if cost_basis > 0 else 0.0

        self.cash += gross
        self.equity = self.cash
        if self.position == 1 and self.shares > 0:
            # In case of long position closed, cash already includes proceeds
            pass
        # Recalculate equity: cash + market value of position
        # Since we closed, equity = cash
        old_equity = self.equity_curve[-1] if self.equity_curve else self.initial_equity
        new_equity = self.cash
        ret = (new_equity - old_equity) / old_equity if old_equity > 0 else 0.0
        self.daily_returns.append(ret)

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

        self.position = 0
        self.entry_price = 0.0
        self.shares = 0
        self.sl_price = 0.0
        self.tp_price = 0.0
        self.time_in_trade = 0

        return pnl_pct

    def _check_sl_tp(self) -> float | None:
        if self.position == 0:
            return None

        if self.current_step >= self.n_steps - 2:
            return self._close_trade("END_OF_DATA", float(self.df.loc[self.current_step, "Close"]))

        nh = float(self.df.loc[self.current_step + 1, "High"])
        nl = float(self.df.loc[self.current_step + 1, "Low"])

        if self.position == 1:
            sl_hit = nl <= self.sl_price
            tp_hit = nh >= self.tp_price
            if sl_hit and tp_hit: return self._close_trade("SL_FIRST", self.sl_price)
            if sl_hit: return self._close_trade("SL_HIT", self.sl_price)
            if tp_hit: return self._close_trade("TP_HIT", self.tp_price)
        else:
            sl_hit = nh >= self.sl_price
            tp_hit = nl <= self.tp_price
            if sl_hit and tp_hit: return self._close_trade("SL_FIRST", self.sl_price)
            if sl_hit: return self._close_trade("SL_HIT", self.sl_price)
            if tp_hit: return self._close_trade("TP_HIT", self.tp_price)

        return None

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        if self.random_start:
            mx = self.n_steps - max(self.min_episode_steps, self.window_size) - 2
            self.current_step = (int(np.random.randint(self.window_size, mx)) if mx > self.window_size
                                 else self.window_size)
        else:
            self.current_step = self.window_size
        obs = self._get_observation()
        return (obs, {}) if _GYMNASIUM else obs

    def step(self, action: int):
        if self.terminated or self.truncated:
            obs = self._get_observation()
            return (obs, 0.0, True, False, {}) if _GYMNASIUM else (obs, 0.0, True, {})

        self.steps_in_episode += 1

        # --- Apply breakout filter to action ---
        raw_action = int(action)
        filtered_action = self._apply_breakout_filter(raw_action)

        # --- Record breakout info ---
        breakout_up, breakout_down = self._get_breakout_flags()

        # --- Execute filtered action ---
        if filtered_action == 1 and self.position == 0:    # BUY
            self._open_trade(1)
        elif filtered_action == 2 and self.position == 0:  # SELL
            self._open_trade(-1)

        # --- Check SL/TP ---
        close_pnl = self._check_sl_tp()

        # --- Update equity (mark-to-market for open positions) ---
        if self.position != 0:
            self.time_in_trade += 1
            current_price = float(self.df.loc[self.current_step, "Close"])
            if self.position == 1:
                mtm = self.shares * (current_price - self.entry_price)
            else:
                mtm = self.shares * (self.entry_price - current_price)
            self.equity = self.cash + mtm

        # --- Reward computation ---
        reward = 0.0

        # PnL from closes
        if close_pnl is not None:
            reward += close_pnl

        # Hold duration penalty (discourage indefinite holding)
        if self.position != 0:
            reward -= self.hold_duration_penalty * self.time_in_trade

        # Slippage cost penalty (embedded in open/close, but add small explicit cost)
        # Small reward for being flat in no-breakout days
        if filtered_action == 0 and self.position == 0:
            reward += 0.001  # Tiny bonus for discipline

        # --- Advance ---
        self.current_step += 1

        # --- Termination ---
        if self.current_step >= self.n_steps - 1:
            # Close any open position
            if self.position != 0:
                self._close_trade("END_OF_DATA", float(self.df.loc[self.current_step - 1, "Close"]))
            self.terminated = True

        if self.episode_max_steps and self.steps_in_episode >= self.episode_max_steps:
            self.truncated = True

        if self.peak_equity > 0:
            dd = (self.peak_equity - self.equity) / self.peak_equity
            if dd >= 0.25:
                if self.position != 0:
                    self._close_trade("DRAWDOWN_LIMIT", float(self.df.loc[self.current_step - 1, "Close"]))
                self.terminated = True
                reward -= 5.0

        self.equity_curve.append(float(self.equity))
        obs = self._get_observation()
        reward *= self.reward_scale

        info = {
            "equity": float(self.equity),
            "cash": float(self.cash),
            "position": int(self.position),
            "raw_action": raw_action,
            "filtered_action": filtered_action,
            "breakout_up": breakout_up,
            "breakout_down": breakout_down,
            "time_in_trade": int(self.time_in_trade),
        }

        return (obs, float(reward), self.terminated, self.truncated, info) if _GYMNASIUM else (obs, float(reward), bool(self.terminated or self.truncated), info)

    def render(self):
        ps = {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(self.position, "?")
        bu, bd = self._get_breakout_flags()
        print(f"Step={self.current_step} | Eq=${self.equity:,.0f} | Pos={ps} | "
              f"BO_Up={bu} BO_Dn={bd} | Entry={self.entry_price:.2f} | "
              f"SL={self.sl_price:.2f} TP={self.tp_price:.2f}")
