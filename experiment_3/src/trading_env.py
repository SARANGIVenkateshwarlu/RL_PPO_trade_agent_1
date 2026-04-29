"""
Regime-Aware Stock Trading Environment (Experiment 3)

Market Regime Filter:
  - BULL (SPX > 50 EMA & > 150 EMA) → BUY only (long)
  - BEAR (SPX < 50 EMA & < 150 EMA) → SELL only (short)
  - NEUTRAL → HOLD only

Strategy: 20/50 EMA Crossover + BB Squeeze, 1:3 Risk/Reward
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


class RegimeStockEnv(gym.Env):
    """
    Regime-aware stock trading environment.

    Action space (regime-dependent):
      - HOLD (0): always available
      - CLOSE (1): close any open position
      - BUY(sl%, tp%) (2..N): only available in BULL regime; tp = 3*sl (1:3 RR)
      - SELL(sl%, tp%) (N+1..M): only available in BEAR regime; tp = 3*sl (1:3 RR)

    Invalid actions (e.g. BUY in bear regime) are treated as HOLD.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        window_size: int = 30,
        sl_options_pct=None,         # SL percentages [2, 3, 5, 7, 10]
        feature_columns=None,
        commission_pct: float = 0.001,
        max_slippage_pct: float = 0.001,
        reward_scale: float = 1.0,
        random_start: bool = True,
        min_episode_steps: int = 200,
        episode_max_steps: int | None = None,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
        hold_reward_weight: float = 0.02,
        trade_penalty_pct: float = 0.1,
        time_penalty_pct: float = 0.005,
        max_drawdown_pct: float = 0.25,
        drawdown_penalty_weight: float = 2.0,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)

        self.feature_columns = [c for c in (feature_columns or []) if c in self.df.columns]
        if not self.feature_columns:
            raise ValueError("No valid feature columns found.")

        if sl_options_pct is None:
            sl_options_pct = [2, 3, 5, 7, 10]
        self.sl_options_pct = list(sl_options_pct)
        # 1:3 risk/reward → TP = 3 * SL
        self.tp_options_pct = [sl * 3 for sl in self.sl_options_pct]

        if self.n_steps <= window_size + 2:
            raise ValueError("Dataframe too short.")

        self.window_size = int(window_size)
        self.commission_pct = float(commission_pct)
        self.max_slippage_pct = float(max_slippage_pct)
        self.reward_scale = float(reward_scale)
        self.hold_reward_weight = float(hold_reward_weight)
        self.trade_penalty_pct = float(trade_penalty_pct)
        self.time_penalty_pct = float(time_penalty_pct)
        self.max_drawdown_pct = float(max_drawdown_pct)
        self.drawdown_penalty_weight = float(drawdown_penalty_weight)

        self.random_start = bool(random_start)
        self.min_episode_steps = int(min_episode_steps)
        self.episode_max_steps = episode_max_steps

        self.feature_mean = feature_mean
        self.feature_std = feature_std

        # --- Build action map ---
        # 0: HOLD, 1: CLOSE
        # 2..K: BUY(sl%, 3*sl%)
        # K+1..M: SELL(sl%, 3*sl%)
        self.action_map = [("HOLD", None, None), ("CLOSE", None, None)]
        for sl in self.sl_options_pct:
            tp = sl * 3  # 1:3 risk/reward
            self.action_map.append(("BUY", float(sl) / 100, float(tp) / 100))
        for sl in self.sl_options_pct:
            tp = sl * 3
            self.action_map.append(("SELL", float(sl) / 100, float(tp) / 100))

        self.action_space = spaces.Discrete(len(self.action_map))

        self.base_num_features = len(self.feature_columns)
        self.state_num_features = 5  # position, dir, time, unreal, dd
        self.num_features = self.base_num_features + self.state_num_features

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, self.num_features),
            dtype=np.float32,
        )
        self._reset_state()

        n_buy = len(self.sl_options_pct)
        n_sell = len(self.sl_options_pct)
        print(f"[RegimeEnv] {len(self.action_map)} actions: HOLD, CLOSE, "
              f"BUY({n_buy} SL, 1:3 RR), SELL({n_sell} SL, 1:3 RR)")
        print(f"[RegimeEnv] SL options: {self.sl_options_pct}% | TP = 3x SL")

    # ------------------------------------------------------------------
    def _reset_state(self):
        self.current_step = 0
        self.steps_in_episode = 0
        self.terminated = False
        self.truncated = False

        self.position = 0       # 0=flat, 1=long, -1=short
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None
        self.time_in_trade = 0
        self.prev_unreal_pct = 0.0

        self.initial_equity = 100000.0
        self.equity = self.initial_equity
        self.peak_equity = self.initial_equity

        self.equity_curve = []
        self.trade_history = []
        self.last_trade_info = None

    def _get_regime(self) -> int:
        """Get market regime at current step: 1=BULL, -1=BEAR, 0=NEUTRAL."""
        if "regime" in self.df.columns:
            return int(self.df.loc[self.current_step, "regime"])
        return 0

    def _get_state_features(self):
        pos = float(self.position)
        direction = 0.0 if self.position == 0 else (1.0 if self.position == 1 else -1.0)
        t_norm = float(self.time_in_trade) / 252.0

        if self.position != 0 and self.entry_price:
            close = float(self.df.loc[self.current_step, "Close"])
            unreal = (close - self.entry_price) / self.entry_price * 100 * self.position
        else:
            unreal = 0.0
        unreal_s = unreal / 20.0

        dd = (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        return np.array([pos, direction, t_norm, unreal_s, dd], dtype=np.float32)

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
    def _open_trade(self, direction: int, sl_pct: float, tp_pct: float):
        close = float(self.df.loc[self.current_step, "Close"])
        slip = np.random.uniform(0, self.max_slippage_pct) * close

        if direction == 1:  # Long
            entry = close + slip
            self.sl_price = entry * (1.0 - sl_pct)
            self.tp_price = entry * (1.0 + tp_pct)
        else:  # Short
            entry = close - slip
            self.sl_price = entry * (1.0 + sl_pct)
            self.tp_price = entry * (1.0 - tp_pct)

        self.position = direction
        self.entry_price = entry
        self.time_in_trade = 0
        self.prev_unreal_pct = 0.0

        self.last_trade_info = {
            "event": "BUY" if direction == 1 else "SELL",
            "step": self.current_step,
            "direction": direction,
            "entry": entry, "sl": self.sl_price, "tp": self.tp_price,
        }

    def _close_trade(self, reason: str, exit_price: float):
        pnl_pct = (exit_price - self.entry_price) / self.entry_price * self.position
        cost_pct = self.commission_pct * 2
        net_pct = pnl_pct - cost_pct

        old_eq = self.equity
        self.equity *= (1.0 + net_pct)
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        info = {
            "event": "CLOSE", "reason": reason, "step": self.current_step,
            "direction": self.position,
            "entry": self.entry_price, "exit": exit_price,
            "pnl_pct": float(pnl_pct * 100),
            "net_pct": float(net_pct * 100),
            "equity": float(self.equity),
            "time_in_trade": int(self.time_in_trade),
        }
        self.trade_history.append(info)

        self.position = 0
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None
        self.time_in_trade = 0
        self.prev_unreal_pct = 0.0
        self.last_trade_info = info
        return net_pct * 100

    def _check_sl_tp(self) -> float | None:
        if self.position == 0:
            return None
        if self.current_step >= self.n_steps - 2:
            return self._close_trade("END_OF_DATA", float(self.df.loc[self.current_step, "Close"]))

        nh = float(self.df.loc[self.current_step + 1, "High"])
        nl = float(self.df.loc[self.current_step + 1, "Low"])

        if self.position == 1:
            sl = nl <= self.sl_price
            tp = nh >= self.tp_price
            if sl and tp: return self._close_trade("SL_FIRST", self.sl_price)
            if sl: return self._close_trade("SL_HIT", self.sl_price)
            if tp: return self._close_trade("TP_HIT", self.tp_price)
        else:
            sl = nh >= self.sl_price
            tp = nl <= self.tp_price
            if sl and tp: return self._close_trade("SL_FIRST", self.sl_price)
            if sl: return self._close_trade("SL_HIT", self.sl_price)
            if tp: return self._close_trade("TP_HIT", self.tp_price)
        return None

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
        reward = 0.0
        info = {}
        regime = self._get_regime()

        act_type, sl_pct, tp_pct = self.action_map[int(action)]

        # --- Apply action based on regime ---
        if act_type == "HOLD":
            pass

        elif act_type == "CLOSE":
            if self.position != 0:
                close = float(self.df.loc[self.current_step, "Close"])
                slip = np.random.uniform(0, self.max_slippage_pct) * close
                exit_p = close - slip if self.position == 1 else close + slip
                reward += self._close_trade("MANUAL_CLOSE", exit_p)

        elif act_type == "BUY":
            if regime == 1 and self.position == 0:
                # Only BUY in BULL regime
                self._open_trade(1, sl_pct, tp_pct)
                reward -= self.trade_penalty_pct
            # If regime != BULL, BUY action is ignored (treated as HOLD)

        elif act_type == "SELL":
            if regime == -1 and self.position == 0:
                # Only SELL in BEAR regime
                self._open_trade(-1, sl_pct, tp_pct)
                reward -= self.trade_penalty_pct
            # If regime != BEAR, SELL action is ignored

        # --- Check SL/TP ---
        realized = self._check_sl_tp()
        if realized is not None:
            reward += realized

        # --- Reward shaping while in trade ---
        if self.position != 0:
            self.time_in_trade += 1
            close = float(self.df.loc[self.current_step, "Close"])
            unreal = (close - self.entry_price) / self.entry_price * 100 * self.position
            delta = unreal - self.prev_unreal_pct

            if unreal > 0:
                reward += self.hold_reward_weight * unreal
            reward -= self.time_penalty_pct
            self.prev_unreal_pct = unreal

        # --- Advance ---
        self.current_step += 1

        # --- Termination ---
        if self.current_step >= self.n_steps - 1:
            self.terminated = True
        if self.episode_max_steps and self.steps_in_episode >= self.episode_max_steps:
            self.truncated = True
        if self.peak_equity > 0:
            dd = (self.peak_equity - self.equity) / self.peak_equity
            if dd >= self.max_drawdown_pct:
                self.terminated = True
                reward -= self.drawdown_penalty_weight * dd * 100
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        self.equity_curve.append(float(self.equity))
        obs = self._get_observation()
        reward *= self.reward_scale

        info.update({
            "equity": float(self.equity),
            "position": int(self.position),
            "regime": regime,
            "time_in_trade": int(self.time_in_trade),
        })

        return (obs, float(reward), self.terminated, self.truncated, info) if _GYMNASIUM else (obs, float(reward), bool(self.terminated or self.truncated), info)

    def render(self):
        r = self._get_regime()
        rl = {1: "BULL", -1: "BEAR", 0: "NEUTRAL"}.get(r, "?")
        ps = {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(self.position, "?")
        print(f"Step={self.current_step} | Regime={rl} | Pos={ps} | "
              f"Equity=${self.equity:,.0f} | Entry={self.entry_price} | SL={self.sl_price} | TP={self.tp_price}")
