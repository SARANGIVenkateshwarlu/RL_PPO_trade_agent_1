"""
Stock Trading Environment (Buy-Only / Long-Only)
Uses EMA crossover + Bollinger Band squeeze as core strategy.

Key properties:
  - Buy-only: No short selling (action space limits to HOLD + BUY with SL/TP)
  - Position persistence: position held until CLOSE, SL hit, or TP hit
  - BB squeeze integration: lower volatility periods incentivized
  - Reward: realized PnL + shaping for holding winners
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


class StockTradingEnv(gym.Env):
    """
    RL Stock Trading Environment — Buy-Only with BB Squeeze + EMA Crossover.

    Observations:
      - Rolling window of normalized features
      - State features: position, time_in_trade, unrealized_pnl%, drawdown%

    Actions (buy-only):
      0: HOLD
      1: CLOSE (close if holding)
      2..N: BUY with SL% and TP%

    Position persistence: once bought, held until CLOSE, SL, or TP.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        window_size: int = 30,
        sl_options_pct=None,       # Stop loss as % of entry (e.g., [2, 3, 5, 7, 10])
        tp_options_pct=None,       # Take profit as % of entry (e.g., [3, 5, 8, 12, 15])
        feature_columns=None,
        commission_pct: float = 0.001,   # 0.1% commission per trade
        max_slippage_pct: float = 0.001, # 0.1% max slippage
        reward_scale: float = 1.0,
        random_start: bool = True,
        min_episode_steps: int = 200,
        episode_max_steps: int | None = None,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
        hold_reward_weight: float = 0.01,
        buy_penalty_pct: float = 0.1,
        time_penalty_pct: float = 0.01,
        max_drawdown_pct: float = 0.25,
        drawdown_penalty_weight: float = 2.0,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)

        if feature_columns is None:
            self.feature_columns = list(self.df.columns)
        else:
            self.feature_columns = [c for c in feature_columns if c in self.df.columns]

        if sl_options_pct is None or tp_options_pct is None:
            raise ValueError("sl_options_pct and tp_options_pct are required.")
        self.sl_options_pct = list(sl_options_pct)
        self.tp_options_pct = list(tp_options_pct)

        if self.n_steps <= window_size + 2:
            raise ValueError("Dataframe too short.")

        self.window_size = int(window_size)

        # Costs
        self.commission_pct = float(commission_pct)
        self.max_slippage_pct = float(max_slippage_pct)

        # Reward params
        self.reward_scale = float(reward_scale)
        self.hold_reward_weight = float(hold_reward_weight)
        self.buy_penalty_pct = float(buy_penalty_pct)
        self.time_penalty_pct = float(time_penalty_pct)
        self.max_drawdown_pct = float(max_drawdown_pct)
        self.drawdown_penalty_weight = float(drawdown_penalty_weight)

        # Episode
        self.random_start = bool(random_start)
        self.min_episode_steps = int(min_episode_steps)
        self.episode_max_steps = (
            episode_max_steps if episode_max_steps is None else int(episode_max_steps)
        )

        # Normalization
        self.feature_mean = feature_mean
        self.feature_std = feature_std

        # Action space: HOLD(0), CLOSE(1), BUY(sl, tp)...
        self.action_map = [("HOLD", None, None)]
        for sl in self.sl_options_pct:
            for tp in self.tp_options_pct:
                self.action_map.append(("BUY", float(sl) / 100.0, float(tp) / 100.0))
        self.action_map.append(("CLOSE", None, None))

        self.action_space = spaces.Discrete(len(self.action_map))

        # Observation
        self.base_num_features = len(self.feature_columns)
        self.state_num_features = 4
        self.num_features = self.base_num_features + self.state_num_features

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.num_features),
            dtype=np.float32,
        )

        self._reset_state()

        print(
            f"[StockEnv] {len(self.action_map)} actions "
            f"(BUY: {len(self.sl_options_pct)} SL x {len(self.tp_options_pct)} TP + HOLD + CLOSE)"
        )

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def _reset_state(self):
        self.current_step = 0
        self.steps_in_episode = 0
        self.terminated = False
        self.truncated = False

        self.position = 0      # 0=flat, 1=long
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None
        self.time_in_trade = 0
        self.prev_unrealized_pct = 0.0

        self.initial_equity = 100000.0
        self.equity = self.initial_equity
        self.peak_equity = self.initial_equity

        self.equity_curve = []
        self.trade_history = []
        self.last_trade_info = None

    def _get_state_features(self):
        pos = float(self.position)
        t_norm = float(self.time_in_trade) / 252.0  # Days in trade / trading days per year

        if self.position == 1 and self.entry_price:
            close = float(self.df.loc[self.current_step, "Close"])
            unreal = (close - self.entry_price) / self.entry_price * 100  # % profit
        else:
            unreal = 0.0
        unreal_scaled = unreal / 20.0  # Scale to ~[-1, 1] range

        if self.peak_equity > 0:
            dd = (self.peak_equity - self.equity) / self.peak_equity
        else:
            dd = 0.0

        return np.array([pos, t_norm, unreal_scaled, dd], dtype=np.float32)

    def _get_observation(self):
        start = self.current_step - self.window_size
        if start < 0:
            start = 0

        obs_df = self.df.iloc[start : self.current_step]
        obs_data = obs_df[self.feature_columns].values.astype(np.float32)

        if len(obs_data) == 0:
            row = self.df.iloc[0][self.feature_columns].values.astype(np.float32)
            base = np.tile(row, (self.window_size, 1))
        elif obs_data.shape[0] < self.window_size:
            pad_rows = self.window_size - obs_data.shape[0]
            pad = np.tile(obs_data[0], (pad_rows, 1))
            base = np.vstack([pad, obs_data])
        else:
            base = obs_data

        state_feat = self._get_state_features()
        state_block = np.tile(state_feat, (self.window_size, 1))
        obs = np.hstack([base, state_block]).astype(np.float32)

        if self.feature_mean is not None and self.feature_std is not None:
            n_feat = self.base_num_features
            mean = self.feature_mean.reshape(1, -1)
            std = self.feature_std.reshape(1, -1)
            std = np.where(std == 0, 1.0, std)
            obs[:, :n_feat] = (obs[:, :n_feat] - mean) / std

        return obs

    # ------------------------------------------------------------------
    # Trading logic
    # ------------------------------------------------------------------
    def _open_long(self, sl_pct: float, tp_pct: float):
        close = float(self.df.loc[self.current_step, "Close"])
        slip = np.random.uniform(0, self.max_slippage_pct) * close
        entry = close + slip

        self.position = 1
        self.entry_price = entry
        self.sl_price = entry * (1.0 - sl_pct)
        self.tp_price = entry * (1.0 + tp_pct)
        self.time_in_trade = 0
        self.prev_unrealized_pct = 0.0

        self.last_trade_info = {
            "event": "BUY",
            "step": self.current_step,
            "entry_price": self.entry_price,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
        }

    def _close_long(self, reason: str, exit_price: float):
        pnl_pct = (exit_price - self.entry_price) / self.entry_price

        # Commission (round-trip)
        cost_pct = self.commission_pct * 2  # Buy + sell commission
        net_pct = pnl_pct - cost_pct

        # Update equity
        old_equity = self.equity
        self.equity *= (1.0 + net_pct)

        # Track peak
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        trade_info = {
            "event": "CLOSE",
            "reason": reason,
            "step": self.current_step,
            "entry_price": self.entry_price,
            "exit_price": exit_price,
            "pnl_pct": float(pnl_pct * 100),
            "cost_pct": float(cost_pct * 100),
            "net_pct": float(net_pct * 100),
            "equity": float(self.equity),
            "time_in_trade": int(self.time_in_trade),
        }

        self.trade_history.append(trade_info)

        self.position = 0
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None
        self.time_in_trade = 0
        self.prev_unrealized_pct = 0.0

        self.last_trade_info = trade_info
        return net_pct * 100  # Return % as pips-like units

    def _check_sl_tp_and_maybe_close(self) -> float | None:
        if self.position == 0:
            return None

        # End of data
        if self.current_step >= self.n_steps - 2:
            exit_price = float(self.df.loc[self.current_step, "Close"])
            return self._close_long("END_OF_DATA", exit_price)

        # Check next bar's high/low against SL/TP
        next_high = float(self.df.loc[self.current_step + 1, "High"])
        next_low = float(self.df.loc[self.current_step + 1, "Low"])

        sl_hit = next_low <= self.sl_price
        tp_hit = next_high >= self.tp_price

        if sl_hit and tp_hit:
            return self._close_long("SL_AND_TP_SAME_BAR_SL_FIRST", self.sl_price)
        elif sl_hit:
            return self._close_long("SL_HIT", self.sl_price)
        elif tp_hit:
            return self._close_long("TP_HIT", self.tp_price)

        return None

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()

        if self.random_start:
            max_start = self.n_steps - max(self.min_episode_steps, self.window_size) - 2
            if max_start <= self.window_size:
                self.current_step = self.window_size
            else:
                self.current_step = int(np.random.randint(self.window_size, max_start))
        else:
            self.current_step = self.window_size

        obs = self._get_observation()
        if _GYMNASIUM:
            return obs, {}
        return obs

    def step(self, action: int):
        if self.terminated or self.truncated:
            obs = self._get_observation()
            if _GYMNASIUM:
                return obs, 0.0, True, False, {}
            return obs, 0.0, True, {}

        self.steps_in_episode += 1
        reward = 0.0
        info = {}

        act_type, sl_pct, tp_pct = self.action_map[int(action)]

        # 1) Apply action
        if act_type == "HOLD":
            pass

        elif act_type == "CLOSE":
            if self.position == 1:
                close_price = float(self.df.loc[self.current_step, "Close"])
                slip = np.random.uniform(0, self.max_slippage_pct) * close_price
                exit_price = close_price - slip
                reward += self._close_long("MANUAL_CLOSE", exit_price)

        elif act_type == "BUY":
            if self.position == 0:
                self._open_long(sl_pct, tp_pct)
                reward -= self.buy_penalty_pct  # Small penalty for buying
            # If already holding, BUY is ignored (no flipping for buy-only)

        # 2) Check SL/TP
        realized = self._check_sl_tp_and_maybe_close()
        if realized is not None:
            reward += realized

        # 3) Reward shaping while holding
        if self.position == 1:
            self.time_in_trade += 1
            close = float(self.df.loc[self.current_step, "Close"])
            unreal = (close - self.entry_price) / self.entry_price * 100
            delta = unreal - self.prev_unrealized_pct

            # Bonus for holding winners
            if unreal > 0:
                reward += self.hold_reward_weight * unreal

            # Time penalty
            reward -= self.time_penalty_pct

            self.prev_unrealized_pct = unreal

        # 4) Advance time
        self.current_step += 1

        # 5) Termination
        if self.current_step >= self.n_steps - 1:
            self.terminated = True

        if self.episode_max_steps and self.steps_in_episode >= self.episode_max_steps:
            self.truncated = True

        # Drawdown-based termination
        if self.peak_equity > 0:
            dd = (self.peak_equity - self.equity) / self.peak_equity
            if dd >= self.max_drawdown_pct:
                self.terminated = True
                reward -= self.drawdown_penalty_weight * dd * 100

        # Track peak
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        # 6) Log
        self.equity_curve.append(float(self.equity))

        # 7) Observation
        obs = self._get_observation()

        reward *= self.reward_scale

        info.update(
            {
                "equity": float(self.equity),
                "position": int(self.position),
                "time_in_trade": int(self.time_in_trade),
                "last_trade_info": self.last_trade_info,
            }
        )

        if _GYMNASIUM:
            return obs, float(reward), self.terminated, self.truncated, info
        else:
            done = bool(self.terminated or self.truncated)
            return obs, float(reward), done, info

    def render(self):
        pos_str = "LONG" if self.position == 1 else "FLAT"
        print(
            f"Step={self.current_step} | Equity=${self.equity:,.2f} | "
            f"Pos={pos_str} | Entry={self.entry_price} | SL={self.sl_price} | TP={self.tp_price}"
        )
