from __future__ import annotations

import numpy as np
import random

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYMNASIUM = True
except ImportError:
    import gym
    from gym import spaces
    _GYMNASIUM = False


class ForexTradingEnv(gym.Env):
    """
    Reinforcement Learning Forex Trading Environment with EMA Crossover Strategy.

    Key properties:
      - Observation: rolling window of normalized features + state features
      - Actions:
          0: HOLD
          1: CLOSE (close any open position)
          2..: OPEN(direction, SL, TP) — opens a new position when flat
      - Position persistence: once opened, stays until CLOSE or SL/TP hit
      - Enhanced reward with risk-adjusted returns
      - Multiple SL/TP options for flexible risk management
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        window_size: int = 30,
        sl_options=None,
        tp_options=None,
        feature_columns=None,
        pip_value: float = 0.0001,
        spread_pips: float = 1.0,
        commission_pips: float = 0.0,
        max_slippage_pips: float = 0.2,
        lot_size: float = 100000.0,
        reward_scale: float = 1.0,
        unrealized_delta_weight: float = 0.02,
        random_start: bool = True,
        min_episode_steps: int = 300,
        episode_max_steps: int | None = None,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
        allow_flip: bool = False,
        hold_reward_weight: float = 0.005,
        open_penalty_pips: float = 0.5,
        time_penalty_pips: float = 0.02,
        # Enhanced parameters
        position_sizing: float = 1.0,
        max_drawdown_pct: float = 0.30,
        drawdown_penalty_weight: float = 1.0,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)

        if feature_columns is None:
            self.feature_columns = list(self.df.columns)
        else:
            self.feature_columns = list(feature_columns)

        if sl_options is None or tp_options is None:
            raise ValueError("sl_options and tp_options must be provided.")
        self.sl_options = list(sl_options)
        self.tp_options = list(tp_options)

        if self.n_steps <= window_size + 2:
            raise ValueError("Dataframe too short for given window_size.")

        self.window_size = int(window_size)
        self.pip_value = float(pip_value)

        # Friction parameters
        self.spread_pips = float(spread_pips)
        self.commission_pips = float(commission_pips)
        self.max_slippage_pips = float(max_slippage_pips)
        self.lot_size = float(lot_size)
        self.usd_per_pip = self.pip_value * self.lot_size

        # Reward parameters
        self.reward_scale = float(reward_scale)
        self.unrealized_delta_weight = float(unrealized_delta_weight)
        self.hold_reward_weight = float(hold_reward_weight)
        self.open_penalty_pips = float(open_penalty_pips)
        self.time_penalty_pips = float(time_penalty_pips)

        # Enhanced parameters
        self.position_sizing = float(position_sizing)
        self.max_drawdown_pct = float(max_drawdown_pct)
        self.drawdown_penalty_weight = float(drawdown_penalty_weight)

        # Episode handling
        self.random_start = bool(random_start)
        self.min_episode_steps = int(min_episode_steps)
        self.episode_max_steps = (
            episode_max_steps if episode_max_steps is None else int(episode_max_steps)
        )

        # Normalization
        self.feature_mean = feature_mean
        self.feature_std = feature_std

        self.allow_flip = bool(allow_flip)

        # --- Action space ---
        self.action_map = [("HOLD", None, None, None), ("CLOSE", None, None, None)]
        for direction in [0, 1]:  # 0=short, 1=long
            for sl in self.sl_options:
                for tp in self.tp_options:
                    self.action_map.append(("OPEN", direction, float(sl), float(tp)))

        self.action_space = spaces.Discrete(len(self.action_map))
        print(f"[Env] Action space: {len(self.action_map)} actions "
              f"({len(self.sl_options)} SL x {len(self.tp_options)} TP x 2 dirs + HOLD + CLOSE)")

        # Observation space
        self.base_num_features = len(self.feature_columns)
        self.state_num_features = 4  # position, time_in_trade, unrealized_pnl, drawdown
        self.num_features = self.base_num_features + self.state_num_features

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.num_features),
            dtype=np.float32,
        )

        self._reset_state()

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def _reset_state(self):
        self.current_step = 0
        self.steps_in_episode = 0
        self.terminated = False
        self.truncated = False

        self.position = 0
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None
        self.time_in_trade = 0
        self.prev_unrealized_pips = 0.0

        self.initial_equity_usd = 10000.0
        self.equity_usd = self.initial_equity_usd
        self.peak_equity_usd = self.initial_equity_usd
        self.max_drawdown_usd = 0.0

        self.equity_curve = []
        self.trade_history = []
        self.last_trade_info = None

    def _get_state_features(self):
        pos = float(self.position)
        t_norm = float(self.time_in_trade) / 1000.0
        unreal_pips = (
            float(self._compute_unrealized_pips()) if self.position != 0 else 0.0
        )
        unreal_scaled = unreal_pips / 100.0

        # Drawdown ratio
        if self.peak_equity_usd > 0:
            dd_ratio = (self.peak_equity_usd - self.equity_usd) / self.peak_equity_usd
        else:
            dd_ratio = 0.0

        return np.array([pos, t_norm, unreal_scaled, dd_ratio], dtype=np.float32)

    def _compute_unrealized_pips(self):
        if self.position == 0 or self.entry_price is None:
            return 0.0
        close_price = float(self.df.loc[self.current_step, "Close"])
        if self.position == 1:
            return (close_price - self.entry_price) / self.pip_value
        else:
            return (self.entry_price - close_price) / self.pip_value

    def _apply_optional_normalization(self, obs: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_std is None:
            return obs
        # Only normalize the feature columns (first base_num_features), not state features
        n_feat = self.base_num_features
        feat_part = obs[:, :n_feat]
        mean = self.feature_mean.reshape(1, -1)
        std = self.feature_std.reshape(1, -1)
        std = np.where(std == 0, 1.0, std)
        normalized_feat = (feat_part - mean) / std
        obs = obs.copy()
        obs[:, :n_feat] = normalized_feat
        return obs

    def _get_observation(self):
        start = self.current_step - self.window_size
        if start < 0:
            start = 0

        obs_df = self.df.iloc[start : self.current_step].copy()
        obs_df = obs_df[self.feature_columns]

        if len(obs_df) == 0:
            base = np.tile(
                self.df.iloc[0][self.feature_columns].values.astype(np.float32),
                (self.window_size, 1),
            )
        else:
            base = obs_df.values.astype(np.float32)
            if base.shape[0] < self.window_size:
                pad_rows = self.window_size - base.shape[0]
                pad = np.tile(base[0], (pad_rows, 1))
                base = np.vstack([pad, base])

        state_feat = self._get_state_features()
        state_block = np.tile(state_feat, (self.window_size, 1))
        obs = np.hstack([base, state_block]).astype(np.float32)
        obs = self._apply_optional_normalization(obs)
        return obs

    # ------------------------------------------------------------------
    # Trading logic
    # ------------------------------------------------------------------
    def _sample_slippage_pips(self) -> float:
        if self.max_slippage_pips <= 0:
            return 0.0
        return float(np.random.uniform(0.0, self.max_slippage_pips))

    def _cost_pips_round_trip(self) -> float:
        return self.spread_pips + self.commission_pips

    def _open_position(self, direction: int, sl_pips: float, tp_pips: float):
        close_price = float(self.df.loc[self.current_step, "Close"])
        slip_pips = self._sample_slippage_pips()
        slip_price = slip_pips * self.pip_value

        if direction == 1:  # long
            entry = close_price + slip_price
            sl_price = entry - sl_pips * self.pip_value
            tp_price = entry + tp_pips * self.pip_value
            self.position = 1
        else:  # short
            entry = close_price - slip_price
            sl_price = entry + sl_pips * self.pip_value
            tp_price = entry - tp_pips * self.pip_value
            self.position = -1

        self.entry_price = entry
        self.sl_price = sl_price
        self.tp_price = tp_price
        self.time_in_trade = 0
        self.prev_unrealized_pips = 0.0

        self.last_trade_info = {
            "event": "OPEN",
            "step": self.current_step,
            "position": self.position,
            "entry_price": self.entry_price,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
        }

    def _close_position(self, reason: str, exit_price: float):
        if self.position == 1:
            pnl_price = exit_price - self.entry_price
        else:
            pnl_price = self.entry_price - exit_price
        realized_pips = pnl_price / self.pip_value

        cost_pips = self._cost_pips_round_trip()
        net_pips = realized_pips - cost_pips

        self.equity_usd += net_pips * self.usd_per_pip * self.position_sizing

        # Track peak equity and drawdown
        if self.equity_usd > self.peak_equity_usd:
            self.peak_equity_usd = self.equity_usd
        dd = self.peak_equity_usd - self.equity_usd
        if dd > self.max_drawdown_usd:
            self.max_drawdown_usd = dd

        trade_info = {
            "event": "CLOSE",
            "reason": reason,
            "step": self.current_step,
            "position": self.position,
            "entry_price": self.entry_price,
            "exit_price": exit_price,
            "realized_pips": float(realized_pips),
            "cost_pips": float(cost_pips),
            "net_pips": float(net_pips),
            "equity_usd": float(self.equity_usd),
            "time_in_trade": int(self.time_in_trade),
            "drawdown_usd": float(self.max_drawdown_usd),
        }

        self.trade_history.append(trade_info)

        self.position = 0
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None
        self.time_in_trade = 0
        self.prev_unrealized_pips = 0.0

        self.last_trade_info = trade_info
        return net_pips

    def _check_sl_tp_intrabar_and_maybe_close(self) -> float | None:
        if self.position == 0:
            return None

        if self.current_step >= self.n_steps - 2:
            exit_price = float(self.df.loc[self.current_step, "Close"])
            return self._close_position("END_OF_DATA", exit_price)

        next_high = float(self.df.loc[self.current_step + 1, "High"])
        next_low = float(self.df.loc[self.current_step + 1, "Low"])

        if self.position == 1:
            sl_hit = next_low <= self.sl_price
            tp_hit = next_high >= self.tp_price
            if sl_hit and tp_hit:
                return self._close_position("SL_AND_TP_SAME_BAR_SL_FIRST", self.sl_price)
            elif sl_hit:
                return self._close_position("SL_HIT", self.sl_price)
            elif tp_hit:
                return self._close_position("TP_HIT", self.tp_price)
        else:
            sl_hit = next_high >= self.sl_price
            tp_hit = next_low <= self.tp_price
            if sl_hit and tp_hit:
                return self._close_position("SL_AND_TP_SAME_BAR_SL_FIRST", self.sl_price)
            elif sl_hit:
                return self._close_position("SL_HIT", self.sl_price)
            elif tp_hit:
                return self._close_position("TP_HIT", self.tp_price)

        return None

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()

        if self.random_start:
            max_start = (
                self.n_steps - max(self.min_episode_steps, self.window_size) - 2
            )
            if max_start <= self.window_size:
                self.current_step = self.window_size
            else:
                self.current_step = int(
                    np.random.randint(self.window_size, max_start)
                )
        else:
            self.current_step = self.window_size

        self.steps_in_episode = 0
        self.terminated = False
        self.truncated = False

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
        reward_pips = 0.0
        info = {}

        act_type, direction, sl_pips, tp_pips = self.action_map[int(action)]

        # 1) Apply action
        if act_type == "HOLD":
            pass

        elif act_type == "CLOSE":
            if self.position != 0:
                close_price = float(self.df.loc[self.current_step, "Close"])
                slip_pips = self._sample_slippage_pips()
                slip_price = slip_pips * self.pip_value
                exit_price = (
                    close_price - slip_price
                    if self.position == 1
                    else close_price + slip_price
                )
                reward_pips += self._close_position("MANUAL_CLOSE", exit_price)

        elif act_type == "OPEN":
            if self.position == 0:
                self._open_position(
                    direction=direction, sl_pips=sl_pips, tp_pips=tp_pips
                )
                reward_pips -= self.open_penalty_pips
            elif self.allow_flip:
                close_price = float(self.df.loc[self.current_step, "Close"])
                reward_pips += self._close_position("FLIP_CLOSE", close_price)
                self._open_position(
                    direction=direction, sl_pips=sl_pips, tp_pips=tp_pips
                )
                reward_pips -= self.open_penalty_pips

        # 2) Check SL/TP on next bar
        realized_now = self._check_sl_tp_intrabar_and_maybe_close()
        if realized_now is not None:
            reward_pips += realized_now

        # 3) Reward shaping for open positions
        if self.position != 0:
            self.time_in_trade += 1
            unreal_now = self._compute_unrealized_pips()
            delta_unreal = unreal_now - self.prev_unrealized_pips

            # Bonus for holding winning trades
            if unreal_now > 0:
                reward_pips += self.hold_reward_weight * unreal_now

            # Shaping on delta unrealized
            if self.unrealized_delta_weight != 0.0:
                reward_pips += self.unrealized_delta_weight * delta_unreal

            # Time penalty to discourage indefinite holding
            reward_pips -= self.time_penalty_pips

            self.prev_unrealized_pips = unreal_now

        # 4) Advance time
        self.current_step += 1

        # 5) Termination / truncation
        if self.current_step >= self.n_steps - 1:
            self.terminated = True

        if (
            self.episode_max_steps is not None
            and self.steps_in_episode >= self.episode_max_steps
        ):
            self.truncated = True

        # Drawdown-based early termination
        if self.peak_equity_usd > 0:
            current_dd_pct = (
                (self.peak_equity_usd - self.equity_usd) / self.peak_equity_usd
            )
            if current_dd_pct >= self.max_drawdown_pct:
                self.terminated = True
                # Strong penalty for hitting max drawdown
                reward_pips -= self.drawdown_penalty_weight * abs(current_dd_pct * 100)

        # 6) Log equity
        self.equity_curve.append(float(self.equity_usd))

        # Update peak equity
        if self.equity_usd > self.peak_equity_usd:
            self.peak_equity_usd = self.equity_usd

        # 7) Build observation
        obs = self._get_observation()

        # 8) Reward scaling
        reward = float(reward_pips) * self.reward_scale

        # 9) Info dict
        info.update(
            {
                "equity_usd": float(self.equity_usd),
                "position": int(self.position),
                "time_in_trade": int(self.time_in_trade),
                "reward_pips": float(reward_pips),
                "last_trade_info": self.last_trade_info,
                "drawdown_usd": float(self.max_drawdown_usd),
                "peak_equity_usd": float(self.peak_equity_usd),
            }
        )

        if _GYMNASIUM:
            return obs, reward, self.terminated, self.truncated, info
        else:
            done = bool(self.terminated or self.truncated)
            return obs, reward, done, info

    def render(self):
        print(
            f"Step={self.current_step} | Equity=${self.equity_usd:,.2f} | "
            f"Peak=${self.peak_equity_usd:,.2f} | DD=${self.max_drawdown_usd:,.2f} | "
            f"Pos={self.position} | Entry={self.entry_price} | SL={self.sl_price} | TP={self.tp_price}"
        )
