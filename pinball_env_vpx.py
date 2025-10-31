
import time
from typing import Optional, Sequence, Tuple, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import requests

class PinballEnv(gym.Env):
    """
    VPX-backed Gymnasium env.
    Expects a local "bridge" HTTP server that VPX posts state to and that
    returns actions in response (see vpx_bridge.py).
    """
    metadata = {"render_modes": ["human", "rgb_array", None], "render_fps": 60}

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:5000",
        render_mode: Optional[str] = None,
        table_size: Tuple[float, float] = (500.0, 1000.0),
        max_balls: int = 1,
        obs_features: Sequence[str] = (
            "ball_x", "ball_y", "ball_vx", "ball_vy", "ball_speed",
            "on_playfield", "score", "tilt_warning"
        ),
        action_schema: str = "binary_flippers",
        nudge_cooldown: float = 0.25,
        frame_skip: int = 1,
        dt: float = 1.0 / 60.0,
        seed: Optional[int] = None,
        feature_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        reset_timeout_s: float = 6.0,
        step_wait_s: float = 0.0,   # extra wait after sending action (in addition to frame_skip*dt)
    ) -> None:
        super().__init__()
        self.server_url = server_url.rstrip("/")
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be one of {self.metadata['render_modes']}; got {render_mode}")
        self.render_mode = render_mode

        self.table_w, self.table_h = float(table_size[0]), float(table_size[1])
        self.max_balls = int(max_balls)
        self.obs_features = tuple(obs_features)
        self.action_schema = action_schema
        if self.action_schema not in ("binary_flippers", "continuous_flippers"):
            raise ValueError("action_schema must be 'binary_flippers' or 'continuous_flippers'.")
        self.nudge_cooldown = float(nudge_cooldown)
        self._last_nudge_ts = -1e9
        self.frame_skip = int(frame_skip)
        self.dt = float(dt)
        self.step_wait_s = float(step_wait_s)
        self.reset_timeout_s = float(reset_timeout_s)

        self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.feature_bounds = feature_bounds or self._default_feature_bounds(self.table_w, self.table_h, self.dt)
        self._validate_feature_bounds(self.obs_features, self.feature_bounds)

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(len(self.obs_features),), dtype=np.float32)

        if self.action_schema == "binary_flippers":
            self.action_space = spaces.MultiBinary(3)  # [L,R,nudge]
            self._action_desc = ("left_flipper", "right_flipper", "nudge")
        else:
            low = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            high = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
            self._action_desc = ("left_force", "right_force", "nudge_impulse")

        self._episode_steps = 0
        self._last_obs: Optional[np.ndarray] = None
        self._last_score: float = 0.0
        self._done = False

    # -------------------- bounds + normalization --------------------
    def _default_feature_bounds(self, w: float, h: float, dt: float) -> Dict[str, Tuple[float, float]]:
        vmax = max(w, h) / dt
        return {
            "ball_x": (0.0, w),
            "ball_y": (0.0, h),
            "ball_vx": (-vmax, vmax),
            "ball_vy": (-vmax, vmax),
            "ball_speed": (0.0, vmax * 1.5),
            "on_playfield": (0.0, 1.0),
            "score": (0.0, 1e8),
            "tilt_warning": (0.0, 1.0),
        }

    def _validate_feature_bounds(self, features: Sequence[str], bounds: Dict[str, Tuple[float, float]]) -> None:
        for f in features:
            if f not in bounds:
                raise KeyError(f"feature_bounds missing entry for '{f}'.")
            lo, hi = bounds[f]
            if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
                raise ValueError(f"feature_bounds[{f}] must be finite with hi > lo (got {lo}, {hi}).")

    def normalize_feature(self, name: str, value: float) -> float:
        lo, hi = self.feature_bounds[name]
        x = (value - lo) / (hi - lo)
        x = np.clip(x, 0.0, 1.0)
        return float(x * 2.0 - 1.0)

    def pack_observation(self, raw: Dict[str, float]) -> np.ndarray:
        obs = np.empty((len(self.obs_features),), dtype=np.float32)
        for i, name in enumerate(self.obs_features):
            obs[i] = self.normalize_feature(name, raw[name])
        return obs

    # -------------------- HTTP helpers --------------------
    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(self.server_url + path, json=payload, timeout=2.5)
        r.raise_for_status()
        return r.json() if r.content else {}

    def _get(self, path: str) -> Dict[str, Any]:
        r = requests.get(self.server_url + path, timeout=2.5)
        r.raise_for_status()
        return r.json() if r.content else {}

    # -------------------- Gym API --------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self._episode_steps = 0
        self._done = False
        self._last_score = 0.0

        # Tell bridge to start a new ball/game
        self._post("/control", {"cmd": "new_ball"})

        # Wait for VPX to report a valid state
        t0 = time.time()
        raw = None
        while time.time() - t0 < self.reset_timeout_s:
            try:
                state = self._get("/last_state")
                if state and "raw" in state:
                    raw = state["raw"]
                    # Require at least these fields
                    if all(k in raw for k in self.obs_features):
                        break
            except Exception:
                pass
            time.sleep(0.02)

        if raw is None:
            raise RuntimeError("Timeout waiting for VPX state during reset(). Is the VPX table posting to /state?")

        obs = self.pack_observation(raw)
        self._last_obs = obs
        self._last_score = float(raw.get("score", 0.0))
        info = {"raw": raw}
        return obs, info

    def step(self, action):
        if self._done:
            raise RuntimeError("Call reset() before step() after episode end.")

        action_payload = self._encode_action(action)
        # Send the action that VPX should apply for the next tick(s)
        self._post("/act", action_payload)

        # Wait for new state after frame_skip * dt
        time.sleep(max(0.0, self.frame_skip * self.dt + self.step_wait_s))
        state = self._get("/last_state")
        if not state or "raw" not in state:
            raise RuntimeError("No state available from bridge (/last_state).")
        raw = state["raw"]

        obs = self.pack_observation(raw)
        score = float(raw.get("score", 0.0))
        reward = score - self._last_score  # dense reward = score delta
        self._last_score = score

        on_pf = int(raw.get("on_playfield", 1))  # 1 if ball in play; 0 if drained
        tilt_warn = int(raw.get("tilt_warning", 0))

        terminated = (on_pf == 0)  # ball drained
        truncated = False          # you can set a max step cutoff if desired

        self._episode_steps += 1
        self._done = terminated or truncated
        info = {"raw": raw, "action_desc": self._action_desc}
        self._last_obs = obs
        return obs, float(reward), terminated, truncated, info

    # -------------------- action encoding --------------------
    def _encode_action(self, action):
        if self.action_schema == "binary_flippers":
            # Expect MultiBinary(3): [L, R, nudge]
            a = np.asarray(action, dtype=np.int32).clip(0, 1)
            left, right, nudge = int(a[0]), int(a[1]), int(a[2])
            now = time.time()
            do_nudge = 0
            if nudge and (now - self._last_nudge_ts) >= self.nudge_cooldown:
                self._last_nudge_ts = now
                do_nudge = 1
            return {"left": left, "right": right, "nudge": do_nudge, "duration_s": self.frame_skip * self.dt}
        else:
            a = np.asarray(action, dtype=np.float32)
            left_f = float(np.clip(a[0], 0.0, 1.0))
            right_f = float(np.clip(a[1], 0.0, 1.0))
            nudge_imp = float(np.clip(a[2], -1.0, 1.0))
            now = time.time()
            nudge_out = 0.0
            if abs(nudge_imp) > 1e-6 and (now - self._last_nudge_ts) >= self.nudge_cooldown:
                self._last_nudge_ts = now
                nudge_out = nudge_imp
            return {"left_f": left_f, "right_f": right_f, "nudge_imp": nudge_out, "duration_s": self.frame_skip * self.dt}

    def render(self):
        if self.render_mode is None:
            return None
        # Hand rendering to VPX window; optionally grab a screen capture here.
        return None

    def close(self):
        try:
            self._post("/control", {"cmd": "end_episode"})
        except Exception:
            pass
