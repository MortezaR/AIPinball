
import time
from typing import Optional, Sequence, Tuple, Dict, Any, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import requests

class PinballEnv(gym.Env):
    """
    VPX-backed Gymnasium env (robust key handling).
    - Accepts multiple possible raw key names from VPX and computes fallbacks.
    - Prints a one-time warning listing which keys were missing & how they were resolved.
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
        step_wait_s: float = 0.0,
        feature_aliases: Optional[Dict[str, List[str]]] = None,
        verbose_missing_once: bool = True,
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

        # alias map for flexible input keys
        default_aliases = {
            "ball_x": ["ball_x", "x", "X", "BallX"],
            "ball_y": ["ball_y", "y", "Y", "BallY"],
            "ball_vx": ["ball_vx", "vx", "VelX", "ball_vel_x"],
            "ball_vy": ["ball_vy", "vy", "VelY", "ball_vel_y"],
            "ball_speed": ["ball_speed", "speed", "Speed"],
            "on_playfield": ["on_playfield", "in_play", "onpf", "OnPlayfield"],
            "score": ["score", "Score", "POINTS"],
            "tilt_warning": ["tilt_warning", "tilt", "TiltWarning", "Tilt"],
        }
        if feature_aliases:
            for k, v in feature_aliases.items():
                default_aliases[k] = v + default_aliases.get(k, [])
        self.feature_aliases = default_aliases

        self._episode_steps = 0
        self._last_obs: Optional[np.ndarray] = None
        self._last_score: float = 0.0
        self._done = False
        self._warn_once_done = not verbose_missing_once

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

    # Resolve a raw numeric value for a logical feature name.
    def _resolve_raw(self, raw: Dict[str, Any], name: str) -> float:
        # direct/alias lookups
        for k in self.feature_aliases.get(name, [name]):
            if k in raw:
                try:
                    return float(raw[k])
                except Exception:
                    pass
        # fallbacks
        if name == "ball_speed":
            vx = self._resolve_raw(raw, "ball_vx")
            vy = self._resolve_raw(raw, "ball_vy")
            return float(np.hypot(vx, vy))
        if name == "on_playfield":
            # if we can read Y within table, assume in play
            y = None
            for k in self.feature_aliases.get("ball_y", ["ball_y"]):
                if k in raw:
                    try:
                        y = float(raw[k])
                        break
                    except Exception:
                        pass
            if y is not None and (0.0 <= y <= self.table_h * 1.2):
                return 1.0
            return 0.0
        if name == "tilt_warning":
            return 0.0
        if name == "score":
            return 0.0
        if name in ("ball_x", "ball_y", "ball_vx", "ball_vy"):
            return 0.0
        raise KeyError(name)

    def pack_observation(self, raw: Dict[str, Any]) -> np.ndarray:
        obs = np.empty((len(self.obs_features),), dtype=np.float32)
        missing = []
        resolved = {}
        for i, name in enumerate(self.obs_features):
            try:
                val = self._resolve_raw(raw, name)
                resolved[name] = val
                obs[i] = self.normalize_feature(name, val)
            except Exception:
                missing.append(name)
                obs[i] = 0.0
        if missing and not self._warn_once_done:
            self._warn_once_done = True
            have = ", ".join(sorted(raw.keys()))
            need = ", ".join(self.obs_features)
            miss = ", ".join(missing)
            print(f"[PinballEnv] Warning: missing features: {miss}. Raw keys present: {have}. "
                  f"Using fallbacks/zeros where possible.")
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
        self._warn_once_done = False

        self._post("/control", {"cmd": "new_ball"})

        t0 = time.time()
        raw = None
        while time.time() - t0 < self.reset_timeout_s:
            try:
                state = self._get("/last_state")
                if state and "raw" in state:
                    raw = state["raw"]
                    break
            except Exception:
                pass
            time.sleep(0.02)

        if raw is None:
            raise RuntimeError("Timeout waiting for VPX state during reset(). Is the VPX table posting to /state?")

        obs = self.pack_observation(raw)
        self._last_obs = obs
        self._last_score = float(self._resolve_raw(raw, "score"))
        info = {"raw": raw}
        return obs, info

    def step(self, action):
        if self._done:
            raise RuntimeError("Call reset() before step() after episode end.")

        self._post("/act", self._encode_action(action))

        time.sleep(max(0.0, self.frame_skip * self.dt + self.step_wait_s))
        state = self._get("/last_state")
        if not state or "raw" not in state:
            raise RuntimeError("No state available from bridge (/last_state).")
        raw = state["raw"]

        obs = self.pack_observation(raw)
        score = float(self._resolve_raw(raw, "score"))
        reward = score - self._last_score
        self._last_score = score

        on_pf = int(round(self._resolve_raw(raw, "on_playfield")))
        terminated = (on_pf == 0)
        truncated = False

        self._episode_steps += 1
        self._done = terminated or truncated
        info = {"raw": raw, "action_desc": self._action_desc}
        self._last_obs = obs
        return obs, float(reward), terminated, truncated, info

    # -------------------- action encoding --------------------
    def _encode_action(self, action):
        if self.action_schema == "binary_flippers":
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
        return None

    def close(self):
        try:
            self._post("/control", {"cmd": "end_episode"})
        except Exception:
            pass
