import time
from typing import Optional, Sequence, Tuple, Dict, Any, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import requests


class PinballEnv(gym.Env):

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:5000",
        table_size: Tuple[float, float] = (952.0, 2162.0),
        obs_features: Sequence[str] = (
            "ball_x", "ball_y", "ball_vx", "ball_vy",
        ),
        dt: float = 1.0 / 60.0,
    ) -> None:
        super().__init__()
        self.server_url = server_url.rstrip("/")

        self.table_w, self.table_h = float(table_size[0]), float(table_size[1])
        self.obs_features = tuple(obs_features)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(len(self.obs_features),),
            dtype=np.float32,
        )

        self.dt = float(dt)
        self.feature_bounds = self._default_feature_bounds(
            self.table_w, self.table_h, self.dt
        )
        # action: left, right
        self.action_space = spaces.MultiBinary(2)
        self.old_score = 0

        self._episode_steps = 0
        self._done = False

    # ----------------------------- feature bounds -----------------------------
    def _default_feature_bounds(
        self, w: float, h: float, dt: float
    ) -> Dict[str, Tuple[float, float]]:
        vmax = max(w, h) / dt
        return {
            "ball_x": (0.0, w),
            "ball_y": (0.0, h),
            "ball_vx": (-vmax, vmax),
            "ball_vy": (-vmax, vmax),
        }
    def normalize_feature(self, name: str, value: float) -> float:
        lo, hi = self.feature_bounds[name]
        x = (value - lo) / (hi - lo)
        return float(np.clip(x, 0.0, 1.0) * 2.0 - 1.0)

    # ----------------------------- raw resolution -----------------------------
    def _resolve_raw(self, raw: Dict[str, Any], name: str) -> float:
        if name in raw:
            return float(raw[name])
        if name in ("score", "ball_x", "ball_y", "ball_vx", "ball_vy", "drain_got_hit"):
            return 0.0
        raise KeyError(name)

    def pack_observation(self, raw: Dict[str, Any]) -> np.ndarray:
        obs = np.empty((len(self.obs_features),), dtype=np.float32)
        missing = []
        for i, name in enumerate(self.obs_features):
            try:
                obs[i] = self.normalize_feature(name, self._resolve_raw(raw, name))
            except Exception:
                obs[i] = 0.0
                missing.append(name)
        if missing:
            print(f"[PinballEnv] Warning: missing features {missing}. Keys: {list(raw.keys())}")
        return obs

    # ----------------------------- HTTP helpers -----------------------------
    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(self.server_url + path, json=payload, timeout=2.5)
        r.raise_for_status()
        return r.json() if r.content else {}

    def _get(self, path: str) -> Dict[str, Any]:
        r = requests.get(self.server_url + path, timeout=2.5)
        r.raise_for_status()
        return r.json() if r.content else {}

    #----------------------------- Reset -----------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset()
        self._episode_steps = 0        # track steps
        self.old_score = 0
        self._done = False

        state = self._get("/last_state")  # or "/last_state" depending on how your bridge works
        raw = state["raw"]
        obs = self.pack_observation(raw)
        info = {}
        return obs, info


    # ----------------------------- Step -----------------------------
    def step(self, action):

        # send action to bridge
        self._post("/act", self._encode_action(action))
        time.sleep(max(0.0, self.dt))

        # pull latest state
        state = self._get("/last_state")
        if not state or "raw" not in state:
            raise RuntimeError("No state available from bridge (/last_state).")

        raw = state["raw"]
        obs = self.pack_observation(raw)

        score = float(self._resolve_raw(raw, "score"))
        drain = int(round(self._resolve_raw(raw, "drain_got_hit")))

        truncated = False
        terminated = False

        info = {}
        self._episode_steps += 1

        if drain == 1:
            terminated = True
            self._done = True
            time.sleep(1.0)

            return obs, -1000, terminated, truncated, info
        else:
            reward = score - self.old_score
            self.old_score = score

        self._done = terminated or truncated

        #this is an action penalty so it doesn't take random actions
        left, right = action
        if (left == 1 or right == 1):
            reward -= 0.8
        if (left == 1 and right == 1):
            reward -=0.4



        return obs, reward, terminated, truncated, info

    # ----------------------------- Action Encoding -----------------------------
    def _encode_action(self, action):
        """
        Expect action as length-2 iterable [left, right] of 0/1.
        """
        a = np.asarray(action, dtype=np.int32).clip(0, 1)
        left, right = a
        return {
            "left": int(left),
            "right": int(right),
        }
