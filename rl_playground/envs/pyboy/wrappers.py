import gymnasium


class FrameSkip(gymnasium.Wrapper):
    """Return only every `skip`-th frame"""

    def __init__(self, env, skip):
        self._skip = skip

        super().__init__(env)

    def step(self, action):
        """Repeat action, and sum reward"""

        total_reward = 0.0
        for _ in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info
