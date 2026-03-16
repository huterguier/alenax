from functools import partial

import jax
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.wrappers.purerl import GymnaxWrapper


class ClipReward(GymnaxWrapper):
    """Clip the reward to a specified range."""

    def __init__(self, env, min_reward=-1.0, max_reward=1.0):
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward

    @partial(jax.jit, static_argnames=("self",))
    def step(
        self,
        key: jax.Array,
        state: environment.EnvState,
        action: int | float,
        params: environment.EnvParams | None = None,
    ):
        next_obs, next_state, reward, done, info = self._env.step(
            key, state, action, params
        )
        clipped_reward = jnp.clip(reward, self.min_reward, self.max_reward)
        return next_obs, next_state, clipped_reward, done, info
