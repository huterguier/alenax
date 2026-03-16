from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.wrappers.purerl import GymnaxWrapper

from alenax.atari_env import AtariState


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EpisodicLifeState:
    env_state: environment.EnvState
    lives: int
    time: int

    def __getattr__(self, name: str) -> Any:
        if name in self.__dataclass_fields__:
            return super().__getattribute__(name)
        else:
            return getattr(self.env_state, name)


class EpisodicLife(GymnaxWrapper):

    @partial(jax.jit, static_argnames=("self",))
    def reset(
        self, key: jax.Array, params: environment.EnvParams | None = None
    ) -> tuple[jax.Array, EpisodicLifeState]:
        obs, state = self._env.reset(key, params)
        return obs, EpisodicLifeState(
            env_state=state, lives=state.info["lives"], time=0
        )

    @partial(jax.jit, static_argnames=("self",))
    def step(
        self,
        key: jax.Array,
        state: EpisodicLifeState,
        action: int | float,
        params: environment.EnvParams | None = None,
    ) -> tuple[jax.Array, EpisodicLifeState, jax.Array, jax.Array, dict[Any, Any]]:
        prev_lives = state.env_state.info["lives"]
        next_obs, next_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        lives = info["lives"]
        episodic_done = jnp.logical_or(done, lives < prev_lives)
        return (
            next_obs,
            EpisodicLifeState(env_state=next_state, lives=lives, time=state.time + 1),
            reward,
            episodic_done,
            info,
        )
