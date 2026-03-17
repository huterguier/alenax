from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.wrappers.purerl import GymnaxWrapper


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RecordEpisodeStatisticsState:
    env_state: environment.EnvState
    episode_returns: jax.Array
    episode_lengths: jax.Array
    returned_episode_returns: jax.Array
    returned_episode_lengths: jax.Array

    def __getattr__(self, name: str) -> Any:
        if name in self.__dataclass_fields__:
            return super().__getattribute__(name)
        else:
            return getattr(self.env_state, name)


class RecordEpisodeStatistics(GymnaxWrapper):

    @partial(jax.jit, static_argnames=("self",))
    def reset(
        self, key: jax.Array, params: None = None
    ) -> tuple[jax.Array, RecordEpisodeStatisticsState]:
        obs, env_state = self._env.reset(key, params)
        state = RecordEpisodeStatisticsState(
            env_state,
            jnp.float32(0),
            jnp.int32(0),
            jnp.float32(0),
            jnp.int32(0),
        )
        return obs, state

    @partial(jax.jit, static_argnames=("self",))
    def step(
        self,
        key: jax.Array,
        state: RecordEpisodeStatisticsState,
        action: int | float | jax.Array,
        params: None = None,
    ) -> tuple[
        jax.Array, RecordEpisodeStatisticsState, jax.Array, jax.Array, dict[Any, Any]
    ]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = RecordEpisodeStatisticsState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = done
        return obs, state, reward, done, info
