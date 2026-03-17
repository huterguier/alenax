from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from ale_py.vector_env import AtariVectorEnv
from gymnax.environments.environment import Environment, EnvState
from gymnax.environments.spaces import Box, Discrete

_alenax_environments: dict[int, Any] = {}


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AtariState(EnvState):
    env_id: jax.Array
    info: dict[Any, Any]


class AtariEnv(Environment):

    id: str
    """ The Gymnasium environment ID. """
    result_shape_dtype: Any
    """ The shape and dtype of the returned EnvironmentState and Timestep. """
    kwargs: Any
    """ The kwargs used to create the Gymnasium environment. """

    def __init__(self, id: str, **kwargs):
        self.id = id
        env = AtariVectorEnv(self.id, num_envs=1, **kwargs)
        obs: Any
        obs, _ = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, _, info = env.step(action)
        result = (
            jnp.asarray(obs),
            AtariState(
                env_id=jnp.int32(0),
                info=jax.tree.map(jnp.asarray, info),
                time=jnp.int32(0),
            ),
            jnp.asarray(reward, dtype=jnp.float32),
            jnp.asarray(terminated, dtype=jnp.bool),
            jax.tree.map(jnp.asarray, info),
        )
        self.result_shape_dtype = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape[1:], x.dtype), (result)
        )
        self._action_space = Discrete(env.single_action_space.n)  # type: ignore[attr-defined]
        self._observation_space = Box(
            jnp.asarray(env.single_observation_space.low),  # type: ignore[attr-defined]
            jnp.asarray(env.single_observation_space.high),  # type: ignore[attr-defined]
            env.single_observation_space.shape,
        )
        self.kwargs = kwargs

    @partial(jax.jit, static_argnames=("self",))
    def reset(
        self, key: jax.Array, params: None = None
    ) -> tuple[jax.Array, AtariState]:
        del params

        def callback(key):
            global envs_gymnasium, current_env_id
            shape = key.shape[:-1]
            keys_flat = jnp.reshape(key, (-1, key.shape[-1]))
            num_envs = keys_flat.shape[0]
            envs = AtariVectorEnv(self.id, num_envs=num_envs, **self.kwargs)
            obs, info = envs.reset(seed=0)
            env_id = len(_alenax_environments)
            _alenax_environments[env_id] = envs
            obs = np.reshape(obs, shape + obs.shape[1:])
            info = jax.tree.map(lambda i: np.reshape(i, shape + i.shape[1:]), info)
            state = AtariState(
                env_id=jnp.full(shape, env_id, dtype=jnp.int32), info=info, time=0
            )

            return obs, state

        obs, state = jax.pure_callback(
            callback,
            (self.result_shape_dtype[0], self.result_shape_dtype[1]),
            jax.random.key_data(key),
            vmap_method="broadcast_all",
        )
        return obs, state

    @partial(jax.jit, static_argnames=("self",))
    def step(
        self,
        key: jax.Array,
        state: AtariState,
        action: int | float | jax.Array,
        params: None = None,
    ) -> tuple[jax.Array, AtariState, jax.Array, jax.Array, dict[Any, Any]]:
        del key, params

        def callback(env_id, time, action):
            global envs_gymnasium
            shape = env_id.shape
            envs = _alenax_environments[np.ravel(env_id)[0]]
            actions = np.reshape(np.asarray(action), (-1,))
            obs, reward, terminated, truncated, info = envs.step(actions)
            obs = np.reshape(obs, shape + obs.shape[1:])
            reward = jnp.reshape(reward, shape).astype(np.float32)
            done = jnp.reshape(jnp.logical_or(terminated, truncated), shape).astype(
                np.bool_
            )
            info = jax.tree.map(lambda i: np.reshape(i, shape + i.shape[1:]), info)
            state = AtariState(env_id=env_id, info=info, time=(1 - done) * (time + 1))
            return obs, state, reward, done, info

        obs, state, reward, done, info = jax.pure_callback(
            callback,
            self.result_shape_dtype,
            state.env_id,
            state.time,
            action,
            vmap_method="broadcast_all",
        )

        return obs, state, reward, done, info

    def action_space(self, params: None = None) -> Discrete:
        return self._action_space

    def observation_space(self, params: None = None) -> Box:
        return self._observation_space
