from dataclasses import dataclass
from functools import partial
from typing import Any

import ale_py
import gymnasium
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments.environment import Environment
from gymnax.environments.spaces import Box, Discrete, Space

_alenax_environments = {}


@jax.tree_util.register_dataclass
@dataclass
class AtariState:
    env_id: jax.Array


AtariParams = Any


class AtariEnv(Environment):

    id: str
    """ The Gymnasium environment ID. """
    result_shape_dtype: Any
    """ The shape and dtype of the returned EnvironmentState and Timestep. """
    kwargs: Any
    """ The kwargs used to create the Gymnasium environment. """

    def __init__(self, id: str, **kwargs):
        self.id = id
        env = gymnasium.make_vec(self.id, num_envs=1, **kwargs)
        obs, _ = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        state = AtariState(env_id=jnp.int32(0))
        result = (
            jnp.array(obs),
            state,
            jnp.array(reward, dtype=jnp.float32),
            jnp.array(terminated, dtype=jnp.bool),
            jax.tree.map(jnp.array, info),
        )
        print(len(jax.tree.leaves(result)))
        self.result_shape_dtype = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape[1:], x.dtype), (result)
        )
        self._action_space = self.gymnasium_to_gymnax_space(env.single_action_space)
        self._observation_space = self.gymnasium_to_gymnax_space(
            env.single_observation_space
        )
        self.kwargs = kwargs

    @partial(jax.jit, static_argnames=("self",))
    def reset(
        self, key: jax.Array, params: AtariParams | None = None
    ) -> tuple[jax.Array, AtariState]:
        del params

        def callback(key):
            global envs_gymnasium, current_env_id
            shape = key.shape[:-1]
            keys_flat = jnp.reshape(key, (-1, key.shape[-1]))
            num_envs = keys_flat.shape[0]
            envs = gymnasium.make_vec(self.id, num_envs=num_envs, **self.kwargs)
            obs, info = envs.reset(seed=0)
            env_id = len(_alenax_environments)
            _alenax_environments[env_id] = envs
            obs = np.reshape(obs, shape + obs.shape[1:])
            state = AtariState(env_id=jnp.full(shape, env_id, dtype=jnp.int32))

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
        params: AtariParams | None = None,
    ) -> tuple[jax.Array, AtariState, jax.Array, jax.Array, dict[Any, Any]]:
        del key, params

        def callback(env_id, action):
            global envs_gymnasium
            shape = env_id.shape
            envs = _alenax_environments[np.ravel(env_id)[0]]
            actions = np.reshape(np.asarray(action), (-1,))
            obs, reward, terminated, truncated, info = envs.step(actions)
            obs = np.reshape(obs, shape + obs.shape[1:])
            state = AtariState(env_id=env_id)
            reward = jnp.reshape(reward, shape).astype(np.float32)
            done = jnp.reshape(jnp.logical_or(terminated, truncated), shape).astype(
                np.bool
            )
            info = jax.tree.map(lambda i: np.reshape(i, shape + i.shape[1:]), info)
            print(len(jax.tree.leaves((obs, state, reward, done, info))))
            return obs, state, reward, done, info

        obs, state, reward, done, info = jax.pure_callback(
            callback,
            self.result_shape_dtype,
            state.env_id,
            action,
            vmap_method="broadcast_all",
        )

        return obs, state, reward, done, info

    @classmethod
    def gymnasium_to_gymnax_space(cls, gymnasium_space: Any) -> Space:
        """Convert a Gymnasium space to a Gymnax space."""
        if isinstance(gymnasium_space, gymnasium.spaces.Discrete):
            return Discrete(int(gymnasium_space.n))
        elif isinstance(gymnasium_space, gymnasium.spaces.Box):
            return Box(
                jnp.asarray(gymnasium_space.low),
                jnp.asarray(gymnasium_space.high),
                gymnasium_space.shape,
            )
        else:
            raise NotImplementedError(
                f"Gymnasium space {gymnasium_space} not supported."
            )
