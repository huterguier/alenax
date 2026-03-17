import jax
import jax.numpy as jnp
import pytest

from alenax import AtariEnv
from alenax.atari_env import AtariState

GAME = "pong"


@pytest.fixture
def env():
    return AtariEnv(GAME)


@pytest.fixture
def reset(env):
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    return obs, state


class TestInit:
    def test_id(self, env):
        assert env.id == GAME

    def test_action_space(self, env):
        assert env.action_space().n > 0

    def test_observation_space(self, env):
        assert len(env.observation_space().shape) == 3  # (H, W, C)


class TestReset:
    def test_obs_shape(self, env, reset):
        obs, _ = reset
        assert obs.shape == env.observation_space().shape

    def test_obs_dtype(self, reset):
        obs, _ = reset
        assert obs.dtype == jnp.uint8

    def test_state_type(self, reset):
        _, state = reset
        assert isinstance(state, AtariState)

    def test_state_time_zero(self, reset):
        _, state = reset
        assert int(state.time) == 0


class TestStep:
    def test_output_shapes(self, env, reset):
        obs, state = reset
        key = jax.random.PRNGKey(1)
        next_obs, next_state, reward, done, info = env.step(key, state, jnp.int32(0))
        assert next_obs.shape == env.observation_space().shape
        assert reward.shape == ()
        assert done.shape == ()

    def test_output_dtypes(self, env, reset):
        obs, state = reset
        key = jax.random.PRNGKey(1)
        next_obs, next_state, reward, done, info = env.step(key, state, jnp.int32(0))
        assert reward.dtype == jnp.float32
        assert done.dtype == jnp.bool_

    def test_state_time_increments(self, env, reset):
        obs, state = reset
        key = jax.random.PRNGKey(1)
        _, next_state, _, done, _ = env.step(key, state, jnp.int32(0))
        if not done:
            assert int(next_state.time) == 1

    def test_info_has_lives(self, env, reset):
        obs, state = reset
        key = jax.random.PRNGKey(1)
        _, _, _, _, info = env.step(key, state, jnp.int32(0))
        assert "lives" in info


class TestVmap:
    def test_vmap_reset(self, env):
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        obs, state = jax.vmap(env.reset)(keys)
        expected = (4,) + env.observation_space().shape
        assert obs.shape == expected

    def test_vmap_step(self, env):
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        obs, state = jax.vmap(env.reset)(keys)
        step_keys = jax.random.split(jax.random.PRNGKey(1), 4)
        actions = jnp.zeros(4, dtype=jnp.int32)
        next_obs, next_state, reward, done, info = jax.vmap(env.step)(
            step_keys, state, actions
        )
        assert next_obs.shape == obs.shape
        assert reward.shape == (4,)
        assert done.shape == (4,)
