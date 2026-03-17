import jax
import jax.numpy as jnp
import pytest

from alenax import AtariEnv, ClipReward, EpisodicLife, RecordEpisodeStatistics
from alenax.wrappers.episodic_life import EpisodicLifeState
from alenax.wrappers.record_episode_statistics import RecordEpisodeStatisticsState

GAME = "pong"


@pytest.fixture
def env():
    return AtariEnv(GAME)


class TestClipReward:
    def test_default_range(self, env):
        wrapped = ClipReward(env)
        key = jax.random.PRNGKey(0)
        obs, state = wrapped.reset(key)
        step_key = jax.random.PRNGKey(1)
        _, _, reward, _, _ = wrapped.step(step_key, state, jnp.int32(0))
        assert reward >= -1.0
        assert reward <= 1.0

    def test_custom_range(self, env):
        wrapped = ClipReward(env, min_reward=-0.5, max_reward=0.5)
        key = jax.random.PRNGKey(0)
        obs, state = wrapped.reset(key)
        step_key = jax.random.PRNGKey(1)
        _, _, reward, _, _ = wrapped.step(step_key, state, jnp.int32(0))
        assert reward >= -0.5
        assert reward <= 0.5

    def test_obs_unchanged(self, env):
        wrapped = ClipReward(env)
        key = jax.random.PRNGKey(0)
        obs_wrapped, state_wrapped = wrapped.reset(key)
        obs_raw, state_raw = env.reset(key)
        assert obs_wrapped.shape == obs_raw.shape


class TestEpisodicLife:
    def test_reset_state_type(self, env):
        wrapped = EpisodicLife(env)
        key = jax.random.PRNGKey(0)
        obs, state = wrapped.reset(key)
        assert isinstance(state, EpisodicLifeState)

    def test_reset_obs_shape(self, env):
        wrapped = EpisodicLife(env)
        key = jax.random.PRNGKey(0)
        obs, state = wrapped.reset(key)
        assert obs.shape == env.observation_space().shape

    def test_step_returns_five(self, env):
        wrapped = EpisodicLife(env)
        key = jax.random.PRNGKey(0)
        obs, state = wrapped.reset(key)
        result = wrapped.step(jax.random.PRNGKey(1), state, jnp.int32(0))
        assert len(result) == 5

    def test_lives_tracked(self, env):
        wrapped = EpisodicLife(env)
        key = jax.random.PRNGKey(0)
        obs, state = wrapped.reset(key)
        assert state.lives is not None

    def test_state_delegation(self, env):
        wrapped = EpisodicLife(env)
        key = jax.random.PRNGKey(0)
        obs, state = wrapped.reset(key)
        # env_id is delegated from the inner AtariState
        assert hasattr(state, "env_id")


class TestRecordEpisodeStatistics:
    def test_reset_state_type(self, env):
        wrapped = RecordEpisodeStatistics(env)
        key = jax.random.PRNGKey(0)
        obs, state = wrapped.reset(key)
        assert isinstance(state, RecordEpisodeStatisticsState)

    def test_reset_zeroed(self, env):
        wrapped = RecordEpisodeStatistics(env)
        key = jax.random.PRNGKey(0)
        obs, state = wrapped.reset(key)
        assert float(state.episode_returns) == 0.0
        assert int(state.episode_lengths) == 0
        assert float(state.returned_episode_returns) == 0.0
        assert int(state.returned_episode_lengths) == 0

    def test_step_accumulates(self, env):
        wrapped = RecordEpisodeStatistics(env)
        key = jax.random.PRNGKey(0)
        obs, state = wrapped.reset(key)
        step_key = jax.random.PRNGKey(1)
        _, state, _, _, info = wrapped.step(step_key, state, jnp.int32(0))
        assert int(state.episode_lengths) >= 0
        assert "returned_episode_returns" in info
        assert "returned_episode_lengths" in info
        assert "returned_episode" in info

    def test_episode_length_increments(self, env):
        wrapped = RecordEpisodeStatistics(env)
        key = jax.random.PRNGKey(0)
        obs, state = wrapped.reset(key)
        step_key = jax.random.PRNGKey(1)
        _, state, _, done, _ = wrapped.step(step_key, state, jnp.int32(0))
        if not done:
            assert int(state.episode_lengths) == 1

    def test_state_delegation(self, env):
        wrapped = RecordEpisodeStatistics(env)
        key = jax.random.PRNGKey(0)
        obs, state = wrapped.reset(key)
        assert hasattr(state, "env_id")


class TestWrapperComposition:
    def test_all_wrappers(self, env):
        wrapped = RecordEpisodeStatistics(EpisodicLife(ClipReward(env)))
        key = jax.random.PRNGKey(0)
        obs, state = wrapped.reset(key)
        step_key = jax.random.PRNGKey(1)
        next_obs, next_state, reward, done, info = wrapped.step(
            step_key, state, jnp.int32(0)
        )
        assert next_obs.shape == env.observation_space().shape
        assert reward >= -1.0
        assert reward <= 1.0
        assert "returned_episode_returns" in info
