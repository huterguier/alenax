# alenax

A JAX wrapper for the [Arcade Learning Environment](https://github.com/Farama-Foundation/Arcade-Learning-Environment), compatible with [Gymnax](https://github.com/RobertTLange/gymnax).

## Installation

```bash
pip install alenax
```

Requires Python >= 3.13.

## Usage

```python
import jax
from alenax import AtariEnv

env = AtariEnv("pong")
key = jax.random.PRNGKey(0)

obs, state = env.reset(key)
obs, state, reward, done, info = env.step(key, state, action)
```

All environment methods are JIT-compiled and support `jax.vmap` for batched execution.

## Wrappers

```python
from alenax import ClipReward, EpisodicLife, RecordEpisodeStatistics

env = AtariEnv("pong")
env = ClipReward(env)              # Clip rewards to [-1, 1]
env = EpisodicLife(env)            # Treat life loss as episode end
env = RecordEpisodeStatistics(env) # Track episode returns and lengths
```
