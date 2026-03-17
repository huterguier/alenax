<div align="center">
    <img src="https://github.com/huterguier/alenax/blob/main/images/alenax.png" width="300">
</div>

# Arcade Learning Environment for JAX
[![PyPI version](https://img.shields.io/pypi/v/alenax.svg)](#installation)
[![License: MIT](https://img.shields.io/badge/license-MIT-1d8a50.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/codestyle-black-black.svg)](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)

[`alenax`](https://github.com/huterguier/alenax) is a fast and easy-to-use JAX wrapper for the [Arcade Learning Environment](https://github.com/Farama-Foundation/Arcade-Learning-Environment), fully compatible with [`gymnax`](https://github.com/RobertTLange/gymnax). 
All you need to do is import the environment and `alenax` takes care of the rest. While it interfaces with the CPU-based ALE under the hood, it exposes a purely JAX-compatible API so you can trace, compile, and batch your environments effortlessly.

## Features
- ✨ **Easy to Use:** Simply initialize `AtariEnv` with your target game and start stepping. `alenax` handles all the complex boilerplate of plumbing the environment data through JAX's transformations.
- 🔌 **Seamless Integration:** Designed to mimic the `gymnax` API natively, meaning it acts as a drop-in replacement for any `gymnax`-compatible RL pipeline.
- 🚀 **Fast and Efficient:** All environment methods (`reset`, `step`) are fully JIT-compiled and natively support `jax.vmap` for batched execution out of the box.

## Sharp Bits 🔪
Due to the nature of wrapping a CPU-based C++ emulator in JAX, there are a few limitations to keep in mind:
- 🛤️ **Sequential Rollouts Only:** Natively supports sequential forward execution.
- 💾 **Environment State:** The JAX `state` object does not save the true underlying emulator state. Instead, it merely holds a reference to the environment instance running on the CPU.
- 🌳 **No MCTS:** Because the true state cannot be perfectly saved and restored within JAX, Monte Carlo Tree Search and similar planning algorithms are not possible.
- 🔄 **Reset Behavior:** Calling `reset()` internally instantiates a completely new environment object rather than resetting an existing one.

## Installation
`alenax` requires Python >= 3.10 and can be installed via pip:
```bash
pip install alenax
```

## Quick Start
Getting up and running with `alenax` is designed to be as frictionless as possible for those familiar with standard JAX and RL environments. Below you will find a brief walkthrough on how to initialize an environment, interact with it using random keys, and apply standard Atari wrappers.

### Basic Usage
The API follows the standard JAX-based environment patterns. You initialize the environment and pass PRNG keys to `reset` and `step`. 

```python
import jax
from alenax import AtariEnv

env = AtariEnv("pong")
key = jax.random.key(0)

obs, state = env.reset(key)

action = env.action_space().sample(key)
obs, state, reward, done, info = env.step(key, state, action)
```

Because all methods are natively JAX-compatible, you can easily use `jax.vmap` to run multiple environments in parallel:

```python
keys = jax.random.split(key, 4)
obs, state = jax.vmap(env.reset)(keys)
```

### Wrappers
`alenax` provides several out-of-the-box JAX wrappers specifically geared toward standard Atari preprocessing.

```python
from alenax import AtariEnv, ClipReward, EpisodicLife, RecordEpisodeStatistics

env = AtariEnv("pong")
env = ClipReward(env)              # Clip rewards to [-1, 1]
env = EpisodicLife(env)            # Treat life loss as episode end
env = RecordEpisodeStatistics(env) # Track episode returns and lengths
```

## Citation
If you use `alenax` in your work, feel free to cite it as follows:
```bibtex
@software{alenax2026github,
  author = {huterguier},
  title = {{alenax}: Arcade Learning Environment for JAX.},
  url = {[https://github.com/huterguier/alenax](https://github.com/huterguier/alenax)},
  version = {0.1.3},
  year = {2026},
}
```
