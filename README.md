# pinball-env

The Pinball Domain for skill discovery in reinforcement learning.

## Installation

You can download and make the library available in your project using _pip_.

```bash
$ pip install git+https://github.com/luizth/pinball-env.git
```


## Usage

Simple instructions on how to import and instanciate an environment using available configurations.

```python
from pinball_env.pinball import PinballEnv
from pinball_env.pinball_config import available_configs

print(available_configs)  # dict_keys(['pinball_hard_single', 'pinball_simple_single'])

env = PinballEnv(available_configs.sample())
env_simple = PinballEnv(available_configs.get('pinball_simple_single'))
```


## Feature Construction

There are feature constructions approaches included to use for adapting classical RL algorithms to
work with the continous state space data.

### Tile Coding

Provides both generalization and discrimination of the state space. It converts the 4-dimensional
continous state space in a np.array of IHT size. The features are zero (deactivated) except for active
tiles that are 1. This approach is especially useful for working with linear function approximation.

```python
from pinball_env.feature_construction.tile_coder import PinballTileCoder

coder = PinballTileCoder(32, 8, 4096)

state, info = env.reset()
coder.get_state_features(*state)  # returns np.array of size coder.iht_size
```


## Reference

The Pinball Domain was introduced by G. D. Konidaris and A. G. Barto in early skill discovery work [1].

This implementation was adapted from code in PyRL (https://github.com/amarack/python-rl), from Will Dabney and Pierre-Luc Bacon.

You can find more about the Pinball Domain in the Brown University website (http://irl.cs.brown.edu/pinball/).

[1] George Konidaris and Andrew Barto. 2009. Skill discovery in continuous reinforcement learning domains using skill chaining. In Proceedings of the 23rd International Conference on Neural Information Processing Systems (NIPS'09). Curran Associates Inc., Red Hook, NY, USA, 1015–1023.
