# ma-gym
A collection of multi agent environments based on OpenAI gym.

[![Build Status](https://travis-ci.com/koulanurag/ma-gym.svg?token=DM2fKnVJXEZDaszt9oHm&branch=master)](https://travis-ci.com/koulanurag/ma-gym)

## Installation
```bash
cd ma-gym
pip install -e .
```

## Usage:
```python
import gym

env = gym.make('Switch2-v0')
done_n = [False for _ in range(env.n_agents)]
ep_reward = 0

obs_n = env.reset()
while not all(done_n):
    env.render()
    obs_n, reward_n, done_n, info = env.step(env.action_space.sample())
    ep_reward += sum(reward_n)
env.close()
```

Please refer to [Wiki](https://github.com/koulanurag/ma-gym/wiki/Usage) for complete usage details

## Environments:
- [x] Checkers
- [x] Combat
- [x] PredatorPrey
- [x] Pong Duel  ```(two player pong game)```
- [x] Switch

```
Note : openai's environment can be accessed in multi agent form by prefix "ma_".Eg: ma_CartPole-v0
This returns an instance of CartPole-v0 in "multi agent wrapper" having a single agent. 
These environments are helpful during debugging.
```

Please refer to [Wiki](https://github.com/koulanurag/ma-gym/wiki/Environments) for more details.

## Testing:

- Install: ```pip install pytest ```
- Run: ```pytest```
