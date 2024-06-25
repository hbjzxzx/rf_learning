from typing import Tuple, Optional, TypeVar
from pathlib import Path

from loguru import logger
import imageio
import gymnasium as gym
import numpy as np

_logger = logger.bind(name='Env')

Reward = float
State = TypeVar('State')
Action = TypeVar('Action')


# 处理连续的Action空间时，有一种简单的方式是将连续的Action空间离散化，即Q网络输出的Action还是离散的，通过转换后，应用到环境中
# 将离散的Action转换为连续的Action
class Discrete1ContinuousAction: 
    def __init__(self, low, high, bins: int, round: int = 1) -> None:
        self._low = low
        self._high = high
        self._bins = bins
        self._round = round

        self._step = (high - low) / bins

    def to_continuous_action(self, action_of_integer: np.ndarray) -> float:
        return np.round(
            self._low + self._step * action_of_integer,
            self._round
        )
 

def get_action_discreter(e: 'Env', bin: int):
    action_space = e._gym_env.action_space
    return Discrete1ContinuousAction(
        action_space.low,
        action_space.high,
        bin
    )


class Env:
    def __init__(self, gym_env: gym.Env):
        self._gym_env = gym_env

    def step(self, action) ->  Tuple[Reward, Optional[State]]: 
        next_state, reward, is_terminated, is_truncated, _ = self._gym_env.step(action)
        if is_terminated or is_truncated:
            return reward, None
        else:
            return reward, next_state

    def reset(self) -> State:
        init_state, _ = self._gym_env.reset()
        return init_state

    @classmethod
    def from_env_name(cls, name: str, render_mode: Optional[str] = None):
        if render_mode is not None:
            return cls(gym.make(name))
        else:
            return cls(gym.make(name, render_mode=render_mode))

    def print_state_action_dims(self):
        action_space, state_space = self._gym_env.action_space, self._gym_env.observation_space
        show_dim = lambda x: x.n if hasattr(x, 'n') else x
        _logger.info(f'action: {show_dim(action_space)}, space: {show_dim(state_space)}')
    
    def get_state_dim(self):
        state_space = self._gym_env.observation_space
        if hasattr(state_space, 'n'):
            return (state_space.n, )
        else:
            return state_space.shape
    
    def get_action_dim(self):
        action_space = self._gym_env.action_space
        if hasattr(action_space, 'n'):
            return (action_space.n, )
        else:
            return action_space.shape
    
    def to_gif(self, path: Path, duration: float = 1/30):
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        frames = self._gym_env.render()
        imageio.mimsave(path, frames, duration=duration, format='.gif')
