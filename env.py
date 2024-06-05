import gymnasium as gym
from typing import Tuple, Optional, TypeVar
from pathlib import Path
import imageio


Reward = float
State = TypeVar('State')
Action = TypeVar('Action')

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

    
    def to_gif(self, path: Path, duration: float = 1/30):
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        frames = self._gym_env.render()
        imageio.mimsave(path, frames, duration=duration, format='.gif')
