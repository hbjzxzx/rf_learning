from typing import Callable, List, Tuple, Optional, TypeVar
from pathlib import Path

import numpy as np
from tqdm import tqdm
import gymnasium as gym

from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.optim as optim
from collections import deque
import random


# 表示状态特征向量的维度
StateDim = int
StateVector = torch.Tensor # one-dimensional tensor, representing state feature vector

# 这里的动作是离散的，所以我们用整数表示动作
Action = int
ActionsValueVector = torch.Tensor # one-dimensional tensor, each items is the state-action value

# 这里的接口，默认都为批处理方式，即张量的第一个维度我们默认为 batch_index
Batched = List


# 策略 Q函数是基于值的，通过这一组转换函数，转换为固定策略函数
State = TypeVar('State')
Action = TypeVar('Action')
Strategy = Callable[[State], Action]


class AbstractQFunc():
    def get_action_distribute(self, state_batch: Batched[StateVector]) -> Batched[ActionsValueVector]:
        raise NotImplementedError()
    
    def get_optimal_action(self, state: Batched[StateVector]) -> Batched[Action]:
        raise NotImplementedError()
    
    def get_actions_count(self) -> int:
        raise NotImplementedError()



def to_strategy(f: AbstractQFunc) -> Strategy:
    def _strategy(s: Batched[StateVector]) -> Batched[Action]:
        x = f.get_action_distribute(s).detach().numpy()
        return x
    return _strategy

def to_strategy_epsilon_greedy(f: DeepQFunc, epsilon: float) -> Strategy:
    def _strategy(s: State) -> ActionProbDistribution:
        # e-greedy 策略
        if np.random.uniform(0, 1) > epsilon:
            # 这里选择最优动作（没有随机性）
            optimal_action = f.get_optimal_action(s)
            # 创建一个one-hot编码的动作分布
            action_distribution = np.zeros(f.get_actions_count(), dtype=np.float32)
            action_distribution[optimal_action] = 1.0
            return action_distribution
        else:
            # 随机选择动作 
            return np.ones(f.get_actions_count(), dtype=np.float32) / f.get_actions_count()
    return _strategy


class DeepQFunc(AbstractQFunc, torch.nn.Module):
    def __init__(self, state_dim: int, action_nums: int, hidden_dim: int = 128) -> None:
        # here use full-connect layer to represent Q function
        super().__init__() 
        self._state_dims = state_dim 
        self._action_nums = action_nums
        
        self._fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self._fc2 = torch.nn.Linear(hidden_dim, action_nums)

    def forward(self, x): 
        x = torch.nn.functional.relu(self._fc1(x))
        return self._fc2(x)
        
    def get_action_distribute(self, state_batch: Batched[StateVector] ) -> Batched[ActionsValueVector]:
        out = self.forward(torch.tensor(state_batch))
        return torch.nn.functional.softmax(out, dim=0).detach()
    
    def get_optimal_action(self, state_batch: Batched[StateVector]) -> Batched[Action]:    
        out = self.forward(torch.tensor(state_batch))
        return torch.argmax(out).item()

    def get_actions_count(self) -> int:
        return self._action_nums