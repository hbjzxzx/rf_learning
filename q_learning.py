#!/usr/bin/env python
# coding: utf-8

# ### 首先考虑离散的 State、Action 空间组成的Q函数

# In[41]:


from collections import defaultdict
from typing import Callable, List, Tuple, Optional

import numpy as np
from tqdm import tqdm
import gymnasium as gym
from tensorboardX import SummaryWriter


State = int
Action = int
Reward = float
ActionProbDistribution = List[float]

class AbstractQFunc():
    def get_value(self, state: State, action: Action) -> float:
        raise NotImplementedError()
    
    def get_action_distribute(self, state: State) -> ActionProbDistribution:
        raise NotImplementedError()

    def get_actions_count(self) -> int:
        raise NotImplementedError()
    
    def set_value(self, state: State, action: Action, value: float) -> None:
        raise NotImplementedError()

class DiscreteQFunc(AbstractQFunc):
    def __init__(self, state_nums: int, action_nums: int) -> None:
        self._q_table = np.zeros((state_nums, action_nums))
        self._state_nums = state_nums 
        self._action_nums = action_nums

    def get_value(self, state, action) -> float:
        return self._q_table[state][action]

    def set_value(self, state: State, action: Action, value: float) -> None:
        self._q_table[state][action] = value

    def get_action_distribute(self, state: State) -> ActionProbDistribution:
        return self._q_table[state]

    def get_actions_count(self) -> int:
        return self._action_nums


# ### 我们定义策略函数Pi(s) = P(a | s)；策略函数实际返回一个Action空间的分布函数，在离散的情况下，我们用一个数组表示这个分布， 下面定义一组函数，用于将Q转换为对应的策略

# In[58]:


# 策略函数
# todo: change the right type
ActionProbDistribution = List[float]
Strategy = Callable[[State], ActionProbDistribution]


def to_strategy(f: AbstractQFunc) -> Strategy:
    def _strategy(s: State) -> ActionProbDistribution:
        return f.get_action_distribute(s)

def to_strategy_epsilon_greedy(f: AbstractQFunc, epsilon: float) -> Strategy:
    def _strategy(s: State) -> ActionProbDistribution:
        # e-greedy 策略
        if np.random.uniform(0, 1) > epsilon:
            # 这里选择最优动作（没有随机性）
            optimal_action = np.argmax(f.get_action_distribute(s))
            # 创建一个one-hot编码的动作分布
            action_distribution = np.zeros(f.get_actions_count(), dtype=np.float32)
            action_distribution[optimal_action] = 1.0
            return action_distribution
        else:
            # 随机选择动作 
            return np.ones(f.get_actions_count(), dtype=np.float32) / f.get_actions_count()
    return _strategy


# ### 最后是训练流程，在一个环境中，首先根据当前环境进行决策，再执行动作&观察反馈，最后根据信息更新

# In[3]:


class AbstractEnv():
    # 如果返回的State部分是None，则表示Terminal 状态
    def step(self, action: Action) ->  Tuple[Reward, Optional[State]]: 
        raise NotImplementedError()
    
    def reset(self) -> State:
        return NotImplementedError() 
    

class AbstractTrainer():
    def train(self):
        raise NotImplementedError()
    

class AbstractTester():
    def test(self):
        raise NotImplementedError()
 
    


# In[55]:


# 我们实现一个使用 epsilon-greedy 策略的Q-Learning 训练。（ps， 只针对离散的Q Learning）
class QLearningTrainer(AbstractTrainer):
    def __init__(self, gamma: float, learning_rate: float, epsilon_list: List[float],
                 q_func: AbstractQFunc, env: AbstractEnv):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon_list = epsilon_list
        
        self.q_func = q_func
        self.env = env
        
        self.current_state = None

        self.reward_record = []
    def train(self, epoch_cnt: int, max_steps: int):
        for epoch in tqdm(range(epoch_cnt)):
            init_state = self.env.reset()  # 回合开始前先重制环境          
            self.current_state = init_state
            self.acc_reward = 0
            # print(f'state change to: {self.current_state}')
            for s in range(max_steps): # 复杂的环境设置最大步数，也就是Horizon
                # 获取此时Q 对应的epsilon-greedy 的策略 
                e_greedy_s = to_strategy_epsilon_greedy(self.q_func, self.epsilon_list[epoch])
                # 使用此时的策略进行决策
                action_dis = e_greedy_s(self.current_state)
                # 选择概率最大的action
                action = np.random.choice(self.q_func.get_actions_count(), p=action_dis)
                # 执行此时的action
                reward, next_state = self.env.step(action)
                self.acc_reward += reward
                if next_state is None:
                    # 达到terminal状态
                    q_target = reward 
                else:
                    q_target = reward + self.gamma * np.argmax(self.q_func.get_action_distribute(next_state))
                
                # 更新Q 函数
                current_value = self.q_func.get_value(self.current_state, action)
                self.q_func.set_value(self.current_state, action, 
                                       current_value + self.learning_rate * (q_target - current_value)
                                    )
                self.current_state = next_state
                if self.current_state is None:
                    break
            self.reward_record.append(self.acc_reward)


class QFuncTester(AbstractTester):
    def __init__(self, q_func: AbstractQFunc, env: AbstractEnv) -> None:
        self._q_func = q_func
        self._gym_env = env
    
    def test(self):
        init_state = self.env.reset()  # 回合开始前先重制环境  
        self.current_state = init_state 
        self.acc_reward = 0
        greedy_strateggy = to_strategy(self._q_func)

        while True:
            action = greedy_strateggy(self.current_state) 
            np.argmax()
            # do it
            reward, next_state = self._gym_env.step(action)
            self.acc_reward += reward
            if next_state is None:
                break
            else:
                self.current_state = next_state
        print(f"Test reward: {self.acc_reward}")
    

    def test_batch(self):
        ...
        


# In[56]:


class Env(AbstractEnv):
    def __init__(self, gym_env: gym.Env):
        self._gym_env = gym_env

    def step(self, action: Action) ->  Tuple[Reward, Optional[State]]: 
        next_state, reward, is_terminated, is_truncated, _ = self._gym_env.step(action)
        if is_terminated or is_truncated:
            return reward, None
        else:
            return reward, next_state

    def reset(self) -> State:
        init_state, _ = self._gym_env.reset()
        return init_state
        


# ### 开始使用Q-learning 训练

# In[57]:


GYM_ENV_NAME = 'CliffWalking-v0'
_gym_env = gym.make(GYM_ENV_NAME)

action_nums, state_nums = _gym_env.action_space.n, _gym_env.observation_space.n
print(f'action num: {action_nums}, space num: {state_nums}')

TRAIN_EPOCH = 50000
LEARNING_RATE = 1e-3
GAMMA = 0.8
# EPSILON_LIST = [1.0 * 1.0/(i+1) for i in range(TRAIN_EPOCH)]
EPSILON_LIST = [0.5 for i in range(TRAIN_EPOCH)]

q_func = DiscreteQFunc(state_nums=state_nums, action_nums=action_nums)
env = Env(_gym_env)
q_trainer = QLearningTrainer(
    GAMMA,
    LEARNING_RATE,
    EPSILON_LIST,
    q_func,
    env
)

q_trainer.train(epoch_cnt=TRAIN_EPOCH, max_steps=1000)


# In[45]:


print(f'action num: {action_nums}, space num: {state_nums}')


# In[29]:


import matplotlib.pyplot as plt
# 创建一个新的图形
plt.figure()
# 绘制折线图
plt.plot(q_trainer.reward_record)
# 显示图形
plt.show()


# In[30]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


plt.figure(figsize=(10,8))
sns.heatmap(q_func._q_table, cmap='viridis')
plt.show()


# In[ ]:




