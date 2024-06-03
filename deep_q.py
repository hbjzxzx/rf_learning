from typing import Callable, List, Optional
from pathlib import Path
from collections import deque
import random
import copy

from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim

from env import Env 


# 表示状态特征向量的维度
StateDim = int
StateVector = torch.Tensor # one-dimensional tensor, representing state feature vector

# 这里的动作是离散的，所以我们用整数表示动作
Action = int
ActionsValueVector = torch.Tensor # one-dimensional tensor, each items is the state-action value

# 默认都为批处理方式，即无论是State，还是ActionValueVector, 这两个张量的第一个维度我们默认为 batch_index
BatchedState = torch.Tensor
BatchedAction = torch.Tensor
BatchedActionsValueVec = torch.Tensor


# 策略 Q函数是基于值的，通过这一组转换函数，转换为固定策略函数
Strategy = Callable[[BatchedState], BatchedAction]


class AbstractQFunc():
    def __init__(self, device: Optional[torch.device]=None) -> None:
        self.__device = torch.device('cpu') if device is None else device

    def get_action_distribute(self, state_batch: BatchedState) -> BatchedActionsValueVec:
        raise NotImplementedError()
    
    def get_optimal_action(self, state: BatchedState) -> BatchedAction:
        raise NotImplementedError()
    
    def get_actions_count(self) -> int:
        raise NotImplementedError()

    def save(self, path: Path):
        raise NotImplementedError()
    
    def load(self, path: Path):
        raise NotImplementedError()
    
    def get_device(self):
        return self.__device

class ValueFunc2Strategy():

    @staticmethod
    def to_strategy(f: AbstractQFunc) -> Strategy:
        def _strategy(s: BatchedState) -> BatchedActionsValueVec: 
            x = f.get_action_distribute(s).detach()
            return x
        return _strategy

    @staticmethod
    def to_strategy_epsilon_greedy(f: AbstractQFunc, epsilon: float) -> Strategy:
        def _strategy(s: BatchedState) -> BatchedActionsValueVec:
            with torch.device(f.get_device()):
                batch_size = s.size(0)
                random_choice_action = torch.randint(0, f.get_actions_count(), (batch_size, ))
                optimal_choice_action = f.get_optimal_action(s)
                # e-greedy 策略
                use_random_choice = (torch.rand(batch_size) <= epsilon)
                # 根据随机结果，选择随机策略 or 最优策略（贪心策略）
                final_choice = torch.where(use_random_choice, random_choice_action, optimal_choice_action)

            return final_choice
        return _strategy


class DeepQFunc(AbstractQFunc, torch.nn.Module):
    def __init__(self, state_dim: int, action_nums: int, hidden_dim: int = 128, device: Optional[torch.device]=None) -> None:
        AbstractQFunc.__init__(self, device=device) 
        torch.nn.Module.__init__(self)

        # here use full-connect layer to represent Q function
        self._state_dims = state_dim 
        self._action_nums = action_nums

        with torch.device(self.get_device()): 
            self._fc1 = torch.nn.Linear(state_dim, hidden_dim)
            self._fc2 = torch.nn.Linear(hidden_dim, action_nums)

    def forward(self, x): 
        x = torch.nn.functional.relu(self._fc1(x))
        return self._fc2(x)
        
    def get_action_distribute(self, state_batch: BatchedState ) -> BatchedActionsValueVec:
        out = self.forward(torch.tensor(state_batch)).detach()
        return torch.nn.functional.softmax(out, dim=0)
    
    def get_optimal_action(self, state_batch: BatchedState) -> BatchedAction:    
        out = self.forward(state_batch)
        return torch.argmax(out, dim=1)

    def get_actions_count(self) -> int:
        return self._action_nums

    def save(self, path: Path):
        torch.save(self.state_dict(), path)
    
    def load(self, path: Path):
        self.load_state_dict(torch.load(path))

class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, weight):
        self.buffer.append((state, action, reward, next_state, weight))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        return zip(*transitions)


class DeepQFuncTrainer:
    def __init__(self, q_func: DeepQFunc, 
                 env: Env, 
                 replay_buffer: ReplayBuffer,
                 learning_rate: float,
                 batch_size: int,
                 gamma: float,
                 epsilon_list: List[float],
                 logger_folder: Optional[Path] = None,
                 ) -> None:
        self._q_func = copy.deepcopy(q_func)
        self._target_q_func = q_func
        
        self._env = env
        self._replay_buffer = replay_buffer
        self._gamma = gamma
        self._learning_rate = learning_rate
        self._optimizer = optim.Adam(self._q_func.parameters(), lr=self._learning_rate)
        self._batch_size = batch_size
        self._epsilon_list = epsilon_list

        self._logger_folder = logger_folder if logger_folder is not None else Path('./logs')

    def train(self, train_epoch: int, 
              max_steps: int, 
              target_q_update_freq: int,
              minimal_replay_size_to_train: int):
        writer = SummaryWriter(self._logger_folder)
        progress_bar = tqdm(range(train_epoch))

        update_q_index = 1
        for epoch in progress_bar:
            init_state = self._env.reset()
            current_state = init_state
            acc_reward = 0
            step_cnt = 0
            
            for _ in range(max_steps):
                step_cnt += 1
                # 获取此时DeepQFunc的策略 
                e_greedy_s = ValueFunc2Strategy.to_strategy_epsilon_greedy(self._q_func, self._epsilon_list[epoch])
                # 使用该策略进行决策, 注意传入的state 需要是torch.tensor类型，并且第一个维度是batch_index
                with torch.device(self._q_func.get_device()):
                    batched_action = e_greedy_s(torch.tensor([current_state]))
                action = batched_action.cpu().item()

                # 执行这个动作，获取下一个状态      
                reward, next_state = self._env.step(action)
                acc_reward += reward

                # 将这次的经验存储到经验回放池中 
                if next_state is not None:
                    self._replay_buffer.add(current_state, action, reward, next_state, 1)
                else:
                    self._replay_buffer.add(current_state, action, reward, current_state, 0)

                
                if len(self._replay_buffer.buffer) > minimal_replay_size_to_train:
                    self.update_q_func(update_q_index % target_q_update_freq == 0)
                    update_q_index += 1
                
                current_state = next_state
                if current_state is None: 
                    break

            writer.add_scalar('reward', acc_reward, epoch)
            writer.add_scalar('step', step_cnt, epoch)
            progress_bar.set_postfix({'reward': acc_reward, 'step': step_cnt})
        
        writer.close() 


    def update_q_func(self, update_target_q_func=False):
        state, action, reward, next_state, weight = self._replay_buffer.sample(self._batch_size)
        with self._q_func.get_device():
            state = torch.tensor(state, dtype=torch.float32)
            action = torch.tensor(action, dtype=torch.int64)
            reward = torch.tensor(reward, dtype=torch.float32)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            weight = torch.tensor(weight, dtype=torch.int)
        
        q_values = self._q_func(state).gather(1, action.unsqueeze(1))

        next_q_values = self._target_q_func(next_state)
        target_q_values = reward + self._gamma * torch.max(next_q_values, dim=1).values * weight
        target_q_values = target_q_values.detach()

        loss = torch.nn.functional.mse_loss(q_values, target_q_values.unsqueeze(1))

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        # 间隔一定次数，更新一次target_q_func
        if update_target_q_func:
            self._target_q_func.load_state_dict(self._q_func.state_dict())


class DeepQFuncTester:
    def __init__(self, q_func: AbstractQFunc, env: Env) -> None:
        self._q_func = q_func
        self._env = env
        
    def test(self, max_step: int):
        init_state = self._env.reset()
        current_state = init_state
        acc_reward = 0
        reward_list = []
        # greedy_strategy = to_strategy(self._q_func)

        for _ in range(max_step):
            action = self._q_func.get_optimal_action(torch.tensor([current_state])).item()
            
            reward, next_state = self._env.step(action)
            acc_reward += reward
            reward_list.append(reward)
            current_state = next_state
            if current_state is None:
                break
    
        print(f'Test reward: {acc_reward}')
        print(f'Step Rewards: {reward_list}')
    
