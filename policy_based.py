
from pathlib import Path
from typing import Optional
from itertools import count
import copy

import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from deep_q import AbstractQFunc, BatchedAction, BatchedActionProbVec, BatchedState, Discrete1ContinuousAction, TrainTracerInterface
from env import Env


class PolicyNetFunc(AbstractQFunc, torch.nn.Module):
    def __init__(self, state_dim: int, 
                 action_nums: int, 
                 hidden_dim: int,
                 device: Optional[torch.device] = None):
        AbstractQFunc.__init__(self, device=device)
        torch.nn.Module.__init__(self)
        
        self._state_dims = state_dim 
        self._action_nums = action_nums
        self._hidden_dim = hidden_dim

        with self.get_device():
            self._fc1 = torch.nn.Linear(state_dim, hidden_dim)
            self._fc2 = torch.nn.Linear(hidden_dim, action_nums)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self._fc1(x))
        x = torch.nn.functional.softmax(self._fc2(x), dim=1)
        return x

    def get_action_distribute(self, state_batch: BatchedState) -> BatchedActionProbVec:
        x = self.forward(state_batch)
        return x
   
    def get_optimal_action(self, state: BatchedState) -> BatchedAction:
        out = self.forward(state)
        return out.argmax(dim=1).detach()
    
    def get_actions_count(self) -> int:
        return self._action_nums

    def save(self, path: Path):
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                'model_state_dict': self.state_dict(),
                'meta_info': {
                    'state_dim': self._state_dims,
                    'action_nums': self._action_nums,
                    'hidden_dim': self._hidden_dim
                }
            }, path)
    
    def load(self, path: Path):
        checkpoint = torch.load(path)
        action_nums = checkpoint['meta_info']['action_nums']
        state_dim = checkpoint['meta_info']['state_dim']
        hidden_dim = checkpoint['meta_info']['hidden_dim']
        weight_state_dict = checkpoint['model_state_dict']

        assert self._action_nums == action_nums and self._state_dims == state_dim and self._hidden_dim == hidden_dim, 'Model structure is not match'

        self._action_nums = action_nums        
        self._state_dims = state_dim
        self.load_state_dict(weight_state_dict)
    
    @classmethod
    def from_file(cls, path: Path, device: Optional[torch.device]=None):
        checkpoint = torch.load(path)
        action_nums = checkpoint['meta_info']['action_nums']
        state_dim = checkpoint['meta_info']['state_dim']
        hidden_dim = checkpoint['meta_info']['hidden_dim']
        weights = checkpoint['model_state_dict']
        policy_func = cls(state_dim=state_dim,
                     action_nums=action_nums, hidden_dim=hidden_dim,  device=device)
        policy_func.load_state_dict(weights)
        return policy_func

    
class PolicyNetTrainer:
    def __init__(self, policy_func: PolicyNetFunc,
                 env: Env,
                 learning_rate: float,
                 gamma: float,
                 action_converter: Optional[Discrete1ContinuousAction] = None,
                 logger_folder: Optional[Path] = None,
                 train_tracer: Optional[TrainTracerInterface] = None,
                 ) -> None:
        self._policy_func = policy_func
        self._env = env

        self._learning_rate = learning_rate 
        self._gamma = gamma
        
        self._optimizer = torch.optim.Adam(self._policy_func.parameters(), lr=learning_rate)
        self._action_converter = action_converter
        self._logger_folder = logger_folder if logger_folder is not None else Path('./logs')
        self._train_tracer = train_tracer
    
    def train(self, train_epoch: int):
        with SummaryWriter(self._logger_folder) as writer:
            progress_bar = tqdm(range(train_epoch))
            for epoch in progress_bar:
                init_state = self._env.reset()
                current_state = init_state

                acc_reward = 0
                acc_step_cnt = 0
                trajectory_record_list = []

                l =  copy.deepcopy(self._policy_func).to('cpu')
                for step in count(0):
                    acc_step_cnt += 1
                    # with torch.device(self._policy_func.get_device()):
                    action_distribute = l.get_action_distribute(torch.tensor(np.array([current_state])))
                    action = torch.distributions.Categorical(action_distribute).sample().item()
                    reward, next_state = self._env.step(
                        action if self._action_converter is None else self._action_converter.to_continuous_action(action)
                        )
                    acc_reward += reward
                    trajectory_record_list.append(
                        (current_state, action, reward, next_state)
                    )

                    current_state = next_state
                    
                    if current_state is None:
                        break

                # start to update the policy net
                self.update(trajectory_record_list)
                
                # write the log
                writer.add_scalar('reward', acc_reward, epoch)
                writer.add_scalar('step', acc_step_cnt, epoch)
                
                progress_bar.set_postfix({
                    'reward': f'{acc_reward:.2f}',
                    'step': f'{acc_step_cnt}'
                })
                
    
    def update(self, trajectory_record_list):
        states, actions, rewards, next_states = zip(*trajectory_record_list)
        T = len(states)

        # 这里我们使用向量化的方式来计算
        with torch.device(self._policy_func.get_device()):
            self._optimizer.zero_grad()

            batched_rewards = torch.tensor(np.array(rewards), dtype=torch.float)
            batched_states = torch.tensor(np.array(states), dtype=torch.float)
            batched_action_choosed = torch.tensor(np.array(actions), dtype=torch.int64)

            # 就算全长gamma 衰减权重
            full_gamma_weight_vec = torch.pow(
                torch.full((T,), self._gamma, dtype=torch.float),
                torch.arange(1, T+1))        
            # 矩阵的每一个列j, 代表的是从j开始的全长gamma衰减权重
            full_gamma_weight_matrix = torch.stack(
                [full_gamma_weight_vec.roll(shift) for shift in range(T)],
                dim=1
            )
            # 使用下三角矩阵处理每一列中多余的位置
            full_gamma_weight_matrix = torch.tril(full_gamma_weight_matrix)
            
            weights_on_prob =batched_rewards.view(1, -1).mm(full_gamma_weight_matrix).squeeze()
            
            # 计算每一个状态对应的Action的概率 
            batched_action_prob = self._policy_func.get_action_distribute(batched_states).gather(1, batched_action_choosed.unsqueeze(1))
            batched_action_log_prob = torch.log(batched_action_prob)

            # 计算总体的损失，注意这里要加上负号，因为我们要最大化这个值

            loss = (-1 * batched_action_log_prob * weights_on_prob).sum()
            loss.backward()

            self._optimizer.step()

    
class PolicyNetTester():
    def __init__(self, 
                 policy_fun: AbstractQFunc, 
                 env: Env,
                 action_converter: Optional[Discrete1ContinuousAction] = None,
                 ):
        self._policy_func = policy_fun
        self._env = env
        self._action_converter = action_converter
    
    def test(self, max_step: int):
        init_state = self._env.reset()
        current_state = init_state
        acc_reward = 0
        reward_list = []
        
        for _ in range(max_step):
            action = self._policy_func.get_optimal_action(torch.tensor(np.array([current_state]))).item()
            
            reward, next_state = self._env.step(
                 action if self._action_converter is None else self._action_converter.to_continuous_action(action)
            )
            
            acc_reward += reward    
            reward_list.append(reward) 
            
            current_state = next_state
            if current_state is None:
                break
        
        print(f'Test Reward: {acc_reward}')
        print(f'Step Rewards: {reward_list}')
            