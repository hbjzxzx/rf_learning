
from pathlib import Path
from typing import Optional, Callable
from itertools import count
import copy

import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from deep_q import AbstractQFunc, BatchedAction, BatchedActionProbVec, BatchedState, Discrete1ContinuousAction, TrainTracerInterface
from env import Env
from utils import get_logger

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
   
    def __deepcopy__(self, memo):
        # Create a new instance of the class
        new_copy = type(self)(
            self._state_dims, 
            self._action_nums, 
            self._hidden_dim, 
            self.get_device()
        )
        
        # Deep copy the attributes
        new_copy._fc1 = copy.deepcopy(self._fc1, memo)
        new_copy._fc2 = copy.deepcopy(self._fc2, memo)
        
        return new_copy 

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

EpochEndCallback = Callable[[int, float, PolicyNetFunc], bool]
#  REINFORCE
class PolicyNetTrainer:
    def __init__(self, policy_func: PolicyNetFunc,
                 env: Env,
                 learning_rate: float,
                 gamma: float,
                 action_converter: Optional[Discrete1ContinuousAction] = None,
                 logger_folder: Optional[Path] = None,
                 train_tracer: Optional[TrainTracerInterface] = None,
                 epoch_end_callback: Optional[EpochEndCallback] = None
                 ) -> None:
        self._policy_func = policy_func
        self._env = env

        self._learning_rate = learning_rate 
        self._gamma = gamma
        
        self._optimizer = torch.optim.Adam(self._policy_func.parameters(), lr=learning_rate)
        self._action_converter = action_converter
        self._logger_folder = logger_folder if logger_folder is not None else Path('./logs')
        self._train_tracer = train_tracer
        self._epoch_end_callback = epoch_end_callback
    
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

                    if next_state is not None:
                        trajectory_record_list.append(
                            (current_state, action, reward, next_state, 1)
                        )
                    else:
                        trajectory_record_list.append(
                            (current_state, action, reward, current_state, 0)
                        )

                    current_state = next_state
                    
                    if current_state is None:
                        break

                # start to update the policy net
                self.update(trajectory_record_list, writer, epoch)
                
                # write the log
                writer.add_scalar('reward', acc_reward, epoch)
                writer.add_scalar('step', acc_step_cnt, epoch)
                
                progress_bar.set_postfix({
                    'reward': f'{acc_reward:.2f}',
                    'step': f'{acc_step_cnt}'
                })
                if self._epoch_end_callback is not None:
                    self._epoch_end_callback(epoch, acc_reward, self._policy_func)
                
    
    def update(self, trajectory_record_list):
        states, actions, rewards, next_states, _ = zip(*trajectory_record_list)
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
            # REIFORCE 还要在梯度前乘以gamma 衰减
            loss = (-1 * full_gamma_weight_vec * batched_action_log_prob * weights_on_prob).sum()
            loss.backward()

            self._optimizer.step()


class ValueNetFunc(AbstractQFunc, torch.nn.Module):
    def __init__(self, 
                 state_dim: int, 
                 hidden_dim: int,
                 device: torch.device | None = None) -> None:
        AbstractQFunc.__init__(self, device=device)
        torch.nn.Module.__init__(self)
        
        self._state_dim = state_dim
        self._hidden_dim = hidden_dim
        
        with self.get_device():
            self._fc1 = torch.nn.Linear(state_dim, hidden_dim)
            self._fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self._fc1(x))
        return self._fc2(x)


# REINFORCE with base
class PolicyNetTrainerWithBase(PolicyNetTrainer):
    def __init__(self, policy_func: PolicyNetFunc,
                 value_func: ValueNetFunc,
                 env: Env,
                 value_learning_rate: float,
                 learning_rate: float,
                 gamma: float,
                 action_converter: Optional[Discrete1ContinuousAction] = None,
                 logger_folder: Optional[Path] = None,
                 train_tracer: Optional[TrainTracerInterface] = None,
                 ) -> None:
        super().__init__(policy_func, env, learning_rate, gamma, action_converter, logger_folder, train_tracer)
        self._value_func = value_func
        self._value_optimizer = torch.optim.Adam(self._value_func.parameters(), lr=value_learning_rate)

        self._target_value_func = copy.deepcopy(self._value_func).to(self._value_func.get_device())
        self._update_cnt = 1


    def update(self, trajectory_record_list, writer: SummaryWriter, epoch: int):
        states, actions, rewards, next_states, next_state_ok = zip(*trajectory_record_list)
        T = len(states)

        # 这里我们使用向量化的方式来计算
        with torch.device(self._policy_func.get_device()):
            self._optimizer.zero_grad()
            self._value_optimizer.zero_grad()

            batched_rewards = torch.tensor(np.array(rewards), dtype=torch.float)
            batched_states = torch.tensor(np.array(states), dtype=torch.float)
            batched_action_choosed = torch.tensor(np.array(actions), dtype=torch.int64)
            batched_next_state = torch.tensor(np.array(next_states), dtype=torch.float)
            batched_next_is_ok = torch.tensor(np.array(next_state_ok), dtype=torch.int64) 

            # record: action hist
            writer.add_histogram('action_hist', batched_action_choosed, epoch)
            writer.add_histogram('rewards', batched_rewards, epoch)


            # 就算全长gamma 衰减权重
            full_gamma_weight_vec = torch.pow(
                torch.full((T,), self._gamma, dtype=torch.float),
                torch.arange(0, T))       
            # 矩阵的每一个列j, 代表的是从j开始的全长gamma衰减权重
            full_gamma_weight_matrix = torch.stack(
                [full_gamma_weight_vec.roll(shift) for shift in range(T)],
                dim=-1
            )
            # 使用下三角矩阵处理每一列中多余的位置
            full_gamma_weight_matrix = torch.tril(full_gamma_weight_matrix)
            decays_return =batched_rewards.view(1, -1).mm(full_gamma_weight_matrix).squeeze()


            # 更新价值网路            

            # 使用TD 方式更新V
            # now_estimate_value = self._value_func.forward(batched_states).squeeze()
            # td_target = batched_rewards + self._gamma * self._value_func.forward(batched_next_state).squeeze() * batched_next_is_ok
                
            # value_loss = 0.5 * torch.nn.functional.mse_loss(td_target, now_estimate_value)

            # # 使用回归方式更新V
            value_loss = torch.nn.functional.mse_loss(decays_return, self._value_func.forward(batched_states).squeeze()) 
            value_loss.backward()

            writer.add_scalar('value_loss', value_loss, epoch)
            self._value_optimizer.step()

            # 更新策略网络

            now_estimate_value = self._value_func.forward(batched_states).squeeze()
            real_weights = decays_return - now_estimate_value.detach()

            # 计算每一个状态对应的Action的概率 
            batched_action_prob = self._policy_func.get_action_distribute(batched_states).gather(1, batched_action_choosed.unsqueeze(1))
            batched_action_log_prob = torch.log(batched_action_prob)

            # 计算总体的损失，注意这里要加上负号，因为我们要最大化这个值
            # REIFORCE 还要在梯度前乘以gamma 衰减
            loss = (-1 * full_gamma_weight_vec * batched_action_log_prob * real_weights).sum()
            loss.backward()

            writer.add_scalar('policy_target_value', loss, epoch)

            self._optimizer.step() 


class PolicyNetTester():
    def __init__(self, 
                 policy_fun: AbstractQFunc, 
                 env: Env,
                 action_converter: Optional[Discrete1ContinuousAction] = None,
                 stochastic: bool = False
                 ):
        self._policy_func = policy_fun
        self._env = env
        self._action_converter = action_converter
        self._stochastic = stochastic
        self._logger = get_logger('PolicyNetTester')
    
    def test(self, max_step: int):
        init_state = self._env.reset()
        current_state = init_state
        acc_reward = 0
        reward_list = []

        for _ in range(max_step):
            if not self._stochastic:
                action = self._policy_func.get_optimal_action(torch.tensor(np.array([current_state]))).item()
            else:
                action_dis = self._policy_func.get_action_distribute(torch.tensor(np.array([current_state])))
                action = torch.distributions.Categorical(action_dis).sample().item()
            
            reward, next_state = self._env.step(
                 action if self._action_converter is None else self._action_converter.to_continuous_action(action)
            )
            
            acc_reward += reward    
            reward_list.append(reward) 
            
            current_state = next_state
            if current_state is None:
                break
        
        return acc_reward, reward_list


class ActionStateValueNetFunc(AbstractQFunc, torch.nn.Module):
    def __init__(self, 
                 state_dim: int,
                 action_nums: int,
                 hidden_dim: int,
                 device: Optional[torch.device] = None):
        AbstractQFunc.__init__(self, device=device)
        torch.nn.Module.__init__(self) 

        self._state_dim = state_dim
        self._hidden_dim = hidden_dim

        with self.get_device():        
            self._fc1 = torch.nn.Linear(state_dim, hidden_dim)
            self._fc2 = torch.nn.Linear(hidden_dim, action_nums)
    
    def forward(self, x):
        with self.get_device():
            x = torch.nn.functional.relu(self._fc1(x))
            return self._fc2(x)
    
    def get_values(self, state_batch: BatchedState, action_batch: BatchedAction) -> torch.Tensor:
        return self.forward(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze()
    
# Actor-Critic
class PolicyValueNetTrainer(PolicyNetTrainer):
    def __init__(self, policy_func: PolicyNetFunc,
                 value_func: ActionStateValueNetFunc,
                 env: Env,
                 learning_rate: float,
                 vlearning_rate: float,
                 gamma: float,
                 action_converter: Optional[Discrete1ContinuousAction] = None,
                 logger_folder: Optional[Path] = None,
                 train_tracer: Optional[TrainTracerInterface] = None,
                 epoch_end_callback: Optional[EpochEndCallback] = None
                 ) -> None:
        super().__init__(policy_func, env, learning_rate, gamma, action_converter, logger_folder, train_tracer,
                         epoch_end_callback=epoch_end_callback)

        self._value_func = value_func
        self._value_optimizer = torch.optim.Adam(self._value_func.parameters(), lr=vlearning_rate)
        self._target_func = copy.deepcopy(value_func)

    
    def update(self, trajectory_record_list, writer: SummaryWriter, epoch: int):
        states, actions, rewards, next_states, next_state_ok = zip(*trajectory_record_list)
        T = len(states)

        # 这里我们使用向量化的方式来计算
        with torch.device(self._policy_func.get_device()):
            self._optimizer.zero_grad()
            self._value_optimizer.zero_grad()

            batched_rewards = torch.tensor(np.array(rewards), dtype=torch.float)
            batched_states = torch.tensor(np.array(states), dtype=torch.float)
            batched_action_choosed = torch.tensor(np.array(actions), dtype=torch.int64)
            batched_next_state = torch.tensor(np.array(next_states), dtype=torch.float)
            batched_next_is_ok = torch.tensor(np.array(next_state_ok), dtype=torch.int64)

            # TD target:
            maybe_next_action_prob = self._policy_func.get_action_distribute(batched_next_state).detach()
            maybe_next_action = torch.distributions.Categorical(maybe_next_action_prob).sample()
            td_target = batched_rewards + self._gamma * self._target_func.get_values(batched_next_state, maybe_next_action) * batched_next_is_ok
            now_value_estimated = self._value_func.get_values(batched_states, batched_action_choosed)
            value_net_loss = torch.nn.functional.mse_loss(now_value_estimated, td_target.detach())
            value_net_loss.backward()
            
            writer.add_scalar('critic net loss', value_net_loss, epoch)
            self._value_optimizer.step()
            if epoch % 10 == 9:
                self._target_func.load_state_dict(self._value_func.state_dict()) 

            # 就算全长gamma 衰减权重
            full_gamma_weight_vec = torch.pow(
                torch.full((T,), self._gamma, dtype=torch.float),
                torch.arange(0, T))        

            # 计算每一个状态对应的Action的概率 
            batched_action_prob = self._policy_func.get_action_distribute(batched_states).gather(1, batched_action_choosed.unsqueeze(1))
            batched_action_log_prob = torch.log(batched_action_prob)

            # 计算总体的损失，注意这里要加上负号，因为我们要最大化这个值
            loss = (-1 * full_gamma_weight_vec * batched_action_log_prob * now_value_estimated.detach()).sum()
            loss.backward()
            
            writer.add_scalar('policy net loss', loss, epoch)
            self._optimizer.step()
            

# TRPO
class PolicyNetTrainerWithTRPO(PolicyNetTrainer):
    def __init__(self, policy_func: PolicyNetFunc,
                 value_func: ValueNetFunc,
                 env: Env,
                 value_learning_rate: float,
                 learning_rate: float,
                 gamma: float,
                 kl_threadhold: float,
                 search_alpha: float,
                 action_converter: Optional[Discrete1ContinuousAction] = None,
                 logger_folder: Optional[Path] = None,
                 train_tracer: Optional[TrainTracerInterface] = None,
                 ) -> None:
        super().__init__(policy_func, env, learning_rate, gamma, action_converter, logger_folder, train_tracer)
        self._value_func = value_func
        self._value_optimizer = torch.optim.Adam(self._value_func.parameters(), lr=value_learning_rate)

        self._target_value_func = copy.deepcopy(self._value_func).to(self._value_func.get_device())
        self._update_cnt = 1

        self._kl_threadhold = kl_threadhold
        self._search_alpha = search_alpha

    def update(self, trajectory_record_list, writer: SummaryWriter, epoch: int):
        with torch.device(self._policy_func.get_device()):
            self._update_with_trpo(trajectory_record_list, writer, epoch)

    
    def _compute_surrogate_obj(self, 
                               batched_state, 
                               batched_action, 
                               batched_weights,
                               old_policy,
                               new_policy): 
        old_log_prob = torch.log(old_policy(batched_state).gather(1, batched_action.view(-1, 1)))
        new_log_prob = torch.log(new_policy(batched_state).gather(1, batched_action.view(-1, 1)))
        return torch.mean(
            batched_weights * (old_log_prob - new_log_prob)
        )
    
    def _hessian_vector_product(self, batched_states, 
                                old_policy: torch.nn.Module, 
                                new_policy: torch.nn.Module, 
                                vector: torch.Tensor):
        kl_divergence = torch.mean(
            torch.distributions.kl.kl_divergence(
                torch.distributions.Categorical(old_policy(batched_states)),
                torch.distributions.Categorical(new_policy(batched_states))
            )
        )
        print('kl_divergence is: ', kl_divergence)
        kl_grad = torch.autograd.grad(kl_divergence, old_policy.parameters() , create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        print('kl_grad_vector is: ', kl_grad_vector)
        kl_grad_vector_product = kl_grad_vector.dot(vector) 
        grad2 = torch.autograd.grad(kl_grad_vector_product, old_policy.parameters())
        grad2_vector = torch.cat([grad.view(-1) for grad in grad2])

        return grad2_vector
        
    def _conjugate_gradient(self, grad, state, old_policy, new_policy):
        x = torch.zeros_like(grad)
        print('x ini: : ', x)
        r = grad.clone()
        p = grad.clone()
        rdotr = r.dot(r)
        for i in range(10):
            hp = self._hessian_vector_product(state, old_policy, new_policy, p)
            print(f'hp at {i} is: ', hp)
            alpha = rdotr / p.dot(hp)
            x += alpha * p
            print(f'x at {i} is: ', x)
            r -= alpha * hp
            new_rdotr = r.dot(r)
            if new_rdotr < 1e-10:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
        print('x finally: : ', x)
        return x


    def _linear_search(self, batched_states, batched_action, weights, old_policy: PolicyNetFunc, descent_dir):
        old_obj = self._compute_surrogate_obj(batched_states, batched_action, weights, old_policy, old_policy)
        old_params = torch.nn.utils.convert_parameters.parameters_to_vector(old_policy.parameters())
                
        for i in range(15):
            coef = self._search_alpha ** i
            new_params = old_params + coef * descent_dir
            new_actor = copy.deepcopy(old_policy)
            torch.nn.utils.convert_parameters.vector_to_parameters(new_params, new_actor.parameters())
            print('coef is: ', coef)
            print('descent is: ', descent_dir)
            x = new_actor.get_action_distribute(batched_states)
            print(x)
            y = old_policy.get_action_distribute(batched_states)
            print('y is: ', y)
            new_action_dist = torch.distributions.Categorical(x)
            kl_divergence = torch.mean(
                torch.distributions.kl.kl_divergence(
                    torch.distributions.Categorical(old_policy(batched_states)),
                    new_action_dist
                )
            )
            new_obj = self._compute_surrogate_obj(batched_states, batched_action, weights, old_policy, new_actor)
            if new_obj > old_obj and kl_divergence < self._kl_threadhold:
                return new_params
        return old_params


    def _update_with_trpo(self, trajectory_record_list, writer: SummaryWriter, epoch: int):
        states, actions, rewards, next_states, next_state_ok = zip(*trajectory_record_list)
        T = len(states)

        batched_rewards = torch.tensor(np.array(rewards), dtype=torch.float)
        batched_states = torch.tensor(np.array(states), dtype=torch.float)
        batched_action_choosed = torch.tensor(np.array(actions), dtype=torch.int64)
        batched_next_state = torch.tensor(np.array(next_states), dtype=torch.float)
        batched_next_is_ok = torch.tensor(np.array(next_state_ok), dtype=torch.int64) 

        # record: action hist
        writer.add_histogram('action_hist', batched_action_choosed, epoch)
        writer.add_histogram('rewards', batched_rewards, epoch)

        # 计算gamma衰减的权重
            # 就算全长gamma 衰减权重
        full_gamma_weight_vec = torch.pow(
                torch.full((T,), self._gamma, dtype=torch.float),
                torch.arange(0, T))       
            # 矩阵的每一个列j, 代表的是从j开始的全长gamma衰减权重
        full_gamma_weight_matrix = torch.stack(
                [full_gamma_weight_vec.roll(shift) for shift in range(T)],
                dim=-1
            )
            # 使用下三角矩阵处理每一列中多余的位置
        full_gamma_weight_matrix = torch.tril(full_gamma_weight_matrix)
        decays_return =batched_rewards.view(1, -1).mm(full_gamma_weight_matrix).squeeze()


        # old_action_choosed_prob = self._policy_func.get_action_distribute(batched_states).gather(1, batched_action_choosed.unsqueeze(1)).suqeeze()
        # old_action_dist = torch.distributions.Categorical(self._policy_func.get_action_distribute(batched_states))

        surrogate_obj = self._compute_surrogate_obj(
            batched_state=batched_states,
            batched_action=batched_action_choosed,
            batched_weights=decays_return,
            old_policy=self._policy_func,
            new_policy=self._policy_func
        )
        grads = torch.autograd.grad(surrogate_obj, self._policy_func.parameters())
        # 计算此时的梯度向量 g
        object_grad = torch.cat([grad.view(-1) for grad in grads]).detach()

        # 使用共轭梯度计算 x = H^-1 * g
        descent_dir = self._conjugate_gradient(object_grad, batched_states, self._policy_func, self._policy_func) 
        hd = self._hessian_vector_product(batched_states, self._policy_func, self._policy_func, descent_dir)
        max_coef = torch.sqrt(2 * self._kl_threadhold /
                              (torch.dot(descent_dir, hd) + 1e-8))
        
        new_para = self._linear_search(
            batched_states, 
            batched_action_choosed, 
            decays_return, 
            self._policy_func, 
            descent_dir * max_coef)

        torch.nn.utils.convert_parameters.vector_to_parameters(new_para, self._policy_func.parameters())
        
        