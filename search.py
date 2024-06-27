from pathlib import Path
import datetime

import gymnasium as gym
import torch
import optuna

from policy_based import PolicyNetFunc, PolicyNetTrainer, PolicyNetTester, PolicyNetTrainerWithBase, ValueNetFunc
from deep_q import Discrete1ContinuousAction
from env import Env, get_action_discreter
from utils import clear_target_path, show_gif_on_jupyternb, to_gif
from train_test_util import start_test, start_train, StandarTestProcess, StandarTrainProcess
from policy_based import PolicyValueNetTrainer, ActionStateValueNetFunc, EpochEndCallback


GYM_ENV_NAME = 'CartPole-v1'
RESULT_DIR_NAME='cartpoleV1'

_USE_CUDA = True and torch.cuda.is_available()
# _USE_CUDA = False and torch.cuda.is_available()

def objective(trial: optuna.Trial):
    t_number = trial.number
    env = Env.from_env_name(GYM_ENV_NAME)

    LOG_PATH = Path(f'./run/logs/{RESULT_DIR_NAME}/{t_number}/AC')
    MODEL_PATH = Path(f'./run/model/{RESULT_DIR_NAME}/AC_{t_number}/AC.pth')

    
    TRAIN_EPOCH = trial.suggest_int(name='train_epoch', low=1000, high=5000, step=1000)
    HIDDEN_DIM_POLICY = trial.suggest_categorical(name='HIDDEN_DIM_POLICY', choices=[128, 256, 512])
    HIDDEN_DIM_VALUE = trial.suggest_categorical(name='HIDDEN_DIM_VALUE', choices=[128, 256, 512])
    
    LEARNING_RATE = trial.suggest_float('p_learn_rate', low=5e-5, high=1e-1, log=True)
    VLEARNING_RATE = trial.suggest_float('v_learn_rate', low=5e-5, high=1e-1, log=True)
    GAMMA = trial.suggest_categorical(name='gamma', choices=[0.8, 0.9, 0.95, 0.99])


    policy_func = PolicyNetFunc(env.get_state_dim()[0], 
                       action_nums=env.get_action_dim()[0], 
                       hidden_dim=HIDDEN_DIM_POLICY,
                       device=torch.device('cuda') if _USE_CUDA else None)

    value_func = ActionStateValueNetFunc(env.get_state_dim()[0],
                              action_nums=env.get_action_dim()[0],
                              hidden_dim=HIDDEN_DIM_VALUE,
                              device=torch.device('cuda') if _USE_CUDA else None)

    def epoch_end_callback(epoch, avg_reward, policy):
        trial.report(avg_reward, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    policy_func_trainer = PolicyValueNetTrainer(
                                      policy_func=policy_func,
                                      value_func=value_func,
                                      vlearning_rate=VLEARNING_RATE,
                                      env=env,
                                      learning_rate=LEARNING_RATE,
                                      gamma=GAMMA,
                                      logger_folder=LOG_PATH,
                                      epoch_end_callback=epoch_end_callback
                                      )
    
    start_train(
        StandarTrainProcess(
            trainer=policy_func_trainer,
            model=policy_func,
            train_epoch=TRAIN_EPOCH,
            log_path=LOG_PATH,
            model_path=MODEL_PATH
        )
    )

    policy_func_tester = PolicyNetTester(
        policy_fun=policy_func,
        env=env
    )
    
    avg_reward = start_test(
        StandarTestProcess(
            model=policy_func,
            tester=policy_func_tester,
            env=env,
            test_epoch=100,
            show_result=False
        )
    )

    return avg_reward


class CombinedPruner(optuna.pruners.BasePruner):
    def __init__(self, custom_threshold, median_n_startup_trials=5, median_n_warmup_steps=1500):
        self.custom_threshold = custom_threshold
        self.median_pruner = optuna.pruners.MedianPruner(
            n_startup_trials=median_n_startup_trials,
            n_warmup_steps=median_n_warmup_steps
        )

    def prune(self, study, trial):
        # 自定义剪枝逻辑
        step = trial.last_step
        if step is not None:
            intermediate_value = trial.intermediate_values.get(step)
            if step > 1000 and intermediate_value < self.custom_threshold:
                return True

        # MedianPruner的剪枝逻辑
        return self.median_pruner.prune(study, trial)


if __name__ == '__main__':
    # 使用组合剪枝器
    pruner = CombinedPruner(custom_threshold=80)
    study = optuna.create_study(direction='maximize', pruner=pruner, storage='sqlite:///example_study.db')
    study.optimize(objective, n_trials=100, n_jobs=8)
