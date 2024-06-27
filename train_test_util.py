from pathlib import Path
import datetime
from typing import Any, Optional
from dataclasses import dataclass

from tqdm import tqdm

from env import Env
from utils import show_gif_on_jupyternb, to_gif, get_logger, clear_target_path


_logger = get_logger(name='train_test_util')


@dataclass
class StandarTrainProcess:
    trainer: Any
    model: Any
    train_epoch: int

    log_path: Path
    model_path: Path


@dataclass
class StandarTestProcess:
    model: Any
    tester: Any
    env: Env
    test_output_path: Optional[Path] = None

    test_epoch: int = 1
    show_result: bool = True


def start_test(test_process: StandarTestProcess) -> None:
    if test_process.show_result:
        _logger.warning('show_result is True, this will slow down the test process and only test 1 epoch')
        test_process.test_epoch = 1

    reward_lst = [] 
    _logger.info(f'start testing, now datetime: {datetime.datetime.now()}, test_epoch: {test_process.test_epoch}')
    for e in tqdm(range(test_process.test_epoch), unit='epoch'):
        reward, _ = test_process.tester.test(1000)
        reward_lst.append(reward) 
    _logger.info(f'end testing, now datetime: {datetime.datetime.now()}') 
    
    if test_process.show_result:
        test_output_path = test_process.test_output_path
        RESULT_GIF = test_output_path / 'result.gif'
        clear_target_path(RESULT_GIF)
        to_gif(test_process.env._gym_env, RESULT_GIF, 1/30)
        show_gif_on_jupyternb(RESULT_GIF)
    
    return sum(reward_lst) / len(reward_lst)


def start_train(train_process: StandarTrainProcess) -> None:

    log_path = train_process.log_path 
    model_path = train_process.model_path
    _logger.info(f'start training, now datetime: {datetime.datetime.now()}')
    _logger.info(f'First, clean log path: {log_path}, and clean model path: {model_path}')
    clear_target_path(log_path)
    clear_target_path(model_path)

    try:
        _logger.info('train started') 
        train_process.trainer.train(train_epoch=train_process.train_epoch)
        _logger.info(f'end training, now datetime: {datetime.datetime.now()}')
    except KeyboardInterrupt:
        _logger.warning('training interrupted')
    except Exception as e:
        _logger.exception(f'error occured: {e}')
    finally:
        _logger.info(f'saving model to: {model_path},') 
        train_process.model.save(model_path)
