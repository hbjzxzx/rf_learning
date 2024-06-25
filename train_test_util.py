from pathlib import Path
import datetime
from typing import Any

from utils import clear_target_path
from loguru import logger
from dataclasses import dataclass

from env import Env
from utils import show_gif_on_jupyternb, to_gif


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
    test_output_path: Path 

    test_epoch: int = 1


def start_test(test_process: StandarTestProcess) -> None:
    test_output_path = test_process.test_output_path

    RESULT_GIF = test_output_path / 'result.gif'
    clear_target_path(RESULT_GIF)
    test_process.tester.test(1000)

    to_gif(test_process.env._gym_env, RESULT_GIF, 1/30)
    show_gif_on_jupyternb(RESULT_GIF)


def start_train(train_process: StandarTrainProcess) -> None:
    _logger = logger.bind(name='train_test_util')

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
