from pathlib import Path
import shutil
import sys
import traceback

import gymnasium as gym
import imageio
from loguru import logger


def traceback_filter(record):
    # Filter out frames from 'ipykernel' and 'jupyter' modules
    if not(record["exception"] and record["exception"].traceback):
        return True
    tb_list = traceback.extract_tb(record["exception"].traceback)
    tb_list = [tb for tb in tb_list if 'ipykernel' not in tb.filename and 'jupyter' not in tb.filename]
    return ''.join(traceback.format_list(tb_list))

# Configure the logger format
logger.configure(handlers=[
    {
        "sink": sys.stdout, 
        "format": "<level>{level: <8}</level> | <cyan>{name}</cyan>: - <level>{message}</level>; <green>{time:YYYY-MM-DD HH:mm:ss}</green> <cyan>{function}</cyan>:<cyan>{line}</cyan>",
        "filter": traceback_filter
    }
])

_logger = logger.bind(name='utils')


def clear_target_path(p: Path) -> None:
    if p.exists():
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
    else:
        _logger.info(f'clear_target_path: {str(p)} dose not exist')
        


def show_gif_on_jupyternb(p: Path) -> None:
    from IPython.display import Image, display
    image = Image(p, format='gif')
    return display(image)
    

def to_gif(e: gym.Env, path: Path, duration: float = 1/30) -> None:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    frames = e.render()
    imageio.mimsave(path, frames, duration=duration, format='.gif')
