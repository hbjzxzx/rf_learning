from pathlib import Path
import shutil
import gymnasium as gym
import imageio

def clear_target_path(p: Path) -> None:
    if p.exists():
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()


def show_gif_on_jupyternb(p: Path) -> None:
    from IPython.display import Image, display
    image = Image(p, format='gif')
    return display(image)
    

def to_gif(e: gym.Env, path: Path, duration: float = 1/30) -> None:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    frames = e.render()
    imageio.mimsave(path, frames, duration=duration, format='.gif')
    