from pathlib import Path
import shutil

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
    