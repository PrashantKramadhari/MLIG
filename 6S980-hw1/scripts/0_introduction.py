from jaxtyping import install_import_hook

import sys
import os

#SCRIPT_DIR = os.path.dirname(os.path.abspath('/media/ubuntu/Data/project/MLIG/Ass1/6S980-hw1/src/provided_code.py'))
sys.path.append(os.path.dirname("src"))


# Add runtime type checking to all imports.
with install_import_hook(("src",), ("beartype", "beartype")):
    from src.provided_code import get_bunny, plot_point_cloud

if __name__ == "__main__":
    vertices, _ = get_bunny()
    plot_point_cloud(
        vertices,
        xlim=(-1.0, 1.0),
        ylim=(-1.0, 1.0),
        zlim=(-1.0, 1.0),
    )
