from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import os
import gdown
import tyro

from nerfstudio.utils.rich_utils import CONSOLE

@dataclass
class Args:
    """Download pretrained monocular prediction models."""
    dst: Path
    """Path to the directory to save the models."""
    proxy: Optional[str] = None

def download_models(args: Args):

    if not args.dst.exists():
        args.dst.mkdir(parents=True, exist_ok=True)
    # sam_vit_h_4b8939
    if not (args.dst / "sam_vit_h_4b8939.pth").exists():
        cmd = f"wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P {args.dst}"
        CONSOLE.log("Downloading sam_vit_h_4b8939.pth: " + cmd)
        os.system(cmd)
    else:
        CONSOLE.log("Skipping: sam_vit_h_4b8939.pth already exists.")

    if not (args.dst / "snu_scannet.pt").exists():
        url = "https://drive.google.com/file/d/1lOgY9sbMRW73qNdJze9bPkM2cmfA8Re-/view?usp=sharing"
        CONSOLE.log("Downloading snu_scannet.pt: gdown" + url)
        gdown.download(
            url=url,
            output=str(args.dst / "snu_scannet.pt"),
            fuzzy=True,
            proxy=args.proxy
        )
    else:
        CONSOLE.log("Skipping: snu_scannet.pt already exists.")

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    download_models(tyro.cli(Args))

if __name__ == "__main__":
    entrypoint()