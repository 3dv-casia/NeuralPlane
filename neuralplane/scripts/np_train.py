from dataclasses import dataclass, field
from pathlib import Path
from easydict import EasyDict as edict

import tyro

from neuralplane.neuralplane_config import neuralplane_method
from neuralplane.utils.yaml import load_yaml, update_yaml

from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.scripts.train import main as train
from nerfstudio.utils.rich_utils import CONSOLE

@dataclass
class Args:
    config: Path = Path("config.yaml")
    '''Path to the configuration file'''
    CFG: TrainerConfig = field(default_factory=lambda: neuralplane_method.config)
    """[Nerfstudio Configurations] rebuilds indoor scenes as arrangements of planar primitives"""

class NeuralPlane:
    def __init__(self, args: Args):
        yaml_config = load_yaml(args.config)

        CFG = args.CFG

        CFG.output_dir = yaml_config.DATA.OUTPUT
        CFG.timestamp = yaml_config.TIMESTAMP

        CFG.experiment_name = yaml_config.DATA.SCENE_ID
        CFG.machine.seed = yaml_config.MACHINE.SEED

        CFG.pipeline.datamanager.cache_path = yaml_config.DATA.CACHE
        CFG.pipeline.datamanager.dataparser.data = yaml_config.DATA.SOURCE / yaml_config.DATA.SCENE_ID

        out_dir = CFG.output_dir / CFG.experiment_name / "neuralplane" / CFG.timestamp
        if out_dir.exists():
            CONSOLE.log(f"Output directory {out_dir} already exists. Remove ...")
            import shutil
            shutil.rmtree(out_dir)
            CONSOLE.log(f"Output directory {out_dir} removed.")

        train(CFG)

        yaml_config.TRAIN = edict()
        yaml_config.TRAIN.NS_CONFIG = CFG.get_base_dir() / "config.yml"

        CONSOLE.log(f"Updating configuration file: {args.config}")
        update_yaml(args.config, yaml_config)

        CONSOLE.rule(f"[bold yellow]scene{yaml_config.DATA.SCENE_ID}[/]: Training Completed")

        pass

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    NeuralPlane(
        tyro.cli(Args)
    )

if __name__ == "__main__":
    entrypoint()