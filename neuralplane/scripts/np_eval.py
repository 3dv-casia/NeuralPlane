from dataclasses import dataclass
from pathlib import Path

import json
import tyro

from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.scripts.train import _set_random_seed

from neuralplane.evaluation.eval_geo import GeoEval
from neuralplane.evaluation.eval_segm import SegmEval
# from neuralplane.evaluation.eval_planar import PlanarEval

from neuralplane.utils.yaml import load_yaml, update_yaml

@dataclass
class Args:
    config: Path = Path("config.yaml")
    '''Path to the configuration file'''

class Eval:
    def __init__(self, args: Args):
        self.config_yaml_path = args.config
        self.config = load_yaml(args.config)
        self.seed = self.config.MACHINE.SEED
        self.write_file = self.config.TRAIN.NS_CONFIG.parent / f"{self.config.DATA.SCENE_ID}_eval.json"

        _set_random_seed(self.seed)
    def run(self):

        CONSOLE.log(f"Running evaluation for {self.config.DATA.SCENE_ID}")

        metrics = {}
        metrics.update(
            GeoEval(save_error_map=True).run(
                gt_anno_path=self.config.DATA.ANNOTATION,
                pred_path=self.config.EVAL.EXPORT_DIR / self.config.EVAL.GEO_VERTS
            )
        )
        metrics.update(
            SegmEval().run(
                gt_anno_path=self.config.DATA.ANNOTATION,
                pred_path=self.config.EVAL.EXPORT_DIR / self.config.EVAL.SEGM_LABELS
            )
        )

        data = {
            "scene_id": self.config.DATA.SCENE_ID,
            "metrics": metrics
        }

        CONSOLE.log(f"Result written to {self.write_file}")
        with open(self.write_file, "w") as f:
            json.dump(data, f, indent=2)

        self.config.EVAL.RESULT = self.write_file
        update_yaml(self.config_yaml_path, self.config)

        CONSOLE.rule(f"[bold yellow]scene{self.config.DATA.SCENE_ID}[/]: Evaluation Completed")

        pass

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    Eval(
        tyro.cli(Args)
    ).run()

if __name__ == "__main__":
    entrypoint()