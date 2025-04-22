from dataclasses import dataclass
from pathlib import Path

import json
import numpy as np
import tyro

from neuralplane.utils.yaml import load_yaml

from nerfstudio.utils.rich_utils import CONSOLE

@dataclass
class Args:
    split_file: Path=Path("./data/8scenes.txt")
    '''Path to the split file'''
    config_path: Path=Path("./configs/2025-Mar-25-Tue")
    '''Path to the configuration file'''

class EvalStats:
    def __init__(self, args: Args):
        self.split_file = args.split_file
        self.scenes = []
        with open(self.split_file, "r") as f:
            for line in f.readlines():
                scene = line.strip()
                self.scenes.append(scene)
        self.config_path = args.config_path

    def run(self):
        failed = []
        metrics_count = 0
        metrics = {"geo": [], "segm": []}
        for scene in self.scenes:
            config_file = self.config_path / f"{scene}.yaml"
            if not config_file.exists():
                print(f"Config file {config_file} not found, skipping scene {scene}")
                failed.append(scene)
                continue
            config = load_yaml(config_file)

            try:
                result_file = config.EVAL.RESULT
                with open(result_file, 'r') as f:
                    data = json.load(f)

                geo = data["metrics"]["geo"]
                segm = data["metrics"]["segm"]

                metrics_count += 1

            except:
                print(f"Result file not found, skipping scene {scene}")
                failed.append(scene)
                continue

            metrics["geo"].append(geo)
            metrics["segm"].append(segm)

        # Calculate average metrics
        def cal_mean(metrics_dicts):
            metrics = {}
            for key in metrics_dicts[0].keys():
                metrics[key] = np.mean([metrics_dict[key] for metrics_dict in metrics_dicts])
            return metrics

        metric_avg = {}
        for key in metrics.keys():
            metric_avg[key] = cal_mean(metrics[key])

        CONSOLE.log(f"Average metrics for {self.split_file}:")
        CONSOLE.log(f"Geo: {metric_avg['geo']}")
        CONSOLE.log(f"Segm: {metric_avg['segm']}")
        CONSOLE.log(f"Number of scenes: {metrics_count}")
        CONSOLE.log(f"Failed scenes: {failed}")

        # Save the average metrics to a file
        output_file = self.split_file.parent / f"{self.split_file.stem}_avg.json"
        with open(output_file, 'w') as f:
            json.dump(metric_avg, f, indent=2)
        CONSOLE.log(f"Average metrics saved to {output_file}")

        pass

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    args = tyro.cli(Args)
    EvalStats(args).run()
    pass


if __name__ == "__main__":
    entrypoint()