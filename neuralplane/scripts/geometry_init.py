from pathlib import Path
from dataclasses import dataclass, field

import tyro
import torch
import numpy as np
import h5py

from typing import Literal
from rich.progress import track

from hloc import extract_features, pairs_from_retrieval, match_dense, triangulation
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.scannet_dataparser import ScanNetDataParserConfig
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.scripts.train import _set_random_seed

from neuralplane.frontend.keypoints.colmap import create_empty_pycolmap_reconstruction
from neuralplane.utils.yaml import load_yaml, update_yaml

match_conf = {
    "output": "matches-loftr",
    "model": {
        "name": "loftr",
        "weights": "indoor"
        # "weights": "outdoor"
    },
    "preprocessing": {"grayscale": True, "resize_max": 1024, "dfactor": 8},
    "max_error": 1,  # max error for assigned keypoints (in px)
    "cell_size": 1,  # size of quantization patch (max 1 kp/patch)
}

@dataclass
class Args:
    config: Path = Path("config.yaml")
    '''Path to the configuration file'''
    mode: Literal["sfm"] = "sfm"
    vis: bool = False
    '''Visualize the triangulation results. (colmap gui required)'''

class Basis:
    def __init__(self, args: Args):
        self.config_yaml_path = args.config
        self.config = load_yaml(args.config)
        self.vis = args.vis

        dconfig = self.config.DATA
        CONSOLE.log(f"[Geometry Initialization] {dconfig.SOURCE}-{dconfig['SCENE_ID']}")
        self.images_dir = dconfig.SOURCE/dconfig.SCENE_ID/'color'
        self.datamanager = VanillaDataManagerConfig(
            dataparser=ScanNetDataParserConfig(data=dconfig.SOURCE/dconfig.SCENE_ID, train_split_fraction=1, scale_factor=1.0, scene_scale=1.0, auto_scale_poses=False, center_method="none", load_3D_points=False)
        ).setup()
        self.output_dir = dconfig.OUTPUT/dconfig.SCENE_ID
        self.seed = self.config.MACHINE.SEED

        _set_random_seed(self.seed)
        pass

class Triangulation(Basis):
    def triangulate_from_given_poses(self) -> Path:
        retrieval_conf = extract_features.confs["netvlad"]
        out_dir = self.output_dir / 'triangulation_netvlad_loftr'
        out_dir.mkdir(parents=True, exist_ok=True)
        create_empty_pycolmap_reconstruction(out_dir, self.datamanager)

        # Find image pairs via image retrieval
        retrieval_path = extract_features.main(retrieval_conf, self.images_dir, out_dir/'matches')
        pairs_from_retrieval.main(retrieval_path, out_dir/'matches'/"pairs-netvlad.txt", num_matched=5)

        # Extract and match local features
        feature_path, match_path = match_dense.main(match_conf, out_dir/'matches'/"pairs-netvlad.txt", self.images_dir, out_dir/'matches', max_kps=8192, overwrite=False)

        # Triangulate a new SfM model from the given poses
        triangulation.main(
            out_dir/'sparse', out_dir/'reference', self.images_dir, out_dir/'matches'/"pairs-netvlad.txt", feature_path, match_path
        ).export_PLY(out_dir/'sparse'/'3d-keypoints.ply')

        return out_dir

    def main(self):
        out_dir = self.triangulate_from_given_poses()
        self.config.DATA.KEYPOINTS_REPO = out_dir

        CONSOLE.log(f"Updating configuration file: {self.config_yaml_path}")
        update_yaml(self.config_yaml_path, self.config)

        if self.vis:
            import os
            os.system(f"colmap gui --import_path {out_dir/'sparse'} --database_path {out_dir/'sparse'/'database.db'} --image_path {self.images_dir} &")

        CONSOLE.log(f"[bold yellow]scene{self.config.DATA.SCENE_ID}[/]: Triangulation Completed.")

        pass

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    args = tyro.cli(Args)
    if args.mode == "sfm":
        Triangulation(args).main()
    else:
        raise NotImplementedError("Monocular depth prediction is not implemented yet.")

    pass

if __name__ == "__main__":
    entrypoint()