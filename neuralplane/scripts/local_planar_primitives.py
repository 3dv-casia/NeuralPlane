from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Literal
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn, track

from neuralplane.frontend.planes.plane_excavator import PlaneExcavatorConfig, PlaneExcavator
from neuralplane.utils.yaml import load_yaml, update_yaml

from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.scannet_dataparser import ScanNetDataParserConfig
from nerfstudio.data.utils.colmap_parsing_utils import (
    qvec2rotmat,
    read_images_binary,
    read_points3D_binary,
)
from nerfstudio.scripts.train import _set_random_seed

import tyro
import torch
import numpy as np
import h5py
import os

@dataclass
class Args:
    config: Path = Path("config.yaml")
    '''Path to the configuration file'''
    vis: bool = False
    '''Visualize the results. (detectron2 required)'''
    init_mode: Literal["sfm"] = "sfm"
    skipping: bool = False
    '''Skip the excavation process if the cache exists'''

class LocalPlanarPrimitives:
    def __init__(self, args: Args):
        self.config_yaml_path = args.config
        self.config = load_yaml(args.config)
        self.skipping = args.skipping
        self.vis = args.vis
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seed = self.config.MACHINE.SEED
        self.init_mode = args.init_mode
        dconfig = self.config.DATA

        CONSOLE.log(f"[Pre] Triangulating {dconfig.SOURCE}-{dconfig.SCENE_ID}")
        self.images_dir = dconfig.SOURCE/dconfig.SCENE_ID/'color'
        self.datamanager = VanillaDataManagerConfig(
            dataparser=ScanNetDataParserConfig(data=dconfig.SOURCE/dconfig.SCENE_ID, train_split_fraction=1, scale_factor=1.0, scene_scale=1.0, auto_scale_poses=False, center_method="none", load_3D_points=False)
        ).setup()
        self.output_dir = dconfig.OUTPUT/dconfig.SCENE_ID/'local_planar_primitives'
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.vis:
            try:
                from detectron2.utils.visualizer import Visualizer
            except ImportError:
                raise ImportError("Please install detectron2 to visualize the results.")
            Path(self.output_dir / "vis").mkdir(parents=True, exist_ok=True)

        _set_random_seed(self.seed)

    def excavate_planes(self):
        dataset = self.datamanager.train_dataset
        image_filenames = dataset.image_filenames
        num_images = len(dataset)
        img_height, img_width = dataset[0]["image"].shape[:2]
        excavator: PlaneExcavator = PlaneExcavatorConfig().setup(
            device=self.device,
            img_height=img_height,
            img_width=img_width
        )

        pan_seg_lists: List[np.ndarray] = []  # the panoptic segmentation result of shape (h, w, 1).
        # seg_list: List[Tuple[List[np.ndarray], List[int]]] = []  # each element (tuple) consists of a list of pixel coordinates (useful for pixel sampling;) as well as a list of sizes of instances
        normals_list: List[np.ndarray] = []  # each element is an array of surface normals

        progress = Progress(
            TextColumn(":ten_o’clock: Excavating Local Planar Primitives from images :ten_o’clock:"),
            BarColumn(),
            TaskProgressColumn(show_speed=True),
            ItersPerSecColumn(suffix="fps"),
            TimeRemainingColumn(elapsed_when_finished=True, compact=True),
        )

        with progress:
            for i in progress.track(range(num_images), description=None):
                rgb = dataset[i]["image"]
                results = excavator((rgb.numpy() * 255).astype(np.uint8), c2w=None, vis=self.vis)

                pan_seg_lists.append(results["seg_mask"])
                # masks_list.append((results["indices"], results["areas"]))
                normals_list.append(results["normal"])

                if self.vis:
                    import matplotlib.pyplot as plt
                    vis_batch = results['vis']
                    fig = plt.figure(figsize=(8, 6))
                    gs = fig.add_gridspec(2,2)
                    ax1 = fig.add_subplot(gs[0, 0])
                    ax1.imshow(vis_batch["image"])
                    ax1.set_title("(1) Image")
                    ax2 = fig.add_subplot(gs[0, 1])
                    ax2.imshow(vis_batch["pred_norm"])
                    ax2.set_title("(2) Estimated Normal Map")
                    ax3 = fig.add_subplot(gs[1, 0])
                    ax3.imshow(vis_batch["normal_mask"])
                    ax3.set_title("(3) Normal Clusters")
                    ax4 = fig.add_subplot(gs[1, 1])
                    ax4.imshow(vis_batch["plane_mask"])
                    ax4.set_title("(4) Plane Segmentation")
                    plt.tight_layout()
                    fig.savefig(
                        self.output_dir / "vis" / f"{image_filenames[i].stem}.png",
                        transparent=False,
                        format='png',
                    )
                    plt.close()

        return pan_seg_lists, normals_list

    def assign_3d_keypoints(self, colmap_repo: Path, segs_per_image: List, pan_seg_lists: List[np.ndarray]):
        ptid_to_info = read_points3D_binary(colmap_repo / "points3D.bin")
        im_id_to_image = read_images_binary(colmap_repo / "images.bin")

        scale_factor = self.datamanager.train_dataset._dataparser_outputs.dataparser_scale

        # Find the 3D keypoints in each plane instance, and compute their depths.
        keypoints_to_segs: List[List[np.ndarray]] = [None] * len(segs_per_image)  # np.ndarray of shape (N, 4): N * (v, u, z, e)

        iter_images = track(
            im_id_to_image.items(), total=len(im_id_to_image.items()), description=":ten_o’clock: Processing 3D keypoints :ten_o’clock:"
        )
        for im_id, im_data in iter_images:
            points_list = [[] for _ in range(segs_per_image[im_id])]
            pids = [pid for pid in im_data.point3D_ids if pid != -1]
            xyz_world = np.array([ptid_to_info[pid].xyz for pid in pids])
            if xyz_world.shape[0] == 0:
                keypoints_to_segs[im_id] = [np.array([]) for _ in range(segs_per_image[im_id]) ]
                continue
            rotation = qvec2rotmat(im_data.qvec)
            z = scale_factor * ((rotation @ xyz_world.T)[-1] + im_data.tvec[-1])
            errors = np.array([ptid_to_info[pid].error for pid in pids])
            uv = np.array([im_data.xys[i] for i in range(len(im_data.xys)) if im_data.point3D_ids[i] != -1])
            uu, vv = uv[:, 0].astype(int), uv[:, 1].astype(int)

            instance_id = pan_seg_lists[im_id][vv, uu].reshape(-1)
            for i in range(len(pids)):
                if instance_id[i] != 0:
                    points_list[instance_id[i] - 1].append(np.array([vv[i], uu[i], z[i], errors[i]], dtype=np.float32))

            points_list = [np.array(x) for x in points_list]
            keypoints_to_segs[im_id] = points_list

        return keypoints_to_segs

    def main(self):
        cache_dir = self.output_dir / "cache.h5"
        try:
            if not self.skipping:
                raise FileNotFoundError
            with h5py.File(cache_dir, "r") as f:
                CONSOLE.log(f"Local Planar Primitives Excavation has already been done. Skipping...")
                pan_seg_lists = [f[img.stem]["pan_seg"][()] for img in self.datamanager.train_dataset.image_filenames]
                normals_list = [f[img.stem]["normals"][()] for img in self.datamanager.train_dataset.image_filenames]
        except:
            pan_seg_lists, normals_list = self.excavate_planes()

            CONSOLE.log(f"Saving the results to {cache_dir}...")
            with h5py.File(cache_dir, "w") as f:
                for i, img in enumerate(self.datamanager.train_dataset.image_filenames):
                    img = img.stem
                    f.create_group(img)
                    f[img].create_dataset("normals", data=normals_list[i])
                    f[img].create_dataset("pan_seg", data=pan_seg_lists[i])

        segs_per_images = [n.shape[0] for n in normals_list]
        if self.init_mode == "sfm":
            try:
                keypoints_repo = Path(self.config.DATA.KEYPOINTS_REPO) / 'sparse'
                assert keypoints_repo.exists(), f"Keypoints repo {self.keypoints_repo} does not exist."
            except:
                cmd = f"np-pre-geo-init --config {self.config_yaml_path} --mode {self.init_mode}"
                CONSOLE.log(f"Keypoints repo not found. Run `{cmd}`.")
                os.system(cmd)
                self.config = load_yaml(self.config_yaml_path)
            finally:
                keypoints_repo = Path(self.config.DATA.KEYPOINTS_REPO) / 'sparse'
                keypoints_to_segs = self.assign_3d_keypoints(keypoints_repo, segs_per_images, pan_seg_lists)

        else:
            raise NotImplementedError(f"Init mode {self.init_mode} not supported.")

        with h5py.File(cache_dir, "a") as f:
            for i, img in enumerate(self.datamanager.train_dataset.image_filenames):
                img = img.stem
                assert img in f
                if "keypoints" in f[img]:
                    del f[img]["keypoints"]
                f[img].create_group("keypoints")
                for j, keypoints in enumerate(keypoints_to_segs[i]):
                    f[img]["keypoints"].create_dataset(f"{j:03d}", data=keypoints)

        CONSOLE.log(f"Updating configuration file: {self.config_yaml_path}")
        self.config.DATA.CACHE = cache_dir
        update_yaml(self.config_yaml_path, self.config)

        CONSOLE.rule(f"[bold yellow]scene{self.config.DATA.SCENE_ID}[/]: Excavating Local Planar Primitives Completed")

        pass

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    args = tyro.cli(Args)
    LocalPlanarPrimitives(args).main()


if __name__ == "__main__":
    entrypoint()