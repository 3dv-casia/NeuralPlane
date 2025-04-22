import os, struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple, Type
from datetime import datetime
from joblib import Parallel, delayed

import cv2
import yaml
import imageio.v2 as imageio
import numpy as np
import tyro
from easydict import EasyDict as edict

# import ray
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.utils.rich_utils import CONSOLE
from generate_planes import generate_planes

COMPRESSION_TYPE_COLOR = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
COMPRESSION_TYPE_DEPTH = {
    -1: "unknown",
    0: "raw_ushort",
    1: "zlib_ushort",
    2: "occi_ushort",
}


class RGBDFrame:
    def load(self, file_handle):
        self.camera_to_world = np.asarray(
            struct.unpack("f" * 16, file_handle.read(16 * 4)), dtype=np.float32
        ).reshape(4, 4)
        self.timestamp_color = struct.unpack("Q", file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack("Q", file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.color_data = b"".join(
            struct.unpack(
                "c" * self.color_size_bytes, file_handle.read(self.color_size_bytes)
            )
        )
        self.depth_data = b"".join(
            struct.unpack(
                "c" * self.depth_size_bytes, file_handle.read(self.depth_size_bytes)
            )
        )

    def decompress_depth(self, compression_type):
        if compression_type == "zlib_ushort":
            return self.decompress_depth_zlib()
        else:
            raise

    def decompress_depth_zlib(self):
        return zlib.decompress(self.depth_data)

    def decompress_color(self, compression_type):
        if compression_type == "jpeg":
            return self.decompress_color_jpeg()
        else:
            raise

    def decompress_color_jpeg(self):
        return imageio.imread(self.color_data)


@dataclass
class SensorDataConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: SensorData)
    src_dir: Path = Path("raw_data")
    """directory in which the raw data is downloaded"""
    out_dir: Path = Path("datasets/scannet")
    task_data: bool = False
    """download task data (v1)"""
    label_map: bool = False
    """download label map file"""
    v1: bool = False
    """download v1 data instead of v2"""
    id: str = "scene0000_00"
    """specific scan id to download"""
    skip: bool = False
    """skip if file already exists"""
    crop_size: Optional[Tuple[int, int]] = (1248, 936)
    """size of cropped images"""
    scale_factor: float = 1.95  # 640x480
    """downscale factor"""
    interval: int = 8
    """interval for frames to keep"""


class SensorData:
    def __init__(self, config: SensorDataConfig):
        self.version = 4
        self.config = config
        self.out_dir = config.out_dir
        self.scene_path = config.src_dir

        CONSOLE.log(f"Loading RGB-D sensor stream of {config.id}")
        self.load(self.scene_path / f"{config.id}.sens")

        if config.crop_size is None:
            self.W_target, self.H_target = self.color_width, self.color_height
        else:
            self.W_target, self.H_target = config.crop_size
        if config.scale_factor is not None:
            self.W_target, self.H_target = int(
                self.W_target // config.scale_factor
            ), int(self.H_target // config.scale_factor)

    def load(self, filename):
        with open(filename, "rb") as f:
            version = int(struct.unpack("I", f.read(4))[0])
            assert self.version == version
            strlen = int(struct.unpack("Q", f.read(8))[0])
            self.sensor_name = b"".join(struct.unpack("c" * strlen, f.read(strlen)))
            self.intrinsic_color = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_color = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.intrinsic_depth = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_depth = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[
                struct.unpack("i", f.read(4))[0]
            ]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[
                struct.unpack("i", f.read(4))[0]
            ]
            self.color_width = struct.unpack("I", f.read(4))[0]
            self.color_height = struct.unpack("I", f.read(4))[0]
            self.depth_width = struct.unpack("I", f.read(4))[0]
            self.depth_height = struct.unpack("I", f.read(4))[0]
            self.depth_shift = struct.unpack("f", f.read(4))[0]
            num_frames = struct.unpack("Q", f.read(8))[0]
            self.frames = []
            for i in range(num_frames):
                frame = RGBDFrame()
                frame.load(f)
                self.frames.append(frame)

    def export_depth_images(self):
        depth_path = self.scene_path / "depth"
        if depth_path.exists():
            CONSOLE.log(f"Skipping {depth_path}")
            return
        CONSOLE.log(f"Exporting depth images to {depth_path}")
        depth_path.mkdir(parents=True)
        for f in range(0, len(self.frames)):
            depth_data = self.frames[f].decompress_depth(self.depth_compression_type)
            depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(
                self.depth_height, self.depth_width
            )
            depth = cv2.resize(
                depth, (self.W_target, self.H_target), interpolation=cv2.INTER_NEAREST
            )
            filename = os.path.join(depth_path, "{0:06d}.png".format(f))
            imageio.imwrite(filename, depth)

    def export_color_images(self):
        color_path = self.scene_path / "color"
        if color_path.exists():
            CONSOLE.log(f"Skipping {color_path}")
            return
        CONSOLE.log(f"Exporting color images to {color_path}")
        color_path.mkdir(parents=True)
        for f in range(0, len(self.frames)):
            color = self.frames[f].decompress_color(self.color_compression_type)
            if self.config.crop_size:
                assert (self.color_width - self.config.crop_size[0]) % 2 == 0 and (
                    self.color_height - self.config.crop_size[1]
                ) % 2 == 0
                crop_width_half = (self.color_width - self.config.crop_size[0]) // 2
                crop_height_half = (self.color_height - self.config.crop_size[1]) // 2
                # crop
                color = color[
                    crop_height_half : self.color_height - crop_height_half,
                    crop_width_half : self.color_width - crop_width_half,
                    :,
                ]
            if self.config.scale_factor is not None:
                color = cv2.resize(
                    color,
                    (self.W_target, self.H_target),
                    interpolation=cv2.INTER_LINEAR,
                )
            filename = os.path.join(color_path, "{0:06d}.jpg".format(f))
            imageio.imwrite(filename, color)

    def save_mat_to_file(self, matrix, filename):
        with open(filename, "w") as f:
            for line in matrix:
                np.savetxt(f, line[np.newaxis], fmt="%f")

    def export_poses(self):
        pose_path = self.scene_path / "pose"
        if pose_path.exists():
            CONSOLE.log(f"Skipping {pose_path}")
            return
        CONSOLE.log(f"Exporting poses to {pose_path}")
        pose_path.mkdir(parents=True)
        for f in range(0, len(self.frames)):
            filename = os.path.join(pose_path, "{0:06d}.txt".format(f))
            self.save_mat_to_file(self.frames[f].camera_to_world, filename)

    def export_intrinsics(self):
        intrin_path = self.scene_path / "intrinsic"
        if intrin_path.exists():
            CONSOLE.log(f"Skipping {intrin_path}")
            return
        CONSOLE.log(f"Exporting intrinsics to {intrin_path}")
        intrin_path.mkdir(parents=True)
        if self.config.crop_size:
            crop_width_half = (self.color_width - self.config.crop_size[0]) // 2
            crop_height_half = (self.color_height - self.config.crop_size[1]) // 2
            self.intrinsic_color[0, 2] -= crop_width_half
            self.intrinsic_color[1, 2] -= crop_height_half
        if self.config.scale_factor:
            self.intrinsic_color[:2, :3] /= self.config.scale_factor

        self.save_mat_to_file(
            self.intrinsic_color, os.path.join(intrin_path, "intrinsic_color.txt")
        )
        self.save_mat_to_file(
            self.extrinsic_color, os.path.join(intrin_path, "extrinsic_color.txt")
        )
        self.save_mat_to_file(
            self.intrinsic_depth, os.path.join(intrin_path, "intrinsic_depth.txt")
        )
        self.save_mat_to_file(
            self.extrinsic_depth, os.path.join(intrin_path, "extrinsic_depth.txt")
        )

    def symlink(self):
        poses = sorted((self.scene_path / 'pose').iterdir(), key=lambda p: p.stem)
        ids = []
        for p in poses:
            pose = np.loadtxt(p)
            if np.isnan(pose).any() or np.isinf(pose).any():
                CONSOLE.log(f"Pose {p} contains NaN or Inf values. Skipping.")
                continue
            ids.append(p.stem)

        for dtype, suff in zip(["color", "depth", "pose"], ['jpg', 'png', 'txt']):
            src = self.scene_path / dtype
            dst = self.out_dir / dtype
            if dst.exists() and self.config.skip:
                return
            dst.mkdir(parents=True, exist_ok=True)
            for i, id in enumerate(ids[::self.config.interval]):
                f = src / f"{id}.{suff}"
                assert f.exists(), f"{f} does not exist."
                if not (dst / f.name).exists():
                    (dst / f.name).symlink_to(f)
        if not (self.out_dir / "intrinsic").exists():
            (self.out_dir / "intrinsic").symlink_to(self.scene_path / "intrinsic")

    def process(self):
        self.export_color_images()
        self.export_depth_images()
        self.export_poses()
        self.export_intrinsics()
        self.symlink()


@dataclass
class MultiProcessorConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: MultiProcessor)

    src_dir: Path = Path("/data/Datasets")
    """directory to which the raw data is downloaded"""
    dst_dir: Path = Path("./datasets")
    """directory to which the processed data is saved"""
    out_dir: Path = Path("./outputs")
    """directory to which the reconstruction cache and results are saved"""
    timestamp: str = datetime.now().strftime("%Y-%b-%d-%a")
    '''timestamp for the experiment'''
    id: Optional[str] = None
    """specific scan id to process"""
    split_file: Optional[Path] = None
    """file containing list of scan ids. If specified, id is ignored."""
    n_proc: int = 1
    """processes launched to process scenes. NOT WELL TESTED"""

    skipping: bool = False
    """whether to skip if the config file already exists"""

class MultiProcessor:
    def __init__(self, config: MultiProcessorConfig, **kwargs) -> Any:
        self.split = config.split_file
        self.id = config.id
        self.config = config
        self.src_dir = config.src_dir / "scannetv2"
        assert self.src_dir.exists(), f"{self.src_dir} does not exist."
        self.dst_dir = config.dst_dir / "scannetv2"
        self.out_dir = config.out_dir / "scannetv2"

    def main(self):
        if self.split:
            scenes = self.split.read_text().splitlines()
        elif self.id:
            scenes = [self.id]
        else:
            CONSOLE.log("Specify either split file or id.")
            raise ValueError

        Parallel(n_jobs=self.config.n_proc, verbose=0)(
            delayed(self.process_with_single_worker)(scene) for scene in scenes
        )

    def process_with_single_worker(self, scene):
        scene_name = f"scene{scene}"
        try:
            path = self.src_dir / "scans" / scene_name
            assert path.exists()
        except AssertionError:
            path = self.src_dir / "scans_test" / scene_name
        assert path.exists(), f"{path} does not exist."
        config_file = Path("./configs") / self.config.timestamp / f'{scene}.yaml'

        if not config_file.parent.exists():
            config_file.parent.mkdir(parents=True, exist_ok=True)

        dst_dir = self.dst_dir / scene
        dst_dir.mkdir(parents=True, exist_ok=True)

        if not (self.config.skipping and config_file.exists()):
            SD = SensorDataConfig(
                id=scene_name,
                src_dir=path,
                out_dir=dst_dir,
                skip=self.config.skipping,
            ).setup()
            SD.process()
            pass
        else:
            CONSOLE.log(f"Skipping reading sensor data of {scene_name}")

        anno_path = dst_dir / "annotations"
        if not anno_path.exists():
            anno_path.mkdir(parents=True, exist_ok=True)
            gt_planes_args = edict({"data_raw_path": path, "annotation_path": anno_path})
            generate_planes(gt_planes_args, scene_id=scene_name, save_mesh=True)
        else:
            CONSOLE.log(f"Skipping {anno_path}")

        yml_config = {"_BASE_": "../base.yaml"}
        yml_config['DATA'] = {
            "DATASET": "scannetv2",
            "SCENE_ID": scene,
            "ANNOTATION": anno_path,
            "SOURCE": self.dst_dir,
            "OUTPUT": self.out_dir,
        }
        yml_config['TIMESTAMP'] = self.config.timestamp
        CONSOLE.log(f"Generating configuration file: {config_file}")
        config_file.write_text(yaml.dump(yml_config), "utf-8")

def main(config: MultiProcessorConfig):
    MP = config.setup()
    MP.main()


def entrypoint():
    tyro.extras.set_accent_color("bright_yellow")
    main(
        tyro.cli(
            MultiProcessorConfig,
            description=convert_markup_to_ansi(__doc__),
        )
    )


if __name__ == "__main__":
    entrypoint()
