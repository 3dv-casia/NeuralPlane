from pathlib import Path
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.utils.rich_utils import CONSOLE

import pycolmap
import numpy as np

def create_empty_pycolmap_reconstruction(dst: Path, datamanager: VanillaDataManager) -> None:
    cameras = datamanager.train_dataparser_outputs.cameras
    cameras.camera_to_worlds = datamanager.train_dataparser_outputs.transform_poses_to_original_space(cameras.camera_to_worlds)
    image_filenames = datamanager.train_dataparser_outputs.image_filenames
    num_images = len(cameras)

    dst = dst / "reference"
    dst.mkdir(parents=True, exist_ok=True)

    CONSOLE.log(f"Creating empty pycolmap reconstruction at {dst}. Number of images: {num_images}")

    rec = pycolmap.Reconstruction()
    for i in range(num_images):
        fx, fy, cx, cy = cameras.fx[i].item(), cameras.fy[i].item(), cameras.cx[i].item(), cameras.cy[i].item()
        img_height, img_width = cameras.image_height[i].item(), cameras.image_width[i].item()
        cam = pycolmap.Camera(model='PINHOLE', width=img_width, height=img_height, params=[fx, fy, cx, cy], camera_id=i)
        rec.add_camera(cam)

        c2w = cameras.camera_to_worlds[i].numpy()
        c2w = np.concatenate([c2w, np.array([[0, 0, 0, 1]])], 0)
        w2c = np.linalg.inv(c2w)
        image = pycolmap.Image(name=image_filenames[i].name, points2D=pycolmap.ListPoint2D(), cam_from_world=pycolmap.Rigid3d(w2c[:3, :]), camera_id=i, id=i)
        rec.add_image(image)
        rec.register_image(i)
    rec.write(dst)

    return
