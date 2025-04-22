from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Tuple, Type, Union, List
from pathlib import Path
from jaxtyping import Int, Float

import torch
import torch.nn as nn
import h5py
import numpy as np

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.data.utils.dataloaders import RandIndicesEvalDataloader
from nerfstudio.cameras.lie_groups import exp_map_SO3xR3

from neuralplane.neuralplane_pixel_sampler import PlaneInstancePixelSamplerConfig, PlaneInstancePixelSampler
from neuralplane.utils.geometry import transformPlanes
from neuralplane.utils.disp import writePlanarPrimitive2File
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation

@dataclass
class NeuralPlaneDataManagerConfig(VanillaDataManagerConfig):
    _target: Type = field(default_factory=lambda: NeuralPlaneDataManager)
    """target class to instantiate"""
    pixel_sampler: PlaneInstancePixelSamplerConfig = field(default_factory=lambda: PlaneInstancePixelSamplerConfig())
    """config for the pixel sampler: num_rays_per_image, num_rays_per_seg"""
    num_pts_threshold: int = 50
    """only estimate offsets of plane segments with more than this number of 3D keypoints detected."""
    max_plane_offsets: float = 5.0
    """maximum offset of the plane segments to be considered valid."""
    local_planar_primitives_opt_mode: Literal["offset_only", "SO3xR3", "addon"] = "addon"
    """optimization strategy to use."""
    cache_path: Path = field(default_factory=Path)
    """path to the cache file containing the plane segs and normals."""

class NeuralPlaneDataManager(VanillaDataManager):
    train_pixel_sampler: PlaneInstancePixelSampler
    def __init__(
        self,
        config: NeuralPlaneDataManagerConfig,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self.cache_path = config.cache_path.parent  # keep this since we are to save the refined local planar primitives.
        self.num_images = len(self.train_dataset)
        self.opt_mode = config.local_planar_primitives_opt_mode

        self.eval_dataset = self.create_train_dataset()  # no image for evaluation. use training images.
        self.eval_dataloader = RandIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
        )

        # precomute a ray bundle from an identity camera pose
        _camera = self.train_dataset.cameras[0][None, ...]
        c2w_identity = torch.zeros((1, 3, 4))
        c2w_identity[0, :, :3] = torch.eye(3)
        _camera.camera_to_worlds = c2w_identity
        self._ray_bundle = _camera.generate_rays(camera_indices=0)

        # load preprocessed results
        self.train_pixel_sampler.plane_seg_uv_list, self.train_pixel_sampler.instance_weight_list, init_instance_normal, instance_keypoints = self._load_preprocessed_results(config.cache_path)
        assert self.num_images == len(init_instance_normal), "Number of images in the cache does not match the number of images in the dataset."

        if self.test_mode == "val":  # training mode
            self.init_instance_params: List = self._init_local_planar_primitives_params(init_instance_normal, instance_keypoints)

            # init residuals for the refinement
            self.local_planar_primitives_residuals: List = [
                nn.Parameter(
                    torch.zeros(param.shape[0], 6, device=self.device)
                ) for param in self.init_instance_params
            ]

            pass

        elif self.test_mode == "test":  # export mode
            self.init_instance_params = self._load_from_refined_state(self.cache_path / (config.cache_path.stem + "_refined.h5"))

    def get_param_groups(self) -> Dict[str, List[nn.Parameter]]:
        """Returns the param groups for the data manager."""
        param_groups = super().get_param_groups()
        param_groups["local_planar_primitives_residuals"] = self.local_planar_primitives_residuals
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = super().get_training_callbacks(training_callback_attributes)

        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN],
                func=self.save_local_planar_primitives,
            )
        )

        return callbacks

    def next_train(self, step: int, sample_on_instances: bool=False, local_planar_primitives_opt: bool=False) -> Tuple[RayBundle, Dict[str, Any]]:
        """Returns the next batch of data for train.

        Args:
            sample_on_instances: `True` to sample rays from the plane segments according to `num_rays_per_image`, `num_rays_per_seg`
            local_planar_primitives_opt: `True` to turn on optimization of the local geometry.
        """
        if not sample_on_instances:
            ray_bundle, batch = super().next_train(step)
        else:
            self.train_count += 1
            image_batch = next(self.iter_train_image_dataloader)
            batch = self.train_pixel_sampler.sample(image_batch, sample_on_instances=True)
            ray_indices = batch["indices"]

            ray_bundle = self.train_ray_generator(ray_indices)

            batch["K_inv_dot_xy1"] = self._get_identity_camera_ray_directions(ray_indices)
            batch["plane_params"], batch["plane_params_w"], batch["valid_instances"] = self._get_local_planar_primitive_params(
                batch,
                param_opt=local_planar_primitives_opt
            )
        return ray_bundle, batch

    @torch.no_grad()
    def update_local_planar_primitives(self, batch: Dict[str, Any], model_outputs: Dict[str, Any]) -> None:
        """Updates the parameters of those invalid local planar primitives."""
        num_rays_per_seg = batch["num_rays_per_seg"]
        valid_mask = batch["valid_instances"].to(self.device)
        seg_idx = batch["seg_idx"].to(self.device)
        plane_params = batch["plane_params"].to(self.device)
        K_inv_dot_xy1 = batch["K_inv_dot_xy1"].to(self.device)

        invalid = ~valid_mask
        plane_normals = plane_params[invalid, :3].view(-1, 1, 3)
        K_inv_dot_xy1 = K_inv_dot_xy1[invalid].view(-1, 3, 1)

        normal_xyz = torch.bmm(plane_normals, K_inv_dot_xy1).view(-1, 1)  # bs 1

        depth = model_outputs["depth"][invalid].view(-1, 1)
        plane_offsets = - depth * normal_xyz
        plane_offsets = plane_offsets.view(-1, num_rays_per_seg, 1).mean(dim=1)

        # update the plane offsets
        # TODO: update all or only the invalid
        seg_idx = seg_idx[invalid][::num_rays_per_seg]
        for i, (im_id, seg_id) in enumerate(seg_idx):
            self.init_instance_params[im_id][seg_id, 3] = plane_offsets[i]

    @torch.no_grad()
    def save_local_planar_primitives(self, step: int) -> None:
        """Saves the refined local planar primitives."""
        batch = {"num_rays_per_seg": 1}
        seg_idx = [ (im_id, seg_id) for im_id in range(self.num_images) for seg_id in range(len(self.init_instance_params[im_id])) ]
        batch["seg_idx"] = torch.as_tensor(seg_idx)

        params, params_w, valid = self._get_local_planar_primitive_params(batch, param_opt=True)

        params = params.cpu().numpy()
        cache_file = self.cache_path / (self.config.cache_path.stem + "_refined.h5")
        count = 0
        with h5py.File(cache_file, "w") as f:
            for i, img in enumerate(self.train_dataset.image_filenames):
                f.create_dataset(img.stem, data=params[count:count + len(self.init_instance_params[i])])
                count += len(self.init_instance_params[i])

        CONSOLE.log(f"Local planar primitives saved to {cache_file}")
        pass

    def _get_identity_camera_ray_directions(self, indices: Int[torch.Tensor, "bs 3"]) -> Float[torch.Tensor, "bs 3"]:
        """Returns the identity camera ray directions for the given indices. Similar to get K_inv_dot_xy1🤔."""
        x_ind = indices[:, 1]
        y_ind = indices[:, 2]

        return self._ray_bundle.directions[x_ind, y_ind, :]

    @staticmethod
    def _load_preprocessed_results(cache_path) -> Tuple[List, List, List, List]:
        """Loads the preprocessed results from the cache."""
        plane_seg_uv_list = []
        instance_normal_list = []
        instance_weight_list = []
        instance_keypoints_list = []

        CONSOLE.log(f":ten_o’clock: Loading plane seg cache from {cache_path}...")
        with h5py.File(cache_path, "r") as f:
            for img_id in f:
                normals = f[img_id]["normals"][()]
                pan_seg = f[img_id]["pan_seg"][()]
                seg_uv_list = [np.stack(list(np.where(pan_seg == i + 1)), axis=-1) for i in range(normals.shape[0])]
                keypoints_list = [f[img_id]["keypoints"][f'{j:03d}'][()] for j in range(normals.shape[0])]
                seg_area_list = [seg_uv.shape[0] for seg_uv in seg_uv_list]
                segs_sum = sum(seg_area_list)
                seg_weight_list = [seg_area / segs_sum for seg_area in seg_area_list]

                plane_seg_uv_list.append(seg_uv_list)  # [num_images, num_instances, instance_size, 2]
                instance_weight_list.append(seg_weight_list)  # [num_images, num_instances, 1]

                instance_normal_list.append(normals)  # [num_images, num_instances, 3]
                instance_keypoints_list.append(keypoints_list) # [num_images, num_instances, instance_size, 4]

        return plane_seg_uv_list, instance_weight_list, instance_normal_list, instance_keypoints_list

    @torch.no_grad()
    def _init_local_planar_primitives_params(self, init_instance_normal: List, instance_keypoints: List, num_pts_threshold: int = 50) -> List:
        """Initializes parameters for all of the plane instances in their own local coordinates."""

        init_instance_params = []
        for img_id, (pts_3d_list, normals) in enumerate(zip(instance_keypoints, init_instance_normal)):
            assert len(pts_3d_list) == normals.shape[0], "Number of instances does not match the number of normals."
            num_instances = len(pts_3d_list)
            normals = torch.from_numpy(normals).float()

            params = torch.zeros((num_instances, 4), requires_grad=False)
            params[:, :3] = normals

            for ins_id, (pts_3d, normal) in enumerate(zip(pts_3d_list, normals)):
                offset = -1.  # valid offsets are possitive
                if len(pts_3d) >= num_pts_threshold:  # TODO: according to the depths of the keypoints. Close to the camera, the point is more reliable.
                    pts_3d = torch.from_numpy(pts_3d)
                    u, v, z, e = pts_3d.T
                    u, v = u.long(), v.long()

                    euclidean_depth = z * self._ray_bundle.metadata['directions_norm'][u, v].squeeze(-1)
                    weights = 1.0 / e

                    offsets = -euclidean_depth.view(-1, 1) * self._ray_bundle.directions[u, v] @ normal.view(3, 1)
                    offset = torch.dot(offsets.view(-1), weights) / weights.sum()
                params[ins_id, 3] = offset
            init_instance_params.append(params)

        return init_instance_params

    @staticmethod
    def _load_from_refined_state(cache_path: Path) -> List:
        init_instance_params = []
        CONSOLE.log(f"Loading refined local planar primitives from {cache_path}...")
        with h5py.File(cache_path, "r") as f:
            for img_id in f:
                init_instance_params.append(torch.from_numpy(f[img_id][()]))
        return init_instance_params

    def _get_local_planar_primitive_params(self, batch: Dict[str, Any], param_opt: bool=False) -> Tuple:
        assert "seg_idx" in batch

        num_rays_per_seg = batch["num_rays_per_seg"]
        seg_idx: Int[torch.Tensor, "bs 2"] = batch["seg_idx"][::num_rays_per_seg]  # avoid duplicate computation

        params = [self.init_instance_params[im_id][seg_id] for im_id, seg_id in seg_idx]
        params: torch.Tensor = torch.stack(params, dim=0).to(self.device)
        # valid (accurate) planar primitives should be not too close or not to far from the camera
        valid = (params[:, 3] > 0.1) & (params[:, 3] < self.config.max_plane_offsets)

        if param_opt and self.test_mode == "val":  # training mode
            residuals = [self.local_planar_primitives_residuals[im_id][seg_id] for im_id, seg_id in seg_idx]
            residuals = torch.stack(residuals, dim=0)

            if self.opt_mode == "offset_only":
                params[:, 3] += residuals[:, 0]
            elif self.opt_mode == "SO3xR3":
                params = transformPlanes(exp_map_SO3xR3(residuals), params)
            elif self.opt_mode == "addon":
                params += residuals[:, :4]
                params = params / torch.norm(params[:, :3], dim=1, keepdim=True)
            else:
                raise ValueError(f"Invalid optimization mode: {self.opt_mode}")
            assert params.shape == (seg_idx.shape[0], 4)

        # Compute params in the world space for the geometry verification defined in Eq. (7).
        c2w = self.train_dataset.cameras[seg_idx[:, 0]].camera_to_worlds.to(self.device)
        with torch.no_grad():
            params_w = transformPlanes(c2w, params)

        return params.repeat_interleave(num_rays_per_seg, dim=0), params_w.repeat_interleave(num_rays_per_seg, dim=0), valid.repeat_interleave(num_rays_per_seg, dim=0)

    def writePlanarPrimitive2File(self, out_folder: Path, image_id: int) -> None:
        """ Writes local planar primitives to a PLY file. For debugging purposes. """
        seg_uv_list = self.train_pixel_sampler.plane_seg_uv_list[image_id]
        parms = self.init_instance_params[image_id]
        image = self.train_dataset.image_filenames[image_id]
        camera = self.train_dataset.cameras[image_id]

        K = camera.get_intrinsics_matrices()

        c2w = self.train_dataparser_outputs.transform_poses_to_original_space(
            camera.camera_to_worlds.view(1, 3, 4),
            camera_convention="opengl"
        ).view(3, 4)

        writePlanarPrimitive2File(out_folder, seg_uv_list, parms, image, K, c2w)

        pass

# ==============debugging================
if __name__ == '__main__':
    from nerfstudio.data.dataparsers.scannet_dataparser import ScanNetDataParserConfig

    # Init: load posed images and the plane seg cache, and init the local planar primitives
    datamanager = NeuralPlaneDataManagerConfig(
        dataparser=ScanNetDataParserConfig(
            data = Path("datasets/scannetv2/0084_00"),
            train_split_fraction=1.0,
            scale_factor=1.0,
            scene_scale=1.0,
            auto_scale_poses=False,
            load_3D_points=False,
            center_method="none",
        ),
        local_planar_primitives_opt_mode="SO3xR3",
        cache_path = Path("outputs/scannetv2/0084_00/local_planar_primitives/cache.h5"),
    ).setup(
        device = "cuda",
    )

    # Sample
    ray_bundle, batch = datamanager.next_train(0, sample_on_instances=True, local_planar_primitives_opt=True)

    # Vis
    datamanager.writePlanarPrimitive2File(
        out_folder = Path("outputs/scannetv2/0084_00/local_planar_primitives/ply"),
        image_id = 1712 // 8
    )