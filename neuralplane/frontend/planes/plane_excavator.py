from dataclasses import dataclass, field
from typing import Type, Union
from pathlib import Path
from easydict import EasyDict as edict

import torch
import numpy as np
import cv2
from sklearn.cluster import KMeans

from nerfstudio.configs.base_config import InstantiateConfig

from neuralplane.frontend.monocular.basis import MonocularPredictorConfig
from neuralplane.frontend.monocular.snu import SNUConfig
from neuralplane.frontend.monocular.stable_normal import StableNormalConfig
from neuralplane.frontend.segment.sam_tools import setup_sam
from neuralplane.frontend.segment.mask_generator import infer_masks
from neuralplane.frontend.monocular.tools import to_world_space, remove_small_isolated_areas, merge_normal_clusters


@dataclass
class PlaneExcavatorConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: PlaneExcavator)
    """target class to instantiate"""

    normal_predictor: MonocularPredictorConfig = field(default_factory=lambda: SNUConfig(ckpt_dir=Path("checkpoints")))
    # normal_predictor: SurfaceNormalPredictorConfig = field(default_factory=lambda: StableNormalConfig(ckpt_dir=Path("./checkpoints")))
    "normal predictor config"

    min_size_ratio: float = 0.004 # 0.4%
    """The minimum size of a desired plane segment, as a ratio of the total number of pixels in the image."""
    n_init_normal_clusters: int = 8
    """The number of clusters to form as well as the number of centroids to generate when use KMeans to cluster the surface normals."""
    n_normal_clusters: int = 6
    """The number of normal clusters to keep after KMeans, i.e., we only keep the first `num_max_clusters` clusters c1, c2, ..., where Size(c1) > Size(c2) > ... (sorted by size).
    """
    num_sam_prompts: int = 256
    """The number of SAM prompts to use for inference."""

class PlaneExcavator:
    def __init__(self, config: PlaneExcavatorConfig, device, img_height: int, img_width: int):
        self.n_init_normal_clusters = config.n_init_normal_clusters
        self.n_normal_clusters = config.n_normal_clusters
        self.num_sam_prompts = config.num_sam_prompts
        self.img_shape = (img_height, img_width)  # Currently only support images of the same size
        self.min_plane_size = self.img_shape[0] * self.img_shape[1] * config.min_size_ratio
        self.device = device

        self.normal_predictor = config.normal_predictor.setup(
            device=self.device,
            img_height=img_height,
            img_width=img_width
        )

        self.sam_model = setup_sam(device=self.device)

    def _normals_cluster(self, normals: np.ndarray):
        """
        Cluster the surface normals.
        """
        kmeans = KMeans(n_clusters=self.n_init_normal_clusters, random_state=0, n_init=1).fit(normals.reshape(-1, 3))
        pred = kmeans.labels_
        centers = kmeans.cluster_centers_

        # Select the first `num_max_clusters` clusters c1, c2, ..., where Size(c1) > Size(c2) > ... (sorted by size)
        count_values = np.bincount(pred)
        topk = np.argpartition(count_values,-self.n_normal_clusters)[-self.n_normal_clusters:]
        sorted_topk_idx = np.argsort(count_values[topk])
        sorted_topk = topk[sorted_topk_idx][::-1]

        pred, sorted_topk, num_clusters = merge_normal_clusters(pred, sorted_topk, centers)

        count_valid_cluster = 0
        normal_masks = []
        for i in range(num_clusters):
            mask = (pred==sorted_topk[count_valid_cluster])
            mask_clean = remove_small_isolated_areas((mask>0).reshape(*self.img_shape)*255, min_size=self.min_plane_size).reshape(-1)
            mask[mask_clean==0] = 0

            num_labels, labels = cv2.connectedComponents((mask*255).reshape(self.img_shape).astype(np.uint8))
            for label in range(1, num_labels):
                normal_masks.append(labels == label)
            count_valid_cluster += 1
        return normal_masks

    def __call__(self, img: np.ndarray, c2w: Union[torch.Tensor, np.ndarray], vis: bool = False):
        """
        img: np.ndarray
            The input image, in the form of a numpy array.
        c2w: Union[torch.Tensor, np.ndarray]
            The camera-to-world transformation matrix.
        """
        assert img.shape[0] == self.img_shape[0] and img.shape[1] == self.img_shape[1], f"Input image shape {img.shape} does not match the expected shape {self.img_shape}."

        normals = self.normal_predictor(img)["pred_norm"]
        normals[..., 0] *= -1 # transform normal coordinate to OpenGL convention.
        '''
            y            y
            |            |
        x - o   ----->   o - x (OpenGL)
           /            /
          z            z
        '''
        if c2w:
            normals = to_world_space(normals, c2w)
        normal_clusters = self._normals_cluster(normals)

        # Generate masks
        normalized_prompts = torch.rand(self.num_sam_prompts, 2, device=self.device) * 2 - 1
        sam_outputs = infer_masks(self.sam_model, img, keypoints=normalized_prompts, device=self.device, num_pts_active=0)['masks']
        masks = sam_outputs['masks'].cpu().numpy()
        masks = sorted(masks, key=lambda x: np.sum(x))

        seg_mask = np.zeros(self.img_shape, dtype=np.uint8)  # 0 indicates background (non-plane region)
        count = 0
        for mask in masks:
            for normal_mask in normal_clusters:
                intersect = mask & normal_mask
                size = np.sum(intersect)
                if size < self.min_plane_size:
                    continue
                count += 1
                seg_mask[intersect] = count

        new_seg_mask =np.zeros_like(seg_mask)
        masks_avg_normals = []
        masks_areas = []
        plane_instances = []

        new_count = 0
        for i in range(np.min([100, count])):
            mask = (seg_mask == i + 1)
            area = mask.sum()
            if area < self.min_plane_size:
                continue
            new_count += 1
            new_seg_mask[mask] = new_count
            masks_areas.append(area)
            plane_instances.append(mask)

            avg_normal = np.mean(normals[mask], axis=0)
            avg_normal /= np.sqrt((avg_normal ** 2).sum())  # normalize
            masks_avg_normals.append(avg_normal)

        masks_avg_normals = np.stack(masks_avg_normals)
        masks_areas = np.array(masks_areas)

        outputs = edict(
            {
                "seg_mask": new_seg_mask,
                "normal": masks_avg_normals,
                "areas": masks_areas,
            }
        )

        if vis:
            img_batch = {
                "image": img,
            }
            pred_norm_rgb = ((normals + 1) * 0.5) * 255
            pred_norm_rgb = np.clip(pred_norm_rgb, a_min=0, a_max=255)
            pred_norm_rgb = pred_norm_rgb.astype(np.uint8)
            img_batch["pred_norm"] = pred_norm_rgb

            from neuralplane.utils.disp import overlay_masks
            img_batch["normal_mask"] = overlay_masks(img, normal_clusters)
            img_batch["plane_mask"] = overlay_masks(img, plane_instances)

            outputs["vis"] = img_batch

        return outputs