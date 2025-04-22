from dataclasses import dataclass, field
from typing import Optional, Type, Union, Dict, Tuple

import torch
import numpy as np
from jaxtyping import Int
from torch import Tensor

from nerfstudio.data.pixel_samplers import PixelSamplerConfig, PixelSampler

@dataclass
class PlaneInstancePixelSamplerConfig(PixelSamplerConfig):
    _target: Type = field(default_factory=lambda: PlaneInstancePixelSampler)
    """target class to instantiate"""

    num_rays_per_image: int = 256
    """number of rays to sample per image"""
    num_rays_per_seg: int = 64
    """number of rays to sample per segment"""

class PlaneInstancePixelSampler(PixelSampler):
    def __init__(self, config: PlaneInstancePixelSamplerConfig, **kwargs):
        super().__init__(config, **kwargs)

        self.num_rays_per_image = config.num_rays_per_image
        self.num_rays_per_seg = config.num_rays_per_seg
        self.num_segs_per_image = self.num_rays_per_image // self.num_rays_per_seg
        if self.num_segs_per_image * self.num_rays_per_seg != self.num_rays_per_image:
            raise ValueError("num_rays_per_image should be divisible by num_rays_per_seg.")

        self.plane_seg_uv_list = None
        self.instance_weight_list = None

    def sample_method(
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        image_idx: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu"
    ) -> Tuple[Int[Tensor, "batch_size 3"], Optional[Int[Tensor, "batch_size 1"]]]:
        if image_idx == None:
            indices = super().sample_method(batch_size, num_images, image_height, image_width, mask = None, device = device)
            return indices, None

        sub_bs = batch_size // self.num_rays_per_image  # number of images
        img_idx = torch.randint(low=0, high=num_images, size=(sub_bs,))
        indices_rand = torch.rand((sub_bs, self.num_segs_per_image, self.num_rays_per_seg, 1))

        indices_list = []
        seg_idx_list = []
        for i, _img_id in enumerate(img_idx):
            img_id = image_idx[_img_id].item()
            num_segs = len(self.instance_weight_list[img_id])
            seg_idx = np.random.choice(num_segs, replace=True, size=self.num_segs_per_image, p=self.instance_weight_list[img_id])

            ind_list = []
            for j, _seg_idx in enumerate(seg_idx):
                uv = torch.from_numpy(self.plane_seg_uv_list[img_id][_seg_idx])
                indices = indices_rand[i, j] * uv.shape[0]
                indices = torch.floor(indices).long()
                ind_list.append(uv[indices].squeeze(1))

            indices_list.append(torch.cat(ind_list, dim=0))
            seg_idx_list.append(torch.from_numpy(seg_idx).repeat_interleave(self.num_rays_per_seg).view(-1, 1))

        indices = torch.cat(indices_list, dim=0) # [batch_size, 2]
        img_idx = img_idx.repeat_interleave(self.num_rays_per_image)[..., None]  # [batch_size, 1]
        indices = torch.cat([img_idx, indices], dim=1)  # [batch_size, 3]
        seg_idx_list = torch.cat(seg_idx_list, dim=0)  # [batch_size, 1]

        return indices, seg_idx_list

    def collate_image_dataset_batch(self, batch: Dict, num_rays_per_batch: int, sample_on_instances: bool = False, keep_full_image: bool = False):
        """Simplified version of the collate function in the original code."""
        device = batch["image"].device
        num_images, image_height, image_width, _ = batch["image"].shape

        image_idx = batch["image_idx"] if sample_on_instances else None
        indices, seg_idx_list = self.sample_method(num_rays_per_batch, num_images, image_height, image_width, image_idx=image_idx, device=device)

        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        c, y, x = c.cpu(), y.cpu(), x.cpu()
        collated_batch = {
            key: value[c, y, x] for key, value in batch.items() if key != "image_idx"  and key != "mask" and value is not None
        }
        assert collated_batch["image"].shape[0] == num_rays_per_batch

        indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = indices  # with the abs camera indices
        if keep_full_image:
            collated_batch["full_image"] = batch["image"]
        if sample_on_instances:
            # To locate the plane segment index
            collated_batch["num_images"] = num_rays_per_batch // self.num_rays_per_image
            collated_batch["num_segs_per_image"] = self.num_segs_per_image
            collated_batch["num_rays_per_seg"] = self.num_rays_per_seg
            collated_batch["seg_idx"] = torch.cat([indices[:, 0].view(-1, 1), seg_idx_list], dim=1)  # [batch_size, 2]

        return collated_batch

    def sample(self, image_batch: Dict, sample_on_instances: bool = False):
        """Sample an image batch and return a pixel batch. Simplified version of the sample function in the parent class.

        Args:
            image_batch: batch of images to sample from
            sample_on_instances: whether to sample on plane segments
        """
        if not isinstance(image_batch["image"], torch.Tensor):
            raise ValueError("image_batch['image'] must be a list or torch.Tensor")

        pixel_batch = self.collate_image_dataset_batch(
            image_batch, self.num_rays_per_batch,
            sample_on_instances=sample_on_instances,
            keep_full_image=self.config.keep_full_image
        )
        return pixel_batch


# ================ debugging ================
if __name__ == "__main__":
    sampler = PlaneInstancePixelSamplerConfig(
        num_rays_per_image=256,
        num_rays_per_seg=64
    ).setup()
    indices, seg_idx_list = sampler.sample_method(
        batch_size=4096,
        num_images=256,
        image_height=480,
        image_width=640,
        image_idx=None,
        device=torch.device("cuda")
    )

