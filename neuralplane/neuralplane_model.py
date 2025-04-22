from dataclasses import dataclass, field
from typing import Type, Dict, List
from jaxtyping import Float

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.model_components.renderers import SemanticRenderer
from nerfstudio.model_components.losses import scale_gradients_by_distance_squared, depth_loss, DepthLossType
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction

from neuralplane.neuralplane_coplanarity_field import CoplanarityFieldConfig, CoplanarityField
from neuralplane.neuralplane_parser import ParserConfig, Parser

class NormalizedSemanticRenderer(SemanticRenderer):
    """Render feature embeddings (semantics) along a ray, where the output features are unit norm"""
    @classmethod
    def forward(
        cls,
        embeds: Float[torch.Tensor, "bs num_samples n_dims"],
        weights: Float[torch.Tensor, "bs num_samples 1"],
    ) -> Float[torch.Tensor, "bs n_dims"]:
        """Calculate semantics along the ray."""
        output = torch.sum(weights * embeds, dim=-2)
        output = output / torch.linalg.norm(output, dim=-1, keepdim=True)
        return output

@dataclass
class NeuralPlaneModelConfig(NerfactoModelConfig):
    _target: Type = field(default_factory=lambda: NeuralPlaneModel)
    """target class to instantiate"""

    coplanarity_field: CoplanarityFieldConfig = field(default_factory=lambda: CoplanarityFieldConfig())
    """config for the neural coplanarity field (NCF)."""
    num_ncf_samples_per_ray: int = 24
    """number of samples per ray to use for NCF supervision."""

    parser: ParserConfig = field(default_factory=lambda: ParserConfig())
    """config for the neural parser (NP)."""

    LOSS_WEIGHT_NORMAL: float = 0.01
    """weight for the explicit normal loss. $\lambda_1$ in eq. (9) of the paper."""
    LOSS_WEIGHT_PDEPTH: float = 0.1
    """weight for the pseudo-depth supervision loss. $\lambda_2$ in Eq. (9) of the paper."""
    LOSS_WEIGHT_PULL: float = 0.5
    """weight for the pull loss. $\lambda_3$ in Eq. (9) of the paper. Note that the weight of push loss is consistently set to 1."""

    ds_depth_loss_sigma: float = 5e-3

    normal_threhold: float = 10  # degree
    """normal distance exceeding this threshold are considered as negative samples."""
    offset_threshold: float = 0.08  # meter
    """offset distance exceeding this threshold are considered as negative samples."""

class NeuralPlaneModel(NerfactoModel):
    def populate_modules(self):
        super().populate_modules()  # NefactoModel

        self.t_n = 1 - np.cos(np.deg2rad(self.config.normal_threhold))
        self.t_o = self.config.offset_threshold
        self.margin = 1.5  # margin for the push loss.

        # NCF
        scene_contraction = None if self.config.disable_scene_contraction else SceneContraction(order=float("inf"))
        self.coplanarity_field: CoplanarityField = self.config.coplanarity_field.setup(
            aabb = self.scene_box.aabb,
            spatial_distortion = scene_contraction
        )
        self.ncf_renderer = NormalizedSemanticRenderer()  # NCF renderer

        # Loss & Metrics
        self.normal_loss = lambda x, tgt: (1.0 - F.cosine_similarity(x, tgt, dim=-1)).nanmean()
        self.feature_dist = lambda x, y: F.pairwise_distance(x, y, p=2)
        self.normal_mutual_dist = lambda x: 1.0 - torch.matmul(x, x.T)  # compute mutual distances between normals. In: Nx3, Out: NxN
        self.offset_mutual_dist = lambda x: torch.cdist(x, x, p=1)  # compute mutual distances between offsets. In: Nx1, Out: NxN

        # Parser
        self.parser: Parser = self.config.parser.setup(
            dim_out=self.coplanarity_field.n_dims,
            feature_loss = self.feature_dist,
        )

    def get_param_groups(self) -> Dict[str, List[nn.Parameter]]:
        param_groups = super().get_param_groups()
        param_groups["coplanarity_field"] = list(self.coplanarity_field.parameters())
        param_groups["parser"] = list(self.parser.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        outputs = super().get_outputs(ray_bundle)  # Nefacto outputs

        # Compute NCF outputs

        # In eval mode, ray samples and weights are discarded by the parent class. Recomputing here seems wasteful...
        if self.training:
            ray_samples, weights = outputs["ray_samples_list"][-1], outputs["weights_list"][-1]
        else:
            ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
            field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
            if self.config.use_gradient_scaling:
                field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)
            weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

        ncf_sample_weights, ncf_sample_id = torch.topk(
            weights, self.config.num_ncf_samples_per_ray, dim=-2, sorted=False
        )
        def gather_fn(_tensor):
            return torch.gather(
                _tensor, -2, ncf_sample_id.expand(*ncf_sample_id.shape[:-1], _tensor.shape[-1])
            )

        dataclass_fn = lambda dc: dc._apply_fn_to_fields(gather_fn, dataclass_fn)

        ncf_samples: RaySamples = ray_samples._apply_fn_to_fields(gather_fn, dataclass_fn)
        hash = self.coplanarity_field.get_hash(ncf_samples)
        hash_rendered = self.ncf_renderer(embeds=hash, weights=ncf_sample_weights.detach().half())

        outputs["coplarity_feat"] = self.coplanarity_field.get_mlp(hash_rendered).float()

        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        for loss_fn in [self._get_normal_loss, self._get_pseudo_depth_loss, self._get_pull_push_loss, self._get_parser_loss]:
            loss_dict.update(loss_fn(outputs, batch))

        return loss_dict

    def query_coplarity_feature_at_positions(self, positions):
        """Query the coplanarity feature at the given *positions*. This function is used for the final explicification."""

        hash: torch.Tensor = self.coplanarity_field.get_hash_from_positions(positions)
        hash_normalized = F.normalize(hash, p=2, dim=-1)
        return self.coplanarity_field.get_mlp(hash_normalized).float()

    @torch.no_grad()
    def get_parser_prototype_features(self):
        return self.parser.FFN(self.parser.prototypes.weight)

    def _get_normal_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, Float[torch.Tensor, "1"]]:
        """Compute the normal loss: Eq.(4) in the paper."""
        K_inv_dot_xy1 = batch["K_inv_dot_xy1"].to(self.device)
        num_rays_per_seg = batch["num_rays_per_seg"]
        plane_normals = batch["plane_params"][:, :3].to(self.device).detach()

        X1 = K_inv_dot_xy1 * outputs["expected_depth"]  # bs N 3
        X1 = X1.view(-1, num_rays_per_seg, 3)
        X2 = torch.roll(X1, shifts=1, dims=1)
        X3 = torch.roll(X1, shifts=2, dims=1)

        # Eq. (3) in the paper. NeRF-derived normals using triplets of rays.
        zz = torch.cross(X2 - X1, X3 - X1, dim=-1).view(-1, 3)
        flip_mask = zz[:, 2] < 0
        zz[flip_mask] *= -1

        # Eq. (4)
        return {"normal_loss": self.config.LOSS_WEIGHT_NORMAL * self.normal_loss(zz, plane_normals).float()}

    def _get_pseudo_depth_loss(self, outputs, batch):
        K_inv_dot_xy1 = batch["K_inv_dot_xy1"].to(self.device)  # bs 3
        plane_params = batch["plane_params"].to(self.device)
        valid_mask = batch["valid_instances"].to(self.device)

        plane_normals = plane_params[:, :3].view(-1, 3)  # bs' 3
        plane_offsets = plane_params[:, 3].view(-1, 1)  # bs' 1

        # Compute the pseudo-depth derived from the estimated plane parameters.
        normal_xyz = torch.bmm(plane_normals.view(-1, 1, 3), K_inv_dot_xy1.view(-1, 3, 1)).view(-1, 1)  # bs', 1
        plane_depths = -plane_offsets / (normal_xyz + 1e-5)  # bs' 1
        plane_depths = torch.clamp_min(plane_depths, 0.0)
        if self.config.far_plane is not None:
            valid_mask = valid_mask & (plane_depths.view(-1) < self.config.far_plane)

        return {"pseudo_depth_loss": self.config.LOSS_WEIGHT_PDEPTH * self._depth_loss(outputs, plane_depths, valid_mask)}

    def _get_pull_push_loss(self, outputs, batch):
        n_dims = self.coplanarity_field.n_dims
        num_rays_per_seg = batch["num_rays_per_seg"]

        losses = {}

        """features within the same mask should locate close to each other"""
        feats_i = outputs["coplarity_feat"].view(-1, num_rays_per_seg, n_dims)
        feats_j = torch.roll(feats_i, shifts=1, dims=1)

        losses.update(
            {"pull_loss": self.config.LOSS_WEIGHT_PULL * self.feature_dist(feats_i, feats_j).nanmean().float()}
        )

        """features from different masks AND fail to meet the geometry verification should be as distinct (Eq. (7) in the paper)."""
        valid_mask = batch["valid_instances"].to(self.device)[::num_rays_per_seg]
        plane_params_w = batch["plane_params_w"].to(self.device)[::num_rays_per_seg]  # bs 4

        feats = outputs["coplarity_feat"].view(-1, num_rays_per_seg, n_dims)
        primitive_feats = F.normalize(feats.mean(dim=1)[valid_mask])

        with torch.no_grad():
            normal_dist_mat = self.normal_mutual_dist(plane_params_w[valid_mask, :3])

            offset_dist_mat = self.offset_mutual_dist(plane_params_w[valid_mask, 3].view(-1, 1))

            negative_indicator = torch.triu(
                (normal_dist_mat > self.t_n) | (offset_dist_mat > self.t_o),
                diagonal=1
            )
            indices = torch.where(negative_indicator)  # hard mask

        push_loss = F.relu(
            self.margin - self.feature_dist(primitive_feats[indices[0]], primitive_feats[indices[1]])
        ).nanmean()

        losses.update(
            {"push_loss": push_loss.float()}
        )

        return losses

    def _get_parser_loss(self, outputs, batch):
        n_dims = self.coplanarity_field.n_dims
        num_rays_per_seg = batch["num_rays_per_seg"]
        valid_mask = batch["valid_instances"].to(self.device)[::num_rays_per_seg]

        feats = outputs["coplarity_feat"].view(-1, num_rays_per_seg, n_dims).detach()

        with torch.no_grad():
            primitive_feats = F.normalize(feats.mean(dim=1)[valid_mask])

        return self.parser(primitive_feats)

    def _depth_loss(self, outputs, depths, mask):
        """Helper function to compute the depth loss. Eq. (5) in the paper."""
        loss = 0
        for i in range(len(outputs["weights_list"])):
            loss += depth_loss(
                weights = outputs["weights_list"][i][mask],
                ray_samples=outputs["ray_samples_list"][i][mask],
                termination_depth=depths[mask],
                predicted_depth=outputs["depth"][mask],
                sigma=self.config.ds_depth_loss_sigma,
                directions_norm=None,
                is_euclidean=True,
                depth_loss_type=DepthLossType.DS_NERF
            )  / len(outputs["weights_list"])
        return loss
