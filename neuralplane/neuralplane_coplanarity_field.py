# Modified from Garfield: https://github.com/chungmin99/garfield/blob/main/garfield/garfield_field.py

from dataclasses import dataclass, field
from typing import Type, Dict, Any, Tuple

import torch
import torch.nn as nn
import tinycudann as tcnn
import numpy as np

from nerfstudio.fields.base_field import FieldConfig, Field
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox

@dataclass
class CoplanarityFieldConfig(FieldConfig):
    _target: Type = field(default_factory=lambda: CoplanarityField)
    """A simple feature field class to instantiate."""
    hashgrid_cfg: Dict[str, Any] = field(
        default_factory=lambda: {
            "resolution_range": [(16, 256)],
            "level": [12],
        }
    )
    """Field parameters."""
    n_dims: int = 4
    """dimensions of the field."""
    mlp_hidden_layers: int = 4
    """number of hidden layers in the MLP."""
    mlp_neurons: int = 256
    """number of neurons in the hidden layers."""

class CoplanarityField(Field):
    def __init__(self, config: CoplanarityFieldConfig, aabb, spatial_distortion=None):
        super().__init__()
        self.spatial_distortion = spatial_distortion
        hashgrid_cfg = config.hashgrid_cfg
        self.n_dims = config.n_dims

        self.register_buffer("aabb", aabb)

        # This is a trick to make the hashgrid encoding work with the TCNN library.
        self.enc_list = torch.nn.ModuleList(
            [
                self._get_encoding(
                    hashgrid_cfg["resolution_range"][i], hashgrid_cfg["level"][i]
                )
                for i in range(len(hashgrid_cfg["level"]))
            ]
        )
        tot_out_dims = sum([e.n_output_dims for e in self.enc_list])

        # This is the MLP that takes the hashgrid encoding as input.
        self.instance_net = tcnn.Network(
            n_input_dims=tot_out_dims,
            n_output_dims=config.n_dims,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": config.mlp_neurons,
                "n_hidden_layers": config.mlp_hidden_layers,
            },
        )

    @staticmethod
    def _get_encoding(res_range: Tuple[int, int], levels: int, indim=3, hash_size=19) -> tcnn.Encoding:
        """Helper function to create a HashGrid encoding."""
        start_res, end_res = res_range
        growth = np.exp((np.log(end_res) - np.log(start_res)) / (levels - 1))

        enc = tcnn.Encoding(
            n_input_dims=indim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": levels,
                "n_features_per_level": 8,
                "log2_hashmap_size": hash_size,
                "base_resolution": start_res,
                "per_level_scale": growth,
            },
        )
        return enc

    def get_outputs(self, ray_samples):
        """This function is not supported -- please use `get_hash` and `get_mlp` instead. `get_mlp` assumes that hash values are normalized, which requires the renderer (in the model)."""
        raise NotImplementedError

    def get_hash(self, ray_samples: RaySamples) -> torch.Tensor:
        """Get the hashgrid encoding. Note that this function does *not* normalize the hash values."""

        positions = ray_samples.frustums.get_positions().detach()
        return self.get_hash_from_positions(positions)

    def get_hash_from_positions(self, positions: torch.Tensor) -> torch.Tensor:
        if self.spatial_distortion == None:
            positions = SceneBox.get_normalized_positions(positions, self.aabb)
        else:
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0

        xs = [e(positions.view(-1, 3)) for e in self.enc_list]
        x = torch.concat(xs, dim=-1)
        hash = x.view(positions.shape[:-1] + (-1,))
        return hash

    def get_mlp(self, hash_rendered: torch.Tensor) -> torch.Tensor:
        """This function *does* assume that the hash values are normalized. The MLP output is normalized to unit length."""
        epsilon = 1e-5

        output = self.instance_net(hash_rendered)
        output = output / (torch.norm(output, dim=-1, keepdim=True) + epsilon)
        return output