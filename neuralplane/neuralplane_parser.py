from dataclasses import dataclass, field
from typing import Callable, Union
from jaxtyping import Float

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment

from nerfstudio.configs.base_config import InstantiateConfig

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

@dataclass
class ParserConfig(InstantiateConfig):
    _target: type = field(default_factory=lambda: Parser)
    """target class to instantiate"""

    dim_hidden: int = 8
    """hidden dimensions"""
    num_prototypes: int = 32

    dbscan_eps: float = 0.2

class Parser(nn.Module):
    def __init__(
        self,
        config: ParserConfig,
        dim_out: int,
        feature_loss: Callable[
            [Float[torch.Tensor, "bs n_dim"], Float[torch.Tensor, "bs n_dim"]], Float[torch.Tensor, "bs 1"]
        ],
    ):
        super().__init__()
        self.config = config
        self.dim_hidden = config.dim_hidden
        self.num_prototypes = config.num_prototypes

        self.dim_out = dim_out
        self.feature_loss = feature_loss

        self.prototypes = nn.Embedding(self.num_prototypes, self.dim_hidden)

        generator = torch.Generator()
        generator.manual_seed(2025)
        self.prototypes.weight.data.normal_(0, 0.12, generator=generator)

        self.triu_indices = torch.triu_indices(self.num_prototypes, self.num_prototypes, offset=1)
        self.FFN = FFNLayer(self.dim_hidden, self.dim_out, activation="relu")

    def forward(self, primitive_feats):
        pho = self.FFN(self.prototypes.weight)
        miu = self._cluster(primitive_feats)
        match_indices = self._hungarian_matcher(pho, miu)

        proto_mutual_loss = F.relu(
            2.0 - self.feature_loss(pho[self.triu_indices[1]], pho[self.triu_indices[0]])
        ).nanmean().float()

        # Eq. (8)
        return {
            "parser_loss": self.feature_loss(pho[match_indices[1]], miu[match_indices[0]]).nanmean().float(),
            "proto_mutual_loss": proto_mutual_loss
        }

    @torch.no_grad()
    def _hungarian_matcher(self, outputs: Float[torch.Tensor, "N_x n_dim"], targets: Float[torch.Tensor, "N_y n_dim"]) -> Float[torch.Tensor, "N_x N_y"]:
        # cost matrix
        C = torch.cdist(outputs, targets, p=2).view(-1, self.num_prototypes).cpu()
        indices = linear_sum_assignment(C)

        return torch.from_numpy(np.array(indices)).to(outputs.device)


    @torch.no_grad()
    def _cluster(self, feats):
        clustering = DBSCAN(eps=self.config.dbscan_eps, min_samples=1, metric='euclidean').fit(feats.detach().cpu().numpy())
        labels = clustering.labels_

        cluster_mean = torch.stack(
            [feats[labels == i].nanmean(dim=0) for i in range(labels.max() + 1)]
        ).float()

        return cluster_mean


class FFNLayer(nn.Module):
    def __init__(
        self,
        dim_hidden,
        dim_out,
        dropout=0.0,
        activation="relu",
        post_normalize=True,
    ):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(dim_hidden, dim_hidden)
        self.linear2 = nn.Linear(dim_hidden, dim_hidden)
        self.linear3 = nn.Linear(dim_hidden, dim_out)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self._reset_parameters()

        self.post_normalize = post_normalize

    def _reset_parameters(self):
        generator = torch.Generator()
        generator.manual_seed(2025 + 128439)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, generator=generator)

    def forward(self, x):
        x1 = self.linear1(x)
        x1 = self.activation(x1)
        x1 = self.dropout(x1)

        x2 = self.linear2(x1)
        x2 = self.activation(x2)
        x2 = self.dropout(x2)

        x3 = self.linear3(x2)

        if self.post_normalize:
            x3 = F.normalize(x3, p=2, dim=-1)

        return x3