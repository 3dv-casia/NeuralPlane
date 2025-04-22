import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
import os

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        basemodel_name = 'tf_efficientnet_b5_ap'
        print('Loading base model {}...'.format(basemodel_name), end='')
        model_cache_path = f'{os.path.expanduser("~/.cache/torch/hub/rwightman_gen-efficientnet-pytorch_master/")}'
        if not Path(model_cache_path).exists():
            basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name, pretrained=True)
        else:
            basemodel = torch.hub.load(model_cache_path, basemodel_name, pretrained=True, trust_repo=True, source='local')

        print('Done.')

        # Remove last layer
        print('Removing last two layers (global_pool & classifier).')
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()

        self.original_model = basemodel

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features


