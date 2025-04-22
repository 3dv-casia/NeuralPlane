from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image, ImageFilter

import torch
import os
import numpy as np
import contextlib
import io

from neuralplane.frontend.monocular.basis import MonocularPredictor, MonocularPredictorConfig

@dataclass
class StableNormalConfig(MonocularPredictorConfig):
    _target: type = field(default_factory=lambda: StableNormal)
    """target class to instantiate"""

class StableNormal(MonocularPredictor):
    def __init__(self, config: MonocularPredictorConfig, device, **kwargs):
        super().__init__(config, device, **kwargs)

        basemodel_name = "StableNormal"
        model_cache_path = f'{os.path.expanduser("~/.cache/torch/hub/Stable-X_StableNormal_main/")}'

        if not os.path.exists(model_cache_path):
            with contextlib.redirect_stdout(io.StringIO()):
                model = torch.hub.load("Stable-X/StableNormal", basemodel_name, trust_repo=True, local_cache_dir=self.config.ckpt_dir)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                model = torch.hub.load(model_cache_path, basemodel_name, trust_repo=True, source='local', local_cache_dir=self.config.ckpt_dir)

        self.predictor = model.to(self.device)

    def __call__(self, img):

        img = Image.fromarray(img)

        with contextlib.redirect_stdout(io.StringIO()):
            normal_image = self.predictor(img)

        # blur
        # normal_image = normal_image.filter(ImageFilter.BoxBlur(radius=3))

        pred_norm = np.array(normal_image) / 255 * 2 - 1
        batch = {
            "pred_norm": pred_norm
        }

        return batch


