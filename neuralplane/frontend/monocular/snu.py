# G. Bae, I. Budvytis, and R. Cipolla, “Estimating and Exploiting the Aleatoric Uncertainty in Surface Normal Estimation,” 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 13117–13126, Oct. 2021, doi: 10.1109/ICCV48922.2021.01289.

from dataclasses import dataclass, field
from typing import Type, Literal
from pathlib import Path
from easydict import EasyDict as edict

import torch
import numpy as np
from torchvision import transforms

from nerfstudio.utils.rich_utils import CONSOLE

from neuralplane.frontend.monocular.basis import MonocularPredictor, MonocularPredictorConfig
import thirdparty.surface_normal_uncertainty as snu

@dataclass
class SNUConfig(MonocularPredictorConfig):
    _target: Type = field(default_factory=lambda: SNU)
    """target class to instantiate"""
    pretrain_dataset: Literal["scannet", "nyu"] = "scannet"
    "on which dataset is the checkpoint pretrained"

class SNU(MonocularPredictor):
    def __init__(self, config: SNUConfig, device, **kwargs):
        super().__init__(config, device, **kwargs)

        checkpoint = self.config.ckpt_dir / Path(f"snu_{self.config.pretrain_dataset}.pt")
        CONSOLE.log(f"loading surface normal predictor 'SNU' checkpoint from: {checkpoint}")

        model_configs = edict(
                {
                    "architecture": "BN",
                    "pretrained": self.config.pretrain_dataset,
                    "sampling_ratio": 0.4,
                    "importance_ratio": 0.7,
                    "input_height": self.kwargs.get("img_height", 480),
                    "input_width": self.kwargs.get("img_width", 640),
                }
            )

        self.model = snu.models.snu_NNET(model_configs).to(self.device)
        self.model = snu.utils.load_checkpoint(checkpoint, self.model)
        self.model.eval()

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, img):
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        img = self.normalize(img).to(self.device)

        norm_out_list, _, _ = self.model(img)
        norm_out = norm_out_list[-1]
        pred_norm = norm_out[:, :3, :, :].detach().cpu().permute(0, 2, 3, 1).numpy()  # (B, H, W, 3)
        pred_kappa = norm_out[:, 3:, :, :].detach().cpu().permute(0, 2, 3, 1).numpy()  # (B, H, W, 3)
        pred_alpha = snu.utils.kappa_to_alpha(pred_kappa)

        batch = {
            "pred_norm": pred_norm[0],
            "pred_uncert": pred_alpha[0, :, :, 0]
        }

        return batch