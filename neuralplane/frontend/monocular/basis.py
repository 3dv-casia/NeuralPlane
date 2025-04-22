# Quick wrapper for surface normal prediction model

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Type, Union
from pathlib import Path
from nerfstudio.configs.base_config import InstantiateConfig

import torch
import numpy as np

@dataclass
class MonocularPredictorConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: MonocularPredictor)
    """target class to instantiate"""
    ckpt_dir: Path = Path("checkpoints")
    "checkpoint path"

class MonocularPredictor:
    def __init__(self, config: MonocularPredictorConfig, device: Union[torch.device, str], **kwargs):
        self.config = config
        self.kwargs = kwargs
        self.device = device
        self.model = None

    @abstractmethod
    def __call__(self, img: np.ndarray):

        raise NotImplementedError
