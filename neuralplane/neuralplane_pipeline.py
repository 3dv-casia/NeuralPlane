from dataclasses import dataclass, field
from typing import Type, List

from neuralplane.neuralplane_model import NeuralPlaneModelConfig, NeuralPlaneModel
from neuralplane.neuralplane_datamanager import NeuralPlaneDataManagerConfig, NeuralPlaneDataManager

from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig

@dataclass
class NeuralPlanePipelineConfig(VanillaPipelineConfig):
    """Configuration for NeuralPlane pipeline instantiation"""
    _target: Type = field(default_factory=lambda: NeuralPlanePipeline)

    datamanager: NeuralPlaneDataManagerConfig = field(default_factory=NeuralPlaneDataManagerConfig)
    """prepare pose images and estimated local planar primitives as well as load training batches"""
    model: NeuralPlaneModelConfig = field(default_factory=NeuralPlaneModelConfig)
    """Nerfacto alongside the coplanarity field and the neural parser, where all loss functions are defined"""

    global_activation_step: int = 1000

class NeuralPlanePipeline(VanillaPipeline):
    config: NeuralPlanePipelineConfig
    datamanager: NeuralPlaneDataManager
    model: NeuralPlaneModel

    def __init__(self, config: NeuralPlanePipelineConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.local_planar_primitives_opt = False # whether to optimize local planar primitives. ON when the density field is holistically optimized to a decent state.

    def get_train_loss_dict(self, step: int):

        ray_bundle, batch = self.datamanager.next_train(step, sample_on_instances=True, local_planar_primitives_opt=self.local_planar_primitives_opt)

        model_outputs = self._model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        if step >= self.config.global_activation_step:
            if step == self.config.global_activation_step:
                self.local_planar_primitives_opt = True
                self.model.margin = 2.0

            self.datamanager.update_local_planar_primitives(batch, model_outputs)

        return model_outputs, loss_dict, metrics_dict
