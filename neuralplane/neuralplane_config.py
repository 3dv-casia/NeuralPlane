from pathlib import Path

from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.data.dataparsers.scannet_dataparser import ScanNetDataParserConfig
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig

from neuralplane.neuralplane_pipeline import NeuralPlanePipelineConfig
from neuralplane.neuralplane_datamanager import NeuralPlaneDataManagerConfig
from neuralplane.neuralplane_model import NeuralPlaneModelConfig
from neuralplane.neuralplane_pixel_sampler import PlaneInstancePixelSamplerConfig
from neuralplane.neuralplane_coplanarity_field import CoplanarityFieldConfig
from neuralplane.neuralplane_parser import ParserConfig

neuralplane_method = MethodSpecification(
    config=TrainerConfig(
        method_name="neuralplane",
        steps_per_eval_batch=0,  # never
        steps_per_eval_image=0,  # never
        steps_per_eval_all_images=0,  # never
        steps_per_save=0,  # never
        max_num_iterations=4000,
        mixed_precision=True,
        pipeline = NeuralPlanePipelineConfig(
            datamanager=NeuralPlaneDataManagerConfig(
                dataparser=ScanNetDataParserConfig(
                    data = Path("datasets/scannetv2/0084_00"),
                    train_split_fraction=1.0,
                    scale_factor=1.0,
                    scene_scale=1.0,
                    auto_scale_poses=False,
                    load_3D_points=False,
                ),
                local_planar_primitives_opt_mode="addon",
                cache_path = Path("outputs/scannetv2/0084_00/local_planar_primitives/cache.h5"),
                train_num_rays_per_batch=8192,
                pixel_sampler=PlaneInstancePixelSamplerConfig(
                    num_rays_per_image=2048,
                    num_rays_per_seg=64,
                ),
                num_pts_threshold=50,
            ),
            model=NeuralPlaneModelConfig(
                near_plane=0.1,
                far_plane=6.0,
                background_color="black",
                camera_optimizer=CameraOptimizerConfig(mode="off"),
                disable_scene_contraction=False,
                LOSS_WEIGHT_NORMAL=0.01,
                LOSS_WEIGHT_PDEPTH=0.1,
                LOSS_WEIGHT_PULL=0.5,
                ds_depth_loss_sigma=5e-3,
                normal_threhold=10,
                offset_threshold=0.08,
                coplanarity_field=CoplanarityFieldConfig(
                    n_dims=4
                ),
                parser=ParserConfig(
                    dim_hidden=8,
                    num_prototypes=32, # better vis with 64
                    dbscan_eps=0.2
                )
            ),
            global_activation_step=1000
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-3, max_steps=4000
                ),
            },
            "coplanarity_field": {
                "optimizer": AdamOptimizerConfig(
                    lr=1e-4, eps=1e-15, weight_decay=1e-6, max_norm=1.0
                ),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-5, max_steps=4000
                ),
            },
            "local_planar_primitives_residuals": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=4000
                ),
            },
            "parser": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-5, max_steps=4000
                ),
            }
        },
        vis="tensorboard",
    ),
    description="rebuilds indoor scenes as arrangements of planar primitives"
)
