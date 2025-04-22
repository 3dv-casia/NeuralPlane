from dataclasses import dataclass
from pathlib import Path
from typing import List

import tyro
import trimesh
import numpy as np
import open3d as o3d

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.engine.trainer import TrainerConfig

from neuralplane.evaluation.eval_segm import project_to_mesh
from neuralplane.export.point_cloud_exporter import PointCloudExporter
from neuralplane.export.plane_ransac import PlaneRANSAC
from neuralplane.export.surface_triangulation import SurfaceReconstruction
from neuralplane.utils.yaml import load_yaml, update_yaml
from neuralplane.utils.disp import ColorPalette

from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.scripts.train import _set_random_seed

@dataclass
class Args:
    config: Path = Path("config.yaml")
    '''Path to the configuration file'''

class NeuralPlaneExporter:

    def __init__(self, args: Args):
        self.config_yaml_path = args.config
        self.config = load_yaml(args.config)

        self.out_dir = self.config.TRAIN.NS_CONFIG.parent / "export"
        if self.out_dir.exists():
            CONSOLE.log(f"Export directory {self.out_dir} already exists. Remove ...")
            import shutil
            shutil.rmtree(self.out_dir)
            CONSOLE.log(f"Export directory {self.out_dir} removed.")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.seed = self.config.MACHINE.SEED
        _set_random_seed(self.seed)

        def fix_eval_sampler(config: TrainerConfig):
            config.pipeline.datamanager.train_num_rays_per_batch = 8192
            config.pipeline.datamanager.pixel_sampler.num_rays_per_image = 2048
            config.pipeline.datamanager.pixel_sampler.num_rays_per_seg = 64
            return config

        _, self.pipeline, _, _ = eval_setup(self.config.TRAIN.NS_CONFIG, update_config_callback=fix_eval_sampler)

        self.device = self.pipeline.device

        pass

    def run(self):
        pcd: trimesh.Trimesh = PointCloudExporter(
            output_dir=self.out_dir,
            num_points=self.config.EXPORT.NUM_POINTS,
            fusion_max_depth=self.config.EXPORT.FUSION_MAX_DEPTH,
        ).export(self.pipeline)

        # for development: load from file
        # import numpy as np
        # pcd = trimesh.load(self.out_dir / "point_cloud.ply", process=False)
        # attributes = np.array([[em[3], em[4], em[5], em[-2]] + em[-1][1].tolist() for em in pcd.metadata['_ply_raw']['vertex']['data']])
        # pcd = trimesh.Trimesh(
        #     vertices=pcd.vertices,
        #     vertex_normals=attributes[:, :3],
        #     vertex_colors=pcd.visual.vertex_colors,
        #     vertex_attributes= {"proto_labels": attributes[:, 3].astype(int), "seg_ids": attributes[:, 4:].astype(int)}
        # )

        plane_instances = PlaneRANSAC(
            output_dir=self.out_dir,
            iter=self.config.EXPORT.SEQ_RANSAC.ITER,
            offset_threshold=self.config.EXPORT.SEQ_RANSAC.OFFSET_THRESHOLD,
            normal_threshold=self.config.EXPORT.SEQ_RANSAC.NORMAL_THRESHOLD,
            seed=self.seed,
        ).run(pcd)

        # for development: load from file
        # import h5py
        # import torch
        # plane_instances = []
        # with h5py.File(self.out_dir / "plane_instances.h5", "r") as f:
        #     for key in f.keys():
        #         data = (torch.from_numpy(f[key]["plane_params"][()]).to('cuda'), torch.from_numpy(f[key]["seg_ids"][()]).to('cuda'), torch.from_numpy(f[key]["verts"][()]).to('cuda'))
        #         plane_instances.append(data)

        mesh_instances = SurfaceReconstruction(
            output_dir=self.out_dir,
        ).run(self.pipeline, plane_instances, mode="mask")

        # color the results as well as prepare for evaluation
        eval_path = self.out_dir / "eval"
        eval_path.mkdir(exist_ok=True, parents=True)
        CONSOLE.log(f"Exporting evaluation data to {eval_path}")

        file_geo_verts = Path("eval") / "geo_verts.ply"
        file_segm_label = Path("eval") / "segm_label.txt"

        self.post_processing(mesh_instances, self.out_dir / file_geo_verts, self.out_dir / file_segm_label)

        self.config.EVAL.EXPORT_DIR = self.out_dir
        self.config.EVAL.GEO_VERTS = file_geo_verts  # relative to EXPORT_DIR
        self.config.EVAL.SEGM_LABELS = file_segm_label
        update_yaml(self.config_yaml_path, self.config)

        CONSOLE.rule(f"[bold yellow]scene{self.config.DATA.SCENE_ID}[/]: Exporting Completed")

        pass

    def post_processing(self, piecewise_meshes: List[trimesh.Trimesh], file_geo_verts: Path, file_segm_label: Path) -> None:
        gt_mesh = trimesh.load(self.config.DATA.ANNOTATION / "planes_mesh_vis.ply", process=False)
        gt_labels = np.array([em[-1] for em in gt_mesh.metadata['_ply_raw']['vertex']['data']])

        gt_verts_pc = []
        gt_verts_color = []
        gt_n_ins = np.max(gt_labels) + 1
        for i in range(gt_n_ins):
            mask = gt_labels == i
            gt_verts_pc.append(
                o3d.t.geometry.PointCloud(np.asarray(gt_mesh.vertices[mask], dtype=np.float32)).voxel_down_sample(0.02)
            )
            gt_verts_color.append(
                gt_mesh.visual.vertex_colors[mask][0]
            )

        # Coloring: for each piecewise mesh, find the closest ground truth mesh
        colorMap_vis = ColorPalette(len(piecewise_meshes))
        flag = np.zeros(gt_n_ins, dtype=bool)
        pred_ins_id = []
        for i, mesh in enumerate(piecewise_meshes):
            pts, _ = trimesh.sample.sample_surface_even(mesh, 100, seed=self.seed)
            pc = o3d.t.geometry.PointCloud(np.asarray(pts, dtype=np.float32))
            chamer_distances = []
            for gt_pc in gt_verts_pc:
                # Compute Chamfer distance using open3d: https://www.open3d.org/docs/latest/python_api/open3d.t.geometry.PointCloud.html#open3d.t.geometry.PointCloud.compute_metrics
                dist = gt_pc.compute_metrics(pc, [o3d.t.geometry.Metric.ChamferDistance], o3d.t.geometry.MetricParameters()).cpu().numpy()
                chamer_distances.append(dist)
            ind = np.array(chamer_distances).argmin()
            if not flag[ind]:
                flag[ind] = True
                mesh.visual.vertex_colors = gt_verts_color[ind]
            else:
                mesh.visual.vertex_colors = colorMap_vis(i)

            pred_ins_id.append(np.ones(len(mesh.faces), dtype=np.int32) * i)

        pred_ins_id = np.concatenate(pred_ins_id)

        merged: trimesh.Trimesh = trimesh.util.concatenate(piecewise_meshes)
        merged.export(self.out_dir / f"vis_{self.config.DATA.SCENE_ID}.ply")

        # for geometry evaluation
        eval_points, face_index = trimesh.sample.sample_surface_even(merged, self.config.EVAL.NUM_POINTS, seed=self.seed)
        pred_ins_id = pred_ins_id[face_index]

        eval_ply = trimesh.Trimesh(vertices=np.asarray(eval_points), vertex_colors=merged.visual.face_colors[face_index], vertex_attributes={"pred_ins_id": pred_ins_id})

        eval_ply.export(file_geo_verts)

        # for segmentation evaluation: project the predicted mesh labels to the ground truth mesh (see Appendix A.3.3)
        gt_mesh_transferr_labels, transfered_labels = project_to_mesh(
            from_mesh=eval_ply, to_mesh=gt_mesh, attribute=pred_ins_id, attr_name="pred_ins_id"
        )
        gt_mesh_transferr_labels.export(file_segm_label.parent / "gt_mesh_label_transferred.ply")
        np.savetxt(file_segm_label, transfered_labels, fmt="%d")

        pass

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    NeuralPlaneExporter(
        tyro.cli(Args)
    ).run()

if __name__ == "__main__":
    entrypoint()