from dataclasses import dataclass
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
from pathlib import Path

import open3d as o3d
import trimesh
import torch
import numpy as np

from nerfstudio.scripts.exporter import ExportPointCloud
from nerfstudio.utils.rich_utils import CONSOLE

from neuralplane.neuralplane_pipeline import NeuralPlanePipeline

# Modified based on NerfStudio: https://github.com/nerfstudio-project/nerfstudio/blob/73fe54dda0b743616854fc839889d955522e0e68/nerfstudio/exporter/exporter_utils.py#L83
@torch.no_grad()
def generate_point_cloud(
    pipeline: NeuralPlanePipeline,
    num_points: int = 1000000,
    rgb_output_name: str = "rgb",
    depth_output_name: str = "depth",
    fusion_max_depth: float = 3.0,
    std_ratio: float = 2.0,
)-> o3d.geometry.PointCloud:
    """Generate a point cloud from a nerf."""
    prototypes = pipeline.model.get_parser_prototype_features()

    points = []
    rgbs = []
    normals = []
    plane_cluster_ids = []
    plane_seg_ids = []
    view_directions = []

    progress = Progress(
        TextColumn(":ten_o’clock: Computing Point Cloud :ten_o’clock:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        TimeRemainingColumn(elapsed_when_finished=False, compact=True),
        console=CONSOLE,
    )

    with progress as progress_bar:
        task = progress_bar.add_task("Generating Point Cloud", total=num_points)
        while not progress_bar.finished:
            ray_bundle, batch = pipeline.datamanager.next_train(0, sample_on_instances=True, local_planar_primitives_opt=True)
            model_outputs = pipeline.model(ray_bundle)

            assert rgb_output_name in model_outputs, f"Output {rgb_output_name} not found in model outputs"
            assert depth_output_name in model_outputs, f"Output {depth_output_name} not found in model outputs"

            seg_idx = batch["seg_idx"].to(pipeline.device)

            rgba = pipeline.model.get_rgba_image(model_outputs, rgb_output_name)
            depth = model_outputs[depth_output_name].view(-1, 1)

            ray_o = ray_bundle.origins.view(-1, 3)
            ray_d = ray_bundle.directions.view(-1, 3)
            _points = ray_o + ray_d * depth  # bs 3
            _normals = batch['plane_params_w'][:, :3]  # bs 3
            feats = pipeline.model.query_coplarity_feature_at_positions(_points)

            # filter invalid planar primitives
            num_rays_per_seg = batch["num_rays_per_seg"]
            per_point_offsets = torch.bmm(_points.view(-1, 1, 3), _normals.view(-1, 3, 1)).view(-1, num_rays_per_seg)
            per_point_offsets_var = torch.var(per_point_offsets, dim=1)
            mask_param_valid = batch['valid_instances'] & (per_point_offsets_var < 1e-3).repeat_interleave(num_rays_per_seg, dim=0)

            # filter points with ambiguous coplanar relationship
            C = torch.cdist(feats, prototypes, p=2)
            pts_2_prototype = C.argmin(dim=1).view(-1, num_rays_per_seg, 1)
            stats = [torch.unique(pts_2_prototype[i], return_counts=True) for i in range(pts_2_prototype.shape[0])]  # len: mask_bs

            max_count_id = [torch.argmax(counts) for _, counts in stats]
            mode_prototype_id = torch.stack([s[0][id] for s, id in zip(stats, max_count_id)], dim=0).view(-1, 1)
            mode_count = torch.stack([s[1][id] for s, id in zip(stats, max_count_id)], dim=0)

            _mask = mode_count > 0.5 * num_rays_per_seg

            prototype_id = mode_prototype_id.repeat_interleave(num_rays_per_seg, dim=0)
            mask_coplane = _mask.repeat_interleave(num_rays_per_seg, dim=0) & (pts_2_prototype.view(-1) == prototype_id.view(-1))

            # filter points with opacity lower than 0.5
            mask_opacity = rgba[:, 3] > 0.5

            # filter points with depth larger than fusion_max_depth
            mask_depth = depth.squeeze() < fusion_max_depth

            mask = mask_coplane & mask_opacity & mask_depth & mask_param_valid

            points.append(_points[mask])
            rgbs.append(rgba[mask, :3])
            normals.append(_normals[mask])
            plane_cluster_ids.append(prototype_id[mask])
            plane_seg_ids.append(seg_idx[mask])
            view_directions.append(ray_d[mask])

            progress.advance(task, mask.sum())

    points = torch.cat(points, dim=0)
    rgbs = torch.cat(rgbs, dim=0)
    normals = torch.cat(normals, dim=0)
    plane_cluster_ids = torch.cat(plane_cluster_ids, dim=0)
    plane_seg_ids = torch.cat(plane_seg_ids, dim=0)
    view_directions = torch.cat(view_directions, dim=0)

    pcd = o3d.t.geometry.PointCloud(points.cpu().numpy())
    pcd.point.proto_labels = plane_cluster_ids.cpu().numpy()
    pcd.point.seg_ids = plane_seg_ids.cpu().numpy()
    pcd.point.colors = rgbs.cpu().numpy()
    pcd.point.normals = normals.cpu().numpy()

    CONSOLE.log("Cleaning Point Cloud")
    pcd_clean, ind = pcd.remove_statistical_outliers(nb_neighbors=20, std_ratio=std_ratio)

    # re-orient the normals
    view_directions_clean = view_directions[torch.from_numpy(ind.numpy())]
    normals_clean = torch.from_numpy(pcd_clean.point.normals.numpy()).to(view_directions_clean.device)
    flip_mask = torch.bmm(normals_clean.view(-1, 1, 3), view_directions_clean.view(-1, 3, 1)).squeeze() > 0
    normals_clean[flip_mask] *= -1
    pcd_clean.point.normals = normals_clean.cpu().numpy()

    del pcd # free memory
    torch.cuda.empty_cache()

    return pcd_clean

@dataclass
class PointCloudExporter(ExportPointCloud):
    output_dir: Path = Path("export")
    """Path to the output directory."""
    load_config: Path = None
    fusion_max_depth: float = 3.0
    """depth to limit depth maps to when fusing. This is useful to avoid fusing points that are too far away from the camera"""

    def export(self, pipeline: NeuralPlanePipeline):

        pcd = generate_point_cloud(
            pipeline=pipeline,
            num_points=self.num_points,
            rgb_output_name=self.rgb_output_name,
            depth_output_name=self.depth_output_name,
            fusion_max_depth=self.fusion_max_depth,
            std_ratio=self.std_ratio,
        )

        # apply the inverse dataparser transform to the point cloud
        applied_transform = pipeline.datamanager.train_dataparser_outputs.dataparser_transform.cpu().numpy()
        applied_transform = np.concatenate((applied_transform, np.array([[0, 0, 0, 1]])), 0)
        inv_transform = np.linalg.inv(applied_transform)
        applied_scale = pipeline.datamanager.train_dataparser_outputs.dataparser_scale

        transformed_pcd = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(pcd.point.positions.numpy())
        ).scale(1./applied_scale, center=np.array([0., 0., 0.])).transform(inv_transform)
        transformed_normals = (inv_transform[:3, :3] @ pcd.point.normals.numpy().T).T

        # voxel downsampling
        min_bound, max_bound = transformed_pcd.get_min_bound(), transformed_pcd.get_max_bound()
        downpcd, _, record = transformed_pcd.voxel_down_sample_and_trace(voxel_size=0.02, min_bound=min_bound, max_bound=max_bound)
        record = np.asarray([idx[0] for idx in record])

        downpcd.normals = o3d.utility.Vector3dVector(transformed_normals[record])
        downpcd.colors = o3d.utility.Vector3dVector(pcd.point.colors.numpy()[record])

        # debug
        # o3d.visualization.draw_geometries([downpcd], point_show_normal=True)

        # vis
        vis_ply = trimesh.Trimesh(
            vertices=np.asarray(downpcd.points),
            vertex_normals=pcd.point.normals.numpy()[record],
            vertex_colors=pcd.point.colors.numpy()[record],
            vertex_attributes= {"proto_labels": pcd.point.proto_labels.numpy()[record].astype(np.int32).reshape(-1), "seg_ids": pcd.point.seg_ids.numpy()[record].astype(np.int32).reshape(-1, 2)}
        )
        vis_ply.export(self.output_dir / "vis_verts.ply")

        return vis_ply