from dataclasses import dataclass
from typing import List, Tuple, Literal
from pathlib import Path
from scipy.spatial import Delaunay
from joblib import Parallel, delayed
import mapbox_earcut as earcut

import torch
import torch.nn.functional as F
import numpy as np
import trimesh

import cv2

from nerfstudio.utils.rich_utils import CONSOLE

from neuralplane.neuralplane_pipeline import NeuralPlanePipeline
from neuralplane.neuralplane_datamanager import NeuralPlaneDataManager
from neuralplane.utils.mask import mask2polygon, polygon_to_mask
from neuralplane.utils.geometry import transformPlanes, pose_inverse

class CanonicalPlane:
    def __init__(self, verts: torch.Tensor, plane_params: torch.Tensor, resolution=0.01):
        normal = plane_params[:3].view(1, 3)  # 1x3
        center_3d = verts.mean(dim=0).view(1, 3)  # 1x3
        reference_3d = verts[0].view(1, 3)  # 1x3

        axis_1 = F.normalize(center_3d - reference_3d)  # 1x3
        axis_2 = torch.cross(normal, axis_1, dim=-1)  # 1x3

        proj_mat = torch.cat((axis_1, axis_2), dim=0)  # 2x3

        # Project the points onto the plane
        verts_3d = verts - center_3d
        verts_2d = verts_3d @ proj_mat.T  # Nx2

        # Rasterize
        verts_2d_coord = torch.round(verts_2d / resolution).int()  # nx2
        min_coord = verts_2d_coord.min(dim=0).values - 5
        max_coord = verts_2d_coord.max(dim=0).values + 5

        self.height = max_coord[1] - min_coord[1]
        self.width = max_coord[0] - min_coord[0]
        self.grid = np.zeros((self.height, self.width), dtype=np.uint8)
        self.proj_mat = proj_mat.cpu().numpy()
        self.min_coord = min_coord.cpu().numpy()
        self.center_3d = center_3d.cpu().numpy()
        self.verts = verts.cpu().numpy()
        self.resolution = resolution

        pass

    def add_polygons(self, polygons_3d: np.ndarray):
        polygons_3d -= self.center_3d
        polygons_2d = polygons_3d @ self.proj_mat.T

        polygons = np.round(polygons_2d / self.resolution).astype(np.int16) - self.min_coord

        mask = polygon_to_mask([polygons], self.height, self.width)
        self.grid = self.grid | mask
        pass

    def poly_triangulate(self) -> trimesh.Trimesh:
        # post-processing on the mask
        mask = self.post_process()

        polygons = mask2polygon(mask)

        if len(polygons) == 0:
            return trimesh.Trimesh()
        # assert len(polygons) != 0, "No polygons detected. Check the mask post-processing."

        faces = []
        count = 0
        verts_2d = []
        for poly in polygons:
            poly = poly.reshape(-1, 2)
            if len(poly) < 3:
                continue
            triangles = earcut.triangulate_int32(poly, [len(poly)])
            triangles += count
            triangles = triangles.reshape(-1, 3)
            faces.append(triangles)
            verts_2d.append(poly)

            count += len(poly)
        verts_2d = np.concatenate(verts_2d, axis=0)
        faces = np.concatenate(faces, axis=0)
        verts_2d_coord = (verts_2d + self.min_coord) * self.resolution
        verts_3d = verts_2d_coord @ self.proj_mat + self.center_3d

        return trimesh.Trimesh(vertices=verts_3d, faces=faces)

    def post_process(self):
        ori_mask = (self.grid * 255).copy()

        # you may want to close small holes inside.
        # kernel = np.ones((9, 9), np.uint8)  # tweak the kernel size
        # mask = cv2.morphologyEx(ori_mask, cv2.MORPH_CLOSE, kernel)
        ## and you may also want to smooth the edge
        # mask = cv2.blur(mask, (3, 3), 0)  # tweak the kernel size
        # _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        # ori_mask = mask

        # or PlanarRecon style
        # kernel = np.ones((17, 17), np.uint8)  # tweak the kernel size
        # mask = cv2.morphologyEx(ori_mask, cv2.MORPH_OPEN, kernel)
        # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # mask = np.zeros_like(ori_mask)
        # for c in contours:
        #     hull = cv2.convexHull(c)
        #     cv2.fillConvexPoly(mask, hull, 255)
        # if (mask==255).sum() < 5000:  # if the mask too small, drop it
        #     mask = np.zeros_like(ori_mask)
        # ori_mask = mask

        return ori_mask


@dataclass
class SurfaceReconstruction:
    output_dir: Path = Path("export")
    """Path to the output directory."""

    def run(self, pipeline: NeuralPlanePipeline, plane_instances: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], mode: Literal['point', 'mask']='point') -> List[trimesh.Trimesh]:
        """
        Args:
            pipeline: NeuralPlanePipeline
            plane_instances: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
                List of plane instances. Each tuple contains:
                    - plane params (torch.Tensor): (1, 4)
                    - seg_ids (torch.Tensor): (N1, 3)
                    - verts (torch.Tensor): (N2, 3)
            mode: Literal['point', 'mask']
                If 'point', reconstructs the surface using the point cloud.
                If 'mask', reconstructs the surface using the 2D mask.
        """
        if mode == 'point':
            mesh_instances = self.reconstruct_surface_from_point_cloud(plane_instances)
        elif mode == 'mask':
            mesh_instances = self.reconstruct_surface_from_masks(pipeline.datamanager, plane_instances)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        return mesh_instances

    def reconstruct_surface_from_point_cloud(self, plane_instances: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> List[trimesh.Trimesh]:

        # Note that the surface reconstruction implementation described in the appendix is not gonna be provided in the current version. It is heuristic and lack of memory efficiency. We recommand to use the `reconstruct_surface_from_masks` method instead.
        raise NotImplementedError("Surface reconstruction from point cloud is not implemented yet.")

    def reconstruct_surface_from_masks(self, datamanager: NeuralPlaneDataManager, plane_instances: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> List[trimesh.Trimesh]:
        CONSOLE.log("[Surface Reconstruction] :ten_o’clock: Reconstructing the surface using 2D plane segments...")
        self.device = plane_instances[0][0].device

        self.plane_seg_uv_list = datamanager.train_pixel_sampler.plane_seg_uv_list
        self.instance_weight_list = datamanager.train_pixel_sampler.instance_weight_list
        self.transform_poses_to_original_space = datamanager.train_dataparser_outputs.transform_poses_to_original_space
        self.camera_to_worlds = self.transform_poses_to_original_space(
            datamanager.train_dataparser_outputs.cameras.camera_to_worlds, camera_convention="opengl"
        )
        self.K_inv_dot_xyz = datamanager._ray_bundle.directions.to(self.device)

        # get K_inv_dot_xy1
        _camera = datamanager.train_dataparser_outputs.cameras[0]
        fx, fy, cx, cy = _camera.fx.item(), _camera.fy.item(), _camera.cx.item(), _camera.cy.item()
        self.im_height, self.im_width = _camera.height.item(), _camera.width.item()

        def process_single_instance(plane_param, seg_ids, verts):
            plane_param = plane_param.float()
            verts = verts.float()
            weights = [self.instance_weight_list[i][j] for i, j in seg_ids]
            uv_masks = [self.plane_seg_uv_list[i][j] for i, j in seg_ids]
            c2ws = [self.camera_to_worlds[i].to(self.device) for i, j in seg_ids]
            max_to_min_indices = np.argsort(weights)[::-1]

            # Fuse masks: acquire the polygon mask and project the mask onto the canonical plane.
            canonical_plane = CanonicalPlane(verts, plane_param, resolution=0.005)

            for i in max_to_min_indices:
                mask = np.zeros((self.im_height, self.im_width), dtype=np.uint8)
                uv_mask = uv_masks[i]
                mask[uv_mask[:, 0], uv_mask[:, 1]] = 1
                polygons = mask2polygon(mask)
                c2w = c2ws[i]

                #debug
                # mask_debug = polygon_to_mask(polygons, im_height, im_width)

                for poly in polygons:
                    xy_n2 = torch.from_numpy(poly).reshape(-1, 2).to(self.device)
                    _K_inv_dot_xyz = self.K_inv_dot_xyz[xy_n2[:, 1], xy_n2[:, 0]]  # n 3

                    plane_param_l = transformPlanes(pose_inverse(c2w.unsqueeze(0)), plane_param.unsqueeze(0)).squeeze(0)
                    offset = plane_param_l[3]
                    normal = plane_param_l[:3]
                    depth = -offset / (normal @ _K_inv_dot_xyz.T)

                    poly_3d = (c2w[:3, :3] @ (_K_inv_dot_xyz * depth.view(-1, 1)).T).T + c2w[:3, 3].view(-1, 3)

                    # debug
                    # trimesh.PointCloud(poly_3d.cpu().numpy()).export(self.output_dir / 'test.ply')

                    canonical_plane.add_polygons(poly_3d.cpu().numpy())
                    continue

            # triangulate on the canoncial plane.
            mesh = canonical_plane.poly_triangulate()
            return mesh

        mesh_list = Parallel(n_jobs=1, verbose=0)(
            delayed(process_single_instance)(plane_params, seg_ids, verts) for plane_params, seg_ids, verts in plane_instances
        )

        mesh_list = [mesh for mesh in mesh_list if len(mesh.vertices) > 0]
        # mesh = trimesh.util.concatenate(mesh_list)
        # mesh.export(self.output_dir / "planes.ply")

        return mesh_list