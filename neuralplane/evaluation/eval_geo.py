# This file is derived from [Atlas](https://github.com/magicleap/Atlas).
# Originating Author: Zak Murez (zak.murez.com)
# and modified by [PlanarRecon](https://github.com/neu-vi/PlanarRecon).

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from nerfstudio.utils.rich_utils import CONSOLE

def eval_mesh(file_pred, file_trgt, threshold=.05, down_sample=.02, error_map=True):
    """ Compute Mesh metrics between prediction and target.
    Opens the Meshs and runs the metrics
    Args:
        file_pred: file path of prediction
        file_trgt: file path of target
        threshold: distance threshold used to compute precision/recal
        down_sample: use voxel_downsample to uniformly sample mesh points
    Returns:
        Dict of mesh metrics
    """

    pcd_pred = o3d.io.read_point_cloud(file_pred)
    pcd_trgt = o3d.io.read_point_cloud(file_trgt)
    if down_sample:
        pcd_pred = pcd_pred.voxel_down_sample(down_sample)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)
    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)

    _, dist1 = nn_correspondance(verts_pred, verts_trgt)
    _, dist2 = nn_correspondance(verts_trgt, verts_pred)
    dist1 = np.array(dist1)
    dist2 = np.array(dist2)

    precision = np.mean((dist2 < threshold).astype('float')) * 100
    recal = np.mean((dist1 < threshold).astype('float')) * 100
    fscore = 2 * precision * recal / (precision + recal)

    acc = np.mean(dist2) * 100
    comp = np.mean(dist1) * 100
    chamfer = 0.5 * (acc + comp)
    metrics = {'Accu.': acc,
               'Comp.': comp,
               'Chamfer': chamfer,
               'Prec.': precision,
               'Recall': recal,
               'F-score': fscore,
               }
    if error_map:
        # repeat but without downsampling
        mesh_pred = o3d.io.read_point_cloud(file_pred)
        mesh_trgt = o3d.io.read_point_cloud(file_trgt)
        verts_pred = np.asarray(mesh_pred.points)
        verts_trgt = np.asarray(mesh_trgt.points)
        _, dist1 = nn_correspondance(verts_pred, verts_trgt)
        _, dist2 = nn_correspondance(verts_trgt, verts_pred)
        dist1 = np.array(dist1)
        dist2 = np.array(dist2)

        # recall_err_viz
        from matplotlib import cm
        cmap = cm.get_cmap('jet')
        # cmap = cm.get_cmap('brg')
        dist1_n = dist1 / 0.2
        color = cmap(dist1_n)
        # recal_mask = (dist1 < threshold)
        # color = np.array([[1., 0., 0]]).repeat(verts_trgt.shape[0], axis=0)
        # color[recal_mask] = np.array([[0, 1., 0]]).repeat((recal_mask.sum()).astype(np.int), axis=0)
        mesh_trgt.colors = o3d.utility.Vector3dVector(color[:, :3])

        # precision_err_viz
        dist2_n = dist2 / 0.3
        color = cmap(dist2_n)
        # prec_mask = dist2 < threshold
        # color = np.array([[1., 0., 0]]).repeat(verts_pred.shape[0], axis=0)
        # color[prec_mask] = np.array([[0, 1., 0]]).repeat((prec_mask.sum()).astype(np.int), axis=0)
        mesh_pred.colors = o3d.utility.Vector3dVector(color[:, :3])
    else:
        mesh_pred = mesh_trgt = None
    return metrics, mesh_pred, mesh_trgt


def nn_correspondance(verts1, verts2):
    """ for each vertex in verts2 find the nearest vertex in verts1
    Args:
        nx3 np.array's
    Returns:
        ([indices], [distances])
    """

    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts1)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    for vert in verts2:
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        indices.append(inds[0])
        distances.append(np.sqrt(dist[0]))

    return indices, distances

@dataclass
class GeoEval:

    save_error_map: bool = False
    """Save error map visualization"""

    def run(self, gt_anno_path: Path, pred_path: Path):
        CONSOLE.log(f"[EVAL] Geometry: Accu.↓, Comp.↓, Chamfer↓, Prec.↑, Recall↑, F-score↑.")
        verts_gt_path = gt_anno_path / "eval" /"gt_verts.ply"

        eval_results, precision_pcd, recall_pcd = eval_mesh(pred_path, verts_gt_path, down_sample=None, error_map=self.save_error_map)

        if self.save_error_map and precision_pcd is not None:
            error_map_path = pred_path.parent / "error_map"
            error_map_path.mkdir(exist_ok=True, parents=True)
            CONSOLE.log(f"Saving error map to {error_map_path}")
            o3d.io.write_point_cloud(error_map_path / f'precErr.ply', precision_pcd)
            o3d.io.write_point_cloud(error_map_path / f'recErr.ply', recall_pcd)

        return {"geo": eval_results}

    def run_stats(self, metrics_dicts):
        """Compute mean of metrics"""

        metrics = {}
        for key in metrics_dicts[0].keys():
            metrics[key] = np.mean([metrics_dict[key] for metrics_dict in metrics_dicts])
        return metrics