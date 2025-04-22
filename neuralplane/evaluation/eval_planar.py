# Planar metrics according to Airplanes: https://github.com/nianticlabs/airplanes

from dataclasses import dataclass
from typing import Dict
from numba import cuda
from pathlib import Path
from rich.progress import track

import math
import trimesh
import numpy as np

from neuralplane.evaluation.eval_geo import nn_correspondance
from nerfstudio.utils.rich_utils import CONSOLE

@cuda.jit
def find_nearest_neighbors_gpu(coords_A, coords_B, indices, distances, max_dist):
    tid = cuda.grid(1)

    if tid < coords_A.shape[0]:
        min_dist = max_dist
        min_idx = -1

        for i in range(coords_B.shape[0]):
            dist = 0.0
            for j in range(3):
                dist += (coords_A[tid, j] - coords_B[i, j]) ** 2
            dist = math.sqrt(dist)

            if dist < min_dist:
                min_dist = dist
                min_idx = i

        indices[tid] = min_idx
        distances[tid] = min_dist

def find_nearest_neighbors(points_from: np.ndarray, points_to: np.ndarray, max_dist: float = 1.0):
    """Brute force nearest neighbour search between two point clouds on the GPU."""
    # allocate memory on the device
    indices = cuda.device_array((points_from.shape[0],), dtype="int32")
    distances = cuda.device_array((points_from.shape[0],), dtype="float32")

    # move data to the device
    points_from = cuda.to_device(points_from)
    points_to = cuda.to_device(points_to)

    # compute nearest neighbors
    block_size = 1024
    grid_size = (points_from.shape[0] + block_size - 1) // block_size
    find_nearest_neighbors_gpu[grid_size, block_size](points_from, points_to, indices, distances, max_dist)  # type: ignore

    # move back to host and return distances and indices
    return distances.copy_to_host(), indices.copy_to_host()

def my_find_nearest_neighbors(points_from: np.ndarray, points_to: np.ndarray, max_dist: float = 1.0):
    indices, distances = nn_correspondance(points_to, points_from)
    distances = np.clip(distances, 0, max_dist)
    return distances, indices

def compute_planar_metrics(
    gt_pcd: trimesh.PointCloud,
    gt_labels: np.ndarray,
    pred_pcd: trimesh.PointCloud,
    pred_labels: np.ndarray,
    max_dist: float = 1.0, k: int = 20,
) -> Dict[str, float]:
    """Compute metrics for a predicted and gt point cloud.

    Returns:
        dict[str, float]: Metrics for this point cloud comparison.
    """

    metrics: Dict[str, float] = {}

    if len(pred_pcd.vertices) == 0:
        metrics["top_planes_compl"] = max_dist
        metrics["top_planes_acc"] = max_dist
        return metrics

    pred_points = np.array(pred_pcd.vertices)
    pred_n_ins = np.max(pred_labels) + 1

    pred_planes_pts = []
    for i in range(pred_n_ins):
        mask = pred_labels == i
        if mask.sum() > 20:
            pred_planes_pts.append(pred_points[mask])

    gt_points = np.array(gt_pcd.vertices)
    gt_unique_labels, gt_counts = np.unique(gt_labels, axis=0, return_counts=True)

    # sort gt planes by number of sampled points
    sorted_gt_planes = sorted(
        zip(gt_counts, gt_unique_labels), reverse=True, key=lambda tup: tup[0]
    )
    sorted_gt_planes = sorted_gt_planes[: k]

    final_accuracy = 0
    final_completion = 0
    final_count = 0

    iter_sorted_gt_planes = track(
        sorted_gt_planes, total=len(sorted_gt_planes), description=f":ten_o’clock: Computing..."
    )

    for count, label in iter_sorted_gt_planes:
        gt_plane_pts = gt_points[gt_labels == label]

        best_completion = 1000 * max_dist
        best_matched_plane = -1

        # find closest predicted plane
        for idx, pred_plane_pts in enumerate(pred_planes_pts):
            distances_gt2pred, _ = my_find_nearest_neighbors(
                gt_plane_pts, pred_plane_pts, max_dist
            )

            completion = float(np.mean(distances_gt2pred))
            if completion < best_completion:
                best_completion = completion
                best_matched_plane = idx

        distances_pred2gt, _ = my_find_nearest_neighbors(
            pred_planes_pts[best_matched_plane], gt_plane_pts, max_dist
        )
        accuracy = float(np.mean(distances_pred2gt))
        if accuracy > max_dist or best_completion > max_dist:
            raise ValueError("Accuracy or completion are larger than max_dist")
        final_accuracy += accuracy * count
        final_completion += best_completion * count
        final_count += count

    metrics["top_planes_compl(fidelity)"] = final_completion / final_count * 100  # in cm
    metrics["top_planes_acc"] = final_accuracy / final_count * 100  # in cm
    metrics["top_planes_chamfer"] = 0.5 * (metrics["top_planes_acc"] + metrics["top_planes_compl(fidelity)"])

    return metrics

@dataclass
class PlanarEval:
    k: int = 20
    """number of ground truth planes to use for the top planes metrics"""
    max_dist: float = 1.0
    """max_dist: maximum distance to clip distances to in meter"""

    def run(self, gt_anno_path: Path, pred_path: Path):
        CONSOLE.log(f"[EVAL] for {self.k} largest gt planes, find its closest matched pred plane, and compute planar metrics: top_planes_compl (fidelity)↓, top_planes_acc↓, top_planes_chamfer↓.")


        gt_pcd = trimesh.load(gt_anno_path / "eval" / "gt_verts.ply", process=False)
        gt_labels = np.array([em[-1] for em in gt_pcd.metadata['_ply_raw']['vertex']['data']])

        pred_pcd = trimesh.load(pred_path, process=False)
        pred_labels = np.array([em[-1] for em in pred_pcd.metadata['_ply_raw']['vertex']['data']])

        metrics = compute_planar_metrics(
            gt_pcd=gt_pcd, gt_labels=gt_labels, pred_pcd=pred_pcd, pred_labels=pred_labels,
            max_dist=self.max_dist, k=self.k
        )
        return {'planar': metrics}