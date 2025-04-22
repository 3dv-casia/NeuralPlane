# code developed based on
# https://github.com/fuy34/planerecover/blob/master/eval/compute_sc.m
# https://github.com/fuy34/planerecover/blob/master/eval/compare_segmentations.m
# https://github.com/neu-vi/PlanarRecon
# https://github.com/nianticlabs/airplanes

from dataclasses import dataclass

from sklearn.metrics import rand_score
from skimage.metrics import variation_of_information

import numpy as np
import open3d as o3d

from nerfstudio.utils.rich_utils import CONSOLE

def compute_sc(gt_in, pred_in):
    # to be consistent with skimage sklearn input arrangment

    assert len(pred_in.shape) == 1 and len(gt_in.shape) == 1

    acc, pred, gt  = match_seg(pred_in, gt_in) # n_gt * n_pred

    bestmatch_gt2pred = acc.max(axis=1)
    bestmatch_pred2gt = acc.max(axis=0)

    pred_id, pred_cnt = np.unique(pred, return_counts=True)
    gt_id, gt_cnt = np.unique(gt, return_counts=True)

    cnt_pred, sum_pred = 0, 0
    for i, p_id in enumerate(pred_id):
        cnt_pred += bestmatch_pred2gt[i] * pred_cnt[i]
        sum_pred += pred_cnt[i]

    cnt_gt, sum_gt = 0, 0
    for i, g_id in enumerate(gt_id):
        cnt_gt += bestmatch_gt2pred[i] * gt_cnt[i]
        sum_gt += gt_cnt[i]

    sc = (cnt_pred / sum_pred + cnt_gt / sum_gt) / 2

    return sc

def compute_airplane_sc(gt_in, pred_in):
    gt_ids = np.unique(gt_in)
    pred_ids = np.unique(pred_in)

    def airplane_sc(s, s_prime, s_ids, s_prime_ids):
        scores = np.zeros(s_ids.shape[0])
        for i in range(s_ids.shape[0]):
            s_idx = s_ids[i]
            best = 0.0
            for j in range(s_prime_ids.shape[0]):
                s_prime_idx = s_prime_ids[j]
                size = (s == s_idx).sum()
                intersection = np.logical_and(s == s_idx, s_prime == s_prime_idx).sum()
                union = np.logical_or(s == s_idx, s_prime == s_prime_idx).sum()
                overlap = size * intersection / union if union > 0 else 0
                best = max(best, overlap)
            scores[i] = best
        return scores.sum() / s.shape[0]

    return (airplane_sc(gt_in, pred_in, gt_ids, pred_ids) + airplane_sc(pred_in, gt_in, pred_ids, gt_ids)) / 2

def match_seg(pred_in, gt_in):
    assert len(pred_in.shape) == 1 and len(gt_in.shape) == 1

    pred, gt = compact_segm(pred_in), compact_segm(gt_in)
    n_gt = gt.max() + 1
    n_pred = pred.max() + 1


    # this will offer the overlap between gt and pred
    # if gt == 1, we will later have conf[1, j] = gt(1) + pred(j) * n_gt
    # essential, we encode conf_mat[i, j] to overlap, and when we decode it we let row as gt, and col for pred
    # then assume we have 13 gt label, 6 pred label --> gt 1 will correspond to 14, 1+2*13 ... 1 + 6*13
    overlap =  gt + n_gt * pred
    freq, bin_val = np.histogram(overlap, np.arange(0, n_gt * n_pred+1)) # hist given bins [1, 2, 3] --> return [1, 2), [2, 3)
    conf_mat = freq.reshape([ n_gt, n_pred], order='F') # column first reshape, like matlab

    acc = np.zeros([n_gt, n_pred])
    for i in range(n_gt):
        for j in range(n_pred):
            gt_i = conf_mat[i].sum()
            pred_j = conf_mat[:, j].sum()
            gt_pred = conf_mat[i, j]
            acc[i,j] = gt_pred / (gt_i + pred_j - gt_pred) if (gt_i + pred_j - gt_pred) != 0 else 0
    return acc[1:, 1:], pred, gt

def compact_segm(seg_in):
    seg = seg_in.copy()
    uniq_id = np.unique(seg)
    cnt = 1
    for id in sorted(uniq_id):
        if id == 0:
            continue
        seg[seg==id] = cnt
        cnt += 1

    # every id (include non-plane should not be 0 for the later process in match_seg
    seg = seg + 1
    return seg

def project_to_mesh(from_mesh, to_mesh, attribute, attr_name, color_mesh=None, dist_thresh=None):
    """ Transfers attributes from from_mesh to to_mesh using nearest neighbors

    Each vertex in to_mesh gets assigned the attribute of the nearest
    vertex in from mesh. Used for semantic evaluation.

    Args:
        from_mesh: Trimesh with known attributes
        to_mesh: Trimesh to be labeled
        attribute: Which attribute to transfer
        dist_thresh: Do not transfer attributes beyond this distance
            (None transfers regardless of distacne between from and to vertices)

    Returns:
        Trimesh containing transfered attribute
    """

    if len(from_mesh.vertices) == 0:
        to_mesh.vertex_attributes[attr_name] = np.zeros((0), dtype=np.uint8)
        to_mesh.visual.vertex_colors = np.zeros((0), dtype=np.uint8)
        return to_mesh

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(from_mesh.vertices)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    pred_ids = attribute.copy()
    pred_colors = from_mesh.visual.vertex_colors  if color_mesh is None else color_mesh.visual.vertex_colors

    matched_ids = np.zeros((to_mesh.vertices.shape[0]), dtype=np.uint8)
    matched_colors = np.zeros((to_mesh.vertices.shape[0], 4), dtype=np.uint8)

    for i, vert in enumerate(to_mesh.vertices):
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        if dist_thresh is None or dist[0]<dist_thresh:
            matched_ids[i] = pred_ids[inds[0]]
            matched_colors[i] = pred_colors[inds[0]]

    mesh = to_mesh.copy()
    # mesh.vertex_attributes[attr_name] = matched_ids
    mesh.visual.vertex_colors = matched_colors
    return mesh, matched_ids

@dataclass
class SegmEval:
    def run(self, gt_anno_path, pred_path):
        CONSOLE.log(f"[EVAL] Segmentation: RI↑, VOI↓, SC↑, SC (AirPlane)↑.")

        gt_label_file = gt_anno_path / "eval" / "gt_labels.txt"
        pred_labels = np.loadtxt(pred_path).astype(np.int16)
        gt_labels = np.loadtxt(gt_label_file).astype(np.int16)

        ri = rand_score(gt_labels, pred_labels)
        h1, h2 = variation_of_information(gt_labels, pred_labels)
        voi = h1 + h2

        sc = compute_sc(gt_labels, pred_labels)

        eval_results = {
            'RI': ri,
            'VOI': voi,
            'SC': sc,
        }

        return {'segm': eval_results}
