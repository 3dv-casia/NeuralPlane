# Modified based on PlanarRecon: https://github.com/neu-vi/PlanarRecon/tree/e8e446554e20aafd028190a4130a8d27733c0194/tools/seq_ransac

from dataclasses import dataclass
from typing import List, Tuple
from joblib import Parallel, delayed
from datetime import datetime
from pathlib import Path

import trimesh
import torch
import torch.nn.functional as F
import functools

import numpy as np
import h5py
import open3d as o3d

from nerfstudio.scripts.train import _set_random_seed

from nerfstudio.utils.rich_utils import CONSOLE

def seq_ransac_mesh_approx(verts, normals, occup, planeIns, voxel_sz, origin, connect_kernel, norm_thres, dist_thres, init_verts_thres, connect_verts_thres, cur_planeID, n_iter=100):
    # sequential one-point plane ransac using mesh as input
    # the difference with seq_ransac_mesh() is here, we do the connection check as a post-process
    # instead of one step of fitting, as connection_check take lots of times

    # every verts can at most be assigned once
    valid_mask = planeIns == 0

    voxel_verts = (verts - origin.view(-1, 1)) / voxel_sz.view(-1, 1)
    voxel_verts_ind = torch.round(voxel_verts).long()

    if (valid_mask).sum() == 0:
        return planeIns, False, None

    # sample the seeds in valid_mask, we must ensure the sampled number is <= valid_mask points
    generator = torch.Generator(device=valid_mask.device)
    generator.manual_seed(2025)
    idxs = torch.multinomial(valid_mask.float(), min(n_iter, (valid_mask).sum()), replacement=False, generator=generator)

    resume = False
    best_inlier_vol = 0
    plane_param = None
    best_mask = torch.zeros_like(valid_mask).type(torch.bool)
    for i in idxs:
        sample_pnt = verts[:, i].unsqueeze(1)
        sample_norm = normals[:, i].unsqueeze(1)

        # normal should be similiar,
        norm_mask = (torch.sum((normals * sample_norm), dim=0).abs() > norm_thres)

        # distance to the plane should under threshold
        planeD = (sample_norm * sample_pnt).sum()
        cluster_plane_dist = ((sample_norm * verts).sum(dim=0) - planeD).abs()
        spatial_mask = cluster_plane_dist <= dist_thres

        cluster_mask = valid_mask & norm_mask & spatial_mask
        # occup_area = cluster_mask.clone()

        proposal_vol =  cluster_mask.sum()
        if proposal_vol > best_inlier_vol:
            best_mask = cluster_mask.clone()
            best_inlier_vol = proposal_vol

    # ransac will stop if the best plane_area < area_thres
    if best_inlier_vol >= init_verts_thres:

        # check occupied volume
        fill_volum = torch.zeros_like(occup)

        # convert verts into volume space, and select all voxels which are current inliners and be occupied
        inlier_verts = verts[:, best_mask]
        inlier_voxel_verts = (inlier_verts - origin.view(-1, 1)) / voxel_sz.view(-1, 1)
        inlier_verts_ind = torch.round(inlier_voxel_verts).long()
        fill_volum[inlier_verts_ind[0], inlier_verts_ind[1], inlier_verts_ind[2]] = True

        _occup_area = torch.logical_and(fill_volum, occup)
        occup_area = find_largest_connected_componenet(_occup_area, connect_kernel)

        # pick the verts within the largest connected componenet
        final_mask = torch.where(occup_area[voxel_verts_ind[0], voxel_verts_ind[1], voxel_verts_ind[2]],
                                 torch.ones_like(voxel_verts_ind[0]), torch.zeros_like(voxel_verts_ind[0])).bool()

        # print('instance verts number ', final_mask.sum().item())

        # final lstsq to get the result
        if final_mask.sum() >= connect_verts_thres: # we ask at least 4 verts for plane fittng
            planeIns[final_mask] = cur_planeID
            resume = True
            plane_param = compute_plane_params(verts, normals, final_mask)
            # plane_param = plane_param[:3].reshape(3, 1)  # first 3 is the solution

    return planeIns, resume, plane_param # plane_param == None and will not be used if no plane is found

def find_largest_connected_componenet(occup, connect_kernel_sz = 3):
    h, w, z = occup.shape
    seeds = torch.arange(0, occup.view(-1).shape[0]).reshape([h,w,z]).float().view(1,1,h,w,z).to(occup.device)
    _occup = occup.view(1,1,h,w,z)
    seeds[~_occup] = 0

    pre_mask = seeds.clone()
    candidate_mask = seeds.clone()

    # use 3D max pooling to propgate seed
    for cnt in range(max([h, w, z])):  # longest dist to flood fill equals to the largest dim
        candidate_mask = F.max_pool3d(candidate_mask, kernel_size=connect_kernel_sz, stride=1, padding=connect_kernel_sz//2)
        candidate_mask[~_occup] = 0

        if  (pre_mask == candidate_mask).all():
            break
        pre_mask = candidate_mask.clone()

    # print('\n find largest connection in ', cnt, 'steps')
    # take the most freq value
    freq_val , _ = torch.mode(candidate_mask[candidate_mask > 0].view(-1))

    return (candidate_mask == freq_val).squeeze()

def compute_plane_params(verts, normals, mask):

    points = verts[:, mask].T

    # Method 1
    # normals = normals[:, mask].T

    # normal = torch.median(normals, 0).values
    # normal = normal / torch.norm(normal)

    # offset = -torch.median((points * normal).sum(-1))
    # param = torch.cat([normal, torch.tensor([offset]).to(normal.device)])

    # Method 2
    # https://www.ilikebigbits.com/2015_03_04_plane_from_points.html
    centroid = points.mean(dim=0)
    r = points - centroid
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    xx = (x * x).sum()
    xy = (x * y).sum()
    xz = (x * z).sum()
    yy = (y * y).sum()
    yz = (y * z).sum()
    zz = (z * z).sum()

    det_x = yy * zz - yz * yz
    det_y = xx * zz - xz * xz
    det_z = xx * yy - xy * xy

    if det_x > det_y and det_x > det_z:
        abc = torch.tensor([det_x, xz * yz - xy * zz, xy * yz - xz * yy])
    elif det_y > det_z:
        abc = torch.tensor([xz * yz - xy * zz, det_y, xy * xz - yz * xx])
    else:
        abc = torch.tensor([xy * yz - xz * yy, xy * xz - yz * xx, det_z])

    norm = (abc / abc.norm()).to(points.device)
    d = -(norm * centroid).sum()
    param = torch.cat([norm, torch.tensor([d]).to(norm.device)])

    return param


def get_plane_instance_RANSAC_tsdf(verts, norms, occupancy, voxel_sz, origin, connect_kernel, angle_thres=30, dist_thres = 0.05, init_verts_thres= 200,  connect_verts_thres=4, n_iter= 100, device = torch.device('cpu')):
    verts = verts.to(device).T
    norms = norms.to(device).T
    # occupancy = tsdf.to(device).abs() < 1
    occupancy = occupancy.to(device)
    origin = origin.to(device)
    voxel_sz = voxel_sz.to(device)

    # init necessary variables
    norm_thres =  np.cos(np.deg2rad(angle_thres))
    planeIns = torch.zeros_like(verts[0])

    # start sequential RANSAC for each pred_semantic label
    resume_ransac = True
    cur_planeId = 1
    plane_params  = []
    while resume_ransac:

        planeIns, resume_ransac, plane_param = \
            seq_ransac_mesh_approx(verts, norms, occupancy, planeIns, voxel_sz, origin, connect_kernel, norm_thres, dist_thres, init_verts_thres, connect_verts_thres, cur_planeId, n_iter=n_iter)

        if resume_ransac:
            cur_planeId += 1
            plane_params.append(plane_param.cpu())

    return planeIns.cpu(), plane_params

# ========= planarize ==========
def planarize(verts, plane_ins, plane_params, n_plane):
    new_verts = verts.clone()
    for i in range(1, n_plane): # skip the non_plane
        plane_mask = plane_ins==i
        param = plane_params[i-1] # convert to 0-idx
        _planeD = param.norm()
        planeN, planeD = param/_planeD, 1/_planeD

        plane_verts = new_verts[:, plane_mask]

        # proj
        dist = (planeN.T @ plane_verts - planeD) * planeN
        on_plane_verts = plane_verts - dist
        new_verts[:, plane_mask] = on_plane_verts

    return new_verts


@dataclass
class PlaneRANSAC:
    output_dir: Path

    iter: int = 1500
    offset_threshold: float = 0.08
    normal_threshold: float = 20

    init_verts_threshold: int = 20
    """min verts number for a proposal"""
    connect_verts_threshold: int = 25
    """min_verts number for a final plane instance, once fail, the RANSAC will stop"""
    connect_kernel: int = 7
    seed: int = 0

    def run(self, pcd: trimesh.Trimesh):
        assert "proto_labels" in pcd.vertex_attributes.keys()
        assert "seg_ids" in pcd.vertex_attributes.keys()

        verts = torch.from_numpy(np.array(pcd.vertices))  # n 3
        norms = torch.from_numpy(np.array(pcd.vertex_normals))  # n 3
        proto_labels = torch.from_numpy(np.array(pcd.vertex_attributes["proto_labels"])).view(-1)
        seg_ids = torch.from_numpy(np.array(pcd.vertex_attributes["seg_ids"]))  # n 2

        # get the unique proto_labels
        unique_proto_labels = torch.unique(proto_labels)

        verts_list = []
        norms_list = []
        seg_ids_list = []

        for proto_label in unique_proto_labels:
            mask = (proto_labels == proto_label)
            verts_list.append(verts[mask])
            norms_list.append(norms[mask])
            seg_ids_list.append(seg_ids[mask])

        CONSOLE.log(f"[PlaneRANSAC] :ten_o’clock: Processing {len(verts_list)} instances in parallel...")
        time_start = datetime.now()
        results = Parallel(n_jobs=1, verbose=0)(
            delayed(self.process_single_instance)(_verts, _norms, _seg_ids) for _verts, _norms, _seg_ids in zip(verts_list, norms_list, seg_ids_list)
        )
        merged = functools.reduce(lambda x, y: x + y, results)
        merged = sorted(merged, key=lambda x: x[2].shape[0], reverse=True)

        time_end = datetime.now()
        CONSOLE.log(f"[PlaneRANSAC] Time Duration {time_end - time_start}.")
        CONSOLE.log(f"[PlaneRANSAC] Total {len(merged)} plane instances exported.")

        # save to file
        with h5py.File(self.output_dir / "plane_instances.h5", "w") as f:
            for i, (plane_params, seg_ids, verts) in enumerate(merged):
                name = f'{i:03d}'
                f.create_group(name)
                f[name].create_dataset("plane_params", data=plane_params.cpu().numpy())
                f[name].create_dataset("seg_ids", data=seg_ids.cpu().numpy())
                f[name].create_dataset("verts", data=verts.cpu().numpy())

        merged_verts = []
        ins_id = []
        for i, (plane_params, seg_ids, verts) in enumerate(merged):
            merged_verts.append(verts)
            ins_id.append(torch.ones_like(verts[:, 0]) * i)
        merged_verts = torch.cat(merged_verts, dim=0).cpu().numpy()
        ins_id = torch.cat(ins_id, dim=0).cpu().numpy()

        # save the verts for vis
        merged_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(merged_verts))
        min_bound, max_bound = merged_pcd.get_min_bound(), merged_pcd.get_max_bound()
        downsampled_merged_pcd, _, record = merged_pcd.voxel_down_sample_and_trace(voxel_size=0.02, min_bound=min_bound, max_bound=max_bound)
        record = np.asarray([idx[0] for idx in record])

        vis_ply = trimesh.Trimesh(vertices=np.asarray(downsampled_merged_pcd.points), vertex_attributes={"ins_id": ins_id[record].astype(np.int32).reshape(-1)})
        vis_ply.export(self.output_dir / "vis_plane_ransac.ply")

        return merged

    def process_single_instance(self, verts: torch.Tensor, norms: torch.Tensor, seg_ids: torch.Tensor):

        _set_random_seed(self.seed)

        # Compute Occupancy Volume
        origin = verts.min(dim=0).values
        voxel_size = 0.02
        volume_dims = ((verts.max(axis=0).values - origin) // voxel_size).long() + 5
        occupancy = torch.zeros(volume_dims.tolist(), dtype=torch.bool)
        coords = torch.round((verts - origin) / voxel_size).clamp_(min=torch.Tensor([0, 0, 0]), max=volume_dims).long()

        occupancy[coords[:, 0], coords[:, 1], coords[:, 2]] = True

        plane_instances, plane_params = get_plane_instance_RANSAC_tsdf(
            verts=verts, norms=norms, occupancy=occupancy, voxel_sz=torch.Tensor([voxel_size] * 3), origin=origin, connect_kernel=self.connect_kernel, angle_thres=self.normal_threshold, dist_thres=self.offset_threshold, init_verts_thres=self.init_verts_threshold, connect_verts_thres=self.connect_verts_threshold, n_iter=self.iter,
            device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )

        plane_info: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []  # (plane_params, seg_ids, verts)

        if len(plane_params) == 0:
            return plane_info

        n_ins = plane_instances.max().cpu().int().item() + 1
        for i in range(1, n_ins):
            mask = (plane_instances == i)
            instance_verts = verts[mask]  # bs 3
            instance_seg_ids = seg_ids[mask]  # bs 2

            salient_seg_ids, _ = torch.unique(instance_seg_ids, dim=0, return_counts=True)
            plane_info.append(
                (plane_params[i - 1], salient_seg_ids, instance_verts)
            )

        return plane_info

    def _save_plane_instances(self, plane_instances):
        label = []
        points = []
        for i, (param, seg_ids, verts) in enumerate(plane_instances):
            _pts = verts.cpu().numpy()
            _l = np.ones((_pts.shape[0], )) * i
            points.append(_pts)
            label.append(_l)
        points = np.concatenate(points, axis=0).reshape(-1, 3)
        label = np.concatenate(label, axis=0).reshape(-1)
        trimesh.Trimesh(vertices=points, vertex_attributes= {"label": label}).export(self.output_dir / "plane_ransac.ply")
