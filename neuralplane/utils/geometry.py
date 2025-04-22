import numpy as np
import torch
from jaxtyping import Float

## Fit a 3D plane from points
def fitPlane(points):
    """
    plane: p_x * x + p_y * y + p_z * z = 1

    plane normal: (p_x, p_y, p_z) / sqrt(p_x^2 + p_y^2 + p_z^2)
    """
    if points.shape[0] == points.shape[1]:
        planes = np.linalg.solve(points, np.ones(points.shape[0]))
    else:
        planes = np.linalg.lstsq(points, np.ones(points.shape[0]), rcond=-1)[0]

    return planes

def transformPlanes(transformation: Float[torch.Tensor, "bs 3 4"], planes: Float[torch.Tensor, "bs 4"]) -> Float[torch.Tensor, "bs 4"]:
    """
    Transform planes by the given transformation matrix.
    """
    planeNormals = planes[:, :3]
    planeOffsets = planes[:, 3]

    centers_b3 = -planeNormals * planeOffsets.view(-1, 1)
    newCenters = torch.bmm(transformation[:, :3, :3], centers_b3.view(-1, 3, 1)).squeeze(-1) + transformation[:, :3, 3]  # bs 3

    refPoints = centers_b3 - planeNormals
    newRefPoints = torch.bmm(transformation[:, :3, :3], refPoints.view(-1, 3, 1)).squeeze(-1) + transformation[:, :3, 3]

    NewplaneNormals = newCenters - newRefPoints  # bs, 3
    NewplaneOffsets = - (NewplaneNormals * newCenters).sum(dim=1).view(-1, 1) # bs 1
    newPlanes = torch.cat([NewplaneNormals, NewplaneOffsets], dim=1)  # bs 4

    return newPlanes

def get_K_inv_dot_xy1(K, image_height, image_width, out_height, out_width):
    if type(K) != torch.Tensor:
        K = torch.from_numpy(K).cuda().float()

    out_h = int(out_height)
    out_w = int(out_width)
    ori_h = image_height
    ori_w = image_width

    K_inv = torch.linalg.inv(K)
    K_inv_dot_xy1 = torch.zeros((3, out_h, out_w)).cuda()

    for y in range(out_h):
        for x in range(out_w):
            yy = float(y) / out_h * ori_h
            xx = float(x) / out_w * ori_w

            ray = torch.matmul(K_inv, torch.tensor([xx, yy, 1], device=K_inv.device).reshape(3, 1))
            K_inv_dot_xy1[:, y, x] = ray[:, 0]

    # Switch from OpenCV to OpenGL
    K_inv_dot_xy1[1:, ...] *= -1

    return K_inv_dot_xy1


## The function to compute plane depths from plane parameters (Non-eclidean depth)
def calcPlaneDepths(planes, k_inv_dot_xy1, max_depth=10):
    if type(planes) != torch.Tensor:
        planes = torch.from_numpy(planes).cuda().float()
    planes = planes.to(k_inv_dot_xy1.device)
    planeOffsets = planes[:, 3]  # n, 1
    planeNormals = planes[:, :3]  # n, 3
    normalXYZ = torch.matmul(planeNormals, k_inv_dot_xy1.view(3, -1))  # n, hw
    normalXYZ = normalXYZ.permute(1, 0)
    normalXYZ[normalXYZ == 0] = 1e-4
    planeDepths = -planeOffsets.squeeze(-1) / normalXYZ
    if max_depth > 0:
        planeDepths = torch.clamp(planeDepths, min=0, max=max_depth)
        pass
    planeDepths = planeDepths.permute(1, 0).reshape(-1, k_inv_dot_xy1.shape[1], k_inv_dot_xy1.shape[2])

    return planeDepths  # n, h, w

def pose_inverse(pose: Float[torch.Tensor, "*batch 3 4"]) -> Float[torch.Tensor, "*batch 3 4"]:
    """Invert provided pose matrix.

    Args:
        pose: Camera pose without homogenous coordinate.

    Returns:
        Inverse of pose.
    """
    R = pose[..., :3, :3]
    t = pose[..., :3, 3:]
    R_inverse = R.transpose(-2, -1)
    t_inverse = -R_inverse.matmul(t)
    return torch.cat([R_inverse, t_inverse], dim=-1)