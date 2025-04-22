from detectron2.utils.visualizer import GenericMask
from detectron2.utils.visualizer import Visualizer
from typing import Literal, Union
from pathlib import Path

import numpy as np
import colorsys
import cv2
import torch
import trimesh

from PIL import Image, ImageOps

from neuralplane.utils.geometry import get_K_inv_dot_xy1, calcPlaneDepths

# define camera frustum geometry
focal = 1.0
origin_frustum_verts = np.array([
    (0., 0., 0.),
    (0.375, -0.375, -focal),
    (0.375, 0.375, -focal),
    (-0.375, 0.375, -focal),
    (-0.375, -0.375, -focal),
])

frustum_edges = np.array([
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 2),
]) - 1

COLOR_LIST = [
    [172 / 255., 114 / 255., 82 / 255. ],  # chair
    [1., 1., 1.],  # pic white
    [200 / 255., 54 / 255., 131 / 255.],  # table white
    [0 / 255., 192 / 255., 255 / 255.],  # left wall yellow
    [166 / 255., 56 / 255., 124 / 255.],  # right window
    [226/255., 107/255., 10/255.],  # right wall
    [146 / 255., 111 / 255., 194 / 255.],  # floor
    [78 / 255., 71 / 255., 183 / 255.],
    [79 / 255., 129 / 255., 189 / 255.],
    [92 / 255., 193 / 255., 61 / 255.],
    [238 / 255., 236 / 255., 225 / 255.],
    [166/255., 56/255., 124/255.],
    [11/255., 163/255., 51/255.],
    [140 / 255., 57 / 255., 197 / 255.],
    [202 / 255., 185 / 255., 52 / 255.],
    [51 / 255., 176 / 255., 203 / 255.],
    [200 / 255., 54 / 255., 131 / 255.],
    [158 / 255., 218 / 255., 229 / 255.],  # shower curtain
    [100 / 255., 125 / 255., 154 / 255.],
    [178 / 255., 127 / 255., 135 / 255.],
    [120 / 255., 185 / 255., 128 / 255.],
    [192 / 255., 80 / 255., 77 / 255.],
    [230 / 255., 184 / 255., 183 / 255.],
    [247 / 255., 150 / 255., 70 / 255.],
    [176 / 255., 163 / 255., 190 / 255.],
    [64 / 255., 49 / 255., 80 / 255.],
    [253 / 255., 233 / 255., 217 / 255.],
    [31 / 255., 73 / 255., 125 / 255.],
    [255 / 255., 127 / 255., 14 / 255.],  # refrigerator
    [91 / 255., 163 / 255., 138 / 255.],
    [153 / 255., 98 / 255., 156 / 255.],
    [140 / 255., 153 / 255., 101 / 255.],
    [44 / 255., 160 / 255., 44 / 255.],  # toilet
    [112 / 255., 128 / 255., 144 / 255.],  # sink
    [96 / 255., 207 / 255., 209 / 255.],
    [227 / 255., 119 / 255., 194 / 255.],  # bathtub
    [213 / 255., 92 / 255., 176 / 255.],
    [94 / 255., 106 / 255., 211 / 255.],
    [82 / 255., 84 / 255., 163 / 255.],  # otherfurn
    [100 / 255., 85 / 255., 144 / 255.],
]


class ColorPalette(object):
    def __init__(self, num_of_colors=512, mode: Literal['hls'] = 'hls'):
        self.num_of_colors=num_of_colors

        if mode == 'hls':
            hls_colors = [[j,
                            0.4 + np.random.random()* 0.6,
                            0.6 + np.random.random()* 0.4] for j in self.hue_random()]
            self.colors = np.array([colorsys.hls_to_rgb(*color) for color in hls_colors])
        else:
            raise NotImplementedError

    def __call__(self, idx, type: Literal['int', 'float'] = 'int', rgb=True) -> np.ndarray:
        idx = idx % self.num_of_colors
        if type == 'int':
            color = (self.colors[idx]*255).astype(np.uint8)
        else:
            color = self.colors[idx]

        if not rgb:
            color = color[::-1]
        return color

    def hue_random(self):
        count = 0
        while count < self.num_of_colors:
            # if count % 2 != 0:
            #     yield np.random.randint(60, 120)/360.
            # else:
            #     yield np.random.randint(240, 320)/360.
            yield np.random.random()
            count += 1

def overlay_masks(img, masks, alpha=0.4):
    vis = Visualizer(img)
    masks = [GenericMask(_, vis.output.height, vis.output.width) for _ in masks]

    color_palette = ColorPalette(num_of_colors=len(masks))

    vis.overlay_instances(
        masks=masks,
        assigned_colors=[color_palette(i, type='float').tolist() for i in range(len(masks)) ],
        alpha=alpha
    )

    return vis.output.get_image()

def writePlanarPrimitive2File(
    save_path: Path,
    seg_uv_list: np.ndarray,
    parms: np.ndarray,
    image_path: np.ndarray,
    K: Union[np.ndarray, torch.Tensor],
    c2w: Union[np.ndarray, torch.Tensor],
) -> None:
    if type(c2w) != torch.Tensor:
        c2w = torch.tensor(c2w, dtype=torch.float32)
    if not save_path.exists():
        save_path.mkdir(parents=True)

    # out_h, out_w = depth.shape
    out_h = 192
    out_w = 256

    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]

    K_inv_dot_xy_1 = get_K_inv_dot_xy1(K, image_height, image_width, out_h, out_w)

    segmentation = np.zeros_like(image[:, :, 0])
    for i, seg_uv in enumerate(seg_uv_list):
        segmentation[seg_uv[:, 0], seg_uv[:, 1]] = i + 1

    image = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    segmentation = cv2.resize(
        segmentation, (out_w, out_h), interpolation=cv2.INTER_NEAREST
    )
    depth = calcPlaneDepths(parms, K_inv_dot_xy_1, max_depth=10).cpu().numpy()

    # create face from segmentation
    faces = []
    for y in range(out_h - 1):
        for x in range(out_w - 1):
            segmentIndex = segmentation[y, x]
            # ignore non planar region
            if segmentIndex == 0 or parms[segmentIndex-1][3] <= 0:
                continue

            # add face if three pixel has same segmentatioin
            depths = [depth[segmentIndex-1][y][x], depth[segmentIndex-1][y + 1][x], depth[segmentIndex-1][y + 1][x + 1]]
            if (
                segmentation[y + 1, x] == segmentIndex
                and segmentation[y + 1, x + 1] == segmentIndex
                and min(depths) > 0
                and max(depths) < 10
            ):
                faces.append((x, y, x, y + 1, x + 1, y + 1))

            depths = [depth[segmentIndex-1][y][x], depth[segmentIndex-1][y][x + 1], depth[segmentIndex-1][y + 1][x + 1]]
            if (
                segmentation[y][x + 1] == segmentIndex
                and segmentation[y + 1][x + 1] == segmentIndex
                and min(depths) > 0
                and max(depths) < 10
            ):
                faces.append((x, y, x + 1, y + 1, x + 1, y))

    mesh_verts = torch.empty((out_h, out_w, 3))
    mesh_colors = np.zeros_like(image)

    for y in range(out_h):
        for x in range(out_w):
            segmentIndex = segmentation[y][x]
            if segmentIndex == 0 or parms[segmentIndex-1][3] <= 0:
                mesh_verts[y, x] = 0.
                continue
            ray = K_inv_dot_xy_1[:, y, x]
            mesh_verts[y, x] = ray * depth[segmentIndex-1][y, x]
            mesh_colors[y, x] = image[y, x]

    # transform to world coordinate
    mesh_verts = torch.matmul(c2w[:3, :3], mesh_verts.view(-1, 3).T).T + c2w[:3, 3]
    mesh_verts = mesh_verts.view(out_h, out_w, 3)

    mesh_path = save_path/(image_path.stem + ".ply")
    with open(mesh_path, "w") as f:
        header = """ply
format ascii 1.0
comment VCGLIB generated
element vertex """
        header += str(out_h * out_w)
        header += """
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face """
        header += str(len(faces))
        header += """
property list uchar int vertex_indices
property list uchar float texcoord
end_header
"""
        f.write(header)
        for y in range(out_h):
            for x in range(out_w):
                segmentIndex = segmentation[y][x]
                if segmentIndex == 0:
                    f.write("0.0 0.0 0.0 0 0 0\n")
                    continue
                X, Y, Z = mesh_verts[y, x]
                X, Y, Z = X.item(), Y.item(), Z.item()
                blue, green, red = image[y, x, 0], image[y, x, 1], image[y, x, 2]
                f.write(
                    str(X)
                    + " "
                    + str(Y)
                    + " "
                    + str(Z)
                    + " "
                    + str(red)
                    + " "
                    + str(green)
                    + " "
                    + str(blue)
                    + "\n"
                )

        for face in faces:
            f.write("3 ")
            for c in range(3):
                f.write(str(face[c * 2 + 1] * out_w + face[c * 2]) + " ")
            f.write("6 ")
            for c in range(3):
                f.write(
                    str(float(face[c * 2]) / out_w)
                    + " "
                    + str(1 - float(face[c * 2 + 1]) / out_h)
                    + " "
                )
            f.write("\n")
        f.close()

    # mesh = trimesh.load_mesh(mesh_path, process=True)
    # mesh.export(save_path / (image_path.stem + ".ply"))

    c2w_np = np.eye(4)
    c2w_np[:3, :] = c2w.cpu().numpy()
    image_mesh, aspect_ratio_buffer = get_image_box(
        image_path=image_path,
        frustum_pose=c2w_np,
        flip=False,
        cam_marker_size=0.8
    )
    frustum = generate_frustum_at_position(
        rotation=c2w_np[:3, :3],
        translation=c2w_np[:3, 3],
        color=(249, 187, 118),
        size=0.8,
        aspect_ratio=aspect_ratio_buffer
    )

    frustum_path = save_path / (image_path.stem + "_frustum")
    frustum_path.mkdir(exist_ok=True, parents=True)
    trimesh.util.concatenate([image_mesh, frustum]).export(frustum_path / (image_path.stem + ".obj"))

    return


# copy-paste from ACE: https://github.com/nianticlabs/ace
def get_image_box(
        image_path,
        frustum_pose,
        cam_marker_size=1.0,
        flip=False
):
    """ Gets a textured mesh of an image.

    @param image_path: File path of the image to be rendered.
    @param frustum_pose: 4x4 camera pose, OpenGL convention
    @param cam_marker_size: Scaling factor for the image object
    @param flip: flag whether to flip the image left/right
    @return: duple, trimesh mesh of the image and aspect ratio of the image
    """

    pil_image = Image.open(image_path)
    pil_image = ImageOps.flip(pil_image)  # flip top/bottom to align with scene space

    pil_image_w, pil_image_h = pil_image.size
    aspect_ratio = pil_image_w / pil_image_h

    height = 0.75
    width = height * aspect_ratio
    width *= cam_marker_size
    height *= cam_marker_size

    if flip:
        pil_image = ImageOps.mirror(pil_image)  # flips left/right
        width = -width

    vertices = np.zeros((4, 3))
    # vertices[0, :] = [width / 2, height / 2, -cam_marker_size]
    # vertices[1, :] = [width / 2, -height / 2, -cam_marker_size]
    # vertices[2, :] = [-width / 2, -height / 2, -cam_marker_size]
    # vertices[3, :] = [-width / 2, height / 2, -cam_marker_size]
    vertices[0, :] = [width / 2, height / 2, -focal * cam_marker_size]
    vertices[1, :] = [width / 2, -height / 2, -focal * cam_marker_size]
    vertices[2, :] = [-width / 2, -height / 2, -focal * cam_marker_size]
    vertices[3, :] = [-width / 2, height / 2, -focal * cam_marker_size]

    faces = np.zeros((2, 3))
    faces[0, :] = [0, 1, 2]
    faces[1, :] = [2, 3, 0]
    # faces[2,:] = [2,3]
    # faces[3,:] = [3,0]

    uvs = np.zeros((4, 2))

    uvs[0, :] = [1.0, 0]
    uvs[1, :] = [1.0, 1.0]
    uvs[2, :] = [0, 1.0]
    uvs[3, :] = [0, 0]

    face_normals = np.zeros((2, 3))
    face_normals[0, :] = [0.0, 0.0, 1.0]
    face_normals[1, :] = [0.0, 0.0, 1.0]

    material = trimesh.visual.texture.SimpleMaterial(
        image=pil_image,
        ambient=(1.0, 1.0, 1.0, 1.0),
        diffuse=(1.0, 1.0, 1.0, 1.0),
    )
    texture = trimesh.visual.TextureVisuals(
        uv=uvs,
        image=pil_image,
        material=material,
    )

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        face_normals=face_normals,
        visual=texture,
        validate=True,
        process=False
    )

    # from simple recon code
    def transform_trimesh(mesh, transform):
        """ Applies a transform to a trimesh. """
        np_vertices = np.array(mesh.vertices)
        np_vertices = (transform @ np.concatenate([np_vertices, np.ones((np_vertices.shape[0], 1))], 1).T).T
        np_vertices = np_vertices / np_vertices[:, 3][:, None]
        mesh.vertices[:, 0] = np_vertices[:, 0]
        mesh.vertices[:, 1] = np_vertices[:, 1]
        mesh.vertices[:, 2] = np_vertices[:, 2]

        return mesh

    return transform_trimesh(mesh, frustum_pose), aspect_ratio

# copy-paste from ACE: https://github.com/nianticlabs/ace
def generate_frustum_at_position(rotation, translation, color, size, aspect_ratio):
    """Generates a frustum mesh at a specified (rotation, translation), with optional color
    : rotation is a 3x3 numpy array
    : translation is a 3-long numpy vector
    : color is a 3-long numpy vector or tuple or list; each element is a uint8 RGB value
    : aspect_ratio is a float of width/height
    """
    # assert translation.shape == (3,)
    # assert rotation.shape == (3, 3)
    # assert len(color) == 3

    frustum_verts = origin_frustum_verts.copy()
    frustum_verts[:,0] *= aspect_ratio

    transformed_frustum_verts = \
        size * rotation.dot(frustum_verts.T).T + translation[None, :]

    cuboids = []
    for edge in frustum_edges:
        line_cuboid = cuboid_from_line(line_start=transformed_frustum_verts[edge[0]],
                                       line_end=transformed_frustum_verts[edge[1]],
                                       color=color)
        cuboids.append(line_cuboid)

    return trimesh.util.concatenate(cuboids)

# copy-paste from ACE: https://github.com/nianticlabs/ace
def cuboid_from_line(line_start, line_end, color=(255, 0, 255)):
    """Approximates a line with a long cuboid
    color is a 3-element RGB tuple, with each element a uint8 value
    """
    # create two vectors which are both (a) perpendicular to the direction of the line and
    # (b) perpendicular to each other.

    def normalise_vector(vect):
        """
        Returns vector with unit length.

        @param vect: Vector to be normalised.
        @return: Normalised vector.
        """
        length = np.sqrt((vect ** 2).sum())
        return vect / length

    THICKNESS = 0.010  # controls how thick the frustum's 'bars' are

    direction = normalise_vector(line_end - line_start)
    random_dir = normalise_vector(np.random.rand(3))
    perpendicular_x = normalise_vector(np.cross(direction, random_dir))
    perpendicular_y = normalise_vector(np.cross(direction, perpendicular_x))

    vertices = []
    for node in (line_start, line_end):
        for x_offset in (-1, 1):
            for y_offset in (-1, 1):
                vert = node + THICKNESS * (perpendicular_y * y_offset + perpendicular_x * x_offset)
                vertices.append(vert)

    faces = [
        (4, 5, 1, 0),
        (5, 7, 3, 1),
        (7, 6, 2, 3),
        (6, 4, 0, 2),
        (0, 1, 3, 2),  # end of tube
        (6, 7, 5, 4),  # other end of tube
    ]

    mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))

    for c in (0, 1, 2):
        mesh.visual.vertex_colors[:, c] = color[c]

    return mesh