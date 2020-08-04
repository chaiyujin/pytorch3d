# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import NamedTuple, Optional

import torch
import torch.nn as nn

from .rasterize_meshes import rasterize_meshes, rasterize_meshes_python
from .utils import interpolate_face_attributes


# Class to store the outputs of mesh rasterization
class Fragments(NamedTuple):
    pix_to_face: torch.Tensor
    zbuf: torch.Tensor
    bary_coords: torch.Tensor
    dists: torch.Tensor


# Class to store the mesh rasterization params with defaults
class RasterizationSettings:
    __slots__ = [
        "image_size",
        "blur_radius",
        "faces_per_pixel",
        "bin_size",
        "max_faces_per_bin",
        "perspective_correct",
        "clip_barycentric_coords",
        "cull_backfaces",
    ]

    def __init__(
        self,
        image_size: int = 256,
        blur_radius: float = 0.0,
        faces_per_pixel: int = 1,
        bin_size: Optional[int] = None,
        max_faces_per_bin: Optional[int] = None,
        perspective_correct: bool = False,
        clip_barycentric_coords: bool = False,
        cull_backfaces: bool = False,
    ):
        self.image_size = image_size
        self.blur_radius = blur_radius
        self.faces_per_pixel = faces_per_pixel
        self.bin_size = bin_size
        self.max_faces_per_bin = max_faces_per_bin
        self.perspective_correct = perspective_correct
        self.clip_barycentric_coords = clip_barycentric_coords
        self.cull_backfaces = cull_backfaces


class MeshRasterizer(nn.Module):
    """
    This class implements methods for rasterizing a batch of heterogenous
    Meshes.
    """

    def __init__(self, cameras=None, raster_settings=None):
        """
        Args:
            cameras: A cameras object which has a  `transform_points` method
                which returns the transformed points after applying the
                world-to-view and view-to-screen
                transformations.
            raster_settings: the parameters for rasterization. This should be a
                named tuple.

        All these initial settings can be overridden by passing keyword
        arguments to the forward function.
        """
        super().__init__()
        if raster_settings is None:
            raster_settings = RasterizationSettings()

        self.cameras = cameras
        self.raster_settings = raster_settings

    def transform(self, meshes_world, **kwargs) -> torch.Tensor:
        """
        Args:
            meshes_world: a Meshes object representing a batch of meshes with
                vertex coordinates in world space.

        Returns:
            meshes_screen: a Meshes object with the vertex positions in screen
            space

        NOTE: keeping this as a separate function for readability but it could
        be moved into forward.
        """
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of MeshRasterizer"
            raise ValueError(msg)
        verts_world = meshes_world.verts_padded()

        # NOTE: Retaining view space z coordinate for now.
        # TODO: Revisit whether or not to transform z coordinate to [-1, 1] or
        # [0, 1] range.
        verts_view = cameras.get_world_to_view_transform(**kwargs).to(verts_world.device).transform_points(
            verts_world
        )
        verts_screen = cameras.get_projection_transform(**kwargs).to(verts_world.device).transform_points(
            verts_view
        )
        verts_screen[..., 2] = verts_view[..., 2]
        meshes_screen = meshes_world.update_padded(new_verts_padded=verts_screen)
        return meshes_screen

    def forward(self, meshes_world, **kwargs) -> Fragments:
        """
        Args:
            meshes_world: a Meshes object representing a batch of meshes with
                          coordinates in world space.
        Returns:
            Fragments: Rasterization outputs as a named tuple.
        """
        meshes_screen = self.transform(meshes_world, **kwargs)
        raster_settings = kwargs.get("raster_settings", self.raster_settings)
        # TODO(jcjohns): Should we try to set perspective_correct automatically
        # based on the type of the camera?
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=raster_settings.image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            perspective_correct=raster_settings.perspective_correct,
            clip_barycentric_coords=raster_settings.clip_barycentric_coords,
            cull_backfaces=raster_settings.cull_backfaces,
        )

        # fix bary_coords
        # bary_coords = bary_coords / bary_coords.sum(-1, keepdim=True)

        # # try
        # verts = meshes_screen.verts_packed()
        # faces = meshes_screen.faces_packed()
        # faces_verts = verts[faces]  # F, 3, 3

        # # pixels = interpolate_face_attributes(pix_to_face, bary_coords, faces_verts)
        # for y in range(512):
        #     for x in range(512):
        #         fi = pix_to_face[0, y, x, 0]
        #         if fi < 0:
        #             continue
        #         w0 = bary_coords[0, y, x, 0, 0]
        #         w1 = bary_coords[0, y, x, 0, 1]
        #         w2 = bary_coords[0, y, x, 0, 2]
        #         v0 = faces_verts[fi, 0]
        #         v1 = faces_verts[fi, 1]
        #         v2 = faces_verts[fi, 2]

        #         uv = v0 * w0 + v1 * w1 + v2 * w2
        #         u = (1.0 - uv[0].item()) / 2.0 * 512
        #         v = (1.0 - uv[1].item()) / 2.0 * 512
        #         print((u, v), (x, y))

        #         # u = (1.0-pixels[0, y, x, 0, 0].item()) / 2.0 * 512
        #         # v = (1.0-pixels[0, y, x, 0, 1].item()) / 2.0 * 512
        #         # print((u, v), (x, y))

        return Fragments(
            pix_to_face=pix_to_face, zbuf=zbuf, bary_coords=bary_coords, dists=dists
        )
