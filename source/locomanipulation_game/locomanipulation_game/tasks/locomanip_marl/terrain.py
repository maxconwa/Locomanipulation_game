"""The ground's height under the robot, from a height map of the terrain built once (replaces the 187-ray height scan,
whose mean the rewards, the critic and the arm goals' standing frame read: a scan cast every policy step for one
number per env).

TerrainImporter keeps no copy of the generated mesh, so KeptMeshTerrainImporter (scene_cfg.py's terrain class) holds
it until the first ground_height call, which ray-casts it once on a MAP_RESOLUTION grid; afterwards a lookup is a
bilinear read of that grid at the pelvis. Row 0, where every env starts and stays, is flat: the map reads its height
exactly. On rough rows it is the height under the pelvis, where the scan averaged a 1.6 x 1.0 m patch.
"""

from __future__ import annotations

import torch
import trimesh
from typing import TYPE_CHECKING

from isaaclab.terrains import TerrainImporter
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

MAP_RESOLUTION = 0.05  # m; the generator's height field is on 0.1 m


class KeptMeshTerrainImporter(TerrainImporter):
    """TerrainImporter that also keeps the meshes it imports (dropped once the height map is built)."""

    def import_mesh(self, name: str, mesh: trimesh.Trimesh):
        self.kept_meshes = getattr(self, "kept_meshes", {})
        self.kept_meshes[name] = mesh
        super().import_mesh(name, mesh)


class GroundMap:
    """The terrain's height on a grid over its extent; a plane is 0 everywhere."""

    def __init__(self, terrain: TerrainImporter, device: str):
        self.device = device
        meshes = getattr(terrain, "kept_meshes", None)
        if terrain.cfg.terrain_type == "plane":
            self.grid = None
            return
        if not meshes:
            raise ValueError("ground_height needs the terrain's mesh: set the scene's terrain class_type to"
                             " KeptMeshTerrainImporter (terrain.py)")
        mesh = trimesh.util.concatenate(list(meshes.values()))
        low, high = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
        self.origin = torch.tensor(low[:2], dtype=torch.float, device=device)
        nx = int((high[0] - low[0]) / MAP_RESOLUTION) + 1
        ny = int((high[1] - low[1]) / MAP_RESOLUTION) + 1
        xs = self.origin[0] + MAP_RESOLUTION * torch.arange(nx, device=device)
        ys = self.origin[1] + MAP_RESOLUTION * torch.arange(ny, device=device)
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        starts = torch.stack([gx, gy, torch.full_like(gx, float(high[2]) + 1.0)], dim=-1).reshape(-1, 3)
        down = torch.tensor([0.0, 0.0, -1.0], device=device).expand_as(starts).contiguous()
        warp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=device)
        hits = raycast_mesh(starts.unsqueeze(0), down.unsqueeze(0), warp_mesh)[0].reshape(nx, ny, 3)[..., 2]
        # a ray through a gap in the mesh (none on the generator's terrain): the lowest point found
        finite = torch.isfinite(hits)
        self.grid = torch.where(finite, hits, hits[finite].min() if finite.any() else torch.zeros_like(hits))
        self.size = torch.tensor([nx - 1, ny - 1], dtype=torch.float, device=device)
        for name in list(meshes):
            del meshes[name]

    def height(self, xy: torch.Tensor) -> torch.Tensor:
        """The ground's height (m) at world xy (N, 2), bilinear; off the map, its nearest edge."""
        if self.grid is None:
            return torch.zeros(xy.shape[0], device=xy.device)
        f = ((xy - self.origin) / MAP_RESOLUTION).clamp(min=torch.zeros_like(self.size), max=self.size)
        i = f.floor().clamp(max=self.size - 1).long()
        t = f - i
        g = self.grid
        h00, h10 = g[i[:, 0], i[:, 1]], g[i[:, 0] + 1, i[:, 1]]
        h01, h11 = g[i[:, 0], i[:, 1] + 1], g[i[:, 0] + 1, i[:, 1] + 1]
        tx, ty = t[:, 0], t[:, 1]
        return (h00 * (1 - tx) + h10 * tx) * (1 - ty) + (h01 * (1 - tx) + h11 * tx) * ty


def ground_height(env: ManagerBasedEnv) -> torch.Tensor:
    """The ground's height (m) under each env's pelvis, (num_envs,); the map is built on the first call."""
    ground = getattr(env, "_ground_map", None)
    if ground is None:
        ground = env._ground_map = GroundMap(env.scene.terrain, env.device)
    return ground.height(env.scene["robot"].data.root_pos_w[:, :2])
