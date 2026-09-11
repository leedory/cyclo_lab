# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared rigid collision approximation for FFW soft grippers."""

from __future__ import annotations

import re

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from cyclo_lab.assets.robots.ffw_softgripper_geometry import banana_points

FFW_SOFTGRIPPER_BAND_FRACTIONS = (0.0, 0.31, 0.62, 0.78, 0.90, 1.0)
FFW_SOFTGRIPPER_BAND_OVERLAP_FRACTION = 0.015

_DISTAL_LINK_PATTERN = re.compile(
    r"/gripper_[lr]_[^/]*_(?P<finger>[lr]2)/collisions(?:/|_)",
    re.IGNORECASE,
)
_CONTACT_LAST_ATTRIBUTE = "physxScene:solveArticulationContactLast"
_PROXY_SUFFIX = "_cyclo_banana_proxy"


def _triangulate(face_vertex_counts, face_vertex_indices) -> np.ndarray:
    triangles: list[tuple[int, int, int]] = []
    cursor = 0
    for raw_count in face_vertex_counts:
        count = int(raw_count)
        polygon = [int(value) for value in face_vertex_indices[cursor : cursor + count]]
        cursor += count
        for index in range(1, count - 1):
            triangles.append((polygon[0], polygon[index], polygon[index + 1]))
    return np.asarray(triangles, dtype=np.int64)


def _copy_local_transform(source_prim, target_prim) -> None:
    local_transform = UsdGeom.Xformable(source_prim).GetLocalTransformation()
    if isinstance(local_transform, tuple):
        local_transform = local_transform[0]
    UsdGeom.Xformable(target_prim).AddTransformOp().Set(local_transform)


def _copy_material_bindings(source_prim, target_prim) -> None:
    for relationship in source_prim.GetRelationships():
        if relationship.GetName().startswith("material:binding"):
            target_prim.CreateRelationship(relationship.GetName()).SetTargets(
                relationship.GetTargets()
            )


def _enable_contact_last(stage) -> None:
    physics_scenes = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
    if len(physics_scenes) != 1:
        raise RuntimeError(
            "Expected one PhysX scene while enabling contact-last, found "
            f"{[str(prim.GetPath()) for prim in physics_scenes]}"
        )
    physics_scenes[0].CreateAttribute(
        _CONTACT_LAST_ATTRIBUTE, Sdf.ValueTypeNames.Bool
    ).Set(True)


def _distal_collision_meshes(stage, robot_prim_path: str):
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    if not robot_prim.IsValid():
        raise RuntimeError(f"Robot prim does not exist: {robot_prim_path}")

    matches = []
    for prim in Usd.PrimRange(robot_prim):
        path = str(prim.GetPath())
        if _PROXY_SUFFIX in path:
            continue
        link_match = _DISTAL_LINK_PATTERN.search(path)
        if link_match is None or not prim.IsA(UsdGeom.Mesh):
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        matches.append((prim, link_match.group("finger").lower()))

    if len(matches) != 4:
        raise RuntimeError(
            "Expected four FFW distal collision meshes, found "
            f"{[str(prim.GetPath()) for prim, _ in matches]} below {robot_prim_path}"
        )
    return matches


def _author_convex_piece(stage, path, hull: trimesh.Trimesh, source_prim) -> None:
    proxy = UsdGeom.Mesh.Define(stage, path)
    proxy.CreatePointsAttr().Set(
        [Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in hull.vertices]
    )
    proxy.CreateFaceVertexCountsAttr().Set([3] * len(hull.faces))
    proxy.CreateFaceVertexIndicesAttr().Set(
        [int(value) for value in hull.faces.reshape(-1)]
    )
    bounds_min = hull.vertices.min(axis=0)
    bounds_max = hull.vertices.max(axis=0)
    proxy.CreateExtentAttr().Set(
        [
            Gf.Vec3f(*(float(value) for value in bounds_min)),
            Gf.Vec3f(*(float(value) for value in bounds_max)),
        ]
    )
    proxy.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    proxy.CreateDoubleSidedAttr().Set(False)
    UsdGeom.Imageable(proxy.GetPrim()).CreateVisibilityAttr().Set(
        UsdGeom.Tokens.invisible
    )
    UsdPhysics.CollisionAPI.Apply(proxy.GetPrim()).CreateCollisionEnabledAttr().Set(True)
    UsdPhysics.MeshCollisionAPI.Apply(proxy.GetPrim()).CreateApproximationAttr().Set(
        "convexHull"
    )
    _copy_material_bindings(source_prim, proxy.GetPrim())


def _replace_distal_mesh(stage, source_prim, finger: str) -> tuple[int, float]:
    source_mesh = UsdGeom.Mesh(source_prim)
    points = np.asarray(source_mesh.GetPointsAttr().Get(), dtype=np.float64)
    triangles = _triangulate(
        source_mesh.GetFaceVertexCountsAttr().Get(),
        source_mesh.GetFaceVertexIndicesAttr().Get(),
    )
    if points.ndim != 2 or points.shape[1] != 3 or len(triangles) == 0:
        raise RuntimeError(f"Invalid distal collision mesh: {source_prim.GetPath()}")

    points, long_axis, _, tip_deflection = banana_points(points, finger)
    face_centers = points[triangles].mean(axis=1)[:, long_axis]
    lower = float(points[:, long_axis].min())
    long_span = float(np.ptp(points[:, long_axis]))
    boundaries = lower + long_span * np.asarray(FFW_SOFTGRIPPER_BAND_FRACTIONS)
    overlap = FFW_SOFTGRIPPER_BAND_OVERLAP_FRACTION * long_span

    source_path = source_prim.GetPath()
    container_path = source_path.GetParentPath().AppendChild(source_path.name + _PROXY_SUFFIX)
    if stage.GetPrimAtPath(container_path).IsValid():
        stage.RemovePrim(container_path)
    container = UsdGeom.Xform.Define(stage, container_path)
    _copy_local_transform(source_prim, container.GetPrim())

    piece_count = len(FFW_SOFTGRIPPER_BAND_FRACTIONS) - 1
    for piece_index in range(piece_count):
        band_lower = boundaries[piece_index] - (overlap if piece_index else 0.0)
        band_upper = boundaries[piece_index + 1] + (
            overlap if piece_index + 1 < piece_count else 0.0
        )
        upper_test = (
            face_centers <= band_upper
            if piece_index + 1 == piece_count
            else face_centers < band_upper
        )
        selected_faces = (face_centers >= band_lower) & upper_test
        selected_indices = np.unique(triangles[selected_faces].reshape(-1))
        if len(selected_indices) < 4:
            raise RuntimeError(
                f"Softgripper band {piece_index} has too few points: {source_path}"
            )
        hull = trimesh.points.PointCloud(points[selected_indices]).convex_hull
        if len(hull.vertices) < 4 or len(hull.faces) < 4 or not hull.is_watertight:
            raise RuntimeError(
                f"Could not build closed convex band {piece_index}: {source_path}"
            )
        _author_convex_piece(
            stage,
            container_path.AppendChild(f"piece_{piece_index:02d}"),
            hull,
            source_prim,
        )

    UsdPhysics.CollisionAPI(source_prim).GetCollisionEnabledAttr().Set(False)
    return piece_count, tip_deflection


def apply_ffw_softgripper_collision_defaults(stage, robot_prim_path: str) -> None:
    """Enable contact-last and replace four distal meshes with mild banana proxies."""

    _enable_contact_last(stage)
    piece_count = 0
    deflections = []
    for source_prim, finger in _distal_collision_meshes(stage, robot_prim_path):
        authored, deflection = _replace_distal_mesh(stage, source_prim, finger)
        piece_count += authored
        deflections.append(deflection)
    print(
        "[FFW softgripper physics] enabled contact-last and replaced 4 distal "
        f"colliders with {piece_count} closed banana-proxy pieces; "
        f"tip deflection local units={min(deflections):.6g}..{max(deflections):.6g}."
    )
