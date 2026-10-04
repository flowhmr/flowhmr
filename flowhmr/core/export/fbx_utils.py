"""
fbx_utils.py

Shared helpers for FBX SDK operations.
Includes: IO, Animation, Mesh/Skeleton Construction, Constants.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Dict, List, Optional, Tuple

import fbx
import numpy as np
from transforms3d.euler import mat2euler

# Constants (Centralized)

DEFAULT_EULER_AXES = "sxyz"
DEFAULT_INTERP = fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationConstant


# yapf: disable
# SMPL-H Joint Definitions
SMPLH_JOINT2NUM = {
    "Pelvis": 0, "L_Hip": 1, "R_Hip": 2, "Spine1": 3,
    "L_Knee": 4, "R_Knee": 5, "Spine2": 6,
    "L_Ankle": 7, "R_Ankle": 8, "Spine3": 9,
    "L_Foot": 10, "R_Foot": 11, "Neck": 12, "L_Collar": 13, "R_Collar": 14, "Head": 15,
    "L_Shoulder": 16, "R_Shoulder": 17, "L_Elbow": 18, "R_Elbow": 19,
    "L_Wrist": 20, "R_Wrist": 21,
    "L_Index1": 22, "L_Index2": 23, "L_Index3": 24,
    "L_Middle1": 25, "L_Middle2": 26, "L_Middle3": 27,
    "L_Pinky1": 28, "L_Pinky2": 29, "L_Pinky3": 30,
    "L_Ring1": 31, "L_Ring2": 32, "L_Ring3": 33,
    "L_Thumb1": 34, "L_Thumb2": 35, "L_Thumb3": 36,
    "R_Index1": 37, "R_Index2": 38, "R_Index3": 39,
    "R_Middle1": 40, "R_Middle2": 41, "R_Middle3": 42,
    "R_Pinky1": 43, "R_Pinky2": 44, "R_Pinky3": 45,
    "R_Ring1": 46, "R_Ring2": 47, "R_Ring3": 48,
    "R_Thumb1": 49, "R_Thumb2": 50, "R_Thumb3": 51,
}
# yapf: enable


# Manager / Scene I/O
def create_manager() -> fbx.FbxManager:
    mgr = fbx.FbxManager.Create()
    ios = fbx.FbxIOSettings.Create(mgr, fbx.IOSROOT)
    mgr.SetIOSettings(ios)
    return mgr


def load_scene(mgr: fbx.FbxManager, filepath: str) -> fbx.FbxScene:
    importer = fbx.FbxImporter.Create(mgr, "")
    if not importer.Initialize(filepath, -1, mgr.GetIOSettings()):
        raise RuntimeError(
            f"Failed to initialize FBX importer for: {filepath}\nError: {importer.GetStatus().GetErrorString()}"
        )
    scene = fbx.FbxScene.Create(mgr, "")
    importer.Import(scene)
    importer.Destroy()
    return scene


def export_scene_atomic(
    mgr: fbx.FbxManager, scene: fbx.FbxScene, out_path: str, *, embed_textures: bool = False
) -> None:
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

    ios = mgr.GetIOSettings()
    if embed_textures:
        ios.SetBoolProp(fbx.EXP_FBX_EMBEDDED, True)
        ios.SetBoolProp(fbx.EXP_FBX_MATERIAL, True)
        ios.SetBoolProp(fbx.EXP_FBX_TEXTURE, True)

    # True "atomic" write: always export to a temp file in the same directory, then replace.
    out_dir = os.path.dirname(out_path) if os.path.dirname(out_path) else "."
    with tempfile.NamedTemporaryFile(suffix=".fbx", delete=False, dir=out_dir) as tmp:
        tmp_path = tmp.name

    exporter = fbx.FbxExporter.Create(mgr, "")
    try:
        if not exporter.Initialize(tmp_path, -1, ios):
            raise RuntimeError(f"Exporter failed: {exporter.GetStatus().GetErrorString()}")
        exporter.Export(scene)
    finally:
        exporter.Destroy()

    # Replace destination (atomic on POSIX when same filesystem)
    os.replace(tmp_path, out_path)


# Node traversal / scene edits
def collect_all_nodes(node: fbx.FbxNode, nodes_dict: Optional[Dict[str, fbx.FbxNode]] = None) -> Dict[str, fbx.FbxNode]:
    if nodes_dict is None:
        nodes_dict = {}
    nodes_dict[node.GetName()] = node
    for i in range(node.GetChildCount()):
        collect_all_nodes(node.GetChild(i), nodes_dict)
    return nodes_dict


def collect_skeleton_nodes(node, skeleton_nodes=None):
    """Recursively collect skeleton/bone nodes"""
    if skeleton_nodes is None:
        skeleton_nodes = {}
    attr = node.GetNodeAttribute()
    if attr and attr.GetAttributeType() == fbx.FbxNodeAttribute.EType.eSkeleton:
        skeleton_nodes[node.GetName()] = node
    for i in range(node.GetChildCount()):
        collect_skeleton_nodes(node.GetChild(i), skeleton_nodes)
    return skeleton_nodes


def clear_animations(scene: fbx.FbxScene) -> None:
    count = scene.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId))
    for i in range(count - 1, -1, -1):
        obj = scene.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId), i)
        if obj:
            obj.Destroy()


def strip_mesh_geometry(scene: fbx.FbxScene, safe_detach_if_has_children: bool = True) -> None:
    root = scene.GetRootNode()
    if not root:
        return

    mesh_nodes = []

    def dfs(n):
        for i in range(n.GetChildCount()):
            dfs(n.GetChild(i))
        attr = n.GetNodeAttribute()
        if attr and attr.GetAttributeType() == fbx.FbxNodeAttribute.EType.eMesh:
            mesh_nodes.append(n)

    dfs(root)

    for n in mesh_nodes:
        attr = n.GetNodeAttribute()
        if safe_detach_if_has_children and n.GetChildCount() > 0:
            n.SetNodeAttribute(None)
            if attr:
                attr.Destroy()
        else:
            p = n.GetParent()
            if p:
                p.RemoveChild(n)
            n.Destroy()


# Animation helpers
def mat2euler_deg(R: np.ndarray, axes: str = DEFAULT_EULER_AXES) -> np.ndarray:
    """Convert rotation matrix to Euler angles (degrees), NumPy 2.0 safe."""
    return np.rad2deg(mat2euler(np.asarray(R, dtype=np.float64), axes=axes))


def animate_single_channel(anim_layer, component, axis: str, values: np.ndarray, frame_duration: float) -> None:
    ncomp = {"X": 0, "Y": 1, "Z": 2}.get(axis, 0)
    curve = component.GetCurve(anim_layer, axis, True)
    curve.KeyModifyBegin()
    time = fbx.FbxTime()
    for i, val in enumerate(values):
        time.SetSecondDouble(i * frame_duration)
        idx = curve.KeyAdd(time)[0]
        curve.KeySetValue(idx, float(val[ncomp] if isinstance(val, (list, np.ndarray)) else val))
        curve.KeySetInterpolation(idx, DEFAULT_INTERP)
    curve.KeyModifyEnd()


def animate_rotation_from_rotmats(
    anim_layer, node, rot_mats: np.ndarray, frame_duration: float, axes: str = DEFAULT_EULER_AXES
) -> None:
    eulers = np.array([mat2euler_deg(m, axes) for m in rot_mats])
    for axis in ["X", "Y", "Z"]:
        animate_single_channel(anim_layer, node.LclRotation, axis, eulers, frame_duration)


def animate_translation(anim_layer, node, translations: np.ndarray, frame_duration: float) -> None:
    translations = np.asarray(translations, dtype=np.float64)
    if translations.ndim == 1:
        if translations.shape[0] != 3:
            raise ValueError(f"Translation vector must be length 3, got {translations.shape[0]}")
        translations = translations.reshape(1, -1)
    elif translations.ndim == 2:
        if translations.shape[1] != 3:
            raise ValueError(f"Translation array must have 3 columns, got {translations.shape[1]}")
    else:
        raise ValueError(f"Translation must be 1D or 2D, got {translations.ndim}D")
    for axis in ["X", "Y", "Z"]:
        animate_single_channel(anim_layer, node.LclTranslation, axis, translations, frame_duration)


# Construction Helpers


def build_mesh_node(
    scene: fbx.FbxScene, vertices: np.ndarray, faces: np.ndarray, name: str = "body", uv_coords=None, uv_faces=None
) -> fbx.FbxNode:
    geo_node = fbx.FbxNode.Create(scene, "Geometry")
    scene.GetRootNode().AddChild(geo_node)

    mesh = fbx.FbxMesh.Create(scene, name)
    geo_node.SetNodeAttribute(mesh)

    # Vertices
    mesh.InitControlPoints(len(vertices))
    for i, v in enumerate(vertices):
        mesh.SetControlPointAt(fbx.FbxVector4(v[0], v[1], v[2]), i)

    # Faces: ensure integer indices and validate range
    faces = np.asarray(faces, dtype=np.int32)
    max_idx = len(vertices) - 1
    if np.any(faces < 0) or np.any(faces > max_idx):
        raise ValueError(f"Face indices out of range [0, {max_idx}]: min={faces.min()}, max={faces.max()}")
    for f in faces:
        mesh.BeginPolygon(-1)
        for idx in f:
            mesh.AddPolygon(int(idx))
        mesh.EndPolygon()

    # UVs
    if uv_coords is not None and uv_faces is not None:
        uv_layer = mesh.CreateElementUV("UVSet")
        uv_layer.SetMappingMode(fbx.FbxLayerElement.EMappingMode.eByPolygonVertex)
        uv_layer.SetReferenceMode(fbx.FbxLayerElement.EReferenceMode.eIndexToDirect)
        uv_arr = uv_layer.GetDirectArray()
        for uv in uv_coords:
            uv_arr.Add(fbx.FbxVector2(uv[0], uv[1]))
        uv_idx_arr = uv_layer.GetIndexArray()
        for f_uv in uv_faces:
            for idx in f_uv:
                uv_idx_arr.Add(idx)

    return geo_node


def build_skeleton_nodes(
    mgr: fbx.FbxManager, scene: fbx.FbxScene, joints_pos: np.ndarray, parents: np.ndarray, joint_names: List[str]
) -> List[fbx.FbxNode]:
    """Build skeleton nodes. parents array is not modified (safe to pass original)."""
    root_node = scene.GetRootNode()
    ref_node = fbx.FbxNode.Create(scene, "Reference")
    root_node.AddChild(ref_node)

    # Ensure parents is a copy to avoid modifying caller's array
    parents = np.asarray(parents, dtype=np.int32).copy()
    joints_pos = np.asarray(joints_pos, dtype=np.float64)

    skel_nodes = []
    for i, name in enumerate(joint_names):
        skel = fbx.FbxSkeleton.Create(mgr, "")
        # All joints use eLimbNode for Blender FBX import compatibility
        skel.SetSkeletonType(fbx.FbxSkeleton.EType.eLimbNode)

        node = fbx.FbxNode.Create(scene, name)
        node.SetNodeAttribute(skel)
        node.LclTranslation.Set(
            fbx.FbxDouble3(float(joints_pos[i, 0]), float(joints_pos[i, 1]), float(joints_pos[i, 2]))
        )

        skel_nodes.append(node)
        if parents[i] != -1:
            if parents[i] < 0 or parents[i] >= len(skel_nodes):
                raise ValueError(f"Invalid parent index {parents[i]} for joint {i} ({name})")
            skel_nodes[parents[i]].AddChild(node)

    if len(skel_nodes) == 0:
        raise ValueError("No skeleton nodes created")
    ref_node.AddChild(skel_nodes[0]) # Assuming 0 is root
    return skel_nodes


def apply_skinning(
    scene: fbx.FbxScene, mesh_node: fbx.FbxNode, skel_nodes: List[fbx.FbxNode], weights: np.ndarray
) -> None:
    """
    Apply skinning weights to mesh. weights: (NumVerts, NumBones).
    Creates bind pose including mesh and all skeleton nodes.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape[1] != len(skel_nodes):
        raise ValueError(f"Weights shape mismatch: {weights.shape[1]} bones vs {len(skel_nodes)} skeleton nodes")

    skin = fbx.FbxSkin.Create(scene, "")
    evaluator = scene.GetAnimationEvaluator()
    mesh_matrix = evaluator.GetNodeGlobalTransform(mesh_node)

    for bone_idx, bone_node in enumerate(skel_nodes):
        cluster = fbx.FbxCluster.Create(scene, "")
        cluster.SetLink(bone_node)
        cluster.SetLinkMode(fbx.FbxCluster.ELinkMode.eTotalOne)

        # Add weights (only non-zero weights)
        indices = np.where(weights[:, bone_idx] > 1e-6)[0] # Small threshold to avoid numerical noise
        for v_idx in indices:
            cluster.AddControlPointIndex(int(v_idx), float(weights[v_idx, bone_idx]))

        cluster.SetTransformMatrix(mesh_matrix)
        bone_matrix = evaluator.GetNodeGlobalTransform(bone_node)
        cluster.SetTransformLinkMatrix(bone_matrix)
        skin.AddCluster(cluster)

    mesh_node.GetNodeAttribute().AddDeformer(skin)

    _store_bind_pose(scene, mesh_node)


def _add_node_recursively(node_array: List[fbx.FbxNode], node: fbx.FbxNode) -> None:
    """Recursively add a node and all its parents to the list."""
    if node is None:
        return
    _add_node_recursively(node_array, node.GetParent())
    if node not in node_array:
        node_array.append(node)


def _store_bind_pose(scene: fbx.FbxScene, geometry_node: fbx.FbxNode) -> None:
    """Store bind pose by recursively collecting all related nodes.

    Traverses all skin clusters on the mesh, recursively collects each link bone
    and its parents into the bind pose. This matches the FBX SDK recommended approach
    and ensures correct skinning in Blender and other DCC tools.
    """
    if geometry_node is None or geometry_node.GetNodeAttribute() is None:
        return

    clustered_nodes: List[fbx.FbxNode] = []
    attr = geometry_node.GetNodeAttribute()
    attr_type = attr.GetAttributeType()

    if attr_type in (
        fbx.FbxNodeAttribute.EType.eMesh,
        fbx.FbxNodeAttribute.EType.eNurbs,
        fbx.FbxNodeAttribute.EType.ePatch,
    ):
        skin_count = attr.GetDeformerCount(fbx.FbxDeformer.EDeformerType.eSkin)
        for i in range(skin_count):
            skin = attr.GetDeformer(i, fbx.FbxDeformer.EDeformerType.eSkin)
            for j in range(skin.GetClusterCount()):
                link = skin.GetCluster(j).GetLink()
                _add_node_recursively(clustered_nodes, link)

    clustered_nodes.append(geometry_node)

    if not clustered_nodes:
        return

    pose = fbx.FbxPose.Create(scene, geometry_node.GetName())
    pose.SetIsBindPose(True)
    evaluator = scene.GetAnimationEvaluator()
    for node in clustered_nodes:
        bind_matrix = evaluator.GetNodeGlobalTransform(node)
        pose.Add(node, fbx.FbxMatrix(bind_matrix))
    scene.AddPose(pose)
