"""
smplh2fbx.py (Constructive Converter)
Converts SMPL-H motion to FBX by constructing the mesh and skeleton from scratch.
"""

import logging
import os
from typing import Optional, Tuple

import fbx
import numpy as np
import torch

from ..math.geometry import angle_axis_to_rotation_matrix, rot6d_to_rotation_matrix
from .fbx_utils import (
    SMPLH_JOINT2NUM,
    animate_rotation_from_rotmats,
    animate_translation,
    apply_skinning,
    build_mesh_node,
    build_skeleton_nodes,
    create_manager,
    export_scene_atomic,
    strip_mesh_geometry,
)

logger = logging.getLogger(__name__)

DEFAULT_OBJ_TEMPLATE_PATH = "assets/body_models/smplh/textures/male_smplh.obj"


def blend_shapes(betas: torch.Tensor, shape_disps: torch.Tensor) -> torch.Tensor:
    """Calculates the per vertex displacement due to the blend shapes


    Parameters
    ----------
    betas : torch.tensor Bx(num_betas)
        Blend shape coefficients
    shape_disps: torch.tensor Vx3x(num_betas)
        Blend shapes

    Returns
    -------
    torch.tensor BxVx3
        The per-vertex displacement due to shape deformation
    """

    # Displacement[b, m, k] = sum_{l} betas[b, l] * shape_disps[m, k, l]
    # i.e. Multiply each shape displacement by its corresponding beta and
    # then sum them.
    blend_shape = torch.einsum("bl,mkl->bmk", [betas, shape_disps])
    return blend_shape


def vertices2joints(J_regressor: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """Calculates the 3D joint locations from the vertices

    Parameters
    ----------
    J_regressor : torch.tensor JxV
        The regressor array that is used to calculate the joints from the
        position of the vertices
    vertices : torch.tensor BxVx3
        The tensor of mesh vertices

    Returns
    -------
    torch.tensor BxJx3
        The location of the joints
    """

    return torch.einsum("bik,ji->bjk", [vertices, J_regressor])


def get_offsets_from_beta(
    beta: torch.Tensor, smplx_params: dict, return_template_mesh: bool = True
) -> tuple[np.ndarray, torch.Tensor] | np.ndarray:
    v_template = torch.FloatTensor(smplx_params["v_template"]).unsqueeze(0)
    shape_dirs = torch.FloatTensor(smplx_params["shapedirs"])
    J_regressor = torch.FloatTensor(smplx_params["J_regressor"])

    v_shaped = v_template + blend_shapes(beta, shape_dirs)
    J = vertices2joints(J_regressor, v_shaped).squeeze(0).numpy()

    parents = smplx_params["kintree_table"][0].copy()
    parents[0] = -1
    Translates = J.copy()
    Translates[1:] -= J[parents[1:]]
    if not return_template_mesh:
        return Translates
    else:
        return Translates, v_shaped


def _parse_obj_uv(obj_path: str) -> Tuple[np.ndarray, np.ndarray]:
    uv_coords = []
    uv_faces = []

    with open(obj_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("vt "):
                parts = line.split()
                uv_coords.append([float(parts[1]), float(parts[2])])
            elif line.startswith("f "):
                parts = line.split()
                face_uvs = []
                for part in parts[1:]:
                    indices = part.split("/")
                    if len(indices) > 1 and indices[1]:
                        face_uvs.append(int(indices[1]) - 1)
                if len(face_uvs) == 3:
                    uv_faces.append(face_uvs)

    return np.array(uv_coords, dtype=np.float64), np.array(uv_faces, dtype=np.int32)


class SMPLH2FBX:
    def __init__(
        self,
        smplh_model_path: str = "./assets/body_models/smplh/neutral/model.npz",
        obj_template_path: Optional[str] = DEFAULT_OBJ_TEMPLATE_PATH,
    ):
        self.smpl_data = dict(np.load(smplh_model_path, allow_pickle=True))
        self.mgr = create_manager()
        # Pre-sort joint mapping
        self.joint_names_ordered = [k for k, v in sorted(SMPLH_JOINT2NUM.items(), key=lambda x: x[1])]

        # Load UV coordinates from OBJ template
        self.uv_coords: Optional[np.ndarray] = None
        self.uv_faces: Optional[np.ndarray] = None
        if obj_template_path and os.path.isfile(obj_template_path):
            self.uv_coords, self.uv_faces = _parse_obj_uv(obj_template_path)
            logger.info(
                "Loaded UV from OBJ: %d uv_coords, %d uv_faces",
                len(self.uv_coords), len(self.uv_faces),
            )
        elif obj_template_path:
            logger.warning("OBJ template not found, UV disabled: %s", obj_template_path)

    def convert_npz_to_fbx(self, npz_data, outname: str, fps: int = 30, export_mesh: bool = True) -> bool:
        if isinstance(npz_data, str):
            npz_data = np.load(npz_data, allow_pickle=True)

        # Use mocap_framerate from NPZ if available
        if "mocap_framerate" in npz_data:
            fps = int(npz_data["mocap_framerate"])

        # Prepare Data & Skeleton Structure
        scale = 100.0
        betas_np = np.array(npz_data["betas"])
        # normalize betas to shape (1, 16) to match SMPL-H model shapedirs
        if betas_np.ndim == 1:
            betas_np = betas_np[None, :]
        elif betas_np.ndim >= 2 and betas_np.shape[0] > 1:
            betas_np = betas_np[:1]
        betas = torch.from_numpy(betas_np).float()
        # Calculate customized skeleton based on Betas
        # (Simplified: assuming single beta for the whole sequence)
        rest_pose_trans, v_shaped = get_offsets_from_beta(betas[0:1], self.smpl_data)

        poses = torch.from_numpy(npz_data["poses"]).float()
        rot_mats = angle_axis_to_rotation_matrix(poses.reshape(len(poses), -1, 3)).numpy()
        trans = npz_data["trans"] * scale

        # Build Scene
        scene = fbx.FbxScene.Create(self.mgr, "")
        time_mode = fbx.FbxTime().ConvertFrameRateToTimeMode(fps)
        scene.GetGlobalSettings().SetTimeMode(time_mode)

        # Build Mesh
        faces = self.smpl_data.get("f", None)
        if faces is None:
            faces = self.smpl_data.get("faces", None)
        if faces is None:
            raise KeyError("SMPLH model missing faces key: expected `f` (preferred) or `faces`.")
        geo_node = build_mesh_node(
            scene, v_shaped[0].numpy() * scale, faces,
            uv_coords=self.uv_coords, uv_faces=self.uv_faces,
        )

        # Build Skeleton
        parents = np.array(self.smpl_data["kintree_table"][0]).copy()
        parents[0] = -1
        skel_nodes = build_skeleton_nodes(self.mgr, scene, rest_pose_trans * scale, parents, self.joint_names_ordered)

        # Skinning
        apply_skinning(scene, geo_node, skel_nodes, self.smpl_data["weights"])

        # Animate
        stack = fbx.FbxAnimStack.Create(scene, "Anim")
        layer = fbx.FbxAnimLayer.Create(scene, "Layer0")
        stack.AddMember(layer)

        frame_dur = 1.0 / fps

        # Root Translation: Combine global trans with skeleton root rest pose
        root_trans_final = np.tile(rest_pose_trans[0] * scale, (len(trans), 1))
        root_trans_final += trans
        animate_translation(layer, skel_nodes[0], root_trans_final, frame_dur)

        # Rotations
        for i, node in enumerate(skel_nodes):
            animate_rotation_from_rotmats(layer, node, rot_mats[:, i], frame_dur)

        if not export_mesh:
            strip_mesh_geometry(scene)

        export_scene_atomic(self.mgr, scene, outname, embed_textures=export_mesh)
        scene.Destroy()
        return True


if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=str)
    args = parser.parse_args()

    smplh2fbx = SMPLH2FBX(smplh_model_path="./assets/body_models/smplh/neutral/model.npz")

    if os.path.isdir(args.root):
        npzfiles = sorted(glob.glob(os.path.join(args.root, "*.npz")))
    else:
        if args.root.endswith(".npz"):
            npzfiles = [args.root]
        else:
            raise ValueError(f"Unknown file type: {args.root}")

    for npzfile in npzfiles:
        smplh2fbx.convert_npz_to_fbx(npzfile, npzfile.replace(".npz", ".fbx"), export_mesh=False)
