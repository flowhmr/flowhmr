"""Forward kinematics utilities for SMPL body models.

Provides functions to compute joints and vertices from SMPL parameters (rot6d, trans, shapes).
These were previously in evaluation.metrics and are now properly placed in the bodymodels module.
"""

import torch
from ..math.geometry import rot6d_to_rotation_matrix
from ..motion import matrix


def get_joints_from_smpl_params(smpl_skeleton, batch, joint_num=24):
    rot6d = batch["rot6d"]
    transl = batch["trans"]
    shapes = batch["shapes"]
    rot6d_flat = rot6d.reshape(rot6d.shape[0] * rot6d.shape[1], -1, 6)
    transl_flat = transl.reshape(rot6d.shape[0] * rot6d.shape[1], 3)
    assert shapes.shape[1] == 1, f"shapes shape should be [B,1,16], got {shapes.shape}"
    shapes = shapes.repeat(1, rot6d.shape[1], 1)
    shapes_flat = shapes.reshape(rot6d.shape[0] * rot6d.shape[1], 16)

    local_params = {
        "rot6d": rot6d_flat,
        "trans": torch.zeros(transl.shape[0] * transl.shape[1], 3, device=transl.device, dtype=transl.dtype),
        "shapes": shapes_flat,
    }
    global_params = {
        "rot6d": rot6d_flat,
        "trans": transl_flat,
        "shapes": shapes_flat,
    }

    output_joints = {
        "local_joints": smpl_skeleton(local_params)["keypoints3d"].reshape(rot6d.shape[0], rot6d.shape[1], -1, 3),
        "global_joints": smpl_skeleton(global_params)["keypoints3d"].reshape(rot6d.shape[0], rot6d.shape[1], -1, 3),
    }
    for key in output_joints:
        output_joints[key] = output_joints[key][:, :, :joint_num, :]
    return output_joints


def get_vertices_from_smpl_params(smpl_mesh, batch):
    rot6d = batch["rot6d"]
    transl = batch["trans"]
    shapes = batch["shapes"]
    rot6d_flat = rot6d.reshape(rot6d.shape[0] * rot6d.shape[1], -1, 6)
    transl_flat = transl.reshape(rot6d.shape[0] * rot6d.shape[1], 3)
    assert shapes.shape[1] == 1, f"shapes shape should be [B,1,16], got {shapes.shape}"
    shapes = shapes.repeat(1, rot6d.shape[1], 1)
    shapes_flat = shapes.reshape(rot6d.shape[0] * rot6d.shape[1], -1)

    params = {
        "rot6d": rot6d_flat,
        "trans": transl_flat,
        "shapes": shapes_flat,
    }

    out_vertices = smpl_mesh(params)
    out_vertices["local_vertices"] = out_vertices.pop("vertices_wotrans")
    out_vertices["global_vertices"] = out_vertices.pop("vertices")
    return out_vertices


def get_fkmat_from_smpl_params(smpl_skeleton, batch):
    B, L = batch["rot6d"].shape[:2]
    rotmat = rot6d_to_rotation_matrix(batch["rot6d"])[..., :22, :, :]
    parents = smpl_skeleton.parents[:22]

    shapes = batch["shapes"]
    assert shapes.shape[1] == 1, f"shapes shape should be [B,1,16], got {shapes.shape}"
    shapes = shapes.repeat(1, L, 1)
    shapes = shapes.reshape(B * L, 16)
    skeleton = smpl_skeleton.compute_j_shaped(shapes)[..., :22, :].reshape(B, L, 22, 3)
    local_skeleton = skeleton - skeleton[:, :, parents]
    local_skeleton = torch.cat([skeleton[:, :, :1], local_skeleton[:, :, 1:]], dim=2)
    local_skeleton[..., 0, :] += batch["trans"]

    mat = matrix.get_TRS(rotmat, local_skeleton)
    fk_mat = matrix.forward_kinematics(mat, parents)
    joints = matrix.get_position(fk_mat)

    return joints, mat, fk_mat
