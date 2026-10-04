"""Minimal SMPL/SMPL-H skeleton and mesh models (PyTorch).

SMPLSkeleton: joint positions only (fast, no vertices).
SMPLMesh:      full skinned mesh via LBS (pose blending disabled to match
               the FBX conversion in scripts/smplh2fbx.py).

Both take params = {'poses': (F, J*3) axis-angle, 'shapes': (F|1, B) betas,
optional 'trans': (F, 3)}.
"""

import os
import pickle

import numpy as np
import torch

from .lbs import blend_shapes, batch_rigid_transform, lbs, batch_rodrigues


def to_tensor(array, dtype=torch.float32):
    if isinstance(array, torch.Tensor):
        return array
    return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


def load_model_data(model_path):
    """Load a SMPL-family body model from .npz or .pkl."""
    model_path = os.path.abspath(model_path)
    assert os.path.exists(model_path), 'Path {} does not exist!'.format(model_path)
    if model_path.endswith('.npz'):
        return dict(np.load(model_path))
    if model_path.endswith('.pkl'):
        with open(model_path, 'rb') as f:
            return pickle.load(f, encoding='latin1')
    raise ValueError(f'unsupported body model format: {model_path}')


class SMPLSkeleton(torch.nn.Module):
    """SMPL skeleton with joints only (no vertices or faces)."""

    def register_parents(self, data):
        # parent index of each joint
        kintree_table = data['kintree_table']
        if len(kintree_table.shape) == 2:
            kintree_table = kintree_table[0]
        parents = to_tensor(to_np(kintree_table)).long()
        parents[0] = -1
        self.register_buffer('parents', parents)

    def __init__(self, model_path='body_models/smplh/neutral/model.npz', max_shape=-1):
        super().__init__()
        model = load_model_data(model_path)
        # J_regressor: (nJoints, nVertices)
        J_regressor = to_tensor(to_np(model['J_regressor']))
        # shapedirs: (nVertices, 3, nBetas)
        shapedirs = to_tensor(to_np(model['shapedirs']))
        if max_shape > 0:
            shapedirs = shapedirs[:, :, :max_shape]
        j_shapedirs = torch.einsum('jv,vdb->jdb', [J_regressor, shapedirs])
        v_template = to_tensor(to_np(model['v_template']))
        j_template = J_regressor @ v_template
        self.register_buffer('j_template', j_template)
        self.register_buffer('j_shapedirs', j_shapedirs)
        self.register_parents(model)

    def forward(self, params):
        poses = params['poses']
        batch_size = poses.shape[0]
        rot_mats = batch_rodrigues(poses.view(-1, 3)).view([batch_size, -1, 3, 3])

        # shaped joints directly from the template regressor
        j_shaped = self.j_template[None] + blend_shapes(params['shapes'], self.j_shapedirs)
        if j_shaped.shape[0] == 1 and batch_size > 1:
            j_shaped = j_shaped.repeat(batch_size, 1, 1)

        # j_transformed: (F, J, 3); A: (F, J, 4, 4)
        j_transformed, A = batch_rigid_transform(rot_mats, j_shaped, self.parents, dtype=rot_mats.dtype)
        if 'trans' in params:
            j_transformed = j_transformed + params['trans'][:, None, :]

        return {
            'keypoints3d': j_transformed,
            'j_shaped': j_shaped,
            'transforms': A
        }

    def get_skeleton(self, betas):
        """betas: (*, B) -> rest-pose joints: (*, J, 3)"""
        return self.j_template + torch.einsum("...d, jcd -> ...jc", betas, self.j_shapedirs)


class SMPLMesh(SMPLSkeleton):
    """SMPL mesh: full LBS with shape blending (pose blending disabled)."""

    def __init__(self, model_path='body_models/smplh/neutral/model.npz', max_shape=-1):
        torch.nn.Module.__init__(self)
        model = load_model_data(model_path)
        # J_regressor: (nJoints, nVertices)
        J_regressor = to_tensor(to_np(model['J_regressor']))
        # shapedirs: (nVertices, 3, nBetas)
        shapedirs = to_tensor(to_np(model['shapedirs']))
        if max_shape > 0:
            shapedirs = shapedirs[:, :, :max_shape]
        j_shapedirs = torch.einsum('jv,vdb->jdb', [J_regressor, shapedirs])
        v_template = to_tensor(to_np(model['v_template']))
        self.register_buffer('j_template', J_regressor @ v_template)
        self.register_buffer('j_shapedirs', j_shapedirs)
        self.register_buffer('v_template', v_template)
        self.register_buffer('shapedirs', shapedirs)
        num_pose_basis = model['posedirs'].shape[-1]
        posedirs = np.reshape(model['posedirs'], [-1, num_pose_basis]).T
        self.register_buffer('posedirs', to_tensor(posedirs))
        self.register_buffer('lbs_weights', to_tensor(to_np(model['weights'])))
        self.register_buffer('J_regressor', J_regressor)
        self.register_parents(model)

    def forward(self, params):
        poses = params['poses']
        batch_size = poses.shape[0]
        rot_mats = batch_rodrigues(poses.view(-1, 3)).view([batch_size, -1, 3, 3])

        shapes = params['shapes']
        if shapes.shape[0] == 1 and batch_size > 1:
            shapes = shapes.repeat(batch_size, 1)
        shapedirs = self.shapedirs
        if shapedirs.shape[-1] > shapes.shape[-1]:
            shapedirs = shapedirs[..., :shapes.shape[-1]]

        # pose blending disabled (matches the FBX conversion)
        vertices, _, _, _ = lbs(shapes, rot_mats, self.v_template,
                                shapedirs, self.posedirs,
                                self.J_regressor, self.parents,
                                self.lbs_weights, pose2rot=False, use_pose_blending=False)

        if 'trans' in params:
            vertices = vertices + params['trans'][:, None, :]

        return {
            'vertices': vertices
        }
