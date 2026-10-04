import torch
import torch.nn.functional as F
import numpy as np
import os
from .lbs_function import blend_shapes, batch_rigid_transform, batch_rigid_transform_global, lbs, batch_rodrigues, transform_mat
from ..math.geometry import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis
from ..math.geometry import rot6d_to_rotation_matrix


def to_tensor(array, dtype=torch.float32, device=torch.device("cpu")):
    if "torch.tensor" not in str(type(array)):
        return torch.tensor(array, dtype=dtype).to(device)
    else:
        return array.to(device)


def to_np(array, dtype=np.float32):
    if "scipy.sparse" in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


def read_pickle(name):
    import pickle

    with open(name, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return data


def load_model_data(model_path):
    model_path = os.path.abspath(model_path)
    assert os.path.exists(model_path), "Path {} does not exist!".format(model_path)
    if model_path.endswith(".npz"):
        data = np.load(model_path)
        data = dict(data)
    elif model_path.endswith(".pkl"):
        data = read_pickle(model_path)
    return data


joints_name = [
    "MidHip",  # 0
    "LUpLeg",  # 1
    "RUpLeg",  # 2
    "spine",  # 3
    "LLeg",  # 4
    "RLeg",  # 5
    "spine1",  # 6
    "LFoot",  # 7
    "RFoot",  # 8
    "spine2",  # 9
    "LToeBase",  # 10
    "RToeBase",  # 11
    "Neck",  # 12
    "LShoulder",  # 13
    "RShoulder",  # 14
    "Head",  # 15
    "LArm",  # 16
    "RArm",  # 17
    "LForeArm",  # 18
    "RForeArm",  # 19
    "LHand",  # 20
    "RHand",  # 21
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
]

# official definition from https://github.com/vchoutas/smplx/blob/43561ecabd23cfa70ce7b724cb831b6af0133e6e/smplx/joint_names.py
SMPLX_JOINT_NAMES_127 = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",
    "left_eye_smplhf",
    "right_eye_smplhf",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "nose",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
    "right_eye_brow1",
    "right_eye_brow2",
    "right_eye_brow3",
    "right_eye_brow4",
    "right_eye_brow5",
    "left_eye_brow5",
    "left_eye_brow4",
    "left_eye_brow3",
    "left_eye_brow2",
    "left_eye_brow1",
    "nose1",
    "nose2",
    "nose3",
    "nose4",
    "right_nose_2",
    "right_nose_1",
    "nose_middle",
    "left_nose_1",
    "left_nose_2",
    "right_eye1",
    "right_eye2",
    "right_eye3",
    "right_eye4",
    "right_eye5",
    "right_eye6",
    "left_eye4",
    "left_eye3",
    "left_eye2",
    "left_eye1",
    "left_eye6",
    "left_eye5",
    "right_mouth_1",
    "right_mouth_2",
    "right_mouth_3",
    "mouth_top",
    "left_mouth_3",
    "left_mouth_2",
    "left_mouth_1",
    "left_mouth_5",  # 59 in OpenPose output
    "left_mouth_4",  # 58 in OpenPose output
    "mouth_bottom",
    "right_mouth_4",
    "right_mouth_5",
    "right_lip_1",
    "right_lip_2",
    "lip_top",
    "left_lip_2",
    "left_lip_1",
    "left_lip_3",
    "lip_bottom",
    "right_lip_3",
]

# official definition from https://github.com/vchoutas/smplx/blob/43561ecabd23cfa70ce7b724cb831b6af0133e6e/smplx/joint_names.py
SMPLH_JOINT_NAMES_52 = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
]


class SMPLSkeleton(torch.nn.Module):
    """
    This class implements the SMPL skeleton with 24 joints.
    It does not contain vertices and faces.
    """

    def register_parents(self, data):
        # indices of parents for each joints
        kintree_table = data["kintree_table"]
        if len(kintree_table.shape) == 2:
            kintree_table = kintree_table[0]
        parents = to_tensor(to_np(kintree_table)).long()
        parents[0] = -1
        self.register_buffer("parents", parents)

    def __init__(self, model_path="assets/body_models/smplh/neutral/model.npz", max_shape=-1):
        super().__init__()
        model = load_model_data(model_path)
        if "hands_meanr" in model:
            hands_meanl = torch.FloatTensor(model["hands_meanl"])
            hands_meanr = torch.FloatTensor(model["hands_meanr"])
            self.register_buffer("left_hand_mean", hands_meanl)
            self.register_buffer("right_hand_mean", hands_meanr)
        # J_regressor: (nJoints, nVertices)
        J_regressor = to_tensor(to_np(model["J_regressor"]))
        # shapedirs: (nVertices, 3, nBetas)
        shapedirs = to_tensor(to_np(model["shapedirs"]))
        if max_shape > 0:
            shapedirs = shapedirs[:, :, :max_shape]
        j_shapedirs = torch.einsum("jv,vdb->jdb", [J_regressor, shapedirs])
        v_template = to_tensor(to_np(model["v_template"]))
        j_template = J_regressor @ v_template
        self.register_buffer("j_template", j_template)
        self.register_buffer("j_shapedirs", j_shapedirs)
        self.register_parents(model)

    @property
    def device(self):
        return self.j_template.device

    def compute_j_shaped(self, shapes):
        return self.j_template[None] + blend_shapes(shapes, self.j_shapedirs)

    def forward(self, params, fast_forward=False):
        if "poses" in params:
            poses = params["poses"]
            batch_size = poses.shape[0]
            rot_mats = batch_rodrigues(poses.view(-1, 3)).view([batch_size, -1, 3, 3])
        elif "rot6d" in params:
            rot6d = params["rot6d"]
            batch_size = rot6d.shape[0]
            rot_mats = rot6d_to_rotation_matrix(rot6d)
        elif "rot_mats" in params:
            rot_mats = params["rot_mats"]
            batch_size = rot_mats.shape[0]
            rot_mats = rot_mats.view([batch_size, -1, 3, 3])
        else:
            raise ValueError("poses, rot6d, or rot_mats must be in params")

        shapes = params["shapes"]
        dtype, device = rot_mats.dtype, rot_mats.device
        j_shaped = self.j_template[None] + blend_shapes(shapes, self.j_shapedirs)

        if j_shaped.shape[0] == 1 and batch_size > 1:
            j_shaped = j_shaped.repeat(batch_size, 1, 1)
        # j_transformed: (nframes, njoints, 3)
        j_transformed, A = batch_rigid_transform(rot_mats, j_shaped, self.parents, dtype=dtype)
        # A: batch_size x n_joints x 4 x 4
        if "trans" in params:
            trans = params["trans"]
            j_transformed = j_transformed + trans[:, None, :]

        return {"keypoints3d": j_transformed, "j_shaped": j_shaped, "transforms": A}


class SMPLMesh(SMPLSkeleton):
    def __init__(self, model_path="assets/body_models/smplh/neutral/model.npz", max_shape=-1):
        torch.nn.Module.__init__(self)
        model = load_model_data(model_path)
        # J_regressor: (nJoints, nVertices)
        J_regressor = to_tensor(to_np(model["J_regressor"]))
        # shapedirs: (nVertices, 3, nBetas)
        shapedirs = to_tensor(to_np(model["shapedirs"]))
        self.faces = to_np(model["f"])
        if max_shape > 0:
            shapedirs = shapedirs[:, :, :max_shape]
        j_shapedirs = torch.einsum("jv,vdb->jdb", [J_regressor, shapedirs])
        v_template = to_tensor(to_np(model["v_template"]))
        j_template = J_regressor @ v_template
        self.register_buffer("j_template", j_template)
        self.register_buffer("j_shapedirs", j_shapedirs)
        self.register_buffer("v_template", v_template)
        self.register_buffer("shapedirs", shapedirs)
        num_pose_basis = model["posedirs"].shape[-1]
        posedirs = np.reshape(model["posedirs"], [-1, num_pose_basis]).T
        self.register_buffer("posedirs", to_tensor(posedirs))
        lbs_weights = to_tensor(to_np(model["weights"]), dtype=torch.float32)
        self.register_buffer("lbs_weights", lbs_weights)
        j_regressor = to_tensor(to_np(model["J_regressor"]), dtype=torch.float32)
        self.register_buffer("J_regressor", j_regressor)
        self.register_parents(model)

    def forward(self, params, fast_forward=False, sample_indices=None):
        if "poses" in params:
            poses = params["poses"]
            batch_size = poses.shape[0]
            rot_mats = batch_rodrigues(poses.view(-1, 3)).view([batch_size, -1, 3, 3])
        elif "rot6d" in params:
            rot6d = params["rot6d"]
            batch_size = rot6d.shape[0]
            rot_mats = rot6d_to_rotation_matrix(rot6d).view([batch_size, -1, 3, 3])
        elif "rot_mats" in params:
            rot_mats = params["rot_mats"]
            batch_size = rot_mats.shape[0]
            rot_mats = rot_mats.view([batch_size, -1, 3, 3])
        else:
            raise ValueError("poses, rot6d, or rot_mats must be in params")

        shapes = params["shapes"]
        dtype, device = rot_mats.dtype, rot_mats.device
        if shapes.shape[0] == 1 and batch_size > 1:
            shapes = shapes.repeat(batch_size, 1)
        shapedirs = self.shapedirs
        if shapedirs.shape[-1] > shapes.shape[-1]:
            shapedirs = shapedirs[..., : shapes.shape[-1]]

        if sample_indices is not None:
            j_shaped = self.j_template[None] + blend_shapes(shapes, self.j_shapedirs)
            shapedirs = shapedirs[sample_indices]
            v_template = self.v_template[sample_indices]
            lbs_weights = self.lbs_weights[sample_indices]

            vertices, joints, _, _ = lbs(
                shapes,
                rot_mats,
                v_template,
                shapedirs,
                self.posedirs,
                self.J_regressor,
                self.parents,
                lbs_weights,
                pose2rot=False,
                J_shaped=j_shaped,
                use_shape_blending=False,
                use_pose_blending=False,
            )
        else:
            vertices, joints, _, _ = lbs(
                shapes,
                rot_mats,
                self.v_template,
                shapedirs,
                self.posedirs,
                self.J_regressor,
                self.parents,
                self.lbs_weights,
                pose2rot=False,
                use_pose_blending=False,
            )
        vertices_wotrans = vertices

        if "trans" in params:
            trans = params["trans"]
            vertices = vertices + trans[:, None, :]

        return {
            "vertices": vertices,
            "vertices_wotrans": vertices_wotrans,
        }

    def forward_global_rot(self, params):
        global_rot_mats = params["global_rot_mats"]
        batch_size = global_rot_mats.shape[0]
        shapes = params["shapes"]
        dtype, device = global_rot_mats.dtype, global_rot_mats.device

        if shapes.shape[0] == 1 and batch_size > 1:
            shapes = shapes.repeat(batch_size, 1)
        shapedirs = self.shapedirs
        if shapedirs.shape[-1] > shapes.shape[-1]:
            shapedirs = shapedirs[..., : shapes.shape[-1]]

        # shape blending on vertices
        v_shaped = self.v_template + blend_shapes(shapes, shapedirs)

        # joint locations from shape
        j_shaped = self.j_template[None] + blend_shapes(shapes, self.j_shapedirs)

        # FK with global rotations → joint positions + LBS transforms
        J_transformed, A = batch_rigid_transform_global(
            global_rot_mats, j_shaped, self.parents, dtype=dtype
        )

        # LBS skinning
        W = self.lbs_weights.unsqueeze(0).expand(batch_size, -1, -1)
        num_joints = J_transformed.shape[1]
        T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)

        homogen_coord = torch.ones(
            [batch_size, v_shaped.shape[1], 1], dtype=dtype, device=device
        )
        v_posed_homo = torch.cat([v_shaped, homogen_coord], dim=2)
        v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))
        vertices = v_homo[:, :, :3, 0]

        vertices_wotrans = vertices
        if "trans" in params:
            vertices = vertices + params["trans"][:, None, :]

        return {
            "vertices": vertices,
            "vertices_wotrans": vertices_wotrans,
        }

class SMPLMeshSparseKeypoints(SMPLMesh):
    def __init__(
        self,
        model_path="assets/body_models/smplh/neutral/model.npz",
        max_shape=-1,
        J_regressor_sparse="assets/body_models/J_regressor_body25.npy",
    ):
        torch.nn.Module.__init__(self)
        model = load_model_data(model_path)
        # J_regressor: (nJoints, nVertices)
        J_regressor = to_tensor(to_np(model["J_regressor"]))
        # shapedirs: (nVertices, 3, nBetas)
        shapedirs = to_tensor(to_np(model["shapedirs"]))
        if max_shape > 0:
            shapedirs = shapedirs[:, :, :max_shape]
        j_shapedirs = torch.einsum("jv,vdb->jdb", [J_regressor, shapedirs])
        v_template = to_tensor(to_np(model["v_template"]))
        j_template = J_regressor @ v_template

        J_regressor_sparse = to_tensor(to_np(np.load(J_regressor_sparse)))
        v_template = J_regressor_sparse @ v_template
        shapedirs = torch.einsum("jv,vdb->jdb", [J_regressor_sparse, shapedirs])
        self.register_buffer("j_template", j_template)
        self.register_buffer("j_shapedirs", j_shapedirs)
        self.register_buffer("v_template", v_template)
        self.register_buffer("shapedirs", shapedirs)
        num_pose_basis = model["posedirs"].shape[-1]
        posedirs = np.reshape(model["posedirs"], [-1, num_pose_basis]).T
        self.register_buffer("posedirs", to_tensor(posedirs))
        lbs_weights = to_tensor(to_np(model["weights"]), dtype=torch.float32)
        lbs_weights = J_regressor_sparse @ lbs_weights
        self.register_buffer("lbs_weights", lbs_weights)
        j_regressor = to_tensor(to_np(model["J_regressor"]), dtype=torch.float32)
        self.register_buffer("J_regressor", j_regressor)
        self.register_parents(model)

    def forward(self, params, fast_forward=False):
        if "poses" in params:
            poses = params["poses"]
            batch_size = poses.shape[0]
            rot_mats = batch_rodrigues(poses.view(-1, 3)).view([batch_size, -1, 3, 3])
        elif "rot6d" in params:
            rot6d = params["rot6d"]
            batch_size = rot6d.shape[0]
            rot_mats = rot6d_to_rotation_matrix(rot6d).view([batch_size, -1, 3, 3])
        elif "rot_mats" in params:
            rot_mats = params["rot_mats"]
            batch_size = rot_mats.shape[0]
            rot_mats = rot_mats.view([batch_size, -1, 3, 3])
        else:
            raise ValueError("poses, rot6d, or rot_mats must be in params")

        shapes = params["shapes"]
        dtype, device = rot_mats.dtype, rot_mats.device
        if shapes.shape[0] == 1 and batch_size > 1:
            shapes = shapes.repeat(batch_size, 1)
        shapedirs = self.shapedirs
        if shapedirs.shape[-1] > shapes.shape[-1]:
            shapedirs = shapedirs[..., : shapes.shape[-1]]

        j_shaped = self.j_template[None] + blend_shapes(shapes, self.j_shapedirs)

        vertices, joints, _, _ = lbs(
            shapes,
            rot_mats,
            self.v_template,
            shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            pose2rot=False,
            J_shaped=j_shaped,
            use_shape_blending=False,
            use_pose_blending=False,
        )
        vertices_wotrans = vertices

        if "trans" in params:
            trans = params["trans"]
            vertices = vertices + trans[:, None, :]

        return {
            "vertices": vertices,
            "vertices_wotrans": vertices_wotrans,
        }


class PartialSMPLSkeleton(torch.nn.Module):

    @staticmethod
    def _compute_required_joints(target_joints, parents_list):
        required = set(target_joints)
        queue = list(target_joints)
        while queue:
            j = queue.pop(0)
            p = parents_list[j]
            if p >= 0 and p not in required:
                required.add(p)
                queue.append(p)
        return sorted(required)

    def __init__(self, full_skeleton: "SMPLSkeleton", target_joint_ids: list):
        super().__init__()
        parents_list = full_skeleton.parents.cpu().tolist()
        self.num_joints = len(parents_list)
        self.target_joint_ids = target_joint_ids
        self.required_joints = self._compute_required_joints(target_joint_ids, parents_list)
        req = self.required_joints

        self.register_buffer("j_template", full_skeleton.j_template.clone())
        self.register_buffer("j_shapedirs_partial", full_skeleton.j_shapedirs[req].clone())
        self.register_buffer("j_shapedirs", full_skeleton.j_shapedirs.clone())

        self.req_parents = []
        for j in req:
            p = parents_list[j]
            if p < 0:
                self.req_parents.append(-1)
            else:
                self.req_parents.append(req.index(p))

    def forward(self, params, fast_forward=False):
        if "rot6d" in params:
            rot6d = params["rot6d"]  # (B, 52, 6)
            batch_size = rot6d.shape[0]
            rot6d_partial = rot6d[:, self.required_joints, :] # (B, N_req, 6)
            rot_mats = rot6d_to_rotation_matrix(rot6d_partial) # (B, N_req, 3, 3)
        elif "poses" in params:
            poses = params["poses"]
            batch_size = poses.shape[0]
            rot_mats_full = batch_rodrigues(poses.view(-1, 3)).view([batch_size, -1, 3, 3])
            rot_mats = rot_mats_full[:, self.required_joints, :, :]
        elif "rot_mats" in params:
            rot_mats = params["rot_mats"][:, self.required_joints, :, :]
            batch_size = rot_mats.shape[0]
        else:
            raise ValueError("rot6d, rot_mats or poses must be in params")

        shapes = params["shapes"]
        device = rot_mats.device
        dtype = rot_mats.dtype

        j_shaped_partial = self.j_template[self.required_joints][None] + \
            blend_shapes(shapes, self.j_shapedirs_partial)
        if j_shaped_partial.shape[0] == 1 and batch_size > 1:
            j_shaped_partial = j_shaped_partial.repeat(batch_size, 1, 1)

        # FK loop (only N_req iterations)
        N_req = len(self.required_joints)
        j_col = j_shaped_partial.unsqueeze(-1) # (B, N_req, 3, 1)
        rel_joints = j_col.clone()
        for i in range(N_req):
            if self.req_parents[i] >= 0:
                rel_joints[:, i] = j_col[:, i] - j_col[:, self.req_parents[i]]

        transforms_list = [None] * N_req
        for i in range(N_req):
            local_T = transform_mat(rot_mats[:, i], rel_joints[:, i])
            p = self.req_parents[i]
            if p < 0:
                transforms_list[i] = local_T
            else:
                transforms_list[i] = transforms_list[p] @ local_T

        j_shaped_full = self.j_template[None] + blend_shapes(shapes, self.j_shapedirs)
        if j_shaped_full.shape[0] == 1 and batch_size > 1:
            j_shaped_full = j_shaped_full.repeat(batch_size, 1, 1)
        keypoints3d = j_shaped_full.clone()

        # rel_transforms = transforms - pad(transforms @ j_homogen, ...)
        transforms_full = torch.zeros(batch_size, self.num_joints, 4, 4, device=device, dtype=dtype)
        transforms_full[:, :, 0, 0] = 1.0
        transforms_full[:, :, 1, 1] = 1.0
        transforms_full[:, :, 2, 2] = 1.0
        transforms_full[:, :, 3, 3] = 1.0

        for local_i, global_j in enumerate(self.required_joints):
            keypoints3d[:, global_j, :] = transforms_list[local_i][:, :3, 3]
            transforms_full[:, global_j, :, :] = transforms_list[local_i]

        # rel_transform = transform - pad(transform @ j_homogen, left=3)
        j_shaped_full_col = j_shaped_full.unsqueeze(-1)
        j_homogen = F.pad(j_shaped_full_col, [0, 0, 0, 1])
        correction = torch.matmul(transforms_full, j_homogen)
        rel_transforms = transforms_full.clone()
        rel_transforms[:, :, :, 3:4] = transforms_full[:, :, :, 3:4] - correction

        if "trans" in params:
            trans = params["trans"]
            keypoints3d = keypoints3d + trans[:, None, :]

        return {"keypoints3d": keypoints3d, "j_shaped": j_shaped_full, "transforms": rel_transforms}


class SMPLX2SMPL:
    def __init__(self, model_path="assets/body_models/smplx2smpl_sparse.pt"):
        self.smplx2smpl_verts_map = torch.load(model_path, weights_only=True).to_dense()
        self.smplx2smpl_joints_map = [SMPLX_JOINT_NAMES_127.index(joint_name) for joint_name in SMPLH_JOINT_NAMES_52]

    def smplx2smpl_verts(self, x):
        """
        Args
            x: tensor (..., 10475, 3)
        Return
            tensor (..., 6890, 3)
        """
        return torch.einsum("ij,...jk->...ik", self.smplx2smpl_verts_map, x)

    def smplx2smpl_joints(self, x):
        """
        Args
            x: tensor (..., 127, 3)
        Return
            tensor (..., 52, 3)
        """
        return x[..., self.smplx2smpl_joints_map, :]


def face_z_transform(keypoints3d):
    # keypoints3d: (nframes, njoints, 3)
    # global_orient: (nframes, 3)
    # trans: (nframes, 3)
    root_pos_init = keypoints3d[0]
    joints_name = [
        "MidHip",  # 0
        "LUpLeg",  # 1
        "RUpLeg",  # 2
        "spine",  # 3
        "LLeg",  # 4
        "RLeg",  # 5
        "spine1",  # 6
        "LFoot",  # 7
        "RFoot",  # 8
        "spine2",  # 9
        "LToeBase",  # 10
        "RToeBase",  # 11
        "neck",  # 12
        "LShoulder",  # 13
        "RShoulder",  # 14
        "head",  # 15
        "LArm",  # 16
        "RArm",  # 17
        "LForeArm",  # 18
        "RForeArm",  # 19
        "LHand",  # 20
        "RHand",  # 21
        "LHandIndex1",  # 22
        "RHandIndex1",  # 23
    ]
    r_hip, l_hip, sdr_r, sdr_l = (
        joints_name.index("RUpLeg"),
        joints_name.index("LUpLeg"),
        joints_name.index("RShoulder"),
        joints_name.index("LShoulder"),
    )
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / torch.norm(across, dim=-1, keepdim=True)

    axis_up = torch.FloatTensor([0, 1, 0]).to(keypoints3d.device)
    forward_init = torch.cross(axis_up, across, dim=-1)
    forward_init = forward_init / torch.norm(forward_init, dim=-1, keepdim=True)
    target = torch.FloatTensor([0, 0, 1]).to(keypoints3d.device)
    sign = torch.sign(torch.cross(forward_init, target, dim=-1)[1])
    theta = torch.acos(torch.sum(forward_init * target, dim=-1, keepdim=True))
    rotation = angle_axis_to_rotation_matrix(theta * sign * axis_up)
    return rotation


def process_r_t(R_transform, global_orient, transl, j_shaped):
    global_orient_rot = angle_axis_to_rotation_matrix(torch.FloatTensor(global_orient))
    global_orient_rot_new = R_transform[None] @ global_orient_rot
    if True:
        transl_new = (R_transform[None] @ (j_shaped[..., None] + torch.FloatTensor(transl[..., None]))).reshape(
            -1, 3
        ) - j_shaped
    else:
        transl_new = (R_transform[None] @ torch.FloatTensor(transl[..., None])).reshape(-1, 3)
    global_orient_new = rotation_matrix_to_angle_axis(global_orient_rot_new)

    return global_orient_new.cpu().numpy(), transl_new.cpu().numpy()

if __name__ == "__main__":
    body_model = SMPLMesh(model_path="assets/body_models/smplh/neutral/model.npz")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body_model.to(device)
    num_frames = 360
    bs = 16
    params = {
        "rot6d": torch.randn(bs * num_frames, 52, 6),
        "trans": torch.randn(bs * num_frames, 3),
        "shapes": torch.randn(bs * num_frames, 16),
    }
    params = {k: v.to(device) for k, v in params.items()}
    out = body_model(params)
    print(out["vertices"].shape)
    print(out["vertices_wotrans"].shape)

    import matplotlib.pyplot as plt

    times = []
    mem_usages = []
    num_points_list = []

    for num_points in range(100, 6890, 500):
        sample_indices = torch.randint(0, 6890, (num_points,), device=device)

        import time
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()
        start_time = time.time()
        for repeat in range(10):
            out = body_model(params, sample_indices=sample_indices)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - start_time

        if torch.cuda.is_available():
            mem = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        else:
            mem = 0.0

        print(f"Num points: {num_points}, Time taken: {elapsed:.3f} s, Max Mem: {mem:.2f} MB")
        print(out["vertices"].shape)
        print(out["vertices_wotrans"].shape)

        num_points_list.append(num_points)
        times.append(elapsed)
        mem_usages.append(mem)

    fig, ax1 = plt.subplots()

    color1 = 'tab:blue'
    ax1.set_xlabel('Sampled Vertices')
    ax1.set_ylabel('Time (s)', color=color1)
    ax1.plot(num_points_list, times, color=color1, marker='o', label='Time (s)')
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Max Memory (MB)', color=color2)
    ax2.plot(num_points_list, mem_usages, color=color2, marker='x', label='Max Mem (MB)')
    ax2.tick_params(axis='y', labelcolor=color2)

    fig.tight_layout()
    plt.title("Performance vs Number of Sampled Vertices")
    plt.savefig("smpl_performance_vs_vertices.png")

    print("\n" + "=" * 60)
    print("FP32 vs FP16 Numerical Precision Test")
    print("model weights: fp32 | input params: fp32 vs fp16->fp32")
    print("=" * 60)

    body_model_test = SMPLMesh(model_path="assets/body_models/smplh/neutral/model.npz")
    body_model_test.to(device).float()

    angle_configs = {
        "identity (0°)": 0.0,
        "small (±15°)": np.deg2rad(15),
        "medium (±45°)": np.deg2rad(45),
        "large (±90°)": np.deg2rad(90),
        "extreme (±180°)": np.pi,
        "full random (0~360°)": 2 * np.pi,
    }

    test_bs = 64
    num_joints = 52

    for angle_name, max_angle in angle_configs.items():
        print(f"\n{'─' * 60}")
        print(f"  Rotation range: {angle_name}, batch_size={test_bs}")
        print(f"{'─' * 60}")

        if max_angle == 0.0:
            rot_mats_fp32 = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(test_bs, num_joints, -1, -1).contiguous()
        else:
            random_axis = torch.randn(test_bs, num_joints, 3, device=device)
            random_axis = random_axis / torch.norm(random_axis, dim=-1, keepdim=True)
            random_angle = (torch.rand(test_bs, num_joints, 1, device=device) * 2 - 1) * max_angle
            axis_angle = (random_axis * random_angle).reshape(-1, 3)
            rot_mats_fp32 = batch_rodrigues(axis_angle).reshape(test_bs, num_joints, 3, 3)

        shapes_fp32 = torch.randn(test_bs, 16, device=device)
        trans_fp32 = torch.randn(test_bs, 3, device=device)

        params_fp32 = {
            "rot_mats": rot_mats_fp32,
            "shapes": shapes_fp32,
            "trans": trans_fp32,
        }
        with torch.no_grad():
            out_fp32 = body_model_test(params_fp32)

        params_fp16_roundtrip = {
            "rot_mats": rot_mats_fp32.half().float(),
            "shapes": shapes_fp32.half().float(),
            "trans": trans_fp32.half().float(),
        }
        with torch.no_grad():
            out_fp16_rt = body_model_test(params_fp16_roundtrip)

        verts_fp32 = out_fp32["vertices"]
        verts_fp16_rt = out_fp16_rt["vertices"]

        abs_diff = (verts_fp32 - verts_fp16_rt).abs()
        rel_diff = abs_diff / (verts_fp32.abs() + 1e-8)

        rot_quant_err = (rot_mats_fp32 - rot_mats_fp32.half().float()).abs()
        shapes_quant_err = (shapes_fp32 - shapes_fp32.half().float()).abs()
        trans_quant_err = (trans_fp32 - trans_fp32.half().float()).abs()

        input_err = rot_quant_err.mean().item()
        output_err = abs_diff.mean().item()
        amplification = output_err / input_err if input_err > 0 else float('inf')

        print(f"  Input quant error  | rot_mats: mean={rot_quant_err.mean().item():.6e}, max={rot_quant_err.max().item():.6e}")
        print(f"                     | shapes:   mean={shapes_quant_err.mean().item():.6e}, max={shapes_quant_err.max().item():.6e}")
        print(f"                     | trans:    mean={trans_quant_err.mean().item():.6e}, max={trans_quant_err.max().item():.6e}")
        print(f"  Vertices abs error | mean={abs_diff.mean().item():.6e}, max={abs_diff.max().item():.6e}, median={abs_diff.median().item():.6e}")
        print(f"  Vertices rel error | mean={rel_diff.mean().item():.6e}, max={rel_diff.max().item():.6e}")
        print(f"  Error amplification (output/input): {amplification:.2f}x")

        verts_wt_fp32 = out_fp32["vertices_wotrans"]
        verts_wt_fp16_rt = out_fp16_rt["vertices_wotrans"]
        abs_diff_wt = (verts_wt_fp32 - verts_wt_fp16_rt).abs()
        print(f"  Verts (w/o trans)  | mean={abs_diff_wt.mean().item():.6e}, max={abs_diff_wt.max().item():.6e}")

        thresholds = [1e-4, 1e-3, 1e-2, 1e-1]
        pcts = [f"<{t:.0e}: {(abs_diff < t).float().mean().item() * 100:.1f}%" for t in thresholds]
        print(f"  Error distribution | {' | '.join(pcts)}")

    print(f"\n{'=' * 60}")
    print("Per-joint error analysis (large ±90° rotation, bs=1)")
    print(f"{'=' * 60}")
    random_axis = torch.randn(1, num_joints, 3, device=device)
    random_axis = random_axis / torch.norm(random_axis, dim=-1, keepdim=True)
    random_angle = (torch.rand(1, num_joints, 1, device=device) * 2 - 1) * np.deg2rad(90)
    axis_angle = (random_axis * random_angle).reshape(-1, 3)
    rot_mats_fp32 = batch_rodrigues(axis_angle).reshape(1, num_joints, 3, 3)
    shapes_fp32 = torch.randn(1, 16, device=device)
    trans_fp32 = torch.randn(1, 3, device=device)

    with torch.no_grad():
        out_a = body_model_test({"rot_mats": rot_mats_fp32, "shapes": shapes_fp32, "trans": trans_fp32})
        out_b = body_model_test({"rot_mats": rot_mats_fp32.half().float(), "shapes": shapes_fp32.half().float(), "trans": trans_fp32.half().float()})
    abs_diff = (out_a["vertices"] - out_b["vertices"]).abs()
    dominant_joint = body_model_test.lbs_weights.argmax(dim=-1)
    per_joint_err = torch.zeros(num_joints, device=device)
    per_joint_cnt = torch.zeros(num_joints, device=device)
    vert_abs_diff = abs_diff[0]
    for j in range(num_joints):
        mask = dominant_joint == j
        if mask.sum() > 0:
            per_joint_err[j] = vert_abs_diff[mask].mean()
            per_joint_cnt[j] = mask.sum()
    top10_idx = per_joint_err.argsort(descending=True)[:10]
    print(f"  Top-10 joints by mean abs error:")
    for rank, j in enumerate(top10_idx):
        j = j.item()
        print(f"    #{rank+1} joint {j:2d} ({joints_name[j] if j < len(joints_name) else '?':>16s}): "
              f"err={per_joint_err[j].item():.6e}, verts={int(per_joint_cnt[j].item())}")

    print("\n" + "=" * 60)
    print("Precision test done.")
    print("=" * 60)