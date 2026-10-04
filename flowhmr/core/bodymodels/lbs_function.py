import torch
import torch.nn.functional as F


def blend_shapes(betas, shape_disps):
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


def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    """Calculates the rotation matrices for a batch of rotation vectors
    Parameters
    ----------
    rot_vecs: torch.tensor Nx3
        array of N axis-angle vectors
    Returns
    -------
    R: torch.tensor Nx3x3
        The rotation matrices for the given axis-angle parameters
    """
    if len(rot_vecs.shape) > 2:
        rot_vec_ori = rot_vecs
        rot_vecs = rot_vecs.view(-1, 3)
    else:
        rot_vec_ori = None
    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1).view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    if rot_vec_ori is not None:
        rot_mat = rot_mat.reshape(*rot_vec_ori.shape[:-1], 3, 3)
    return rot_mat


def transform_mat(R, t):
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):

    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(rot_mats.view(-1, 3, 3), rel_joints.contiguous().view(-1, 3, 1)).view(
        -1, joints.shape[1], 4, 4
    )

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms


def batch_rigid_transform_global(global_rot_mats, joints, parents, dtype=torch.float32):
    batch_size = global_rot_mats.shape[0]
    num_joints = joints.shape[1]
    device = global_rot_mats.device

    joints_col = joints.unsqueeze(-1)

    # rel_joints: parent→child offset in rest pose
    rel_joints = joints_col.clone()
    rel_joints[:, 1:] -= joints_col[:, parents[1:]]

    # joint_i_pos = joint_parent_pos + global_rot_parent @ rel_offset_i
    posed_positions = torch.zeros(batch_size, num_joints, 3, 1, dtype=dtype, device=device)
    posed_positions[:, 0] = rel_joints[:, 0]

    for i in range(1, num_joints):
        p = parents[i]
        posed_positions[:, i] = posed_positions[:, p] + global_rot_mats[:, p] @ rel_joints[:, i]

    transforms = transform_mat(
        global_rot_mats.reshape(-1, 3, 3),
        posed_positions.reshape(-1, 3, 1),
    ).reshape(batch_size, num_joints, 4, 4)

    posed_joints = posed_positions.squeeze(-1)

    joints_homogen = F.pad(joints_col, [0, 0, 0, 1])
    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0]
    )

    return posed_joints, rel_transforms


def lbs(
    betas,
    pose,
    v_template,
    shapedirs,
    posedirs,
    J_regressor,
    parents,
    lbs_weights,
    pose2rot=True,
    dtype=torch.float32,
    only_shape=False,
    use_shape_blending=True,
    use_pose_blending=True,
    J_shaped=None,
    return_vertices=True,
):
    """Performs Linear Blend Skinning with the given shape and pose parameters

    Parameters
    ----------
    betas : torch.tensor BxNB
        The tensor of shape parameters
    pose : torch.tensor Bx(J + 1) * 3
        The pose parameters in axis-angle format
    v_template torch.tensor BxVx3
        The template mesh that will be deformed
    shapedirs : torch.tensor 1xNB
        The tensor of PCA shape displacements
    posedirs : torch.tensor Px(V * 3)
        The pose PCA coefficients
    J_regressor : torch.tensor JxV
        The regressor array that is used to calculate the joints from
        the position of the vertices
    parents: torch.tensor J
        The array that describes the kinematic tree for the model
    lbs_weights: torch.tensor N x V x (J + 1)
        The linear blend skinning weights that represent how much the
        rotation matrix of each part affects each vertex
    pose2rot: bool, optional
        Flag on whether to convert the input pose tensor to rotation
        matrices. The default value is True. If False, then the pose tensor
        should already contain rotation matrices and have a size of
        Bx(J + 1)x9
    dtype: torch.dtype, optional

    Returns
    -------
    verts: torch.tensor BxVx3
        The vertices of the mesh after applying the shape and pose
        displacements.
    joints: torch.tensor BxJx3
        The joints of the model
    """

    batch_size = max(betas.shape[0], pose.shape[0])
    device = betas.device

    # Add shape contribution
    if use_shape_blending:
        v_shaped = v_template + blend_shapes(betas, shapedirs)
        # Get the joints
        # NxJx3 array
        J = vertices2joints(J_regressor, v_shaped)
    else:
        v_shaped = v_template.unsqueeze(0).expand(batch_size, -1, -1)
        v_shaped = v_template + blend_shapes(betas, shapedirs)
        assert J_shaped is not None, "J_shaped is required when use_shape_blending is False"
        if len(J_shaped.shape) == 2:
            J_shaped = J_shaped.unsqueeze(0).expand(batch_size, -1, -1)
        J = J_shaped

    if only_shape:
        return v_shaped, J, None, None
    # 3. Add pose blend shapes
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3), dtype=dtype).view([batch_size, -1, 3, 3])
    else:
        rot_mats = pose.view(batch_size, -1, 3, 3)

    if use_pose_blending:
        ident = torch.eye(3, dtype=dtype, device=device)
        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        pose_offsets = torch.matmul(pose_feature, posedirs).view(batch_size, -1, 3)
        v_posed = pose_offsets + v_shaped
    else:
        v_posed = v_shaped
    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    if not return_vertices:
        return None, J_transformed, A, None
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    num_joints = J_transformed.shape[1]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1], dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed, A, T


def vertices2joints(J_regressor, vertices):
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
