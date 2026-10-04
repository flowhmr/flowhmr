from typing import Optional

import numpy as np
import torch
from torch import Tensor

from ..core.bodymodels.fk_utils import (
    get_vertices_from_smpl_params,
    get_fkmat_from_smpl_params,
)
from ..core.math.geometry import rot6d_to_rotation_matrix


def as_np_array(d):
    if isinstance(d, torch.Tensor):
        return d.cpu().numpy()
    elif isinstance(d, np.ndarray):
        return d
    else:
        return np.array(d)


def compute_jpe(S1, S2):
    return torch.sqrt(((S1 - S2) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()


def compute_rte(target_trans, pred_trans):
    # Compute the global alignment
    _, rot, trans = align_pcl(target_trans[None, :], pred_trans[None, :], fixed_scale=True)
    pred_trans_hat = (torch.einsum("tij,tnj->tni", rot, pred_trans[None, :]) + trans[None, :])[0]

    # Compute the entire displacement of ground truth trajectory
    disps, disp = [], 0
    for p1, p2 in zip(target_trans, target_trans[1:]):
        delta = (p2 - p1).norm(2, dim=-1)
        disp += delta
        disps.append(disp)

    # Compute absolute root-translation-error (RTE)
    rte = torch.norm(target_trans - pred_trans_hat, 2, dim=-1)

    # Normalize it to the displacement
    return (rte / disp).cpu().numpy()


def compute_jitter(joints, fps=30):
    pred_jitter = torch.norm(
        (joints[3:] - 3 * joints[2:-1] + 3 * joints[1:-2] - joints[:-3]) * (fps**3),
        dim=2,
    ).mean(dim=-1)

    return pred_jitter.cpu().numpy() / 10.0


def compute_foot_sliding(target_verts, pred_verts, thr_sta=1e-2):
    assert target_verts.shape == pred_verts.shape
    assert target_verts.shape[-2] == 6890

    # Foot vertices idxs
    foot_idxs = [3216, 3387, 6617, 6787]

    # Compute contact label
    foot_loc = target_verts[:, foot_idxs]
    foot_disp = (foot_loc[1:] - foot_loc[:-1]).norm(2, dim=-1)
    contact = foot_disp[:] < thr_sta
    dynamic_005 = foot_disp[:] > 0.05
    dynamic_01 = foot_disp[:] > 0.1

    pred_feet_loc = pred_verts[:, foot_idxs]
    pred_disp = (pred_feet_loc[1:] - pred_feet_loc[:-1]).norm(2, dim=-1)
    pred_contact = pred_disp[:] < thr_sta
    pred_dynamic_005 = pred_disp[:] > 0.05
    pred_dynamic_01 = pred_disp[:] > 0.1

    error = pred_disp[contact]
    correct_contact = ((pred_contact == contact).sum() / contact.numel()).reshape(1, 1)
    correct_dynamic_01 = ((pred_dynamic_01 == dynamic_01).sum() / dynamic_01.numel()).reshape(1, 1)
    correct_dynamic_005 = ((pred_dynamic_005 == dynamic_005).sum() / dynamic_005.numel()).reshape(1, 1)

    return (
        error.cpu().numpy(),
        correct_contact.cpu().numpy(),
        correct_dynamic_01.cpu().numpy(),
        correct_dynamic_005.cpu().numpy(),
    )


def compute_foot_sliding_by_pred_vel(target_j3d, pred_vel, fps=30, thr_sta=1e-2):
    assert target_j3d.shape[0] == pred_vel.shape[0]
    assert target_j3d.shape[-2] == 24

    # Foot vertices idxs
    foot_idxs = [7, 10, 8, 11] # for joints

    # Compute contact label
    foot_loc = target_j3d[:, foot_idxs]
    foot_disp = (foot_loc[1:] - foot_loc[:-1]).norm(2, dim=-1)
    contact = foot_disp[:] < thr_sta
    dynamic_005 = foot_disp[:] > 0.05
    dynamic_01 = foot_disp[:] > 0.1

    pred_disp = (pred_vel[:, : len(foot_idxs)] / fps)[:-1].norm(2, dim=-1)
    pred_contact = pred_disp[:] < thr_sta
    pred_dynamic_005 = pred_disp[:] > 0.05
    pred_dynamic_01 = pred_disp[:] > 0.1

    error = pred_disp[contact]
    correct_contact = ((pred_contact == contact).sum() / contact.numel()).reshape(1, 1)
    correct_dynamic_01 = ((pred_dynamic_01 == dynamic_01).sum() / dynamic_01.numel()).reshape(1, 1)
    correct_dynamic_005 = ((pred_dynamic_005 == dynamic_005).sum() / dynamic_005.numel()).reshape(1, 1)

    return (
        error.cpu().numpy(),
        correct_contact.cpu().numpy(),
        correct_dynamic_01.cpu().numpy(),
        correct_dynamic_005.cpu().numpy(),
    )


def align_pcl(Y, X, weight=None, fixed_scale=False):
    """align similarity transform to align X with Y using umeyama method
    X' = s * R * X + t is aligned with Y
    :param Y (*, N, 3) first trajectory
    :param X (*, N, 3) second trajectory
    :param weight (*, N, 1) optional weight of valid correspondences
    :returns s (*, 1), R (*, 3, 3), t (*, 3)
    """
    *dims, N, _ = Y.shape
    N = torch.ones(*dims, 1, 1, device=Y.device, dtype=Y.dtype) * N

    if weight is not None:
        Y = Y * weight
        X = X * weight
        N = weight.sum(dim=-2, keepdim=True)

    # subtract mean
    my = Y.sum(dim=-2) / N[..., 0]
    mx = X.sum(dim=-2) / N[..., 0]
    y0 = Y - my[..., None, :]
    x0 = X - mx[..., None, :]

    if weight is not None:
        y0 = y0 * weight
        x0 = x0 * weight

    # correlation
    C = torch.matmul(y0.transpose(-1, -2), x0) / N
    U, D, Vh = torch.linalg.svd(C)

    S = torch.eye(3, device=Y.device, dtype=Y.dtype).reshape(*(1,) * (len(dims)), 3, 3).repeat(*dims, 1, 1)
    neg = torch.det(U) * torch.det(Vh.transpose(-1, -2)) < 0
    S[neg, 2, 2] = -1

    R = torch.matmul(U, torch.matmul(S, Vh))

    D = torch.diag_embed(D)
    if fixed_scale:
        s = torch.ones(*dims, 1, device=Y.device, dtype=Y.dtype)
    else:
        var = torch.sum(torch.square(x0), dim=(-1, -2), keepdim=True) / N
        s = torch.diagonal(torch.matmul(D, S), dim1=-2, dim2=-1).sum(dim=-1, keepdim=True) / var[..., 0]

    t = my - s * torch.matmul(R, mx[..., None])[..., 0]

    return s, R, t


def global_align_joints(gt_joints, pred_joints):
    """
    :param gt_joints (T, J, 3)
    :param pred_joints (T, J, 3)
    """
    s_glob, R_glob, t_glob = align_pcl(gt_joints.reshape(-1, 3), pred_joints.reshape(-1, 3))
    pred_glob = s_glob * torch.einsum("ij,tnj->tni", R_glob, pred_joints) + t_glob[None, None]
    return pred_glob


def first_align_joints(gt_joints, pred_joints):
    """
    align the first two frames
    :param gt_joints (T, J, 3)
    :param pred_joints (T, J, 3)
    """
    s_first, R_first, t_first = align_pcl(gt_joints[:2].reshape(1, -1, 3), pred_joints[:2].reshape(1, -1, 3))
    pred_first = s_first * torch.einsum("tij,tnj->tni", R_first, pred_joints) + t_first[:, None]
    return pred_first


def batch_align_by_pelvis(data_list, pelvis_idxs=[1, 2]):
    """
    Assumes data is given as [pred_j3d, target_j3d, pred_verts, target_verts].
    Each data is in shape of (frames, num_points, 3)
    Pelvis is notated as one / two joints indices.
    Align all data to the corresponding pelvis location.
    """

    pred_j3d, target_j3d, pred_verts, target_verts = data_list

    pred_pelvis = pred_j3d[:, pelvis_idxs].mean(dim=1, keepdims=True).clone()
    target_pelvis = target_j3d[:, pelvis_idxs].mean(dim=1, keepdims=True).clone()

    # Align to the pelvis
    pred_j3d = pred_j3d - pred_pelvis
    target_j3d = target_j3d - target_pelvis
    pred_verts = pred_verts - pred_pelvis
    target_verts = target_verts - target_pelvis

    return (pred_j3d, target_j3d, pred_verts, target_verts)


def batch_compute_similarity_transform_torch(S1, S2):
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0, 2, 1)
        S2 = S2.permute(0, 2, 1)
        transposed = True
    assert S2.shape[1] == S1.shape[1]

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0], 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0, 2, 1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0, 2, 1)

    return S1_hat


def compute_error_accel(joints_gt, joints_pred, valid_mask=None, fps=None):
    # (F, J, 3) -> (F-2) per-joint
    accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]
    normed = torch.norm(accel_pred - accel_gt, dim=-1).mean(dim=-1)
    if fps is not None:
        normed = normed * fps**2

    if valid_mask is None:
        new_vis = torch.ones(len(normed)).to(dtype=torch.bool, device=joints_gt.device)
    else:
        invis = torch.logical_not(valid_mask)
        invis1 = torch.roll(invis, -1)
        invis2 = torch.roll(invis, -2)
        new_invis = torch.logical_or(invis, torch.logical_or(invis1, invis2))[:-2]
        new_vis = torch.logical_not(new_invis)
        if new_vis.sum() == 0:
            print("Warning!!! no valid acceleration error to compute.")

    return normed[new_vis]


def compute_camcoord_metrics(batch, pelvis_idxs=[1, 2], fps=30, mask=None):
    # All data is in camera coordinates
    pred_j3d = batch["pred_j3d"].cpu()  # (..., J, 3)
    target_j3d = batch["target_j3d"].cpu()
    pred_verts = batch["pred_verts"].cpu()
    target_verts = batch["target_verts"].cpu()

    if mask is not None:
        mask = mask.cpu()
        pred_j3d = pred_j3d[mask].clone()
        target_j3d = target_j3d[mask].clone()
        pred_verts = pred_verts[mask].clone()
        target_verts = target_verts[mask].clone()
    assert "mask" not in batch

    # Align by pelvis
    pred_j3d, target_j3d, pred_verts, target_verts = batch_align_by_pelvis(
        [pred_j3d, target_j3d, pred_verts, target_verts], pelvis_idxs=pelvis_idxs
    )

    # Metrics
    m2mm = 1000
    S1_hat = batch_compute_similarity_transform_torch(pred_j3d, target_j3d)
    pa_mpjpe = compute_jpe(S1_hat, target_j3d) * m2mm
    mpjpe = compute_jpe(pred_j3d, target_j3d) * m2mm
    pve = compute_jpe(pred_verts, target_verts) * m2mm
    accel = compute_error_accel(joints_pred=pred_j3d, joints_gt=target_j3d, fps=fps)

    camcoord_metrics = {
        "pa_mpjpe": pa_mpjpe,
        "mpjpe": mpjpe,
        "pve": pve,
        "accel": accel,
    }
    return camcoord_metrics


def compute_local_metrics(pred_j3d, target_j3d, pred_verts, target_verts, pelvis_idxs=[1, 2], fps=30):
    # Align by pelvis
    pred_j3d, target_j3d, pred_verts, target_verts = batch_align_by_pelvis(
        [pred_j3d, target_j3d, pred_verts, target_verts], pelvis_idxs=pelvis_idxs
    )
    m2mm = 1000
    S1_hat = batch_compute_similarity_transform_torch(pred_j3d, target_j3d)
    pa_mpjpe = compute_jpe(S1_hat, target_j3d) * m2mm
    # TODO: align by first frame
    mpjpe = compute_jpe(pred_j3d, target_j3d) * m2mm
    pve = compute_jpe(pred_verts, target_verts) * m2mm
    accel = compute_error_accel(joints_pred=pred_j3d, joints_gt=target_j3d, fps=fps)

    camcoord_metrics = {
        "pa_mpjpe": pa_mpjpe,
        "mpjpe": mpjpe,
        "pve": pve,
        "accel": accel,
    }
    return camcoord_metrics


def compute_global_metrics(smpl_skeleton, smpl_mesh, J_regressor, batch, mask=None, enable_timer=False):
    import time

    timings = {}

    def timer_start():
        if enable_timer:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            return time.perf_counter()
        return None

    def timer_end(name, start_time):
        if enable_timer and start_time is not None:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.perf_counter() - start_time
            timings[name] = elapsed

    # Step 1: Get pred joints from SMPL params
    # Step 2: Get pred vertices from SMPL params
    t0 = timer_start()
    pred_verts = get_vertices_from_smpl_params(smpl_mesh, batch["pred"])
    pred_verts_local = pred_verts["local_vertices"].squeeze()
    pred_verts_glob = pred_verts["global_vertices"].squeeze()
    pred_j3d_local = J_regressor[None] @ pred_verts_local
    pred_j3d_glob = J_regressor[None] @ pred_verts_glob
    timer_end("2_get_pred_vertices", t0)

    # Step 3: Get target joints/vertices
    t0 = timer_start()
    target_verts_glob = batch["gt"]["vertices"].squeeze()
    target_transl = batch["gt"]["trans"].squeeze()

    target_verts_local = target_verts_glob - target_transl[:, None]

    target_j3d_glob = J_regressor[None] @ target_verts_glob
    target_j3d_local = J_regressor[None] @ target_verts_local
    target_j3d_glob = target_j3d_glob[:, : pred_j3d_glob.shape[-2], :]
    target_j3d_local = target_j3d_local[:, : pred_j3d_local.shape[-2], :]

    timer_end("3_get_target_joints_verts", t0)

    # Step 4: Apply mask
    t0 = timer_start()
    # All data is in global coordinates
    if mask is not None:
        mask = mask
        pred_j3d_glob = pred_j3d_glob[mask].clone()
        target_j3d_glob = target_j3d_glob[mask].clone()
        pred_verts_glob = pred_verts_glob[mask].clone()
        target_verts_glob = target_verts_glob[mask].clone()
    assert "mask" not in batch
    timer_end("4_apply_mask", t0)

    seq_length = pred_j3d_glob.shape[0]

    # Step 5: Compute local metrics
    t0 = timer_start()
    local_metrics = compute_local_metrics(pred_j3d_local, target_j3d_local, pred_verts_local, target_verts_local)
    timer_end("5_compute_local_metrics", t0)

    # Step 6: Use chunk to compare (alignment metrics)
    t0 = timer_start()
    chunk_length = 100
    wa2_mpjpe, waa_mpjpe = [], []
    for start in range(0, seq_length, chunk_length):
        end = min(seq_length, start + chunk_length)

        target_j3d = target_j3d_glob[start:end].clone()
        pred_j3d = pred_j3d_glob[start:end].clone()

        w_j3d = first_align_joints(target_j3d, pred_j3d)
        wa_j3d = global_align_joints(target_j3d, pred_j3d)

        wa2_mpjpe.append(compute_jpe(target_j3d, w_j3d))
        waa_mpjpe.append(compute_jpe(target_j3d, wa_j3d))
    timer_end("6_chunk_alignment_metrics", t0)

    # Metrics
    t0 = timer_start()
    m2mm = 1000
    wa2_mpjpe = np.concatenate(wa2_mpjpe) * m2mm
    waa_mpjpe = np.concatenate(waa_mpjpe) * m2mm
    timer_end("7_concat_mpjpe", t0)

    # Step 8: Compute RTE
    t0 = timer_start()
    rte = compute_rte(target_j3d_glob[:, 0], pred_j3d_glob[:, 0]) * 1e2
    timer_end("8_compute_rte", t0)

    # Step 9: Compute jitter
    t0 = timer_start()
    jitter = compute_jitter(pred_j3d_glob, fps=30)
    timer_end("9_compute_jitter", t0)

    # Step 10: Compute foot sliding
    t0 = timer_start()
    if True:
        foot_sliding, cc_ratio, cd_ratio_01, cd_ratio_005 = compute_foot_sliding(target_verts_glob, pred_verts_glob)
    else:
        # NOTE: to check fs metrics derived from predicted end effector velocity, use this branch
        foot_sliding, cc_ratio, cd_ratio_01, cd_ratio_005 = compute_foot_sliding_by_pred_vel(
            target_j3d_glob, batch["pred"]["end_effector_vel"].squeeze()
        )
    foot_sliding = foot_sliding * m2mm
    cc_ratio = cc_ratio * 100
    cd_ratio_01 = cd_ratio_01 * 100
    cd_ratio_005 = cd_ratio_005 * 100
    timer_end("10_compute_foot_sliding", t0)

    # Print timing summary if enabled
    if enable_timer:
        print("\n" + "=" * 60)
        print("compute_global_metrics timing statistics:")
        print("=" * 60)
        total_time = sum(timings.values())
        for name, elapsed in timings.items():
            pct = (elapsed / total_time * 100) if total_time > 0 else 0
            print(f"  {name:35s}: {elapsed*1000:8.2f} ms ({pct:5.1f}%)")
        print("-" * 60)
        print(f"  {'Total':35s}: {total_time*1000:8.2f} ms")
        print("=" * 60 + "\n")

    global_metrics = {
        "wa2_mpjpe": wa2_mpjpe,
        "waa_mpjpe": waa_mpjpe,
        "rte": rte,
        "jitter": jitter,
        "fs": foot_sliding,
        "correct_contact": cc_ratio,
        "correct_dynamic_0.1": cd_ratio_01,
        "correct_dynamic_0.05": cd_ratio_005,
        "pred_j3d_glob": pred_j3d_glob,
    }
    global_metrics.update(local_metrics)
    return global_metrics


def compute_2dkp_metrics(smpl_skeleton, smpl_mesh, J_regressor, J_regressor_25, batch, enable_timer=False):
    import time

    import cv2

    from .vertex_ids import coco133tobody25, smpl_to_openpose

    timings = {}

    def timer_start():
        if enable_timer:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            return time.perf_counter()
        return None

    def timer_end(name, start_time):
        if enable_timer and start_time is not None:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.perf_counter() - start_time
            timings[name] = elapsed

    # Step 1: Get 3D joints from predicted SMPL params via J_regressor @ vertices
    t0 = timer_start()
    pred_verts = get_vertices_from_smpl_params(smpl_mesh, batch["pred"])
    pred_verts_glob = pred_verts["global_vertices"]  # (bs*F, 6890, 3)

    pred_j3d = torch.einsum("jv,bvc->bjc", J_regressor_25, pred_verts_glob)  # (bs*F, num_joints, 3) 25,3

    # Reshape to (bs, F, num_joints, 3)
    rot6d = batch["pred"]["rot6d"]
    bs, num_frames = rot6d.shape[:2]
    num_regressor_joints = pred_j3d.shape[1]
    pred_j3d = pred_j3d.reshape(bs, num_frames, num_regressor_joints, 3)

    timer_end("1_get_pred_joints", t0)

    # Step 2: Extract GT 2D keypoints and convert to Body25 format
    gt_kp2d_coco133 = batch["gt"]["keypoints3d"]  # (bs, F, 133, 3)
    K_mat = batch["gt"]["K"]  # (bs, 3, 3)

    bs, num_frames, num_kps, _ = gt_kp2d_coco133.shape

    # Convert COCO133 to Body25 for each batch and frame
    t0 = timer_start()
    gt_kp2d_body25_list = []
    for b in range(bs):
        # coco133tobody25 expects (F, 133, 3)
        kp_np = gt_kp2d_coco133[b].cpu().numpy()
        kp_body25 = coco133tobody25(kp_np)
        gt_kp2d_body25_list.append(kp_body25)
    gt_kp2d_body25 = np.stack(gt_kp2d_body25_list, axis=0) # (bs, F, 25, 3)
    timer_end("2_convert_to_body25", t0)

    gt_xy = gt_kp2d_body25[..., :2] # (bs, F, 25, 2)
    gt_conf = gt_kp2d_body25[..., 2] # (bs, F, 25)

    # Step 3: PnP alignment and projection for each sample and frame
    t0 = timer_start()

    all_proj_kp2d = []
    all_pnp_success = []

    K_mat_np = K_mat.cpu().numpy().astype(np.float64)
    K_per_frame = K_mat_np.ndim == 4 # (bs, F, 3, 3)

    for b in range(bs):
        sample_proj = []
        sample_success = []

        for f in range(num_frames):
            # Get camera intrinsics for current frame
            if K_per_frame:
                K = np.ascontiguousarray(K_mat_np[b, f])
            else:
                K = np.ascontiguousarray(K_mat_np[b])

            # Get corresponding 2D (Body25) and 3D (SMPL regressor joints) points
            pts_2d = gt_xy[b, f].astype(np.float64)
            pts_3d = pred_j3d[b, f].cpu().numpy().astype(np.float64)
            conf = gt_conf[b, f]

            # Filter by confidence
            valid_mask = conf > 0.3
            if valid_mask.sum() < 4:
                # Not enough points for PnP, use all points with conf > 0
                valid_mask = conf > 0

            if valid_mask.sum() < 4:
                # Still not enough, skip this frame
                sample_proj.append(np.zeros((25, 2)))
                sample_success.append(False)
                continue

            pts_2d_valid = np.ascontiguousarray(pts_2d[valid_mask])
            pts_3d_valid = np.ascontiguousarray(pts_3d[valid_mask])

            # Solve PnP with RANSAC
            dist_coeffs = np.zeros(4, dtype=np.float64)
            success, rvec, tvec = cv2.solvePnP(pts_3d_valid, pts_2d_valid, K, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)

            if success:
                # Project all 25 Body25 joints (mapped from SMPL regressor joints) to 2D
                proj_pts, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, dist_coeffs)
                proj_pts_body25 = proj_pts.reshape(-1, 2)

                sample_proj.append(proj_pts_body25)
                sample_success.append(True)
            else:
                sample_proj.append(np.zeros((25, 2)))
                sample_success.append(False)

        all_proj_kp2d.append(np.stack(sample_proj, axis=0))
        all_pnp_success.append(sample_success)

    proj_kp2d = np.stack(all_proj_kp2d, axis=0) # (bs, F, 25, 2)
    pnp_success = np.array(all_pnp_success) # (bs, F)

    timer_end("3_pnp_alignment", t0)

    # Step 4: Compute metrics for all 25 Body25 joints
    t0 = timer_start()

    errors = []
    pck_05_list = []
    pck_10_list = []
    pck_20_list = []

    # Pre-compute bbox sizes for all frames
    bbox_sizes = np.zeros((bs, num_frames))
    for b in range(bs):
        for f in range(num_frames):
            visible_kps = gt_conf[b, f] > 0.3
            if visible_kps.sum() > 1:
                visible_xy = gt_xy[b, f, visible_kps]
                bbox_sizes[b, f] = max(
                    visible_xy[:, 0].max() - visible_xy[:, 0].min(), visible_xy[:, 1].max() - visible_xy[:, 1].min()
                )

    for body25_idx in list(range(25)):
        pred_2d = proj_kp2d[:, :, body25_idx, :] # (bs, F, 2)
        gt_2d = gt_xy[:, :, body25_idx, :] # (bs, F, 2)
        conf = gt_conf[:, :, body25_idx] # (bs, F)

        # Compute per-joint error
        err = np.sqrt(((pred_2d - gt_2d) ** 2).sum(axis=-1)) # (bs, F)

        # Mask by confidence and PnP success
        valid = (conf > 0.3) & pnp_success

        if valid.sum() > 0:
            errors.append(err[valid])

            # Compute PCK with pre-computed bbox sizes
            for b in range(bs):
                for f in range(num_frames):
                    if valid[b, f] and bbox_sizes[b, f] > 0:
                        normalized_err = err[b, f] / bbox_sizes[b, f]
                        pck_05_list.append(normalized_err < 0.05)
                        pck_10_list.append(normalized_err < 0.1)
                        pck_20_list.append(normalized_err < 0.2)

    timer_end("4_compute_metrics", t0)

    # Aggregate metrics
    if len(errors) > 0:
        all_errors = np.concatenate(errors)
        reproj_error = float(all_errors.mean()) # unit: pixels
    else:
        reproj_error = float("nan")

    # PCK in percentage (0-100%)
    pck_05 = float(np.mean(pck_05_list)) * 100 if len(pck_05_list) > 0 else float("nan")
    pck_10 = float(np.mean(pck_10_list)) * 100 if len(pck_10_list) > 0 else float("nan")
    pck_20 = float(np.mean(pck_20_list)) * 100 if len(pck_20_list) > 0 else float("nan")

    kp2d_metrics = {
        "reproj_error_px": reproj_error,  # Mean Per Joint Reprojection Error (pixels)
        "pck@0.05 ": pck_05,  # Percentage of Correct Keypoints @ threshold 0.05
        "pck@0.1 ": pck_10,
        "pck@0.2 ": pck_20,
        "pnp_success_rate (%)": float(pnp_success.mean()) * 100,
    }

    if enable_timer:
        print("Timings:", timings)

    return kp2d_metrics, proj_kp2d, gt_xy, gt_conf
