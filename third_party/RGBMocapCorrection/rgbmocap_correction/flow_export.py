"""Atomic native SMPL-H surface + anatomical Humanoid52 export preparation."""


import numpy as np

def proper_rotations(matrices):
    """Remove only FP32 roundoff before long FK chains / BVH Euler encoding."""
    from scipy.spatial.transform import Rotation
    matrices=np.asarray(matrices,dtype=float)
    if not np.isfinite(matrices).all() or np.max(np.abs(matrices.swapaxes(-1,-2)@matrices-np.eye(3)))>2e-4 or np.min(np.linalg.det(matrices))<.9998:
        raise ValueError('FlowHMR produced invalid source rotation frames')
    return Rotation.from_matrix(matrices.reshape(-1,3,3)).as_matrix().reshape(matrices.shape)
