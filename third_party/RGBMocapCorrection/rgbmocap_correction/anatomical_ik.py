"""Source-preserving, fixed anatomical-axis correction for two leg chains.

Only the corrective knee increment is constrained here: existing inferred
SMPL-X rotations are never rewritten into a different neutral skeleton.
Export fitting separately enforces its complete signed one-axis hinges.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .mixamo_rig import align


def reach_lengths(rest,world):
    """Maximum physical reach with the source's non-hinge components fixed."""
    lengths=[]
    for hip,knee,ankle in ((1,4,7),(2,5,8)):
        upper=rest[knee]-rest[hip];lower=rest[ankle]-rest[knee]
        axis=align([0,-1,0],upper)@np.array([1.,0,0])
        local=world[:,hip].swapaxes(-1,-2)@world[:,knee]
        v=np.einsum('tij,j->ti',local,lower)
        perpendicular=v-(v@axis)[:,None]*axis
        radius=np.hypot(perpendicular@upper,np.cross(axis,v)@upper)
        constant=(upper@axis)*(v@axis)
        lengths.append(np.sqrt(np.maximum(0,np.dot(upper,upper)+np.dot(lower,lower)+2*(constant+radius))))
    return np.stack(lengths,axis=1)


def minimum_reach_lengths(rest,world):
    """Physical minimum reach at the native solver's 160 degree flex limit."""
    lengths=[]
    for hip,knee,ankle in ((1,4,7),(2,5,8)):
        upper=rest[knee]-rest[hip];lower=rest[ankle]-rest[knee]
        axis=align([0,-1,0],upper)@np.array([1.,0,0])
        local=world[:,hip].swapaxes(-1,-2)@world[:,knee]
        v=np.einsum('tij,j->ti',local,lower)
        perpendicular=v-(v@axis)[:,None]*axis
        radius=np.hypot(perpendicular@upper,np.cross(axis,v)@upper)
        constant=(upper@axis)*(v@axis)
        lengths.append(np.sqrt(np.maximum(0,np.dot(upper,upper)+np.dot(lower,lower)+
                                         2*(constant+radius*np.cos(np.deg2rad(160.))))))
    return np.stack(lengths,axis=1)


def solve_legs(rest, source_positions, source_world, roots, targets, active):
    from .correction import PARENTS
    n=len(roots);local=source_world.copy()
    for j,p in enumerate(PARENTS):
        if p>=0:local[:,j]=source_world[:,p].swapaxes(-1,-2)@source_world[:,j]
    original=local.copy()
    root_shift=roots-source_positions[:,0]
    max_delta=0.;max_off_axis=0.;max_flex=0.
    for foot,(hip,knee,ankle) in enumerate(((1,4,7),(2,5,8))):
        ids=np.flatnonzero(active[:,foot])
        if not len(ids):continue
        upper=rest[knee]-rest[hip];lower=rest[ankle]-rest[knee]
        # Same canonical->source neutral calibration as the exported femur.
        calibration=align([0,-1,0],upper)
        axis=calibration@np.array([1.,0,0])
        v=np.einsum('tij,j->ti',original[ids,knee],lower)
        dot=v@axis
        perpendicular=v-dot[:,None]*axis
        a=perpendicular@upper;b=np.cross(axis,v)@upper
        constant=(upper@axis)*dot
        radius=np.hypot(a,b)
        origin=source_positions[ids,hip]+root_shift[ids]
        delta=targets[ids,foot]-origin
        distance=np.linalg.norm(delta,axis=1)
        desired=(distance**2-np.dot(upper,upper)-np.dot(lower,lower))*.5-constant
        # The positive branch flexes around the calibrated anatomical axis.
        # Unlike cross(hip-knee, knee-ankle), rest curvature cannot change it.
        phase=np.arctan2(b,a)
        flex=np.minimum(np.arccos(np.clip(desired/np.maximum(radius,1e-12),-1,1)),np.deg2rad(160))
        correction=phase+flex
        bend=Rotation.from_rotvec(correction[:,None]*axis).as_matrix()
        knee_local=bend@original[ids,knee]
        end_vector=upper+np.einsum('tij,j->ti',knee_local,lower)
        hip_world=source_world[ids,hip].copy()
        for t,i in enumerate(ids):
            hip_world[t]=align(hip_world[t]@end_vector[t],delta[t])@hip_world[t]
        local[ids,hip]=source_world[ids,PARENTS[hip]].swapaxes(-1,-2)@hip_world
        local[ids,knee]=knee_local
        # Ankle rotation controls the foot only, never incoming shin roll.
        local[ids,ankle]=(hip_world@knee_local).swapaxes(-1,-2)@source_world[ids,ankle]
        rv=Rotation.from_matrix(bend).as_rotvec()
        off=rv-(rv@axis)[:,None]*axis
        max_off_axis=max(max_off_axis,float(np.linalg.norm(off,axis=1).max()))
        max_delta=max(max_delta,float(np.abs(correction).max()))
        max_flex=max(max_flex,float(flex.max()))
    if max_off_axis>1e-7 or not np.isfinite(local).all():
        raise ValueError('Anatomical knee correction failed its axis validation')
    return local,dict(max_added_knee_off_axis_deg=float(np.rad2deg(max_off_axis)),
                      max_added_knee_rotation_deg=float(np.rad2deg(max_delta)),
                      max_corrected_knee_flex_deg=float(np.rad2deg(max_flex)))
