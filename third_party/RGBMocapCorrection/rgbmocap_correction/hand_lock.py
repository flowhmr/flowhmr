"""Conservative palm contacts and source-relative arm-only correction.

No hand contact predictor is available. Automatic timing is geometric evidence,
not pressure or an object reconstruction; sparse user timing is authoritative.
The torso and the already solved legs are never moved by this pass.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation


def palm_offsets(shaped,weights,rest):
    offsets=[]
    for wrist,sign in ((20,1),(21,-1)):
        ids=np.flatnonzero((weights[:,wrist]>.7)&((shaped[:,0]-rest[wrist,0])*sign>.015))
        if len(ids)<4:raise ValueError('Registered palm surface is unavailable')
        ids=ids[np.argsort(shaped[ids,1])[:min(24,len(ids))]]
        offsets.append(shaped[ids].mean(0)-rest[wrist])
    return np.asarray(offsets)


def samples(rest,root,quat,offsets):
    from .correction import fk
    j,g=fk(rest,root,quat)
    return j[:,[20,21]]+np.einsum('thij,hj->thi',g[:,[20,21]],offsets),j,g


def plan(points,joints,fps,mode,floor,manual=None,preset='normal'):
    from .correction import intervals
    n=len(points);active=np.zeros((n,2),bool);segments=[]
    anchors=points.copy()
    if n<2:return active,anchors,segments,[[],[]]
    speed=np.linalg.norm(np.gradient(gaussian_filter1d(points,max(.5,fps*.04),axis=0,mode='nearest'),axis=0),axis=-1)*fps
    # A hand resting at the actor's side is not evidence of an external support.
    below=points[...,1]<joints[:,[16,17],1]-.12
    away=np.linalg.norm(points-joints[:,[1,2]],axis=-1)>.30
    active=(speed<(.10 if preset=='weak' else .18))&below&away
    if mode=='flat':active &= np.abs(points[...,1]-floor)<.15
    # Multi Level admits elevated supports below shoulder level, not stationary
    # overhead gestures. Overhead/rung contacts can be entered manually.
    for hand in (0,1):
        for s,e in intervals(~active[:,hand]):
            if s==0 or e==n or e-s>max(1,round(fps*.10)):continue
            if np.linalg.norm(points[e,hand]-points[s-1,hand])<.025 and speed[s:e,hand].max()<.35:
                active[s:e,hand]=True
        for s,e in intervals(active[:,hand]):
            if e-s<max(4,round(fps*.25)):active[s:e,hand]=False
    automatic=[[(int(s),int(e)) for s,e in intervals(active[:,h])] for h in (0,1)]
    if manual is not None:
        from .foot_contacts import normalize_ranges
        active[:]=False
        for h,track in enumerate(normalize_ranges(manual,n)):
            for s,e in track:active[s:e,h]=True
    for h in (0,1):
        for s,e in intervals(active[:,h]):
            # One pin per observed stance; never drag a pin frame by frame.
            anchor=np.median(points[s:e,h],axis=0)
            anchors[s:e,h]=anchor
            segments.append(dict(foot=h,start=int(s),end=int(e),anchor=anchor.tolist()))
    return active,anchors,segments,automatic


def solve(rest,root,quat,targets,active):
    """Fixed-length arms; calibrated elbow increment, nearest source branch."""
    from .correction import fk,PARENTS
    from .mixamo_rig import align
    j,g=fk(rest,root,quat)
    local=Rotation.from_quat(quat[...,[1,2,3,0]].reshape(-1,4)).as_matrix().reshape(-1,22,3,3)
    for h,(shoulder,elbow,wrist,sign) in enumerate(((16,18,20,1),(17,19,21,-1))):
        ids=np.flatnonzero(active[:,h])
        if not len(ids):continue
        u=rest[elbow]-rest[shoulder];v0=rest[wrist]-rest[elbow]
        axis=align([sign,0,0],u)@np.array([0.,-sign,0.])
        v=np.einsum('tij,j->ti',local[ids,elbow],v0)
        a=v@u-(v@axis)*(u@axis);b=np.cross(axis,v)@u
        c=(v@axis)*(u@axis);radius=np.hypot(a,b);phase=np.arctan2(b,a)
        delta=targets[ids,h]-j[ids,shoulder]
        desired=(np.sum(delta*delta,axis=1)-u@u-v0@v0)/2-c
        angle=np.arccos(np.clip(desired/np.maximum(radius,1e-12),-1,1))
        candidates=np.stack((phase+angle,phase-angle),axis=1)
        candidates=(candidates+np.pi)%(2*np.pi)-np.pi
        # Select the solution continuous with the source elbow, avoiding a
        # branch inversion. Bound added correction, never clip source flexion.
        correction=candidates[np.arange(len(ids)),np.argmin(np.abs(candidates),axis=1)]
        correction=np.clip(correction,-np.deg2rad(30),np.deg2rad(30))
        bend=Rotation.from_rotvec(correction[:,None]*axis).as_matrix()
        el=bend@local[ids,elbow]
        end=u+np.einsum('tij,j->ti',el,v0)
        upper=g[ids,shoulder].copy()
        for k,i in enumerate(ids):upper[k]=align(upper[k]@end[k],delta[k])@upper[k]
        local[ids,shoulder]=g[ids,PARENTS[shoulder]].swapaxes(-1,-2)@upper
        local[ids,elbow]=el
        local[ids,wrist]=(upper@el).swapaxes(-1,-2)@g[ids,wrist]
    out=quat.copy()
    for joint in (16,17,18,19,20,21):
        out[:,joint]=Rotation.from_matrix(local[:,joint]).as_quat()[:,[3,0,1,2]]
    return out


def apply(rest,source_root,source_quat,root,quat,offsets,fps,mode,floor,preset,manual=None,enabled=False,emit=lambda *a,**k:None):
    from .palm_contact import apply as apply_palm
    return apply_palm(rest,source_root,source_quat,root,quat,offsets,fps,mode,floor,preset,manual,enabled,emit)
