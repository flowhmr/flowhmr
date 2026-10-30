"""Bounded palm-only contact trajectories, preserving torso and solved feet."""
from types import SimpleNamespace
import time
import numpy as np
from .correction_settings import PRESETS

class Window:
    def __init__(self,context,root,quat):
        from .correction import fk
        self.context=context;self.joints,self.global_r=fk(context.geo.rest,root,quat)
    def hand_predictor(self,h,*unused):
        offset=np.einsum('tij,j->ti',self.global_r[:,20+h],self.context.offsets[h]);point=self.joints[:,20+h]+offset
        rest=self.context.geo.rest
        def evaluate(globals,positions,elbow_local):
            wrist=positions[2]+np.einsum('tij,j->ti',globals[2],rest[20+h]-rest[18+h]);palm=wrist+offset
            return palm,palm[:,1,None]
        return SimpleNamespace(source_sample=point,evaluate=evaluate)

class Tracks:
    def __init__(self,points,ranges,floor,mode):
        self.ranges=ranges;self.phases=[[],[]]
        for h,track in enumerate(ranges):
            for k,(a,b) in enumerate(track):
                anchor=points[a:b,h].mean(0)
                anchor[1]=floor if mode=='flat' else float(np.median(points[a:b,h,1]))
                self.phases[h].append(dict(start=a,end=b,stance=k,kind='palm',anchor=anchor.tolist()))
    def window(self,h,a,b,stance_weights=None):
        phases=self.phases[h];which=np.clip(np.searchsorted([p['start'] for p in phases],np.arange(a,b),side='right')-1,0,len(phases)-1);target=np.array([phases[k]['anchor'] for k in which])
        if stance_weights is not None:
            weights=np.zeros(b-a);mixed=np.zeros((b-a,3))
            for k,w in stance_weights.items():
                p=next(p for p in phases if p['stance']==k);weights+=w;mixed+=np.asarray(p['anchor'])*w[:,None]
            valid=weights>0;target[valid]=mixed[valid]/weights[valid,None]
        return None,None,target

def apply(rest,source_root,source_quat,root,quat,offsets,fps,mode,floor,preset,manual=None,enabled=False,emit=lambda *a,**k:None):
    from .hand_lock import samples,plan
    from .contact_arm_plan import solve,source_reach_gap
    from .foot_contacts import normalize_ranges
    started=time.perf_counter();n=len(root)
    empty=dict(segments=[],automatic=[[],[]],contacts=0,warning='',method='Finite palm-only source-relative contact trajectories')
    if not enabled:return empty
    if n<3:return dict(empty,warning='Short clip: original hand pose retained. Temporal hand correction needs at least three frames.')
    if preset not in PRESETS or mode not in ('flat','multi_level'):raise ValueError('Unsupported hand correction settings')
    if np.shares_memory(source_quat,quat):source_quat=source_quat.copy()
    points=np.empty((n,2,3));joints=np.empty((n,22,3))
    for a in range(0,n,256):
        b=min(n,a+256);points[a:b],joints[a:b],_=samples(rest,root[a:b],source_quat[a:b],offsets)
    active,_,segments,automatic=plan(points,joints,fps,mode,floor,manual,preset)
    ranges=normalize_ranges([[ [s['start'],s['end']] for s in segments if s['foot']==h] for h in (0,1)],n)
    if not active.any():return dict(empty,automatic=automatic,warning='' if manual is not None else 'No confident automatic hand contacts. Add hand lock intervals in the timeline if needed.')
    context=SimpleNamespace(geo=SimpleNamespace(rest=np.asarray(rest)),quat=source_quat,offsets=np.asarray(offsets),strength=PRESETS[preset]['strength'],branch_cache={})
    context.asset=SimpleNamespace(window=lambda a,b,r,q:Window(context,r,q))
    tracks=Tracks(points,ranges,floor,mode);radius=max(2,round(fps*.2));halo=max(64,round(fps*.4));extra=max(10,round(fps*.3)+1)
    planned=np.convolve(active.any(1).astype(int),np.ones(2*radius+1,int),'same')>0;cores=[(s,min(n,s+256)) for s in range(0,n,256) if planned[s:min(n,s+256)].any()]
    arms=[13,14,16,17,18,19,20,21];tile_reports=[]
    for index,(s,e) in enumerate(cores):
        a=max(0,s-halo);b=min(n,e+halo);used=np.flatnonzero(planned[a:b]);end=min(b,a+int(used[-1])+extra+1);a=max(a,a+int(used[0])-extra);b=end
        q,report=solve(context,root[a:b],source_quat[a:b],ranges,tracks,fps,first=a);lo=max(s,a);hi=min(e,b);quat[lo:hi,arms]=q[lo-a:hi-a,arms]
        tile_reports.append(dict(core=[s,e],input=[a,b],report=report));emit('progress',stage='correct',current=index+1,total=len(cores),phase='palm_contacts',message='Motion Correction: planning smooth palm contacts')
    after=np.empty_like(points);rotation_error=0.
    from scipy.spatial.transform import Rotation
    for a in range(0,n,256):
        b=min(n,a+256);after[a:b],_,g=samples(rest,root[a:b],quat[a:b],offsets);_,_,old=samples(rest,root[a:b],source_quat[a:b],offsets);rotation_error=max(rotation_error,float(np.rad2deg(Rotation.from_matrix((old[:,[20,21]].swapaxes(-1,-2)@g[:,[20,21]]).reshape(-1,3,3)).magnitude()).max()))
    targets=points.copy();published=[]
    for h,phases in enumerate(tracks.phases):
        for p in phases:
            a,b=p['start'],p['end'];anchor=p['anchor'];targets[a:b,h]=anchor;published.append(dict(foot=h,start=a,end=b,anchor=anchor))
    error=np.linalg.norm(after-targets,axis=2);maximum=float(error[active].max());warnings=[]
    full_strength=PRESETS[preset]['strength']>=.999
    limited=0
    for a in range(0,n,256):
        b=min(n,a+256);_,_,global_r=samples(rest,root[a:b],source_quat[a:b],offsets)
        local=Rotation.from_quat(source_quat[a:b,...,[1,2,3,0]].reshape(-1,4)).as_matrix().reshape(b-a,22,3,3)
        for h in (0,1):
            gap=source_reach_gap(rest,joints[a:b],global_r,local,h,targets[a:b,h],points[a:b,h])
            limited+=int(((gap>.005)&active[a:b,h]).sum())
    if full_strength and maximum>.01:warnings.append('Some palm targets could not be fully reached; pose changes were limited.')
    def speed(p):
        mask=active[1:]&active[:-1];v=np.linalg.norm(np.diff(p,axis=0),axis=2)*fps
        return float(np.percentile(v[mask],95)) if mask.any() else 0.
    return dict(segments=published,automatic=automatic,contacts=int(active.sum()),warning=' '.join(warnings),full_lock_palm_error_max_m=maximum if full_strength else 0.,partial_lock_palm_error_max_m=maximum if not full_strength else 0.,palm_speed_before_p95_mps=speed(points),palm_speed_after_p95_mps=speed(after),wrist_rotation_max_deg=rotation_error,limited_frames=limited,requested_timings_preserved=True,temporal_tiles=tile_reports,seconds=time.perf_counter()-started,method=empty['method'])
