"""Independent kinematic trajectory planning, not physics or a second leg solve.

The native solver remains the sole publisher of hip/knee/ankle rotations.
This module predicts its response while planning root Y and swing-target Y.
Contact labels, stance targets and ankle frames are fixed. Free trajectories
follow the final planned root at their finite transition boundaries.
"""
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation

CONFIG = dict(root_free=True, hip_weight=0.5,
              target_limit=.02, target_sigma=.02, root_sigma=.01,
              pose_sigma=.175, velocity_sigma=.025, acceleration_sigma=.002,
              root_acceleration_sigma=.0004, iterations=60)

def _optimise_window(rest, root, quat, probes, shift, target_shift, offsets, active,
             transition, flight, floor, fps, mode="flat"):
    from .correction import fk
    from .anatomical_ik import reach_lengths, minimum_reach_lengths
    from .mixamo_rig import align
    from . import contact_temporal_root as temporal_root
    start = time.perf_counter()
    cfg = CONFIG.copy()
    n = len(root)
    # Use the bounded temporal root plan as the warm start.
    base = shift.copy()
    warm, previous = temporal_root.optimise(rest, root, quat, probes, shift,
        target_shift, offsets, active, transition, flight, floor, fps, mode=mode)
    if not active.any():
        return warm, previous
    j, g = fk(rest, root, quat)
    from .contact_trajectory import free_follow_weights
    follow = free_follow_weights(active, transition)
    reference_shift = np.broadcast_to(target_shift[:, None], (n, 2, 3)).copy()
    planned = (active | transition).copy()
    targets = j[:, [7, 8]] + target_shift[:, None] + offsets
    height = probes[..., 1].reshape(n, 2, 2).min(2)
    expected = height + targets[..., 1] - j[:, [7, 8], 1]
    targets[..., 1] += np.where(active,
        np.minimum(np.maximum(floor - expected, 0), .12), 0) if mode=="flat" else 0
    # Admit every free frame that could need clearance inside the bounded
    # root search. The smooth unilateral constraint is evaluated at each
    # trial root, rather than switching a 1 cm correction on after planning.
    original_planned = planned.copy()
    free_base = j[:, [7, 8]] + warm[:, None]
    targets = np.where(planned[..., None], targets, free_base)
    follow[~planned] = 1.
    reference_shift[~planned] = np.broadcast_to(warm[:, None], (n, 2, 3))[~planned]
    possible_height = height + np.minimum(targets[..., 1]-j[:, [7, 8], 1], base[:, 1, None]-.02)
    guard = ~active & ~flight[:, None] & (possible_height < floor+.01) & (mode=="flat")
    planned |= guard
    movable = original_planned & ~active & ~flight[:, None]
    ti, fi = np.nonzero(movable)
    m = len(ti)
    # A finite approach/release parameter must vanish at its free boundary.
    target_gain = 1.-follow[ti, fi]
    distance_xz = ((j[:, [1, 2]][..., [0, 2]] + base[:, None, [0, 2]] -
                    targets[..., [0, 2]]) ** 2).sum(2)
    high_reach = reach_lengths(rest, g) * .9995
    low_reach = minimum_reach_lengths(rest, g) * 1.0005
    high = targets[..., 1] + np.sqrt(np.maximum(high_reach**2-distance_xz, 0)) - j[:, [1, 2], 1]
    low = targets[..., 1] + np.sqrt(np.maximum(low_reach**2-distance_xz, 0)) - j[:, [1, 2], 1]
    bound_active = active & (j[:, [1, 2], 1] + warm[:, 1, None] > targets[..., 1])
    lo = np.maximum(base[:, 1] - .02, np.where(bound_active, low, -np.inf).max(1)).astype('float64')
    hi = np.minimum(base[:, 1] + .02, np.where(bound_active, high, np.inf).min(1)).astype('float64')
    freeze = (~planned.any(1)) | (flight & ~active.any(1)) | (hi-lo < 1e-7)
    if not cfg['root_free']:
        freeze[:] = True
    fixed = warm[:, 1].astype('float64')
    lo[freeze] = fixed[freeze] - 1e-10
    hi[freeze] = fixed[freeze] + 1e-10
    # Keep the existing 2 cm source-relative budget. Flat-floor clearance is
    # applied continuously in predict(), after the bounded target adjustment.
    tlo = np.full(m, -cfg['target_limit'])
    thi = np.full(m, cfg['target_limit'])
    lower = np.r_[lo, tlo]
    upper = np.r_[hi, thi]
    x0 = np.r_[np.clip(warm[:, 1], lo+1e-11, hi-1e-11),
                np.clip(np.zeros(m), tlo+1e-11, thi-1e-11)]
    axes = np.stack([align([0,-1,0], rest[k]-rest[h]) @ np.array([1.,0,0])
                     for h,k in [(1,4),(2,5)]])
    upper_vectors = rest[[4,5]] - rest[[1,2]]
    lower_vectors = rest[[7,8]] - rest[[4,5]]
    local_knee = g[:, [1,2]].swapaxes(-1,-2) @ g[:, [4,5]]
    v = np.einsum('tfij,fj->tfi', local_knee, lower_vectors)
    dot = np.einsum('tfi,fi->tf', v, axes)
    perp = v - dot[..., None]*axes
    aa = np.einsum('tfi,fi->tf', perp, upper_vectors)
    bb = np.einsum('tfi,fi->tf', np.cross(axes, v), upper_vectors)
    radius = np.maximum(np.hypot(aa,bb), 1e-12)
    phase = np.arctan2(bb,aa)
    constant = (upper_vectors**2).sum(1) + (lower_vectors**2).sum(1) + 2*np.einsum('fi,fi->f', upper_vectors, axes)*dot
    # Near collinearity the source's signed bend is not observable. The native
    # solve uses its anatomical pole there; do not make root Y chase the tiny
    # alternating source sign while predicting that solve.
    extension_support = np.clip((np.deg2rad(5)-abs(phase))/np.deg2rad(4),0,1)
    hips = j[:, [1,2]] + base[:, None]
    hips[..., 1] = j[:, [1,2], 1]
    hip_world = g[:, [1,2]]
    # Relative SO(3) increments avoid the +/-pi branch of absolute rotvecs.
    src = Rotation.from_quat(quat[:, [1,2,4,5]][...,[1,2,3,0]].reshape(-1,4)).as_matrix().reshape(n,4,3,3)
    speed = np.zeros((n,2))
    inc = Rotation.from_matrix((src[:-1].swapaxes(-1,-2) @ src[1:]).reshape(-1,3,3)).magnitude().reshape(n-1,4)*fps
    speed[1:] = np.maximum(inc[:, :2], inc[:, 2:])
    motion_weight = .25 + .75/(1+(speed/np.deg2rad(20))**2)
    rate = fps/30.
    # Include joins to the unmodified swing trajectory, unlike a mask that
    # drops every temporal residual exactly where the solve becomes active.
    valid1 = (planned[1:] | planned[:-1]) & ~(flight[1:,None] | flight[:-1,None])
    valid2 = (planned[2:] | planned[1:-1] | planned[:-2]) & ~(flight[2:,None] | flight[1:-1,None] | flight[:-2,None])
    w1 = valid1*np.sqrt(np.minimum(motion_weight[1:],motion_weight[:-1]))*rate*.5/cfg['velocity_sigma']
    w2 = valid2*np.sqrt(np.minimum(np.minimum(motion_weight[2:],motion_weight[1:-1]),motion_weight[:-2]))*rate**2*.5/cfg['acceleration_sigma']
    channel = np.array([1., cfg['hip_weight'],cfg['hip_weight'],cfg['hip_weight']])

    def predict(x, return_geometry=False):
        # Keep finite-difference trial values in float64; storing them in the
        # float32 input shift would quantize away root derivatives.
        dynamic_shift = base.astype(np.float64, copy=True)
        dynamic_shift[:, 1] = x[:n]
        goal = targets + follow[..., None]*(dynamic_shift[:, None]-reference_shift)
        goal[ti,fi,1] += x[n:]*target_gain
        if mode == 'flat':
            gap = floor-(height+goal[..., 1]-j[:, [7, 8], 1])
            lift = np.where(gap>=.01, gap, np.where(gap<=-.01, 0., (gap+.01)**2/.04))
            goal[..., 1] += np.where(planned & ~active & ~flight[:, None], np.minimum(lift,.12), 0.)
        origin = hips.copy()
        origin[...,1] += x[:n,None]
        delta = goal-origin
        squared = (delta**2).sum(2)
        co = (squared-constant)/(2*radius)
        flex = np.minimum(np.arccos(np.clip(co,-1,1)),np.deg2rad(160))
        angle = (phase+flex)*planned
        cv = np.cos(angle)[...,None]
        sv = np.sin(angle)[...,None]
        rotated = v*cv + np.cross(axes,v)*sv + axes*dot[...,None]*(1-cv)
        end = upper_vectors+rotated
        direction = np.einsum('tfij,tfj->tfi',hip_world,end)
        a = direction/np.maximum(np.linalg.norm(direction,axis=2,keepdims=True),1e-12)
        b = delta/np.maximum(np.linalg.norm(delta,axis=2,keepdims=True),1e-12)
        cross = np.cross(a,b)
        sine = np.linalg.norm(cross,axis=2)
        theta = np.arctan2(sine,np.clip((a*b).sum(2),-1,1))
        world_rv = cross*(theta/np.maximum(sine,1e-12))[...,None]
        hip_rv = np.einsum('tfji,tfj->tfi',hip_world,world_rv)*planned[...,None]
        state = np.concatenate(((angle-phase*extension_support)[...,None],hip_rv),2)
        if return_geometry:
            return state, goal, delta, co
        return state

    def residual(x):
        state,goal,delta,co = predict(x,True)
        reach = np.sqrt((delta**2).sum(2))
        violation = (np.maximum(reach-high_reach,0) + np.minimum(reach-low_reach,0))*planned/.0005
        return np.r_[(x[:n]-base[:,1])/cfg['root_sigma'],x[n:]/cfg['target_sigma'],
            (state*channel/cfg['pose_sigma']).ravel(),
            (np.diff(state,axis=0)*w1[...,None]*channel).ravel(),
            (np.diff(state,n=2,axis=0)*w2[...,None]*channel).ravel(),
            np.diff(x[:n],n=2)*rate**2/cfg['root_acceleration_sigma'],violation.ravel()]

    # Sparse coloured finite differences: only neighbouring frame variables
    # affect a residual. There is no dense recording-length squared matrix.
    frame_vars = np.full((n,3),-1,int)
    frame_vars[:,0] = np.arange(n)
    frame_vars[ti,fi+1] = n+np.arange(m)
    rows=[];cols=[];cursor=0
    def block(frames, count, root_only=False):
        nonlocal cursor
        frames=np.asarray(frames)
        if not len(frames):
            return
        if frames.ndim==1:frames=frames[:,None]
        variables = frames if root_only else frame_vars[frames].reshape(len(frames),-1)
        for k in range(variables.shape[1]):
            col=variables[:,k];ok=col>=0
            rr=(cursor+np.arange(len(frames))[:,None]*count+np.arange(count)).repeat(1,axis=0)
            rows.append(rr[ok].ravel());cols.append(np.repeat(col[ok],count))
        cursor += len(frames)*count
    block(np.arange(n),1,True)
    rows.append(cursor+np.arange(m));cols.append(n+np.arange(m));cursor+=m
    block(np.arange(n),8)
    block(np.stack((np.arange(n-1),np.arange(1,n)),1),8)
    block(np.stack((np.arange(n-2),np.arange(1,n-1),np.arange(2,n)),1),8)
    block(np.stack((np.arange(n-2),np.arange(1,n-1),np.arange(2,n)),1),1,True)
    block(np.arange(n),2)
    sparsity=coo_matrix((np.ones(sum(map(len,rows))), (np.concatenate(rows),np.concatenate(cols))),shape=(cursor,n+m)).tocsr()
    before=residual(x0)
    # Column scaling improves conditioning without increasing the iteration
    # cap, changing the objective or introducing dense recording-sized work.
    result=least_squares(residual,x0,jac_sparsity=sparsity,bounds=(lower,upper),
        x_scale='jac',max_nfev=cfg['iterations'],ftol=2e-5,xtol=2e-6,gtol=1e-5,tr_solver='lsmr',
        tr_options={'maxiter':100})
    state,goal,_,_=predict(result.x,True)
    out=warm.copy();out[:,1]=result.x[:n]
    offsets[:] = goal-j[:,[7,8]]-target_shift[:,None]
    transition[:] = planned & ~active
    previous['contact_chain_plan'] = dict(config=cfg,seconds=time.perf_counter()-start,
        evaluations=int(result.nfev),success=bool(result.success),message=result.message,
        cost_before=float(before@before/2),cost_after=float(result.cost),
        root_freeze_frames=int(freeze.sum()),transition_variables=m,
        planned_free_foot_clearance_frames=int(guard.sum()),
        target_y_extra_max_m=float(abs(result.x[n:]*target_gain).max(initial=0)),
        trial_scaling='sparse Jacobian column norms',
        free_goal_basis='final planned root with finite C1 joins',
        root_extra_from_app_max_m=float(abs(result.x[:n]-base[:,1]).max()),
        source_speed_method='relative SO(3)',physics_simulation=False)
    return out,previous

def optimise(rest,root,quat,probes,shift,target_shift,offsets,active,transition,flight,floor,fps,mode='flat',emit=lambda *a,**k:None):
    """Fixed canonical windows with convex overlap; no second leg solve."""
    core=1024;halo=max(4,round(fps*4));n=len(root)
    if n<=core+2*halo:
        return _optimise_window(rest,root,quat,probes,shift,target_shift,offsets,active,transition,flight,floor,fps,mode)
    original_offsets=offsets.copy();original_transition=transition.copy();weight=np.zeros(n);summed=np.zeros((n,3));goals=np.zeros_like(offsets);new_transition=np.zeros_like(transition);reports=[]
    for start in range(0,n,core):
        end=min(n,start+core);a=max(0,start-halo);b=min(n,end+halo);local_offsets=original_offsets[a:b].copy();local_transition=original_transition[a:b].copy()
        value,report=_optimise_window(rest,root[a:b],quat[a:b],probes[a:b],shift[a:b],target_shift[a:b],local_offsets,active[a:b],local_transition,flight[a:b],floor,fps,mode)
        w=np.ones(b-a)
        if start>a:w[:start-a]=.5-.5*np.cos(np.linspace(0,np.pi,start-a))
        if b>end:w[end-a:]=.5+.5*np.cos(np.linspace(0,np.pi,b-end))
        summed[a:b]+=value*w[:,None];goals[a:b]+=local_offsets*w[:,None,None];weight[a:b]+=w;new_transition[a:b]|=local_transition
        reports.append(dict(core=[start,end],input=[a,b],report=report))
        emit('progress',stage='correct',phase='contact_plan',current=len(reports),total=(n+core-1)//core,message='Motion Correction: planning smooth contact transitions')
    assert (weight>0).all()
    offsets[:]=goals/weight[:,None,None];transition[:]=new_transition
    return (summed/weight[:,None]).astype(shift.dtype),dict(bounded_contact_plan=dict(core=core,halo=halo,max_input_frames=max(x['input'][1]-x['input'][0] for x in reports),windows=reports))
