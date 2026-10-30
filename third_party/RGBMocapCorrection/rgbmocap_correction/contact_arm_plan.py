"""Palm contacts with finite source-relative arm/shoulder planning.

No body/root relocation, no dynamics, no leg solve. Registered source wrist
world frames and finger-local motion are retained. Shoulder-girdle depression
is bounded; the source-relative elbow increment uses the application's axis.
"""
import time
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation

CONFIG=dict(iterations=80,elbow_limit_deg=60.,collar_limit_deg=15.,swivel_limit_deg=10.,anchor_limit_m=.01,
 contact_sigma_m=.0007,source_elbow_sigma_m=.03,temporal_weight=.5,
 velocity_sigma=.025,acceleration_sigma=.004,actual_flex_acceleration_sigma=.003)

def source_reach_gap(rest,j,g,local,h,target,sample,collar_deg=15.,elbow_deg=60.):
 """Distance gap for the actual bounded collar circle and elbow axis.

 Derived only from input anatomy/pose/target, never from optimisation success.
 """
 from .mixamo_rig import align
 sign=1 if h==0 else -1;u=rest[18+h]-rest[16+h];v0=rest[20+h]-rest[18+h];axis=align([sign,0,0],u)@np.array([0.,-sign,0.]);v=np.einsum('tij,j->ti',local[:,18+h],v0);dot=v@axis
 aa=v@u-dot*(u@axis);bb=np.cross(axis,v)@u;constant=u@u+v0@v0+2*dot*(u@axis);bound=np.deg2rad(elbow_deg);phase=np.arctan2(bb,aa)
 theta=np.stack([np.full(len(v),-bound),np.full(len(v),bound),np.clip(phase,-bound,bound),np.clip((phase+2*np.pi)%(2*np.pi)-np.pi,-bound,bound),np.clip((phase+np.pi+np.pi)%(2*np.pi)-np.pi,-bound,bound)],1)
 lengths=np.sqrt(np.maximum(constant[:,None]+2*(aa[:,None]*np.cos(theta)+bb[:,None]*np.sin(theta)),0));low=lengths.min(1);high=lengths.max(1)
 offset=np.einsum('tij,j->ti',g[:,13+h],rest[16+h]-rest[13+h]);spin=np.cross(offset,[0.,-1,0]);norm=np.linalg.norm(spin,axis=1);spin/=np.maximum(norm[:,None],1e-12);desired=j[:,20+h]+target-sample-j[:,13+h];parallel=spin*np.einsum('ti,ti->t',spin,offset)[:,None];perp=offset-parallel
 ca=np.einsum('ti,ti->t',desired,perp);cb=np.einsum('ti,ti->t',desired,np.cross(spin,offset));phase=np.arctan2(cb,ca);bound=np.deg2rad(collar_deg)
 theta=np.stack([np.full(len(v),-bound),np.full(len(v),bound),np.clip(phase,-bound,bound),np.clip((phase+np.pi+np.pi)%(2*np.pi)-np.pi,-bound,bound)],1)
 squared=(desired*desired).sum(1)+(offset*offset).sum(1)-2*np.einsum('ti,ti->t',desired,parallel);distances=np.sqrt(np.maximum(squared[:,None]-2*(ca[:,None]*np.cos(theta)+cb[:,None]*np.sin(theta)),0));dmin=distances.min(1);dmax=distances.max(1);degenerate=norm<1e-9;dmin[degenerate]=dmax[degenerate]=np.linalg.norm(desired-offset,axis=1)[degenerate]
 return np.maximum(np.maximum(dmin-high,low-dmax),0)

def align_many(a,b):
 a=a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12);b=b/np.maximum(np.linalg.norm(b,axis=1,keepdims=True),1e-12)
 cross=np.cross(a,b);d=np.clip((a*b).sum(1),-1,1);K=np.zeros((len(a),3,3));K[:,0,1]=-cross[:,2];K[:,0,2]=cross[:,1];K[:,1,0]=cross[:,2];K[:,1,2]=-cross[:,0];K[:,2,0]=-cross[:,1];K[:,2,1]=cross[:,0]
 out=np.eye(3)+K+(K@K)/np.maximum(1+d,1e-10)[:,None,None]
 bad=d<(-1+1e-8)
 if bad.any():
  from .mixamo_rig import align
  for i in np.flatnonzero(bad):out[i]=align(a[i],b[i])
 return out

def solve(scene,root,foot_quat,ranges,tracks,fps,first=0,config=None):
 from .correction import fk
 from .mixamo_rig import align
 start=time.perf_counter();cfg={**CONFIG,**(config or {})};n=len(root);rest=scene.geo.rest
 source_quat=scene.quat[first:first+n];window=scene.asset.window(first,first+n,root,foot_quat)
 j,g=fk(rest[:22],root,source_quat);out=foot_quat.copy();reports=[];segments=[];active=np.zeros((n,2),bool)
 for h,track in enumerate(ranges):
  for index,(a,b) in enumerate(track):
   lo=max(first,a);hi=min(first+n,b)
   if hi>lo:active[lo-first:hi-first,h]=True;segments.append(dict(foot=h,start=lo-first,end=hi-first,stance=index,global_start=a,global_end=b))
 if n<3 or not active.any():return out,dict(contacts=int(active.sum()),segments=segments,seconds=time.perf_counter()-start,bypassed=True)
 local=Rotation.from_quat(source_quat[...,[1,2,3,0]].reshape(-1,4)).as_matrix().reshape(n,22,3,3)
 for h,(collar,shoulder,elbow,wrist,sign) in enumerate([(13,16,18,20,1),(14,17,19,21,-1)]):
  mask=active[:,h];hand_segments=[s for s in segments if s['foot']==h]
  if not mask.any():continue
  group=np.full(n,-1,int)
  for k,s in enumerate(hand_segments):group[s['start']:s['end']]=k
  influence=mask.astype('float64');radius_frames=max(2,round(fps*.20))
  for k,s in enumerate(hand_segments):
   for edge,direction in [(s['start'],-1),(s['end']-1,1)]:
    for distance in range(1,radius_frames+1):
     index=edge+direction*distance
     if index<0 or index>=n or mask[index]:continue
     strength=.5*(1+np.cos(np.pi*distance/(radius_frames+1)))
     if strength>influence[index]:influence[index]=strength;group[index]=k
  planned=influence>0
  # Track each finite contact envelope's owning requested support. Never use
  # a previous stance's anchor while approaching a new one. Suppress nearby
  # halos continuously as one episode reaches full strength.
  envelopes=[];owners=[]
  for segment in hand_segments:
   s,e=segment['start'],segment['end'];w=np.zeros(n);w[s:e]=1
   for edge,direction in [(s,-1),(e-1,1)]:
    for distance in range(1,radius_frames+1):
     index=edge+direction*distance
     if 0<=index<n:w[index]=max(w[index],.5*(1+np.cos(np.pi*distance/(radius_frames+1))))
   owners.append(next(i for i,(a,b) in enumerate(tracks.ranges[h]) if a<=segment['global_start']<b));envelopes.append(w)
  env=np.stack(envelopes,1);support_weights={}
  for k,owner in enumerate(owners):
   others=np.delete(env,k,axis=1);w=env[:,k]*np.prod(1-others,axis=1)
   support_weights[owner]=support_weights.get(owner,np.zeros(n))+w
  material,material_weights,support_target=tracks.window(h,first,first+n,stance_weights=support_weights)
  predictor=window.hand_predictor(h,material,material_weights);src_sample=predictor.source_sample
  u=rest[elbow]-rest[shoulder];v0=rest[wrist]-rest[elbow]
  # Geometry-only reach evidence; do not force an impossible target by
  # spending the entire allowed elbow/collar range. Keep the original target
  # for diagnostics, but soften added motion smoothly when it is unreachable.
  reach_gap=source_reach_gap(rest,j,g,local,h,support_target,src_sample,cfg['collar_limit_deg'],cfg['elbow_limit_deg'])
  reach_gain=gaussian_filter1d(1/(1+(reach_gap/.03)**4),max(.5,fps*.08),truncate=3.,mode='nearest')
  influence*=reach_gain*getattr(scene,'strength',1.)
  axis=align([sign,0,0],u)@np.array([0.,-sign,0.])
  v=np.einsum('tij,j->ti',local[:,elbow],v0);dot=v@axis
  aa=v@u-dot*(u@axis);bb=np.cross(axis,v)@u;radius=np.maximum(np.hypot(aa,bb),1e-12);phase=np.arctan2(bb,aa);constant=u@u+v0@v0+2*dot*(u@axis)
  branch=np.ones(n)
  for k,segment in enumerate(hand_segments):
   # Calibrate one branch from the entire immutable source stance, so window
   # order and partial views never choose a different branch.
   a,b=segment['global_start'],segment['global_end']
   cache=getattr(scene,'branch_cache',None);key=(h,a,b)
   indexes=np.linspace(a,b-1,min(128,b-a),dtype=int)
   full_local=Rotation.from_quat(scene.quat[indexes,elbow][:,[1,2,3,0]]).as_matrix()
   observed_v=np.einsum('tij,j->ti',full_local,v0);observed_dot=observed_v@axis
   observed=np.arctan2(np.cross(axis,observed_v)@u,observed_v@u-observed_dot*(u@axis));reliable=abs(observed)>np.deg2rad(8)
   sign_branch=(cache[key] if cache is not None and key in cache else (1 if np.median(observed[reliable] if reliable.any() else observed)<=0 else -1))
   if cache is not None:cache[key]=sign_branch
   branch[group==k]=sign_branch
  collar_offset=rest[shoulder]-rest[collar];world_offset=np.einsum('tij,j->ti',g[:,collar],collar_offset)
  depression_axis=np.cross(world_offset,np.array([0.,-1,0]));depression_axis/=np.maximum(np.linalg.norm(depression_axis,axis=1,keepdims=True),1e-12)
  src_flex=np.arccos(np.clip((u*v).sum(1)/np.linalg.norm(u)/np.linalg.norm(v0),-1,1))
  rel=local[:-1,elbow].swapaxes(-1,-2)@local[1:,elbow];speed=np.r_[0,Rotation.from_matrix(rel).magnitude()*fps];motion=.25+.75/(1+(speed/np.deg2rad(30))**2)
  # Variables per frame: collar depression, arm swivel, 3D wrist target trim.
  initial_target=j[:,wrist].copy();initial_target[planned]+=(support_target[planned]-src_sample[planned])*influence[planned,None]
  initial_distance=np.linalg.norm(initial_target-j[:,shoulder],axis=1)
  required=(phase+branch*np.arccos(np.clip((initial_distance**2-constant)/(2*radius),-1,1))+np.pi)%(2*np.pi)-np.pi
  expanded_bounds=bool((abs(required[mask])>np.deg2rad(30)).any())
  elbow_limit=cfg['elbow_limit_deg'] if expanded_bounds else 30.
  collar_lower=-np.deg2rad(cfg['collar_limit_deg']) if expanded_bounds else 0.
  nv=n*6+2*len(hand_segments);lo=np.tile([collar_lower,-np.deg2rad(cfg['swivel_limit_deg']),-.006,-.006,-.006,-np.deg2rad(elbow_limit)],(n,1));hi=np.tile([np.deg2rad(cfg['collar_limit_deg']),np.deg2rad(cfg['swivel_limit_deg']),.006,.006,.006,np.deg2rad(elbow_limit)],(n,1));lo[~planned]=-1e-10;hi[~planned]=1e-10
  lower=np.r_[lo.ravel(),np.full(2*len(hand_segments),-1e-10)];upper=np.r_[hi.ravel(),np.full(2*len(hand_segments),1e-10)];x0=np.zeros(nv);x0[:n*6:6]=np.where(mask,np.deg2rad(2.),0);x0[5:n*6:6]=np.clip(required,lo[:,5]+1e-11,hi[:,5]-1e-11)

  def predict(x):
   xx=x[:n*6].reshape(n,6)*influence[:,None];pin=support_target.copy()
   desired=j[:,wrist].copy();desired[planned]+=(pin[planned]-src_sample[planned])*influence[planned,None];desired+=xx[:,2:5]
   C=Rotation.from_rotvec(depression_axis*xx[:,0,None]).as_matrix();collar_world=C@g[:,collar]
   S=j[:,collar]+np.einsum('tij,j->ti',collar_world,collar_offset)
   delta=desired-S;distance=np.linalg.norm(delta,axis=1)
   corr=xx[:,5]  # Optimise the bounded angle itself; no inverse-reach singularity.
   bend=Rotation.from_rotvec(corr[:,None]*axis).as_matrix();E_local=bend@local[:,elbow];end=u+np.einsum('tij,j->ti',E_local,v0)
   source_upper=C@g[:,shoulder];upper=align_many(np.einsum('tij,tj->ti',source_upper,end),delta)@source_upper
   swivel_axis=delta/np.maximum(distance[:,None],1e-12);upper=Rotation.from_rotvec(swivel_axis*xx[:,1,None]).as_matrix()@upper
   E=S+np.einsum('tij,j->ti',upper,u);W=E+np.einsum('tij,tj->ti',upper,np.einsum('tij,j->ti',E_local,v0))
   new_globals=[collar_world,upper,upper@E_local];new_positions=[j[:,collar],S,E]
   sample,heights=predictor.evaluate(new_globals,new_positions,E_local)
   relative=Rotation.from_matrix(g[:,shoulder].swapaxes(-1,-2)@upper).as_rotvec()*planned[:,None]
   state=np.c_[corr,relative,xx[:,:2]]
   physical_flex=np.arccos(np.clip(np.einsum('i,ti->t',u,np.einsum('tij,j->ti',E_local,v0))/np.linalg.norm(u)/np.linalg.norm(v0),-1,1))
   return dict(sample=sample,pin=pin,elbow=E,wrist=W,collar_world=collar_world,upper_world=upper,elbow_local=E_local,state=state,flex=physical_flex,distance=distance)

  one=(planned[1:]|planned[:-1]);two=planned[2:]|planned[1:-1]|planned[:-2];rate=fps/30;tw=cfg['temporal_weight'];w1=one*np.sqrt(np.minimum(motion[1:],motion[:-1]))*tw*rate/cfg['velocity_sigma'];w2=two*np.sqrt(np.minimum(np.minimum(motion[2:],motion[1:-1]),motion[:-2]))*tw*rate**2/cfg['acceleration_sigma'];channel=np.array([1.,.4,.4,.4,1.,.4])
  target=np.zeros((n,3));target[mask]=support_target[mask]
  def residual(x):
   p=predict(x);contact=np.zeros((n,3));goal=src_sample.copy();goal[planned]+=(p['pin'][planned]-src_sample[planned])*influence[planned,None];contact[planned]=p['sample'][planned]-goal[planned]
   # Smooth added changes, with a weak physical-flex acceleration term.
   # Intended source push-up motion retains its own timing and pose prior.
   elbow_error=(p['elbow']-j[:,elbow])*planned[:,None]
   return np.r_[(contact/cfg['contact_sigma_m']).ravel(),
    (p['state']*channel/np.array([.20,.25,.25,.25,.12,.15])).ravel(),
    (elbow_error/cfg['source_elbow_sigma_m']).ravel(),x[n*6:]/.01,
    (np.diff(p['state'],axis=0)*w1[:,None]*channel).ravel(),
    (np.diff(p['state'],n=2,axis=0)*w2[:,None]*channel).ravel(),
    np.diff(p['flex'],n=2)*two*np.sqrt(np.minimum(np.minimum(motion[2:],motion[1:-1]),motion[:-2]))*tw*rate**2/cfg['actual_flex_acceleration_sigma']]
  rows=[];cols=[];cursor=0
  def block(frames,count,anchors_used=False):
   nonlocal cursor
   frames=np.asarray(frames)
   if not len(frames):return
   if frames.ndim==1:frames=frames[:,None]
   variables=(frames[:,:,None]*6+np.arange(6)).reshape(len(frames),-1)
   if anchors_used:
    anchor_cols=np.where(group[frames]>=0,n*6+group[frames]*2,-1);variables=np.c_[variables,anchor_cols, np.where(anchor_cols>=0,anchor_cols+1,-1)]
   rr=cursor+np.arange(len(frames))[:,None]*count+np.arange(count)
   for k in range(variables.shape[1]):
    cc=variables[:,k];ok=cc>=0;rows.append(rr[ok].ravel());cols.append(np.repeat(cc[ok],count))
   cursor+=len(frames)*count
  block(np.arange(n),3,True);block(np.arange(n),6,True);block(np.arange(n),3,True)
  anchor_count=2*len(hand_segments);rows.append(cursor+np.arange(anchor_count));cols.append(n*6+np.arange(anchor_count));cursor+=anchor_count
  block(np.stack((np.arange(n-1),np.arange(1,n)),1),6,True)
  block(np.stack((np.arange(n-2),np.arange(1,n-1),np.arange(2,n)),1),6,True)
  block(np.stack((np.arange(n-2),np.arange(1,n-1),np.arange(2,n)),1),1,True)
  sparsity=coo_matrix((np.ones(sum(map(len,rows))),(np.concatenate(rows),np.concatenate(cols))),shape=(cursor,nv)).tocsr()
  maximum_reach=np.linalg.norm(u)+np.linalg.norm(v0)+2*np.linalg.norm(collar_offset)*np.sin(np.deg2rad(cfg['collar_limit_deg'])/2)+.012
  impossible=mask&(initial_distance>maximum_reach+.002)
  before=residual(x0)
  opt=least_squares(residual,x0,jac_sparsity=sparsity,bounds=(lower,upper),max_nfev=cfg['iterations'],ftol=1e-4,xtol=2e-6,gtol=1e-5,tr_options={'maxiter':150})
  p=predict(opt.x);ids_active=np.flatnonzero(planned)
  matrices={collar:g[:,9].swapaxes(-1,-2)@p['collar_world'],shoulder:p['collar_world'].swapaxes(-1,-2)@p['upper_world'],elbow:p['elbow_local'],wrist:(p['upper_world']@p['elbow_local']).swapaxes(-1,-2)@g[:,wrist]}
  for k,M in matrices.items():out[ids_active,k]=Rotation.from_matrix(M[ids_active]).as_quat()[:,[3,0,1,2]]
  for k,s in enumerate(hand_segments):
   s['anchor']=p['pin'][s['start']].tolist();s['anchor_end']=p['pin'][s['end']-1].tolist();s['anchor_model']='single registered palm proxy; no finger contacts'
  error=np.linalg.norm(p['sample'][mask]-p['pin'][mask],axis=1)
  unreachable=error>.01
  reports.append(dict(hand=h,geometry_unreachable_frames=int((reach_gap[mask]>.005).sum()),reach_gap_max_m=float(reach_gap[mask].max()),reach_gain_min=float(reach_gain[mask].min()),infeasible_requested_frames=int(impossible.sum()),reach_lower_bound_max_m=float(np.maximum(initial_distance-maximum_reach,0)[mask].max()),iterations=int(opt.nfev),success=bool(opt.success),message=opt.message,cost_before=float(before@before/2),cost_after=float(opt.cost),surface_contact_error_max_m=float(error.max()),unresolved_full_contact_frames=int(unreachable.sum()),selected_elbow_limit_deg=elbow_limit,bidirectional_collar=expanded_bounds,
   collar_depression_max_deg=float(np.rad2deg(opt.x[:n*6].reshape(n,6)[:,0]).max()),arm_swivel_max_deg=float(np.rad2deg(abs(opt.x[:n*6].reshape(n,6)[:,1])).max()),
   anchor_xz_adjustments_m=opt.x[n*6:].reshape(-1,2).tolist(),fixed_elbow_branch_per_stance=[int(branch[s['start']]) for s in hand_segments],support_region_phases=[phase for phase in tracks.phases[h] if phase['start']<first+n and phase['end']>first],
   max_added_elbow_deg=float(np.rad2deg(abs(p['state'][:,0])).max()),shoulder_position_change_max_m=float(np.linalg.norm((j[:,collar]+np.einsum('tij,j->ti',p['collar_world'],collar_offset))-j[:,shoulder],axis=1).max())))
 return out,dict(contacts=int(active.sum()),segments=segments,hands=reports,config=cfg,seconds=time.perf_counter()-start,root_and_trunk_preserved=True,foot_pose_preserved=True,physics_simulation=False)
