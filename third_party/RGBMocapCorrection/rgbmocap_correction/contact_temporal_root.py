"""Source-relative temporal contact planning.

JOSH-inspired pose/static consistency, with a known floor instead of a scene
reconstructor. Contact targets are unchanged; original native anatomical IK is
still the sole leg rotation solver. No JOSH or CRISP performance claim.
"""
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation
CONFIG={'position_sigma':.01,'pose_sigma':.175,'velocity_sigma':.025,'acceleration_sigma':.004,'root_acceleration_sigma':.0004,'limit':.02,'iterations':60,'temporal_weight':.5,'total_root_smoothing':True}
def optimise(rest,root,quat,probes,shift,target_shift,goal_offsets,active,transition,flight,floor,fps,mode="flat"):
 from .correction import fk
 from .anatomical_ik import reach_lengths,minimum_reach_lengths
 from .mixamo_rig import align
 start=time.perf_counter();cfg=CONFIG;n=len(root)
 if n>4096:raise ValueError('Contact planner requires bounded input windows')
 j,g=fk(rest,root,quat);planned=active|transition;planned[flight&~active.any(1)]=False
 targets=j[:,[7,8]]+target_shift[:,None]+goal_offsets
 low=probes[...,1].reshape(n,2,2).min(2)
 expected=low+targets[...,1]-j[:,[7,8],1]
 targets[...,1]+=np.where(active,np.minimum(np.maximum(floor-expected,0),.12),0.) if mode=="flat" else 0
 hips=j[:,[1,2]]+shift[:,None];horizontal=((hips[...,[0,2]]-targets[...,[0,2]])**2).sum(2)
 upper=targets[...,1]+np.sqrt(np.maximum((reach_lengths(rest,g)*.9995)**2-horizontal,0))-j[:,[1,2],1]
 lower=targets[...,1]+np.sqrt(np.maximum((minimum_reach_lengths(rest,g)*1.0005)**2-horizontal,0))-j[:,[1,2],1]
 valid=planned&(hips[...,1]>targets[...,1])
 base=shift[:,1].astype('float64');lo=np.maximum(base-cfg['limit'],np.where(valid,lower,-np.inf).max(1));hi=np.minimum(base+cfg['limit'],np.where(valid,upper,np.inf).min(1))
 frozen=(~valid.any(1))|(hi-lo<1e-7);lo[frozen]=base[frozen]-1e-10;hi[frozen]=base[frozen]+1e-10
 a=[];b=[];cc=[];rad=[];phase=[]
 for hip,knee,ankle in [(1,4,7),(2,5,8)]:
  u=rest[knee]-rest[hip];v=rest[ankle]-rest[knee];axis=align([0,-1,0],u)@np.array([1.,0,0]);L=g[:,hip].swapaxes(-1,-2)@g[:,knee];v=np.einsum('tij,j->ti',L,v);dot=v@axis;p=v-dot[:,None]*axis
  av=p@u;bv=np.cross(axis,v)@u;r=np.hypot(av,bv)
  cc.append((u@u+(rest[ankle]-rest[knee])@(rest[ankle]-rest[knee])+2*(u@axis)*dot));rad.append(np.maximum(r,1e-9));phase.append(np.arctan2(bv,av))
 constant=np.stack(cc,1);radius=np.stack(rad,1);phase=np.stack(phase,1);hy=j[:,[1,2],1]-targets[...,1]
 extension_support=np.clip((np.deg2rad(5)-abs(phase))/np.deg2rad(4),0,1)
 # Penalise changes of ADDED correction, not speed of inferred animation.
 src_k=Rotation.from_quat(quat[:,[4,5]][...,[1,2,3,0]].reshape(-1,4)).as_matrix().reshape(n,2,3,3)
 speed=np.r_[np.zeros((1,2)),Rotation.from_matrix((src_k[:-1].swapaxes(-1,-2)@src_k[1:]).reshape(-1,3,3)).magnitude().reshape(n-1,2)*fps]
 static_weight=.25+.75/(1+(speed/np.deg2rad(20))**2)
 mask=valid.astype(float);w0=mask/cfg['pose_sigma']
 rate=fps/30.;tw=cfg['temporal_weight']
 w1=np.minimum(mask[1:],mask[:-1])*np.sqrt(np.minimum(static_weight[1:],static_weight[:-1]))*rate*tw/cfg['velocity_sigma']
 w2=np.minimum(np.minimum(mask[2:],mask[1:-1]),mask[:-2])*np.sqrt(np.minimum(np.minimum(static_weight[2:],static_weight[1:-1]),static_weight[:-2]))*rate**2*tw/cfg['acceleration_sigma']
 wr=rate**2/cfg['root_acceleration_sigma'];m1=n;m2=2*n;m3=2*(n-1);m4=2*(n-2)
 def angle(x):
  dy=hy+x[:,None];co=(horizontal+dy**2-constant)/(2*radius);bounded=np.clip(co,-.999999999,.999999999);value=(phase*(1-extension_support)+np.arccos(bounded))*mask
  derivative=-dy/(radius*np.sqrt(np.maximum(1-bounded**2,1e-12)))*mask*(np.abs(co)<.999999999)
  return value,derivative
 def residual(x):
  v,_=angle(x)
  smooth_root=x if cfg['total_root_smoothing'] else x-base
  return np.r_[(x-base)/cfg['position_sigma'],(v*w0).ravel(),(np.diff(v,axis=0)*w1).ravel(),(np.diff(v,n=2,axis=0)*w2).ravel(),np.diff(smooth_root,n=2)*wr]
 def jac(x):
  _,der=angle(x);rows=[];cols=[];vals=[]
  def add(row,col,value):rows.append(np.asarray(row).ravel());cols.append(np.asarray(col).ravel());vals.append(np.asarray(value).ravel())
  idx=np.arange(n);add(idx,idx,np.full(n,1/cfg['position_sigma']));add(m1+np.arange(m2),np.repeat(idx,2),der*w0)
  off=m1+m2
  for k,c in [(0,-1),(1,1)]:add(off+np.arange(m3),np.repeat(np.arange(n-1)+k,2),c*der[k:k+n-1]*w1)
  off+=m3
  for k,c in [(0,1),(1,-2),(2,1)]:add(off+np.arange(m4),np.repeat(np.arange(n-2)+k,2),c*der[k:k+n-2]*w2)
  off+=m4
  for k,c in [(0,1),(1,-2),(2,1)]:add(off+np.arange(n-2),np.arange(n-2)+k,np.full(n-2,c*wr))
  return coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(off+n-2,n)).tocsr()
 x0=np.clip(base,lo+np.minimum(1e-8,(hi-lo)/4),hi-np.minimum(1e-8,(hi-lo)/4));before=residual(x0)
 jacobian_error=None
 if cfg.get('check_jacobian'):
  test=(lo+hi)/2;J=jac(test);eligible=np.flatnonzero(hi-lo>1e-4);errors=[]
  for k in eligible[np.linspace(0,max(0,len(eligible)-1),min(12,len(eligible))).astype(int)]:
   h=min(1e-7,(hi[k]-lo[k])*.01);plus=test.copy();minus=test.copy();plus[k]+=h;minus[k]-=h;numeric=(residual(plus)-residual(minus))/(2*h);analytic=J[:,k].toarray().ravel();errors.append(float(np.linalg.norm(numeric-analytic)/(1+np.linalg.norm(numeric))))
  jacobian_error=max(errors,default=0.);assert jacobian_error<2e-5,jacobian_error
 result=least_squares(residual,x0,jac=jac,bounds=(lo,hi),method='trf',max_nfev=cfg['iterations'],ftol=1e-5,xtol=1e-6,gtol=1e-5,tr_solver='lsmr')
 out=shift.copy();out[:,1]=result.x.astype('float32')
 return out,{'contact_temporal_root':{'config':dict(cfg),'seconds':time.perf_counter()-start,'evaluations':int(result.nfev),'success':bool(result.success),'cost_before':float(before@before/2),'cost_after':float(result.cost),'additional_root_height_max_m':float(abs(result.x-base).max()),'frozen_frames':int(frozen.sum()),'jacobian_relative_error_max':jacobian_error,'note':'Changes root plan only; retains exact original native anatomical solver and unchanged contact goals.'}}
