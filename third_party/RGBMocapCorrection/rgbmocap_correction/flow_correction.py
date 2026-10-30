"""Native SMPL-H adapter for RGB MoCap Motion Correction.

No inference, SMPL-X conversion, upstream smoothing or second target IK pass."""


import numpy as np

VERSION = 1

def contact_probabilities(params,n):
    # Official Flow representation is a regressed contact value, NOT logits.
    # ODE output may slightly overshoot [0,1]. Order: L ankle/toe, R ankle/toe.
    value=params.get('foot_contacts')
    if value is None or tuple(value.shape)!=(1,n,4):raise ValueError('FlowHMR contact predictions are missing or invalid. Reconstruct the motion again.')
    values=value[0].numpy()
    if not np.isfinite(values).all():raise ValueError('Non-finite FlowHMR contact predictions')
    return np.clip(values,0,1)

class SoleGeometry:
    """Exact upstream SMPL-H LBS (no pose blend), bounded foot-only surface."""
    def __init__(self,body,shape):
        import torch
        self.parents=body.parents.tolist()
        with torch.inference_mode():
            neutral=dict(rot_mats=torch.eye(3).expand(1,52,3,3),shapes=shape,trans=torch.zeros(1,3))
            shaped=body(neutral)['vertices'][0].numpy()
            self.rest=body.compute_j_shaped(shape)[0].numpy()
        weights=body.lbs_weights.numpy();self.groups=[];self.proxies=[]
        from .hand_lock import palm_offsets
        self.palm_offsets=palm_offsets(shaped,weights,self.rest)
        for ankle,toe in ((7,10),(8,11)):
            ids=np.flatnonzero(weights[:,ankle]+weights[:,toe]>.6);z=shaped[ids,2]
            if not len(ids):raise ValueError('SMPL-H foot skin is unavailable')
            for region,proxy in ((ids[z<=np.median(z)],ids[z<=np.quantile(z,.4)]),
                                 (ids[z>=np.median(z)],ids[z>=np.quantile(z,.65)])):
                selected=proxy[np.argsort(shaped[proxy,1])[:24]]
                self.groups.append(region);self.proxies.append(np.searchsorted(region,selected))
        self.ids=np.concatenate(self.groups);self.shaped=shaped[self.ids]
        if np.any(weights[self.ids,22:]>1e-7):raise ValueError('Unexpected hand influence on SMPL-H foot geometry')
        self.weights=weights[self.ids,:22]

    def sample(self,root,quat):
        from .correction import fk
        joints,world=fk(self.rest[:22],root,quat)
        translation=joints-np.einsum('tjik,jk->tji',world,self.rest[:22])
        blend=(self.weights@world.reshape(-1,22,9)).reshape(len(root),len(self.ids),3,3)
        vertices=np.einsum('tvij,vj->tvi',blend,self.shaped)+self.weights@translation
        probes=[];offset=0
        for ids,proxy in zip(self.groups,self.proxies):
            patch=vertices[:,offset:offset+len(ids)];offset+=len(ids)
            value=patch[:,proxy].mean(1);value[:,1]=patch[...,1].min(1);probes.append(value)
        return np.stack(probes,1),vertices
