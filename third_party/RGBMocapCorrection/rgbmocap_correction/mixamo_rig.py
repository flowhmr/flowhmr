"""Source-proportioned Mixamo hierarchy, FK and a rigid beveled mannequin.

No inference or file I/O here. The first 22 joints keep the source body ordering;
the thirty finger joints are real channels, not display-only embellishments."""


import numpy as np

def unit(v, fallback=(0,1,0)):
    v=np.asarray(v,dtype=float);n=np.linalg.norm(v,axis=-1,keepdims=True)
    return np.where(n>1e-10,v/np.maximum(n,1e-10),np.asarray(fallback))

def align(a,b):
    """Proper shortest rotation; deterministic at the antiparallel singularity."""
    a=unit(a);b=unit(b);v=np.cross(a,b);c=float(np.clip(a@b,-1,1))
    if c < -1+1e-8:
        axis=unit(np.cross(a,np.eye(3)[np.argmin(np.abs(a))]))
        return 2*np.outer(axis,axis)-np.eye(3)
    skew=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3)+skew+skew@skew/max(1+c,1e-10)
