"""Source-relative 3D stances; no global ground plane or terrain reconstruction."""
import numpy as np
from scipy.ndimage import gaussian_filter1d


def flight_mask(probes,fps,active):
    """Reject bounded flight only when its raised arc has no stable stance.

    Height alone must not turn an up/down staircase into an airborne interval.
    This remains geometric evidence, not measured ground pressure.
    """
    from .correction import airborne_excursions,intervals
    mask=airborne_excursions(probes,fps)
    height=np.asarray(probes)[...,1].reshape(-1,2,2).min(2)
    stable=np.ones_like(active)
    if len(height)>1:
        vertical=np.abs(np.gradient(gaussian_filter1d(height,max(.5,fps*.04),axis=0,mode='nearest'),axis=0))*fps
        stable=vertical<.1
    for first,end in intervals(mask):
        if any(e-s>=max(3,int(np.ceil(fps*.20))) for foot in (0,1) for s,e in intervals(active[first:end,foot]&stable[first:end,foot])):
            mask[first:end]=False
    return mask


def prepare(probes, probabilities, fps, preset, contact_ranges):
    from .correction import intervals
    from .correction_settings import PRESETS
    from .contact_trajectory import support_weights,support_samples,rolling_anchor,couple_anchors
    config=PRESETS[preset];n=len(probes)
    patches=probes.reshape(n,2,2,3)
    # A rolling heel may move while the forefoot is planted (and vice versa).
    # Both feet are independent: the higher foot can be on another step/rung.
    velocity=np.zeros((n,2,2))
    if n>1:
        velocity[1:]=np.linalg.norm(np.diff(patches,axis=0),axis=-1)*fps
        velocity[0]=velocity[1]
    speed=velocity.min(2)
    confidence=probabilities.reshape(n,2,2).max(2)
    geometry=speed<.45
    active=(confidence>config['confidence'])&geometry
    if config['gap']:
        holding=(confidence>config['release'])&geometry
        for foot in (0,1):
            for first,end in intervals(holding[:,foot]|active[:,foot]):
                seeds=np.flatnonzero(active[first:end,foot])
                if len(seeds):active[first+seeds[0]:end,foot]=True
            for first,end in intervals(~active[:,foot]):
                if first==0 or end==n or end-first>round(fps*config['gap']):continue
                if not geometry[first:end,foot].all():continue
                if np.linalg.norm(patches[end,foot].mean(0)-patches[first-1,foot].mean(0))>.04:continue
                active[first:end,foot]=True
    minimum=max(3,round(fps*max(.10,config['minimum'])))
    for foot in (0,1):
        for first,end in intervals(active[:,foot]):
            if end-first<minimum:active[first:end,foot]=False
    active[flight_mask(probes,fps,active)]=False
    if contact_ranges is not None:
        from .foot_contacts import normalize_ranges
        active[:]=False
        for foot,ranges in enumerate(normalize_ranges(contact_ranges,n)):
            for first,end in ranges:active[first:end,foot]=True
    weights=support_weights(probes,active,fps)
    support=support_samples(probes,weights);anchors=support.copy();segments=[]
    for foot in (0,1):
        for first,end in intervals(active[:,foot]):
            # A single stance cannot span travel to a new height. Split automatic
            # evidence on 3D drift, never split explicit user timing.
            start=first
            while start<end:
                stop=start+1
                while stop<end and (contact_ranges is not None or
                    np.linalg.norm(support[stop,foot]-support[start,foot])<config['anchor_drift']):stop+=1
                if contact_ranges is None and end-stop<minimum:stop=end
                height=float(np.median(support[start:stop,foot,1]))
                path=rolling_anchor(probes[start:stop,2*foot:2*foot+2],weights[start:stop,foot],height)
                anchors[start:stop,foot]=path
                segments.append(dict(foot=foot,start=int(start),end=int(stop),anchor=path[0].tolist(),anchor_end=path[-1].tolist()))
                start=stop
    # Do not smooth absolute heights across separate steps. Adjacent uncertain
    # auto splits blend their *corrections*, in all three dimensions.
    if contact_ranges is None:
        radius=max(2,round(fps*.1))
        for foot in (0,1):
            stances=[s for s in segments if s['foot']==foot]
            for left,right in zip(stances,stances[1:]):
                if left['end']!=right['start']:continue
                boundary=right['start'];first=max(left['start'],boundary-radius);end=min(right['end'],boundary+radius)
                delta=anchors[first:end,foot]-support[first:end,foot]
                anchors[first:end,foot]=support[first:end,foot]+gaussian_filter1d(delta,max(.5,fps*.04),axis=0,mode='nearest')
                right['handoff']=[first,end]
        for s in segments:
            s['anchor']=anchors[s['start'],s['foot']].tolist();s['anchor_end']=anchors[s['end']-1,s['foot']].tolist()
    else:couple_anchors(anchors,support,segments,fps)
    # No global grounding shift: retain the inferred climb/descent trajectory.
    return active,anchors,support,np.zeros((n,3),np.float32),float(probes[...,1].min()),segments
