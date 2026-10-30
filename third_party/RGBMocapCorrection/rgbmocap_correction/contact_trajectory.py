"""Rotation-preserving sole support and bounded contact-transition planning.

This module plans positions only. The native anatomical solver remains the
single owner of leg rotations. No learned pressure model or external cleanup
implementation is used here.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import solve_banded


def runs(mask):
    edge = np.diff(np.r_[False, mask, False].astype(np.int8))
    return zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1))


def support_weights(probes, active, fps=30.):
    """Stable heel/forefoot material weights from geometry, never confidence.

    A small flat-foot deadband avoids interpreting millimetre skin noise as a
    pressure transfer. Smooth only within each observed stance, not across an
    airborne gap. Physical minimum Y is handled independently by the caller.
    """
    height = np.asarray(probes)[..., 1].reshape(-1, 2, 2)
    weights = np.full(active.shape, .5, np.float64)
    for foot in (0, 1):
        for first, end in runs(active[:, foot]):
            delta = height[first:end, foot, 0] - height[first:end, foot, 1]
            delta = gaussian_filter1d(delta, max(.5, fps * .04), mode='nearest')
            x = np.clip((np.abs(delta) - .005) / .02, 0., 1.)
            weights[first:end, foot] += .5 * np.sign(delta) * x*x*(3.-2.*x)
    return weights


def support_samples(probes, weights):
    patches = np.asarray(probes).reshape(-1, 2, 2, 3)
    points = patches[:, :, 0] * (1.-weights[..., None]) + patches[:, :, 1] * weights[..., None]
    points[..., 1] = patches[..., 1].min(2)
    return points


def rolling_anchor(probes, weights, floor):
    """Integrate a continuous contact handoff without sliding a material pin.

    For fixed weights this is an ordinary fixed world anchor. During roll,
    changing which material point supports the foot moves the *contact
    location*, not the planted material itself. The trapezoidal discrete
    no-slip constraint uses the average weights of adjacent frames.
    """
    patches = np.asarray(probes)
    separation = patches[:, 1] - patches[:, 0]
    path = np.zeros((len(patches), 3), np.float64)
    if len(path) > 1:
        path[1:] = np.cumsum(.5 * (separation[1:] + separation[:-1]) * np.diff(weights)[:, None], axis=0)
    observed = patches[:, 0] * (1.-weights[:, None]) + patches[:, 1] * weights[:, None]
    path += np.median(observed-path, axis=0)
    path[:, 1] = floor
    return path


def couple_anchors(anchors, support, segments, fps):
    """Avoid incompatible independent pins across a very short release gap.

    Only constant XZ translations of each stance are allowed: no contact
    rotation changes or drift inside a stance. The closest source placement
    remains the data term. A tridiagonal solve couples neighbouring endpoints;
    each stance adjustment is bounded to 12 cm. Long swing gaps are independent.
    """
    for foot in (0, 1):
        stances = [s for s in segments if s['foot'] == foot]
        count = len(stances)
        if count < 2:
            continue
        diagonal = np.ones(count);off = np.zeros(count-1);rhs = np.zeros((count,2))
        for i, (left, right) in enumerate(zip(stances, stances[1:])):
            gap = right['start']-left['end']
            if gap > fps*.12:
                continue
            weight = min(1000., 10.*(fps*.12/max(1,gap+1))**3)
            a,b = left['end']-1,right['start']
            difference = ((anchors[b,foot]-support[b,foot])-(anchors[a,foot]-support[a,foot]))[[0,2]]
            diagonal[i:i+2] += weight;off[i] = -weight
            rhs[i] += weight*difference;rhs[i+1] -= weight*difference
        bands = np.zeros((3,count));bands[1] = diagonal;bands[0,1:] = off;bands[2,:-1] = off
        adjustment = solve_banded((1,1),bands,rhs,check_finite=False)
        adjustment *= np.minimum(1., .12/np.maximum(np.linalg.norm(adjustment,axis=1),1e-10))[:,None]
        for stance, delta in zip(stances, adjustment):
            first,end = stance['start'],stance['end']
            anchors[first:end,foot,0] += delta[0];anchors[first:end,foot,2] += delta[1]
            stance['anchor'] = anchors[first,foot].tolist()
            stance['anchor_end'] = anchors[end-1,foot].tolist()
            stance['continuity_adjustment_m'] = float(np.linalg.norm(delta))


def handoff_mask(segments,frames):
    mask=np.zeros((frames,2),bool)
    for stance in segments:
        if 'handoff' in stance:
            first,end=stance['handoff'];mask[first:end,stance['foot']]=True
    return mask


def smooth_auto_handoffs(anchors,probes,weights,segments,fps):
    """A drift-limit reset must not teleport an automatically chosen anchor.

    These uncertain travel handoffs are explicitly excluded from fixed-contact
    metrics. They are not pressure-confirmed stance points. Manual stances are
    never split here. Interpolate only the translation between stance anchors,
    relative to a continuous no-slip roll reference. Interpolating source-relative
    offsets here would reintroduce the very source velocity noise being removed.
    """
    radius=max(3,round(fps*.20))
    for foot in (0,1):
        stances=[s for s in segments if s['foot']==foot]
        mask=np.zeros(len(anchors),bool)
        active=np.zeros(len(anchors),bool)
        for stance in stances:active[stance['start']:stance['end']]=True
        reference=anchors[:,foot].copy()
        for first,end in runs(active):
            reference[first:end]=rolling_anchor(probes[first:end,2*foot:2*foot+2],weights[first:end,foot],anchors[first,foot,1])
        for left,right in zip(stances,stances[1:]):
            if left['end']!=right['start']:continue
            boundary=right['start']
            first=max(left['start'],boundary-radius);end=min(right['end'],boundary+radius)
            mask[first:end]=True;right['handoff']=[int(first),int(end)]
        for first,end in runs(mask):
            if end-first<3:continue
            correction=anchors[first:end,foot]-reference[first:end]
            a,b=correction[0],correction[-1]
            va=correction[1]-a;vb=b-correction[-2]
            blended=_hermite(a,va,b,vb,end-first-2)
            anchors[first+1:end-1,foot,0]=reference[first+1:end-1,0]+blended[:,0]
            anchors[first+1:end-1,foot,2]=reference[first+1:end-1,2]+blended[:,2]
        for stance in stances:
            stance['anchor']=anchors[stance['start'],foot].tolist()
            stance['anchor_end']=anchors[stance['end']-1,foot].tolist()


def diagnostic_plan(anchors,weights,segments):
    """Bound metadata independently of clip length; timings stay sparse."""
    frames=len(anchors);indices=np.unique(np.linspace(0,frames-1,min(frames,4096),dtype=np.int64))
    handoffs=np.asarray([[s['foot'],*s['handoff']] for s in segments if 'handoff' in s],np.int64).reshape(-1,3)
    return dict(anchor_frames=indices,anchor_trajectory=np.asarray(anchors[indices],np.float32),
                support_weights=np.asarray(weights[indices],np.float32),handoffs=handoffs)


def _hermite(a, va, b, vb, count):
    """Interior samples of a cubic matching endpoint values and velocities."""
    duration = count + 1
    t = np.arange(1, duration, dtype=np.float64)[:, None] / duration
    return ((2*t**3-3*t**2+1)*a + (t**3-2*t**2+t)*duration*va +
            (-2*t**3+3*t**2)*b + (t**3-t**2)*duration*vb)


def contact_offsets(probes, active, anchors, support, shift, segments, fps, strength, flight):
    """Keep stance targets; connect offsets in neighbouring non-contact frames.

    Contact timings are immutable. Transition samples are *not* new contacts.
    They only carry the outgoing positional/velocity correction into swing.
    Work and temporary storage are linear in frames, with at most 0.5 seconds
    of influence on either side. Never propagate a leg edit into flight.
    """
    offsets = np.zeros_like(support, dtype=np.float64)
    offsets[active] = ((anchors-support-shift[:, None]) * strength[..., None])[active]
    transition = np.zeros_like(active)
    maximum = max(2, round(fps * .5))
    minimum = max(2, round(fps * .12))

    def extent(value):
        # Larger corrections need more time than a two-frame strength ramp.
        return min(maximum, max(minimum, int(np.ceil(np.linalg.norm(value) * fps / .8))))

    def slope(first, end, foot, at_end):
        if end-first < 2:
            return np.zeros(3)
        index = end-1 if at_end else first
        return offsets[index, foot]-offsets[index-1, foot] if at_end else offsets[index+1, foot]-offsets[index, foot]

    # Auto anchor splits can be adjacent. Their offsets must not be smoothed
    # across an explicitly active frame; each side retains its own target.
    for foot in (0, 1):
        stances = list(runs(active[:, foot]))
        by_start = {s:(s,e) for s,e in stances}
        by_end = {e:(s,e) for s,e in stances}
        for first, end in runs(~active[:, foot] & ~flight):
            left = by_end.get(first)
            right = by_start.get(end)
            if left is None and right is None:
                continue
            count = end-first
            lvalue = offsets[first-1, foot].copy() if left else np.zeros(3)
            rvalue = offsets[end, foot].copy() if right else np.zeros(3)
            lextent = extent(lvalue) if left else 0
            rextent = extent(rvalue) if right else 0
            lcount = min(count, lextent)
            rcount = min(count, rextent)
            lv = slope(*left, foot, True) if left else np.zeros(3)
            rv = slope(*right, foot, False) if right else np.zeros(3)
            if left and right and lcount+rcount >= count:
                offsets[first:end, foot] = _hermite(lvalue, lv, rvalue, rv, count)
                transition[first:end, foot] = True
            else:
                if left:
                    # End-of-video is not evidence that the foot returned to
                    # the raw trajectory. Truncate a normal-duration release
                    # instead of squeezing it into the final few samples.
                    duration = lextent if end == len(active) else lcount
                    offsets[first:first+lcount, foot] = _hermite(lvalue, lv, np.zeros(3), np.zeros(3), duration)[:lcount]
                    transition[first:first+lcount, foot] = True
                if right:
                    duration = rextent if first == 0 else rcount
                    offsets[end-rcount:end, foot] = _hermite(np.zeros(3), np.zeros(3), rvalue, rv, duration)[-rcount:]
                    transition[end-rcount:end, foot] = True
    return offsets, transition


def material_slip(before, after, active, weights, fps):
    """XZ speed of the same weighted material during each contact edge.

    Excludes motion from changing the contact label itself, and excludes lifted
    patches. This is a two-patch diagnostic, not a pressure measurement.
    """
    w = .5*(weights[1:]+weights[:-1])
    pairs = active[1:] & active[:-1]
    result = {}
    for label, probes in (('before', before), ('after', after)):
        velocity = np.diff(np.asarray(probes).reshape(-1, 2, 2, 3), axis=0)
        velocity = velocity[:, :, 0]*(1.-w[..., None])+velocity[:, :, 1]*w[..., None]
        speed = np.linalg.norm(velocity[..., [0, 2]], axis=-1)[pairs]*fps
        result['support_material_speed_'+label+'_max_mps'] = float(speed.max()) if len(speed) else None
        result['support_material_speed_'+label+'_p95_mps'] = float(np.quantile(speed, .95)) if len(speed) else None
    return result


def free_follow_weights(active, transition):
    """Finite C1 blend to the actual free trajectory outside contact."""
    out = np.zeros(active.shape, dtype=np.float64)
    for foot in (0, 1):
        for a, b in runs(transition[:, foot] & ~active[:, foot]):
            left = a > 0 and active[a-1, foot]
            right = b < len(active) and active[b, foot]
            t = np.arange(1, b-a+1, dtype=np.float64)/(b-a+1)
            smooth = t*t*(3-2*t)
            out[a:b, foot] = 0 if left and right else smooth if left else 1-smooth if right else 1
    return out
