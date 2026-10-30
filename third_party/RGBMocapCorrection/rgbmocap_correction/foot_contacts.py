"""Sparse foot-lock timing records. Frames are zero-based, end-exclusive."""


def normalize_ranges(ranges, frames):
    if len(ranges) not in (2,4):raise ValueError('Expected two foot or four foot/hand tracks')
    result=[]
    for track in ranges:
        normalized=[]
        for pair in sorted(track):
            if len(pair)!=2:raise ValueError('Invalid foot-lock interval')
            start,end=pair
            if not isinstance(start,int) or not isinstance(end,int) or not 0<=start<end<=frames:
                raise ValueError('Foot-lock intervals must be within the motion frame range')
            if normalized and start<normalized[-1][1]:
                normalized[-1]=(normalized[-1][0],max(end,normalized[-1][1]))
            else:normalized.append((start,end))
        result.append(normalized)
    return result

def segment_ranges(segments):
    return [[(int(s['start']),int(s['end'])) for s in segments if s['foot']==foot] for foot in range(2)]
