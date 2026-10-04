"""Pack rendered tasks into WebDataset shards for base training.

For each task directory written by render_pipeline.py:

    <task_id>.mp4, <task_id>_camera.npz, <task_id>_bbox.npz, <stem>.npz (motion)

extract SAM 3D Body features (3072-d per frame) on the tracked range and write one
sample with motion.npz / camera.npz / bbox.npz / feature.pt (format:
docs/resources.md#webdataset-shards-base-training).

Run from the FlowHMR repo root (needs the FlowHMR environment and
ckpts/sam-3d-body-dinov3):

    python data/rendering_data/scripts/make_webdataset.py \\
        --render_root data/rendering_data/examples_output \\
        --output data/webdataset/synthetic/synthetic-%05d.tar
"""

import argparse
import glob
import io
import json
import os
import os.path as osp
import sys

import numpy as np
import torch

REPO_ROOT = osp.normpath(osp.join(osp.dirname(osp.abspath(__file__)), '..', '..', '..'))
FPS = 30
NUM_BETAS = 16


def npz_bytes(**arrays):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def pt_bytes(tensor):
    buf = io.BytesIO()
    torch.save(tensor, buf)
    return buf.getvalue()


def find_tasks(render_root):
    """Task directories that contain a rendered video."""
    tasks = []
    for mp4 in sorted(glob.glob(osp.join(render_root, '**', '*.mp4'), recursive=True)):
        task_dir = osp.dirname(mp4)
        task_id = osp.splitext(osp.basename(mp4))[0]
        if osp.basename(task_dir) != task_id:
            continue  # e.g. debug overlays
        tasks.append((task_dir, task_id))
    return tasks


def load_task(task_dir, task_id):
    camera = np.load(osp.join(task_dir, f'{task_id}_camera.npz'))
    bbox = np.load(osp.join(task_dir, f'{task_id}_bbox.npz'))
    stem = task_id.rsplit('_', 1)[0]
    motion = np.load(osp.join(task_dir, f'{stem}.npz'))

    poses = np.asarray(motion['poses'], dtype=np.float32)
    poses = poses.reshape(poses.shape[0], -1)
    if poses.shape[1] != 156:
        raise ValueError(f'expected SMPL-H poses (F, 156), got {poses.shape}')
    betas = np.asarray(motion['betas'], dtype=np.float32).reshape(1, -1)
    betas = np.pad(betas, ((0, 0), (0, max(0, NUM_BETAS - betas.shape[1]))))[:, :NUM_BETAS]
    fps = float(motion['mocap_framerate']) if 'mocap_framerate' in motion else FPS
    if abs(fps - FPS) > 1e-3:
        raise ValueError(f'expected {FPS} fps motion, got {fps}')

    return {
        'name': task_id,
        'video': osp.join(task_dir, f'{task_id}.mp4'),
        'poses': poses,
        'trans': np.asarray(motion['trans'], dtype=np.float32),
        'betas': betas,
        'RT': np.asarray(camera['RT'], dtype=np.float32),
        'K': np.asarray(camera['K'], dtype=np.float32),
        'bbox': np.asarray(bbox['bbox'], dtype=np.float32)[:, :4],
        'start_end': np.asarray(bbox['start_end'], dtype=np.int64).reshape(2),
    }


def check_lengths(s):
    num_frames = len(s['poses'])
    lengths = {k: len(s[k]) for k in ('trans', 'RT', 'K', 'bbox')}
    if any(n != num_frames for n in lengths.values()):
        raise ValueError(f'frame count mismatch: poses={num_frames}, {lengths}')
    start, end = s['start_end']
    if not (0 <= start <= end < num_frames) or end - start + 1 < 10:
        raise ValueError(f'invalid or too short tracked range {start}..{end} of {num_frames}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--render_root', required=True, help='--root of render_pipeline.py')
    parser.add_argument('--output', required=True, help='shard pattern, e.g. out/synthetic-%%05d.tar')
    parser.add_argument('--maxcount', type=int, default=1000, help='samples per shard')
    parser.add_argument('--save_features', action='store_true',
                        help='also save <task_id>_sam3d_feat.pt next to each task')
    args = parser.parse_args()

    sys.path.insert(0, REPO_ROOT)
    import webdataset as wds
    from flowhmr.utils.runtime_sam_features import build_sam3d_extractor

    tasks = find_tasks(args.render_root)
    print(f'found {len(tasks)} rendered tasks in {args.render_root}')
    if not tasks:
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor = build_sam3d_extractor(device)

    os.makedirs(osp.dirname(osp.abspath(args.output)), exist_ok=True)
    num_written, skipped = 0, []
    with wds.ShardWriter(args.output, maxcount=args.maxcount) as sink:
        for i, (task_dir, task_id) in enumerate(tasks):
            try:
                s = load_task(task_dir, task_id)
                check_lengths(s)
                start, end = (int(x) for x in s['start_end'])
                feature_path = osp.join(task_dir, f'{task_id}_sam3d_feat.pt')
                if osp.exists(feature_path):
                    feature = torch.load(feature_path, weights_only=True)
                else:
                    tokens = extractor.extract_video_tokens(
                        s['video'],
                        bbox_xyxy=torch.from_numpy(s['bbox']),
                        K_all=torch.from_numpy(s['K']),
                        max_frames=end + 1,
                    )
                    feature = tokens[start:end + 1].float().contiguous()
                    if args.save_features:
                        torch.save(feature, feature_path)
                if feature.shape != (end - start + 1, 3072):
                    raise ValueError(f'feature shape {tuple(feature.shape)}, '
                                     f'expected ({end - start + 1}, 3072)')
            except Exception as e:  # keep going on bad tasks
                print(f'[{i + 1}/{len(tasks)}] skip {task_id}: {e}')
                skipped.append(task_id)
                continue

            key = f'{num_written:08d}_{task_id}'.replace('.', '_')
            sink.write({
                '__key__': key,
                'motion.npz': npz_bytes(poses=s['poses'], trans=s['trans'], betas=s['betas'],
                                        mocap_framerate=FPS),
                'camera.npz': npz_bytes(RT=s['RT'], K=s['K']),
                'bbox.npz': npz_bytes(bbox=s['bbox'], start_end=s['start_end']),
                'feature.pt': pt_bytes(feature),
                'metadata.json': json.dumps({'sequence_name': task_id}).encode('utf-8'),
            })
            num_written += 1
            print(f'[{i + 1}/{len(tasks)}] {task_id}: {len(s["poses"])} frames, '
                  f'tracked {start}..{end}')

    print(f'wrote {num_written} samples, skipped {len(skipped)}')


if __name__ == '__main__':
    main()
