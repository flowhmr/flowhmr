#!/usr/bin/env python3
import argparse
import glob
import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
from isaacgym import gymapi # noqa: F401; Isaac Gym must be imported before torch.
import torch

ROOT = Path(__file__).resolve().parents[1]
PHC_ROOT = Path(os.environ.get(
    "FLOWHMR_PHC_ROOT", str(ROOT / "third_party" / "PerpetualHumanoidControl")))
PHC_DIR = PHC_ROOT / "phc"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PHC_ROOT))
sys.path.insert(0, str(PHC_DIR))

import phc.run_hydra as run_hydra # noqa: E402

ARGS = None


def _as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _deterministic_sample_ref_state(task, env_ids):
    motion_ids = env_ids.long()
    motion_times = torch.zeros(len(env_ids), device=task.device)
    shapes = task.humanoid_shapes[env_ids]
    state = task._get_fixed_smpl_state_from_motionlib(motion_ids, motion_times, shapes)
    return (motion_ids, motion_times, *state)


def _load_batch(task, start_idx):
    task.start_idx = start_idx
    task._motion_lib.load_motions(
        skeleton_trees=task.skeleton_trees,
        gender_betas=task.humanoid_shapes.cpu(),
        limb_weights=task.humanoid_limb_and_weights.cpu(),
        random_sample=False,
        start_idx=start_idx,
    )
    task.ref_motion_cache = {}
    task.reset()


def _rollout_one_batch(task, player, score_scale, termination_distance):
    valid_count = min(task.num_envs, int(task._motion_lib._num_unique_motions) - task.start_idx)
    current_keys = task._motion_lib.curr_motion_keys
    if isinstance(current_keys, (str, bytes)):
        current_keys = [current_keys]
    elif hasattr(current_keys, "tolist"):
        current_keys = current_keys.tolist()
    else:
        current_keys = list(current_keys)
    keys = [str(key) for key in current_keys[:valid_count]]
    steps_per_motion = _as_numpy(
        task._motion_lib.get_motion_num_steps()[:valid_count]).astype(np.int64)
    max_steps = int(steps_per_motion.max())

    obs = player.env_reset()
    player.get_batch_size(obs["obs"], 1)
    if player.is_rnn:
        player.init_rnn()
    done_indices = []
    terminated = np.zeros(valid_count, dtype=bool)
    first_failure = np.full(valid_count, -1, dtype=np.int64)
    mpjpe_frames = []

    with torch.no_grad():
        for step in range(max_steps):
            obs = player.env_reset(done_indices)
            action = player.get_action(obs, True)
            obs, _, done, info = player.env_step(player.env, action)

            body = _as_numpy(info["body_pos"][:valid_count])
            reference = _as_numpy(info["body_pos_gt"][:valid_count])
            mpjpe_frames.append(np.linalg.norm(body - reference, axis=-1).mean(-1))

            active = step < (steps_per_motion - 1)
            terminate_now = _as_numpy(info["terminate"][:valid_count]).astype(bool)
            terminate_now &= active
            newly_failed = terminate_now & ~terminated
            first_failure[newly_failed] = step
            terminated |= terminate_now

            done_indices = done.nonzero(as_tuple=False)[:: player.num_agents]
            done_indices = done_indices[:, 0] if len(done_indices) else []
            if step + 1 >= max_steps:
                break

    mpjpe_frames = np.asarray(mpjpe_frames)
    results = {}
    for env_id, key in enumerate(keys):
        num_steps = max(1, int(steps_per_motion[env_id]) - 1)
        errors = mpjpe_frames[:num_steps, env_id].copy()
        failed_at = int(first_failure[env_id])
        if failed_at >= 0:
            errors[failed_at:] = termination_distance
            survival = failed_at / num_steps
        else:
            survival = 1.0
        capped = np.minimum(errors, termination_distance)
        tracking_score = float(np.exp(-capped.mean() / score_scale))
        results[key] = {
            "tracking_score": round(tracking_score, 6),
            "terminated": bool(failed_at >= 0),
            "survival": round(float(survival), 6),
            "mpjpe_cm": round(float(errors.mean() * 100), 4),
            "num_steps": num_steps,
        }
    return results


def _install_motions(task, motions):
    items = sorted(motions.items(),
                   key=lambda kv: len(kv[1]["pose_quat_global"]), reverse=True)
    ml = task._motion_lib
    ml._motion_data_load = dict(items)
    ml._motion_data_list = np.array([v for _, v in items])
    ml._motion_data_keys = np.array([k for k, _ in items])
    ml._num_unique_motions = len(items)
    ml.setup_constants(fix_height=ml.fix_height, multi_thread=ml.multi_thread)


def _make_converter(max_frames):
    spec = importlib.util.spec_from_file_location(
        "convert_flowhmr_to_phc",
        str(ROOT / "tools" / "convert_flowhmr_to_phc.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.FlowHMRToPHCConverter(fps=30, upright_start=True, yup2zup=True,
                                     max_frames=max_frames)


def _ensure_init_pkl(path):
    import joblib
    if path.exists():
        return
    conv = _make_converter(64)
    T = 32
    poses = np.zeros((T, 52, 3))
    trans = np.zeros((T, 3))
    trans[:, 1] = 0.9
    motions = {"init": conv.convert_data(
        {"poses": poses, "trans": trans, "mocap_framerate": 30})}
    joblib.dump(motions, str(path), compress=False)


def _server_run(player):
    task = player.env.task
    task._sample_ref_state = types.MethodType(_deterministic_sample_ref_state, task)
    task._termination_distances[:] = ARGS.termination_distance
    task._recovery_episode_prob = 0
    task._fall_init_prob = 0
    task.zero_out_far = False
    task.zero_out_far_train = False
    task.cycle_motion = False

    ctrl = Path(ARGS.ctrl_dir)
    ctrl.mkdir(parents=True, exist_ok=True)
    conv = _make_converter(ARGS.max_frames)

    (ctrl / "ready").write_text(str(time.time()))
    print(f"[phc_server] ready @ {ctrl} (num_envs={ARGS.num_envs}, "
          f"device={ARGS.device_id})", flush=True)

    last_active = time.time()
    stats = {"batches": 0, "motions": 0, "errors": 0}
    while True:
        cmds = sorted(glob.glob(str(ctrl / "cmd_*.json")))
        if not cmds:
            if time.time() - last_active > ARGS.idle_timeout:
                print(f"[phc_server] idle {ARGS.idle_timeout}s, exit. stats={stats}",
                      flush=True)
                return
            time.sleep(0.2)
            continue

        last_active = time.time()
        for cmd_path in cmds:
            cmd_path = Path(cmd_path)
            try:
                cmd = json.load(open(cmd_path))
                bundle, out_path = Path(cmd["bundle"]), Path(cmd["out"])
                conv.max_frames = int(cmd.get("max_frames", ARGS.max_frames))
                score_scale = float(cmd.get("score_scale", 0.15))
                term_dist = float(cmd.get("termination_distance",
                                          ARGS.termination_distance))

                data = dict(np.load(str(bundle), allow_pickle=True))
                keys = sorted({k.rsplit("_", 1)[0] for k in data if k.endswith("_poses")})
                motions = {}
                for k in keys:
                    try:
                        motions[k] = conv.convert_data({
                            "poses": data[f"{k}_poses"],
                            "trans": data[f"{k}_trans"],
                            "betas": data.get(f"{k}_betas"),
                            "mocap_framerate": float(data.get(f"{k}_fps", 30)),
                        })
                    except Exception as e:
                        print(f"[phc_server] convert fail {k}: {e}", flush=True)
                ok_keys = list(motions.keys())
                for k in keys:
                    if k not in motions:
                        motions[k] = motions[ok_keys[0]] if ok_keys else None
                if not motions:
                    raise RuntimeError("all converts failed")

                _install_motions(task, motions)
                total = int(task._motion_lib._num_unique_motions)
                results = {}
                for start_idx in range(0, total, task.num_envs):
                    _load_batch(task, start_idx)
                    batch_res = _rollout_one_batch(task, player, score_scale, term_dist)
                    results.update(batch_res)
                for k in keys:
                    if k not in ok_keys:
                        results[k] = {"tracking_score": 0.0, "terminated": True,
                                      "survival": 0.0, "mpjpe_cm": 25.0,
                                      "num_steps": 0, "status": "convert_fail"}
                json.dump(results, open(out_path, "w"))
                stats["batches"] += 1
                stats["motions"] += len(keys)
            except Exception as e:
                stats["errors"] += 1
                import traceback
                print(f"[phc_server] batch error: {e}\n{traceback.format_exc()}",
                      flush=True)
                try:
                    json.dump({"error": str(e)}, open(cmd["out"], "w"))
                except Exception:
                    pass
            finally:
                try:
                    cmd_path.unlink()
                except OSError:
                    pass
            print(f"[phc_server] done {Path(cmd.get('bundle', '?')).name} "
                  f"stats={stats}", flush=True)


def main():
    global ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctrl-dir", required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--termination-distance", type=float, default=0.5)
    parser.add_argument("--idle-timeout", type=int, default=1800)
    ARGS = parser.parse_args()

    ctrl = Path(ARGS.ctrl_dir)
    ctrl.mkdir(parents=True, exist_ok=True)
    init_pkl = ctrl / "init.pkl"
    _ensure_init_pkl(init_pkl)

    player_class = run_hydra.im_amp_players.IMAMPPlayerContinuous
    player_class.run = _server_run
    overrides = [
        "learning=im_mcp_big",
        "exp_name=phc_comp_3",
        "env=env_im_getup_mcp",
        "robot=smpl_humanoid",
        "env.zero_out_far=False",
        "robot.real_weight_porpotion_boxes=False",
        "env.num_prim=3",
        f"env.motion_file={init_pkl}",
        "env.models=['output/HumanoidIm/phc_3/Humanoid.pth']",
        f"env.num_envs={ARGS.num_envs}",
        "env.episode_length=100000",
        "env.enableEarlyTermination=True",
        "env.cycle_motion=False",
        "+env.seq_motions=True",
        "+env.min_length=-1",
        "env.stateInit=Start",
        f"learning.params.config.player.games_num={ARGS.num_envs}",
        "headless=True",
        "no_virtual_display=True",
        f"device_id={ARGS.device_id}",
        f"rl_device=cuda:{ARGS.device_id}",
        "seed=0",
        "torch_deterministic=False",
        "epoch=-1",
        "test=True",
        "im_eval=True",
    ]

    os.chdir(ROOT)
    config_dir = str((PHC_DIR / "data" / "cfg").resolve())
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=overrides)
    run_hydra.main.__wrapped__(cfg)


if __name__ == "__main__":
    main()
