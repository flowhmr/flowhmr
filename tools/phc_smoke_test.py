"""Submit one FlowHMR motion to a running PHC+ score server and print its tracking score.

Usage (simulation env, with tools/phc_score_server.py already running on --ctrl-dir):
    python tools/phc_smoke_test.py --ctrl-dir /tmp/flowhmr_phc_smoke \
        --motion output/example/flowhmr_latest.npz
"""
import argparse
import json
import os
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctrl-dir", required=True, help="--ctrl-dir of the running score server")
    ap.add_argument("--motion", required=True, help="FlowHMR output npz (poses / trans / betas)")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    ctrl = args.ctrl_dir
    if not os.path.exists(os.path.join(ctrl, "ready")):
        sys.exit(f"score server not ready: {ctrl}/ready missing")

    m = np.load(args.motion)
    bundle, out = os.path.join(ctrl, "smoke_bundle.npz"), os.path.join(ctrl, "smoke_scores.json")
    cmd = os.path.join(ctrl, "cmd_smoke.json")
    np.savez(bundle, example_poses=m["poses"], example_trans=m["trans"],
             example_betas=m["betas"], example_fps=30)
    with open(cmd, "w") as f:
        json.dump({"bundle": bundle, "out": out}, f)

    t = time.time()
    while os.path.exists(cmd):
        if time.time() - t > args.timeout:
            sys.exit(f"timeout after {args.timeout}s")
        time.sleep(1)

    res = json.load(open(out))
    print(json.dumps(res))
    r = res.get("example", {})
    ok = "error" not in res and r.get("terminated") is False and r.get("survival") == 1.0
    print("PHC_SMOKE_OK" if ok else "PHC_SMOKE_FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
