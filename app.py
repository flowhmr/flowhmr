"""FlowHMR web demo: drop a video into the page, get SMPL-H motion in a 3D viewer.

Usage (from the repo root, V2M env):
    python app.py                                 # checkpoints/flowhmr_latest, port 8080
    python app.py --ckpt <ckpt> --port 8080 --device cuda:0

Options: python app.py --help. Implementation: tools/web/server.py.
"""
import os
import runpy

if __name__ == "__main__":
    runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "web", "server.py"),
                   run_name="__main__")
