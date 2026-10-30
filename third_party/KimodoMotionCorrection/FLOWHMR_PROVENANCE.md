# Source

Vendored from [toyxyz/toyxyz_RGBMocap](https://github.com/toyxyz/toyxyz_RGBMocap/tree/941acffbd76129a4cf5a2e49bb3c8a8fbe9b5b22/third_party/KimodoMotionCorrection),
commit `941acffbd76129a4cf5a2e49bb3c8a8fbe9b5b22`, directory
`third_party/KimodoMotionCorrection`.

The original `UPSTREAM.json` records the Kimodo source revision and toyxyz patches.
The original Apache-2.0 `LICENSE` and dependency notices in `notices/` are retained.

Local build changes:

- `CMakeLists.txt` discovers pybind11 installed in the selected Python environment.
- `setup.py` accepts `CMAKE_ARGS`, including a local `Eigen3_DIR` for offline builds.
- Built distributions retain the license and dependency notices.
- `.gitignore` excludes generated build files and binaries.

The motion correction implementation is unchanged.
