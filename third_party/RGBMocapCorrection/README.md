# RGB MoCap correction

Computational contact-correction code from
[toyxyz_RGBMocap](https://github.com/toyxyz/toyxyz_RGBMocap/tree/941acffbd76129a4cf5a2e49bb3c8a8fbe9b5b22/mocap),
pinned to `941acffbd76129a4cf5a2e49bb3c8a8fbe9b5b22`.

This package preserves the upstream contact planning, anatomical leg IK,
pelvis and sole clearance handling, bounded temporal windows, and optional
palm correction. `flow_correction.SoleGeometry` supplies upstream SMPL-H
surface probes; `correction.correct_arrays` runs the correction. Project
storage, GUI and application export functions are omitted. `UPSTREAM.json`
records included functions, source hashes and packaging changes.

Install the sibling `KimodoMotionCorrection` package first, then from the
FlowHMR repository root:

```bash
pip install --no-build-isolation ./third_party/RGBMocapCorrection
```

The native extension must provide `rgb_mocap_anatomical_version == 1`.
The upstream MIT license and third-party notices are retained here;
components listed in those original notices are not all included in this
computational subset. The native Kimodo solver retains its separate license
in `../KimodoMotionCorrection/LICENSE`.
