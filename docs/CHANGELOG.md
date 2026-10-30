# Changelog

Notable changes to FlowHMR.

## [2026-10-30]

### Added

- Static camera mode for the CLI and Web demo.
- Configurable SAM chunk size in the CLI and Web demo (default: 16 frames; hand batch: 2× chunk).
- Optional SAM backbone compilation.
- Optional query-batched attention for long-sequence generation.
  Thanks to [@Scar1etCrystal](https://github.com/Scar1etCrystal) for the prototype and experiments in [#4](https://github.com/flowhmr/flowhmr/issues/4).
- [Performance benchmarks](performance.md).

### Changed

- Replace optional post-processing with [toyxyz_RGBMocap correction](https://github.com/toyxyz/toyxyz_RGBMocap),
  based on [Kimodo MotionCorrection](https://github.com/nv-tlabs/kimodo): contact planning,
  pelvis adjustment and anatomical IK, shared by CLI and Web.
- Faster SAM feature extraction with GPU crops and batched body/hand inference.
- Around 10× faster SAM extraction and over 5× faster end-to-end inference (static camera, compiled)
  on a 300-frame A100 benchmark; see the [performance report](performance.md).
- Simplified inference documentation with launch commands and parameters.
- Collapsible Web Settings with SAM/VGGT memory guidance.

### Fixed

- Keep frame limits consistent across tracking, camera and SAM features.
- Refresh caches when person, frame-range or camera settings change.
- Rebuild invalid SAM features and truncated static-camera caches.
- Exclude time spent waiting for person selection from Web detection timing.
- Finalize interrupted Web stage timing when restoring run history.

## [2026-10-26]

### Added

- Persistent Web run history and deletion controls.
- Stop and cancel controls for running and queued jobs.
- Automatic full-video frame mode in the Web demo.

### Fixed

- FFmpeg compatibility during video transcoding.

## [2026-10-21]

### Fixed

- Use first-frame-relative camera rotations for motion generation.

## [2026-10-20]

### Changed

- Source code licensed under Apache 2.0.

## [2026-10-04] — Initial release

### Added

- Initial release with inference, training, the Web demo and setup documentation.
