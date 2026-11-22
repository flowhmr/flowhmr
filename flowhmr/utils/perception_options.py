"""Backend choices and repository-relative model paths for video inference."""

DEFAULT_DETECTOR = "yolo26m"
DEFAULT_CAMERA = "pi4"
DETECTOR_WEIGHTS = {
    "yolo26m": "ckpts/yolo26m/yolo26m.pt",
    "yolox": "ckpts/yolox/yolox_l.pth",
}
CAMERA_WEIGHTS = {
    "pi4": "ckpts/pi4/model.safetensors",
    "vggt": "ckpts/vggt-omega/vggt_omega_1b_512.pt",
    "simplevo": None,
    "static": None,
}


def camera_cache_suffixes(mode):
    feature = "sam3d_feat.pt" if mode == "vggt" else f"sam3d_feat_{mode}.pt"
    return f"{mode}_camera.npz", feature
