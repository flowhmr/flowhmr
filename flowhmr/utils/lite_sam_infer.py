"""Feature-only SAM scheduling using the unchanged upstream model and weights.

Body inference supplies the hand boxes. Hand inference stops at the decoder
tokens consumed by FlowHMR, before the unused fusion and refinement stages.
"""

from contextlib import nullcontext

import numpy as np
import torch


class _HandTokensReady(Exception):
    """Local control flow: the hand decoder has produced its conditioning token."""


def _combine_batches(batches, device):
    # Each frame occupies the batch axis, with its own intrinsics. The person
    # axis stays at one even when left and right crops share a forward pass.
    return {
        key: torch.cat([batch[key] for batch in batches], dim=0).to(device)
        for key, value in batches[0].items()
        if isinstance(value, torch.Tensor)
    }


@torch.no_grad()
def extract_tokens(estimator, images, boxes, cameras, *, gpu_preprocess=False,
                   merge_hands=False):
    """Return CPU float32 ``(frames, 3072)`` tokens in body/left/right order.

    ``images`` contains RGB uint8 frames, one selected person per frame.
    CPU crops and separate hand passes at batch size one preserve the original
    operations leading to the tokens. Batching/GPU crops are optional and may
    introduce floating-point differences. Full SAM outputs use the estimator's
    original ``process_one_image`` entry point instead.
    """
    if not images or len(images) != len(boxes) or len(images) != len(cameras):
        raise ValueError("Expected equally sized, nonempty images, boxes and cameras")
    boxes = torch.as_tensor(boxes).detach().cpu().numpy().reshape(-1, 4)
    cameras = torch.as_tensor(cameras).detach().cpu().reshape(-1, 3, 3)
    if len(boxes) != len(images) or len(cameras) != len(images):
        raise ValueError("Expected one box and one camera per frame")
    if not np.isfinite(boxes).all() or not torch.isfinite(cameras).all():
        raise ValueError("Boxes and cameras must be finite")
    for image in images:
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected RGB uint8 HWC images")

    from sam_3d_body.data.utils.prepare_batch import prepare_batch

    device = torch.device(estimator.device)
    model = estimator.model
    body_tokens, hand_tokens = [], []
    original_single = len(images) == 1 and not gpu_preprocess and not merge_hands

    def capture_body(_module, _args, output):
        body_tokens.append(output[0].detach() if original_single else output[0][:, 0].detach().clone())

    def capture_hand(_module, _args, output):
        hand_tokens.append(output[0].detach() if original_single else output[0][:, 0].detach().clone())
        if not original_single or len(hand_tokens) == 2:
            raise _HandTokensReady

    def collected_tokens():
        if len(body_tokens) != 1 or len(hand_tokens) != 2:
            raise RuntimeError("SAM did not produce body, left hand and right hand tokens")
        pieces = [body_tokens[0], *hand_tokens]
        if original_single:
            pieces = [piece[:, 0] for piece in pieces]
        tokens = torch.cat(pieces, dim=-1)
        if tokens.shape != (len(images), 3072):
            raise RuntimeError(f"Unexpected SAM token shape: {tuple(tokens.shape)}")
        return tokens.to(device="cpu", dtype=torch.float32)

    def prepare(image, box, camera, transform):
        if gpu_preprocess:
            from .sam_gpu_preprocess import prepare_gpu_batch
            return prepare_gpu_batch(image, box[None], camera[None], transform)
        return prepare_batch(image, transform, box[None], cam_int=camera[None])

    # Upstream helpers use the current CUDA device internally. Scope it to the
    # estimator, including when callers select a non-default GPU.
    context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
    state_names = ("_batch_size", "_max_num_person", "_person_valid", "body_batch_idx", "hand_batch_idx")
    state = {name: getattr(model, name) for name in state_names if hasattr(model, name)}
    handles = []
    try:
        with context:
            handles.append(model.decoder.register_forward_hook(capture_body))
            handles.append(model.decoder_hand.register_forward_hook(capture_hand))
            if original_single:
                # Keep the public estimator's exact preprocessing and scheduling
                # for the default path. Only computation after both hand tokens
                # have been produced is omitted.
                try:
                    estimator.process_one_image(
                        images[0], bboxes=boxes, cam_int=cameras, inference_type="full",
                    )
                except _HandTokensReady:
                    pass
                return collected_tokens()
            frames = ([torch.as_tensor(np.ascontiguousarray(img), device=device) for img in images]
                      if gpu_preprocess else images)
            body_batch = _combine_batches([
                prepare(image, box, camera, estimator.transform)
                for image, box, camera in zip(frames, boxes, cameras)
            ], device)
            model._initialize_batch(body_batch)
            body = model.forward_step(body_batch, decoder_type="body")
            # Upstream _get_hand_box indexes affine_trans[0, person]. Expose the
            # flattened frame transforms in that layout without mixing cameras.
            geometry = {"affine_trans": body_batch["affine_trans"].flatten(0, 1).unsqueeze(0)}
            left_boxes, right_boxes = model._get_hand_box(body, geometry)
            del body, body_batch

            def run_hands(indices, sides):
                batches = []
                for side in sides:
                    for index in indices:
                        image = frames[index]
                        box = (left_boxes if side == "left" else right_boxes)[index].copy()
                        if side == "left":
                            box[[0, 2]] = image.shape[1] - box[[2, 0]] - 1
                            image = image.flip(1) if gpu_preprocess else image[:, ::-1]
                        batches.append(prepare(image, box, cameras[index], estimator.transform_hand))
                batch = _combine_batches(batches, device)
                model._initialize_batch(batch)
                try:
                    model.forward_step(batch, decoder_type="hand")
                except _HandTokensReady:
                    pass
                else:
                    raise RuntimeError("SAM hand decoder did not produce conditioning tokens")

            if merge_hands:
                run_hands(range(len(images)), ("left", "right"))
                hand_tokens = list(hand_tokens.pop().split(len(images), dim=0))
            else:
                run_hands(range(len(images)), ("left",))
                run_hands(range(len(images)), ("right",))

            return collected_tokens()
    finally:
        for handle in handles:
            handle.remove()
        for name in state_names:
            if name in state:
                setattr(model, name, state[name])
            elif hasattr(model, name):
                delattr(model, name)
