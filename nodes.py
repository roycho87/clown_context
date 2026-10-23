"""
Clown Context for MiniMax H3
----------------------------
Holds the previous shot's real saved latents at the start of a new shot, in matching
block positions (FRAME_PER_TOKEN = (1, 4, 4, 4, 4) from frame 0; 17-frame blocks), with a
noise mask so the sampler keeps them. An alternative to Motion Context's pinned frames:
the held frames come out as the originals, so the overlap doesn't flash and joins are exact.

Install: put this folder in ComfyUI/custom_nodes/ and restart ComfyUI.
Works next to Clown Anchoring (no node-name clash).
"""

import torch

try:
    from comfy.nested_tensor import NestedTensor
except Exception:  # very old ComfyUI
    NestedTensor = None


def _need_nested(who):
    if NestedTensor is None:
        raise RuntimeError(f"{who}: this ComfyUI has no comfy.nested_tensor; update ComfyUI.")


def _as_tensor(part):
    if isinstance(part, dict) and "samples" in part:
        return part["samples"]
    return part


def _split(value):
    """NestedTensor / list / tensor -> list of tensors, video (5-D) first."""
    if NestedTensor is not None and isinstance(value, NestedTensor):
        parts = list(value.unbind())
    elif isinstance(value, (list, tuple)):
        parts = [_as_tensor(p) for p in value]
    else:
        parts = [value]
    return parts


def _has_samples(lat):
    if lat is None or not isinstance(lat, dict):
        return False
    s = lat.get("samples", None)
    if s is None:
        return False
    if isinstance(s, (list, tuple)) and (len(s) == 0 or any(p is None for p in s)):
        return False
    return True


FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
BLOCK = 17


def _frames_of(n_latents):
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(n_latents))


def _latents_in(frames):
    """Latent count for `frames` frames starting at a block boundary (frames must end on a latent edge)."""
    full, r = divmod(frames, BLOCK)
    return 5 * full + (0 if r == 0 else 1 + (r - 1) // 4)


def _aligned_overlap(overlap, clip_frames):
    """Largest overlap <= requested whose start (clip_frames - overlap) is a block boundary."""
    f = int(overlap) - ((int(overlap) - clip_frames) % BLOCK)
    return max(0, min(f, clip_frames))


def _drop_blocks(drop_frames):
    """drop_frames -> whole 17-frame blocks to drop. Old workflows saved a True/False switch here."""
    if isinstance(drop_frames, bool):
        return 1 if drop_frames else 0
    return max(0, int(drop_frames)) // BLOCK


def _video_first(parts):
    return sorted(parts, key=lambda t: t.ndim, reverse=True)


class H3ClownContext:
    """Front version of Clown Anchoring / alternative to Motion Context's pinned frames.

    Holds the PREVIOUS shot's last `head_frames` frames at the START of this shot,
    in matching block positions, with noise mask 0 so the sampler keeps them as the
    real originals. 39 -> this shot's frames 0-38 = previous shot's frames 204-242.
    drop_frames (17 by default, steps of 17) generates that many frames at the START of
    this shot freely: 39 with 17 -> this shot's frames 0-16 are free and frames 17-38 =
    previous shot's frames 221-242, so the held frames still end on the previous shot's last frame.
    Connect shot_number and set first_shot to your run's first shot number: that shot
    ignores any previous latent, so an old clip left in the folder can't be picked up.
    """

    CATEGORY = "latent/H3"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("av_latent",)
    FUNCTION = "anchor"
    DESCRIPTION = (
        "Holds the previous shot's last frames at the start of this shot, in matching block "
        "positions, masked so the sampler keeps them. Use instead of Motion Context's pinned frames."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Starting AV latent that goes into the sampler."}),
                "head_frames": ("INT", {"default": 39, "min": 0, "max": 4096,
                                        "tooltip": "Overlap in frames (use the audio overlap, e.g. 39). The whole overlap is held."}),
                "drop_frames": ("INT", {"default": 17, "min": 0, "max": 4096, "step": 17,
                                    "tooltip": "Frames at the START of this shot to generate freely, in whole 17-frame blocks (0, 17, 34 ...). The held frames still end on the previous shot's last frame: 39 with 17 -> previous 221-242 held in this shot's 17-38; 39 with 34 -> previous 238-242 held in 34-38. 0 holds the whole overlap."}),
                "first_shot": ("INT", {"default": 1, "min": -9999, "max": 99999,
                                       "tooltip": "Number of the first shot of your run. On that shot (or earlier) the previous latent is ignored."}),
            },
            "optional": {
                "shot_number": ("INT", {"forceInput": True,
                                        "tooltip": "Connect your Shot Number. If left empty, the first-shot check is skipped."}),
                "previous_shot": ("LATENT", {"tooltip": "Previous shot's saved latent (list or packed). If empty, the latent passes through."}),
            },
        }

    def anchor(self, latent, head_frames, drop_frames=17, first_shot=1, shot_number=None, previous_shot=None):
        who = "Clown Context"
        if shot_number is not None and int(shot_number) <= int(first_shot):
            print(f"[{who}] shot {shot_number} is the first shot (first_shot = {first_shot}) - previous latent ignored, latent passed through unchanged")
            return (latent,)
        if not _has_samples(previous_shot):
            print(f"[{who}] no previous shot latent - latent passed through unchanged")
            return (latent,)
        samples = latent["samples"]
        packed = NestedTensor is not None and isinstance(samples, NestedTensor)
        parts = _split(samples)
        vi = max(range(len(parts)), key=lambda i: parts[i].ndim)
        video = parts[vi]
        if video.ndim != 5:
            raise ValueError(f"{who}: expected a 5-D video latent [B, C, T, H, W], got {tuple(video.shape)}.")
        src = _video_first(_split(previous_shot["samples"]))[0]
        if src.ndim != 5 or src.shape[1] != video.shape[1] or src.shape[3:] != video.shape[3:]:
            raise ValueError(
                f"{who}: source latent {tuple(src.shape)} doesn't match this shot {tuple(video.shape)} "
                "(same model and resolution needed)."
            )
        T, Ts = video.shape[2], src.shape[2]
        Lt, Ls = _frames_of(T), _frames_of(Ts)
        F_all = min(_aligned_overlap(head_frames, Ls), Lt, Ls)
        if F_all != int(head_frames):
            print(f"[{who}] overlap {head_frames} doesn't start on a block boundary - using {F_all}")
        # drop_frames frees whole blocks at the START of this shot; the held frames still end on
        # the previous shot's last frame. Whole blocks keep positions aligned.
        k = _drop_blocks(drop_frames)
        F = F_all - BLOCK * k
        n = _latents_in(F_all) - 5 * k
        if F <= 0 or n <= 0:
            print(f"[{who}] nothing to hold (overlap {F_all}, drop_frames {drop_frames}) - latent passed through")
            return (latent,)
        if k:
            print(f"[{who}] drop_frames {BLOCK * k}: this shot's first {BLOCK * k} frames are generated freely, holding the last {F} overlap frames")
        d0 = 5 * k
        s0 = 5 * ((Ls - F_all) // BLOCK) + 5 * k
        video = video.clone()
        video[:, :, d0:d0 + n] = src[:, :, s0:s0 + n].to(device=video.device, dtype=video.dtype)
        parts[vi] = video

        # noise mask: 0 = keep, 1 = generate. Keep any existing mask (audio, Clown Anchoring tail).
        old = latent.get("noise_mask", None)
        masks = _split(old) if old is not None else None
        if masks is not None and len(masks) != len(parts):
            masks = None
        if masks is None:
            masks = [torch.ones_like(p) for p in parts]
            if old is not None and not packed:
                masks[vi] = old
        vmask = masks[vi]
        vmask = torch.ones_like(video) if vmask.shape != video.shape else vmask.clone()
        vmask[:, :, d0:d0 + n] = 0.0
        masks[vi] = vmask

        out = {kk: v for kk, v in latent.items() if kk not in ("samples", "noise_mask")}
        if packed or len(parts) > 1:
            _need_nested(who)
            out["samples"] = NestedTensor(tuple(parts))
            out["noise_mask"] = NestedTensor(tuple(masks))
        else:
            out["samples"] = parts[0]
            out["noise_mask"] = masks[0]

        sf = _frames_of(s0); df = _frames_of(d0)
        print(f"[{who}] held latents {d0}..{d0 + n - 1} = frames {df}..{df + F - 1} "
              f"<- previous shot's frames {sf}..{sf + F - 1}")
        return (out,)


NODE_CLASS_MAPPINGS = {"H3ClownContext": H3ClownContext}
NODE_DISPLAY_NAME_MAPPINGS = {"H3ClownContext": "Clown Context"}
