# Clown Context (experimental, not published)

An alternative to Motion Context's pinned frames for MiniMax H3 continuations.

It holds the previous shot's last `head_frames` frames at the **start** of the new shot. It uses the previous shot's real saved latents, placed in matching block positions and masked so the sampler keeps them.

## Inputs

| Input | What it does |
|---|---|
| `latent` | Starting AV latent going into the sampler. |
| `head_frames` | Overlap in frames. 39 holds this shot's frames 0–38 as the previous shot's frames 204–242. |
| `first_shot` | First shot number of your run (default 1). That shot ignores any previous latent. |
| `shot_number` (optional) | Connect your Shot Number node. |
| `previous_shot` (optional) | Previous shot's saved latent (H3 Motion Context Load Latent). If it's empty, the latent passes through unchanged. |

**For a continuation without Motion Context:** feed the sampler the plain Reference To Video conditioning, and feed it Clown Context's latent output.

## Test results so far

- **Held frames match the previous shot exactly,** so joins are exact and the overlap doesn't flash.
- **Long chains still darken and turn hazy** at about the same rate as Motion Context, measured at shot 8. It does **not** fix chain degradation.

## Install

Put this folder in `ComfyUI/custom_nodes/` and restart ComfyUI. It can sit next to Clown Anchoring.
