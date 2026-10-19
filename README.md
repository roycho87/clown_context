# Clown Context

A ComfyUI node for continuing a **MiniMax H3** shot from the end of the previous one. It's a replacement for Motion Context's pinned frames.

Clown Context takes the previous shot's **real saved latents** and holds them at the start of the new shot. It places them in the same block positions they were made in and masks them so the sampler keeps them as they are. The overlap comes out identical to the end of the previous shot, so joins are exact and the overlap doesn't flash.

Companion node: **[Clown Anchoring](https://github.com/roychodraws87/comfyui-clown-anchoring)** does the same thing at the *end* of a shot, holding the next shot's opening frames. Used together, an in-between shot can be pinned at both ends.

## Why

When you keep Motion Context's pinned frames instead of trimming them, the overlap pulses brighter about every 17 frames. The cause is H3's latent layout. Video latents come in 17-frame blocks of 5 latents (1 + 4 + 4 + 4 + 4 frames), counted from frame 0, and a latent only decodes correctly in the same position within its block. Guide frames end up in different positions than the ones they were made in, so the model redraws them slightly wrong.

Clown Context copies only whole blocks, each into the position it came from. With clip lengths of 5 + 17k frames (for example 243) and an overlap of 5 + 17k (for example 39), the overlap starts on a block boundary, so the whole overlap can be held.

## Inputs

| Input | What it does |
|---|---|
| `latent` | The starting AV latent going into the sampler. |
| `head_frames` | Overlap in frames. With 39 and 243-frame clips, this shot's frames 0–38 are held as the previous shot's frames 204–242. Use the same overlap as your audio splitter. |
| `drop_17_frames` | **Off by default.** Holds one block less, at the end furthest into this shot. With 39, frames 0–21 are held as the previous shot's frames 204–225, and frames 22–38 are generated freely. |
| `first_shot` | The first shot number of your run (default 1). That shot ignores any previous latent, so an old file left in the folder can't be picked up. |
| `shot_number` (optional) | Connect your shot number here to enable the `first_shot` check. |
| `previous_shot` (optional) | The previous shot's saved latent, for example from **H3 Motion Context Load Latent**. Accepts the plain list or a packed latent. If it's empty, the latent passes through unchanged. |

If an overlap wouldn't start on a block boundary, it's rounded down to the nearest one that does, and the console says so. Any existing noise mask, such as an audio mask, is kept.

## Wiring

Feed the sampler the plain **Reference To Video** conditioning, with no Motion Context guide. For its latent:

```
Reference To Video latent -> (audio concat, if any) -> Clown Context -> [Clown Anchoring] -> sampler
```

Save each finished shot's latent, for example with **H3 Motion Context Save Latent**, so the next shot can load it as `previous_shot`.

## What testing showed

- **Joins are exact.** The held frames came back matching the source latents, so the sampler honors the video noise mask, and the overlap doesn't flash. You can keep the overlap frames instead of trimming them.
- **Motion carries through the join.** In a side-by-side test of the same shots, Motion Context froze for about half a second at one join and then jumped to a new pose in a single frame. Clown Context kept the movement continuous through that same join. That's one observed case, not a measured average.
- **It doesn't stop long-chain drift.** In a chain of 8 continuations, background brightness, sharpness and color drifted at about the same rate as with Motion Context.

## Requirements

- A recent ComfyUI with MiniMax H3 support (`comfy.nested_tensor`).
- A way to save and load per-shot latents, such as the save and load nodes from H3 Motion Context.

## Install

Put this folder in `ComfyUI/custom_nodes/` and restart ComfyUI. It can sit next to Clown Anchoring.

## License

MIT
