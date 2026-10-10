# Material training configuration audit

Reviewed on 2026-10-10 against the PBRnxt revision pinned by Texture Studio: `73ab49a0cc0de5ea70e7aa94fb1a7234dd59ab35`. Sources are the published code and README; they do not establish every training run that produced `pbrnxt_402236.pth`.

## Published upstream recipe

| Setting | Published base-generator configuration |
| --- | --- |
| Optimizer | NAdam with decoupled weight decay |
| Betas / decay | `0.98`, `0.99` / `0.01` |
| Learning rate / minimum | `1e-5` / `1e-6` |
| Schedule | Cosine warm restarts, initial period `4000`, multiplier `2` |
| Batch / accumulation | `6` / virtual batch `1` |
| Iterations / patch | `200000` / `128 × 128` |
| Precision / device | BF16 autocast with gradient scaling / CUDA |
| Losses | Huber pixel `0.01`, perceptual `1`, auxiliary `0.001`, GAN `0.01`, render `0` |
| Training scope | Full generator and discriminator |

The effective call appears in [pinned `train.py`](https://github.com/aaf6aa/PBRnxt/blob/73ab49a0cc0de5ea70e7aa94fb1a7234dd59ab35/train.py#L326-L375); optimizer, schedule and loss implementations appear [earlier in that script](https://github.com/aaf6aa/PBRnxt/blob/73ab49a0cc0de5ea70e7aa94fb1a7234dd59ab35/train.py#L21-L82). Function-signature defaults alone are not the effective recipe. `virtual_batch_size` is an accumulation setting separate from DataLoader batch size.

The released mapping combines a 96-channel SCUNet generator and four 12-block RRDB branches with width 32. Its wrapper enlarges outputs 4×. See [pinned `pbrnxt_net.py`](https://github.com/aaf6aa/PBRnxt/blob/73ab49a0cc0de5ea70e7aa94fb1a7234dd59ab35/pbrnxt_net.py#L8-L46) and the [GUI's checkpoint selection](https://github.com/aaf6aa/PBRnxt/blob/73ab49a0cc0de5ea70e7aa94fb1a7234dd59ab35/test_gui.py#L157-L165). Architecture dimensions are properties of the weights, so changing them is not a compatible fine-tuning setting.

## Native configuration and changes

Before this change, the native trainer accepted learning rate through its command interface, but Model Training exposed only LoRA rank and alpha for adapter optimization. Adam coefficients, clipping, loss weights and one-map updates were fixed; the learning rate was constant. Saved configuration did not describe the complete optimizer recipe. The controls now connect to the native trainer, persist as preferences and accompany checkpoints, exports and run records.

| Setting | Native default | Meaning |
| --- | --- | --- |
| Learning rate | `1e-5` | Peak rate with warmup or cosine scheduling |
| Optimizer | AdamW | Adam remains available for earlier recipes |
| Weight decay | `0` | Nonnegative; AdamW decouples decay from the moments |
| Gradient accumulation | `1` | Complete maps averaged per optimizer update |
| Beta 1 / beta 2 | `0.9` / `0.999` | Moment coefficients, each in `[0, 1)` |
| Optimizer epsilon | `1e-8` | Positive numerical stability term |
| Maximum gradient norm | `1` | Global norm clipping; `0` disables clipping |
| Learning rate schedule | Constant | Optional cosine decay |
| Minimum learning rate ratio | `0.1` | Cosine floor as a fraction of the selected learning rate |
| Warmup | `0` updates | Linear warmup before the chosen schedule |
| Random seed | `17` | Adapter initialization and map shuffling |
| LoRA rank / alpha | `8` / `8` | New adapter factors; warm starts retain recorded factors |
| Native grid | `1024 × 1024` | Whole-grid training; source maps are never resized |
| Training scope | Map output branch | Selected RRDB branch; developer scope also adapts its decoder |
| Updates per map / time limit | `100` / `30` minutes | Optimizer applications per map and deadline |
| Quick check / checkpoint interval | `20` / `0` updates | `0` disables the periodic action |

AdamW is a useful configurable extension because its decay does not enter either moment estimate. With the default decay of zero and the same betas, epsilon and clipping, its update matches the earlier Adam recipe. This default avoids silently changing regularization on existing adapters. See the [official AdamW definition](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html). Upstream NAdamW adds Nesterov momentum scheduling; it is not synonymous with AdamW and is not offered by this native implementation. See the [official NAdam definition](https://docs.pytorch.org/docs/2.14/generated/torch.optim.NAdam.html).

Warm starts load learned weights and the recorded completed step, then initialize new optimizer moments. They are a new optimization run rather than exact restoration of the earlier optimizer state. Optimizer updates, scheduling and warmup therefore count from the current run. Export packages intentionally exclude optimizer tensors.

Planned optimizer updates are `training map count × updates per map`. Accumulation does not divide that plan: each update processes up to the selected number of complete maps from the seeded shuffled stream. A stop or deadline applies a partially accumulated group using its actual map count. Warmup and cosine scheduling advance once per optimizer application. The configurable cosine floor must be positive so the final planned update still changes the weights.

## Fixed choices and deliberate differences

The native model uses Float32 with reduced precision fast math disabled and physical batch size one. It freezes the base weights and adapts the selected layers. It retains all learned operations needed by the chosen output, while omitting the wrapper's circular padding, roll and final 4× enlargement. Training uses mean value L1 plus four times multiscale detail L1. Detail compares horizontal and vertical differences at offsets `1`, `2`, `4`, `8`, weighting each by the inverse offset. Validation reports value MAE against the supplied target.

The upstream perceptual, adversarial, auxiliary and rendered losses are not implemented in this adapter workflow. Exposing switches for them would imply nonexistent behavior. Their addition would require the corresponding networks, target outputs and loss validation. Likewise, the native pipeline does not reproduce upstream stochastic depth or feature-noise behavior; it uses deterministic operations so stage replay and frozen-feature caching preserve gradients. The upstream architecture's stochastic components are visible in [pinned `scunetv2_arch.py`](https://github.com/aaf6aa/PBRnxt/blob/73ab49a0cc0de5ea70e7aa94fb1a7234dd59ab35/archs/scunetv2_arch.py#L292-L375) and [RRDB feature-noise code](https://github.com/aaf6aa/PBRnxt/blob/73ab49a0cc0de5ea70e7aa94fb1a7234dd59ab35/archs/rrdbnet_arch.py#L81-L99).

## Speed and validation limits

The [pinned README performance result](https://github.com/aaf6aa/PBRnxt/blob/73ab49a0cc0de5ea70e7aa94fb1a7234dd59ab35/README.md#performance) is about 512 × 512 inference on an RTX 3070. It cannot predict a Float32 native training update on a Mac. A 1024 × 1024 map has 64 times as many pixels as a 128 × 128 patch; backward differentiation, stage replay, compilation and validation add work. This pixel ratio is a sizing comparison, not a measured runtime multiplier.

Larger accumulation processes more maps per optimizer update; it cannot make a single update faster. Changing optimizer or learning rate also does not eliminate forward/backward work. Compare equivalent native grid, layer scope, validation settings and completed optimizer updates when assessing speed. Separate first-update compilation from later update duration, and record compilation count and cache reuse alongside wall time. The [October 10 engine benchmark and regression record](validation/training-settings-performance-2026-10-10.md) measures the pinned weights at 1K, including the saved rank-64 decoder configuration. Representative held-out materials are still required for image quality; small synthetic fixtures establish mechanics only.
