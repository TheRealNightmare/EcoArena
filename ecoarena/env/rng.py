"""Counter-based random numbers.

Every random draw is hash(seed, step, stream, world, slot) on 32-bit integers, so draws need no
stored generator state, run batched on the GPU, and can be reproduced bit-for-bit in TypeScript
(Math.imul + >>> 0). Do not use torch.rand anywhere in the environment.
"""

from __future__ import annotations

import torch

MASK32 = 0xFFFFFFFF

# Stream ids: one per kind of random draw. Never reuse an id for a different purpose.
S_INIT_X = 1
S_INIT_Y = 2
S_INIT_HEADING = 3
S_INIT_AGE = 4
S_GRASS_INIT = 5
S_REFUGE_X = 6
S_REFUGE_Y = 7
S_INIT_ENERGY = 8
S_BIRTH_ANGLE = 10
S_BIRTH_HEADING = 11
S_MIGRANT_X = 12
S_MIGRANT_Y = 13
S_MIGRANT_HEADING = 14
S_POLICY = 20
SPECIES_STREAM_OFFSET = 100  # predator streams = prey stream + 100


def hash32(x: torch.Tensor) -> torch.Tensor:
    """lowbias32 integer hash (Chris Wellons). Input/output: int64 tensors holding uint32 values."""
    x = x & MASK32
    x = x ^ (x >> 16)
    x = (x * 0x7FEB352D) & MASK32
    x = x ^ (x >> 15)
    x = (x * 0x846CA68B) & MASK32
    x = x ^ (x >> 16)
    return x


def uniform(
    seed: int,
    step: int,
    stream: int,
    world: torch.Tensor,
    slot: torch.Tensor,
) -> torch.Tensor:
    """Uniform floats in [0, 1) with the broadcast shape of (world, slot). world/slot are int64."""
    h = hash32(torch.as_tensor(seed & MASK32, dtype=torch.int64, device=world.device))
    h = hash32(h ^ (step & MASK32))
    h = hash32(h ^ (stream & MASK32))
    h = hash32(h ^ world)
    h = hash32(h ^ slot)
    return h.to(torch.float64).div(4294967296.0).to(torch.float32)


def grid(n_worlds: int, n_slots: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Index tensors shaped [W, 1] and [1, N] for broadcasting into uniform()."""
    w = torch.arange(n_worlds, dtype=torch.int64, device=device).unsqueeze(1)
    s = torch.arange(n_slots, dtype=torch.int64, device=device).unsqueeze(0)
    return w, s
