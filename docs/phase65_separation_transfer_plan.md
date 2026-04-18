# Phase65 Separation and Transfer Plan

## Current interpretation
A3 solves most collapse, but two bottlenecks remain:
- slot embeddings are still too similar
- cat/dog transfer still fails

## Next experiments
1. Slot separation loss ablation
2. Multi-view consistency benchmark
3. Harder synthetic shape curriculum
4. Freeze-decoder test
5. Small cat/dog pilot

## Priority 1: Slot separation
Add an explicit loss that pushes entity-0 and entity-1 slots apart while preserving survival.

Metrics:
- cross-slot cosine
- visible survival min
- entity balance
- amodal IoU min

## Priority 2: Multi-view consistency
Group clips with the same scene identity but different cameras.
Same-entity slots should match across views, cross-entity slots should stay apart.

## Priority 3: Harder shapes
Use richer primitives and randomization to reduce shortcut learning.
Examples:
- ellipsoid
- cone
- capsule
- deformed primitives

## Priority 4: Freeze-decoder
Freeze the scene module and train a small decoder on top of the scene maps.
If this works, the scene prior is genuinely useful.

## Priority 5: Cat/Dog pilot
Run only a small pilot set after the separation loss and harder curriculum improve.
