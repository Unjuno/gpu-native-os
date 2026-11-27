# Architecture — research target, not implemented specification

## Goal

A GPU-native computer, with no required general-purpose host CPU after boot as the end target. A minimal CPU for startup, exceptional or sequential work is a comparison design, not a requirement.

## Proposed execution model

1. Guest state holds registers, program counters, memory, dependencies and pending events.
2. Exact integer/reference semantics define each opcode and its observable effects.
3. A hierarchical representation permits individual operations, composed blocks and larger equivalent plans. Affine operations use modular integer matrices; non-affine operations require explicit semantics (e.g. selection, table lookup, tensor or Boolean operations). Do not assume a fixed linear map can represent arbitrary logic on the original state space.
4. A runtime groups ready work by operation and resource constraints; selects a valid representation; executes in spatially and temporally partitioned GPU batches.
5. Changes become visible at specified commit boundaries. Uncommitted work may be discarded or retried. Irreversible external I/O needs a separate commit protocol.
6. Search optimizes representation, layout and batch policy against exact semantics and measured costs. Formal equivalence is required wherever claimed; finite tests are not a general proof.

## Current prototype boundaries

- v01: guest scheduler runs on a CPU-emulated custom wave ISA; sequentially consistent memory, cooperative fixed tasks, no isolation or hardware boot.
- v02: 8-bit affine arithmetic over Z/256Z, private per-context RAM, fixed workload, host-driven epoch scheduling. Non-affine operations are explicit primitives. No GPU device execution. Generated C targets CPU, not CUDA.
- These experiments are independent. v03 should establish a shared guest state and ISA contract before integrating them.

## Next interface to specify

Opcode encoding and word width; address and pointer format; load/store and atomics; commit observability; traps and interrupts; privilege and isolation; device I/O; deterministic reference behavior and memory ordering.

Future execution backends should share this contract: scalar CPU reference, batch CPU model, validated GPU simulator, physical GPU. Performance results are not interchangeable between backends.
