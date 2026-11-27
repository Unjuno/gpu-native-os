# Matrix GPU v0.2 — hierarchical exact ISA and automated optimization

CPU functional prototype, separate from v0.1 (not a drop-in replacement). It demonstrates the design principle of defining a correct reference, then searching equivalent representations. It does not implement a complete OS or GPU hardware timing model.

## Reproduction

Python 3.13.5, NumPy 2.3.5 and GCC 14.2.0 were used. x86_64, Intel Xeon Platinum 8573C; 5 visible CPUs and cgroup quota of 4 CPU-equivalents. Timing was wall clock, not cycle count, clock frequency not fixed. The native benchmark was pinned to CPU 0; change the affinity if needed. Shared library compiled for the local architecture; rebuild it locally.

```sh
python -m pip install numpy==2.3.5
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python evolve.py --generations 24 --population 12
python compare.py
taskset -c 0 python native.py  # Linux; omit taskset if unavailable
```

`results.json`, `comparison.json`, `native.json` contain all reported raw timing samples and chosen plans. `generated.c` is generated automatically from the search results; `native.py` compiles it. The archive does not ship the machine-specific shared object. Timing noise means a replay may choose different winners even with the same mutation seed.

## Machine and semantics

Each context has eight unsigned 8-bit registers, a homogeneous coordinate fixed to one, 256 bytes of private RAM, and a program counter. An epoch groups ready contexts by PC, executes each group's selected representation and commits at basic-block boundaries. Five blocks contain 41 affine instructions; the block exits demonstrate memory reads/writes, XOR, data-dependent branches, looping, and halt. Loop count is initially between 1 and 8. The guest workload is a fixed test program rather than an arbitrary binary loader. No shared RAM, interprocess communication, physical I/O, or autonomous boot. The host interpreter still supplies the scheduling loop.

Affine opcodes (add, subtract, constant add, constant multiply, move) use 9x9 matrices over Z/256Z, not floating point. For the homogeneous vector x, an opcode maps x to Mx; rows encode the updated register and leave other coordinates unchanged. Composition of consecutive opcodes gives M_k ... M_1; each group can be evaluated as a dense matrix, changed rows only, or sparse nonzero terms. Different groups must not cross the block boundary. XOR uses a 256x256 lookup table (implicit columns of a transition matrix). Indexed RAM loads/stores and branches are represented by selection/scatter/condition primitives, not executed by a dense matrix multiplier. They can be interpreted as one-hot selection or finite-state transitions; the prototype is not a matrix-multiplication-only machine.

The affine equivalence argument is exact within the modeled ring: all three NumPy kernels are linear in the full 9 coordinates, so equality on the standard basis implies equality on all coordinate vectors. Matrices are composed in program order. The scalar affine oracle is separately checked on zero and eight register basis points with homogeneous coordinate 1. Addition/multiplication wrap mod 256; generated C uses unsigned arithmetic and truncation to uint8_t. With nine terms bounded by 255*255, sums fit in 32-bit unsigned integers. A preliminary C validation failed because the direct affine implementation was tested at homogeneous coordinates other than 1, outside its specified domain; the matrix validation retains the full domain and the direct implementation is now checked on valid affine states. This is not a proof of NumPy/GCC correctness or of a general OS.

## Search

A genome stores contiguous partition boundaries and one of three backends per group. Mutations split, merge or replace a backend. Elitist selection retains the fastest measured candidates; packed plan size is used as tie-breaker. Every candidate is checked for algebraic equivalence and compared against an independent scalar interpreter on training input, including SHA-256 hashes of every epoch's full register/RAM/PC state and exact final arrays. SHA-256 is used only for test comparison, not security. Finalists are re-timed before selection. The adaptive policy trains separate size buckets with synthetic batches; at runtime the current group size selects an equivalent plan. The native stage automatically generates C for the 68 distinct block partitions from the 242 genomes and compares these to separately generated C for ordinary integer opcodes. All compiled plans are checked on basis and random states. No LLM or neural model participates in this search; this is evolutionary search and measured autotuning, not a search over physical hardware.

## Verification and measured limits

- 24 generations, population 12, 242 unique candidates, all training trace checks passed.
- Six held-out batches, 5,778 contexts, 104,712 committed block transitions; exact final arrays and per-epoch hashes matched scalar oracle. XOR checked on all 65,536 byte pairs.
- Additional comparison and native holdouts use different seeds; all modes matched the oracle in these runs. Native stage includes 16,384 contexts.
- Pre-commit discard on either store branch leaves register/PC/RAM writes uncommitted; batch retries later and final state agrees. This only models one injected abort at an epoch boundary. No asynchronous preemption or irreversible external I/O rollback.
- `results.json` main run: logical packed static-plan representation 3,362 -> 402 bytes; these are the payload of the hypothetical encoding, excluding Python object headers, scheduler, XOR table, RAM and adaptive alternatives. Peak traced allocations for execution at 4,096 contexts: 1,236,980 -> 1,243,559 bytes. Total live RAM does not shrink.
- `native.json` benchmark: 13 randomized-order repetitions per mode and batch size, includes input copying, grouping, dispatch, register and RAM updates; excludes code generation/compilation, autotuning, trace collection, correctness tests and initial workload generation. At 4,096 contexts the matrix reference median is 35.890 ms, C direct 4.155 ms, C fused matrix 4.411 ms, C adaptive matrix 4.185 ms. At 16,384: 143.851, 17.064, 15.048, 16.506 ms respectively. Ranges overlap for the compiled versions. No robust performance advantage over direct integer execution is established. Timing variability and training-size overfit are relevant; evolution itself has not demonstrated a clear advantage over simple full fusion. No GPU utilization, energy, real-world application performance, security or worst-case scheduling guarantees are measured.

## Next steps

Generalize the ISA to wider words, shared memory and controlled side effects; define formal block/interrupt observability; improve layout/search and validate on independent workloads; add GPU code generation and run on real GPU or a validated timing simulator. Treat integrity, confidentiality, availability, and isolation as separate requirements; correctness of compiler transforms cannot establish cryptographic or system security.
