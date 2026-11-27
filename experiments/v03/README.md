# v03 — Exact 32/64-bit guest ISA

Status: CPU functional research prototype; not an integrated OS or hardware timing simulator.

## Run

From the repository root, with Python 3.13 and NumPy 2.3.5:

```sh
python -m pip install -r requirements.txt
python experiments/v03/experiment.py --test
mkdir -p runs/v03
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/v03/experiment.py --budget 24 --seed 20261128 --reps 9 --out runs/v03/first.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/v03/experiment.py --budget 24 --seed 20261129 --reps 9 --out runs/v03/repeat.json
```

Use an unused output path: existing results are not overwritten. For exact reproduction of CPU affinity, run with taskset -c 0 if CPU 0 is available. The experiment measures wall-clock time; the search winners are timing-dependent, not deterministic despite seeded inputs. Do not run with Python -O.

## ISA contract

- Image: little-endian header (4-byte magic GNO3, 1-byte word width, 4-byte unsigned instruction count), followed by fixed 17-byte instructions (1-byte opcode and two unsigned 64-bit operands). Opcodes in machine.OPS define the versioned mapping. Invalid images and operands are rejected.
- Each context: eight unsigned word registers, sixteen private word-addressed RAM locations, PC, remaining fuel, status. Word width is 32 or 64 bits. Arithmetic wraps modulo 2^width. All contexts start at PC 0 with the specified fuel.
- MOV/SET/ADD/SUB/ADDI/MULI update registers. XOR and ANDI are exact bitwise operations. LOAD dst,addrReg and STORE addrReg,src require an in-bounds address. JNZ reg,target and JMP target use absolute instruction indices. HALT consumes fuel and retains its PC.
- Each successfully retired instruction consumes one fuel. A fault leaves that instruction uncommitted. If fuel is exhausted before execution, the status is BUDGET (before PC bounds checking). Out-of-range PC/address yields FAULT. RUN/HALT/FAULT/BUDGET are encoded 0/1/2/3. Terminal states are compared including registers, RAM, PC, fuel and status.
- The independent scalar oracle uses Python integers. Batch execution uses NumPy uint32/uint64. No floating-point arithmetic enters guest semantics.
- Affine instructions are representable as homogeneous 9-by-9 matrices over the modular ring. Composition preserves program order; delta execution writes only changed rows. Non-affine and memory/control operations are explicit boundaries. Fused execution requires enough fuel to retire the entire block; otherwise single-step fallback preserves exact prefix behavior. A branch into an interior position also falls back to the appropriate instruction.
- Contexts have private RAM; no externally observable intermediate effects across contexts. Shared memory, atomicity, externally observable instruction timing, asynchronous interrupt delivery, privilege, physical boot and GPU backend are NOT yet implemented.

## Experiment

For each width and seed: 203 differential cases, 2,699 contexts, three backend comparisons per case, and three malformed images rejected. Includes fuel prefixes, overflow, address faults, branches into fused spans, self-aliasing and random opcode sequences. The finite tests are not a general proof.

Random and mutation-based search each score 24 unique plans per width/seed, with the same two full-fusion starting plans and three training repetitions. Three finalists get five fresh training repetitions. Holdout inputs are distinct from the 128 training contexts. Six modes are verified against the scalar oracle at 1, 64, 512 and 4096 contexts. Nine interleaved timing samples per mode. Two runs, each with both widths.

See [measurements-4096.csv](measurements-4096.csv) for raw timing samples of the largest workload and [report.md](report.md) for results, source/data fingerprints and scope. The full reports contain search traces and all batch sizes and can be regenerated with the commands above.

Next: v04 shared memory, permissions and atomic/commit semantics, followed by mixed workload adaptive scheduling. Do not infer hardware speed or security from these CPU-only results.
