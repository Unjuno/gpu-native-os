# v03 experiment report

Two CPU-only experiment runs completed without state mismatches. Across both widths, each run checked 406 differential cases (1,218 backend comparisons). Each run used 9,346 holdout contexts across both widths, with six execution modes. Across both runs: 192 candidate evaluations, counting candidates independently per strategy/width/run (not necessarily globally unique); 18,692 holdout contexts. Differential cases use fixed seeds and are repeated, not independent new cases.

Environment: Intel Xeon Platinum 8573C, x86_64 Linux 6.18.57 / glibc 2.41, affinity CPU 0, Python 3.13.5, NumPy 2.3.5, OPENBLAS_NUM_THREADS/OMP_NUM_THREADS/MKL_NUM_THREADS=1, clock unfixed. Nine interleaved wall-clock samples per mode; timings exclude compilation/search and include preparation and scheduler overhead.

## Median milliseconds, 4096 contexts

| Width / run | Direct integer batch | Unfused matrix | Fused dense | Fused delta | Random search | Evolution |
|---|---:|---:|---:|---:|---:|---:|
| 32 / 1 | 10.180 | 20.616 | 9.781 | 9.787 | 11.596 | 9.384 |
| 32 / 2 | 10.423 | 21.179 | 9.803 | 8.768 | 10.131 | 10.124 |
| 64 / 1 | 10.825 | 22.321 | 11.031 | 9.158 | 10.957 | 9.906 |
| 64 / 2 | 10.794 | 23.231 | 10.696 | 9.723 | 11.471 | 9.434 |

Raw sample ranges and peak traced allocation can be computed from measurements-4096.csv. Peak traced allocation is not device memory or process RSS. The fused-delta matrix data is 352 B (32-bit) or 640 B (64-bit) versus 3,564 B or 7,128 B for unfused. This measures matrix coefficients and index arrays only, not total plan/VM memory. Peak traced allocation for fused delta exceeds the direct integer baseline (e.g. 1,635,956 B vs 1,091,040 B in run 1, 64-bit).

## H/T/D/C/U

- H: exact modular affine fusion preserves guest state and can reduce runtime; evolution may improve on simple fusion.
- T: fixed 20-op workload and independent random/edge-case differential checks; equal 24-candidate search budgets; two input seed sets; six modes; 9 timing repeats.
- D: PASS on observed functional equivalence, no malformed image accepted in the specified tests. PASS on observed fused-vs-unfused latency reductions. UNCERTAIN on superiority to simple fusion or direct integer batch: timing ranges overlap and evolution does not consistently beat simple fusion. No formal statistical superiority criterion was set in advance.
- C: fusion alone explains much of the gain; Python/NumPy dispatch overhead, matrix allocation, cache behavior and timing noise remain competing explanations.
- U: no fixed clock or hardware cycle/energy counters; no device model or GPU; two runs and one optimization workload do not establish generalization. No combined measurement uncertainty estimated.

## Data provenance

The following SHA-256 fingerprints identify the locally retained full JSON reports and exact benchmark sources:

- machine.py: 87d30ac500a1620aa8e7b8ae472b60a61c2363e81af8e006b569bea968d7e52a
- experiment.py: 932635c362bd80426b2e482ae9af9c62eb4647c634f181694e6d1eb5e13e18de
- first full results.json: 8a4c09ca8f0fca5df99a0b6f9eede553fe4ce7e885861e841204d2d3ed6c143e
- repeat full repeat.json: 71921b0971887c0159b1ab1b429e6db34fff8e04475caf4b8021978a67525851

The full JSON reports are distributed separately with the experiment artifact; this commit includes the exact source and largest-batch raw samples, not the full search trace. Reproduction yields new timing samples rather than identical fingerprints.
