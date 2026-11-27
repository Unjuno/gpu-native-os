# Experiment protocol

Every experiment records hypothesis, test workload, pass/fail/uncertain criteria, counterexamples and uncertainty. Correctness gates are evaluated before performance. All current measurements are CPU-only.

## Historical evidence

| Experiment | Result | Boundary |
| --- | --- | --- |
| v01 | 480/480 functional configurations; 2,048 tasks in stress test | Sequentially consistent virtual GPU, no hardware timing |
| v02 | 242 candidates in 24 generations; 5,778 held-out contexts, no mismatch reported | Fixed byte-sized workload; no general OS proof |
| v02 native | 16,384 contexts: 143.851 ms unfused matrix, 17.064 ms direct C, 15.048 ms fused C | CPU medians; overlapping timing ranges; no established advantage over direct C |

See historical READMEs and raw JSON in each experiment directory. The v02 original archive also contains comparison.json, generated.c and logs; those original files are not imported here. The scripts regenerate them. Do not treat historical samples as newly measured data.

## H/T/D/C/U template

- H: specify a falsifiable claim, workload, backend, threshold, deadline and resource budget.
- T: record environment (hardware, clock or unfixed, affinity, compiler, dependencies), input seeds, sample size, repeat count, stopping rule. Separate training and holdout workloads.
- D: PASS only when the specified functional checks pass and the prespecified performance criterion is met. FAIL on a counterexample; UNCERTAIN when evidence is insufficient or timing overlaps.
- C: document competing explanations such as simple fusion, compiler effects, timing noise, cache or memory behavior.
- U: record raw samples, median and range; quantify measurement uncertainty where feasible; otherwise state what cannot be measured.

## Required controls

Scalar oracle; direct integer batch; unfused exact matrix; simple fused matrix; evolved candidate; adaptive selection. Use independent holdouts. For scheduling and memory tests, record lost/duplicate tasks, commit trace, abort behavior and bounded progress. Record real GPU latency/utilization only when measured on a real GPU or appropriately validated timing model.

Use the runner to isolate experiments. Timing is not a regression-test assertion: CI hardware is not a controlled benchmark environment.
