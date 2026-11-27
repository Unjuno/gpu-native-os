# Development workflow

1. Start with a hypothesis and an issue or experiment ID.
2. Define guest-visible semantics and an independent reference before changing execution.
3. Create a small branch/PR. Keep the reference and optimized execution paths distinct.
4. Run `python -m unittest discover -s tests -v` and a relevant isolated experiment.
5. Attach the run manifest, raw samples, and failed cases to the PR if suitable for public release. Never include credentials or proprietary inputs.
6. Report baseline, hardware, software versions, clocks/affinity, workload, seed, repetitions, and limits.
7. Merge only after correctness checks pass and the claimed improvement has suitable evidence.

Do not silently rewrite historical results in experiments/v01 or experiments/v02. New ISA/runtime work should be versioned independently (start with v03). Timing comparisons must distinguish CPU-only measurements from actual GPU measurements. Security and boot claims require independent evidence. Generated executable binaries are not committed.

For AI-assisted optimization: change the candidate, not the reference or pass/fail criteria; preserve counterexamples and reject unknown/unverified equivalence.
