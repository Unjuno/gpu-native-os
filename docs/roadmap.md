# Roadmap

The version names below are proposed milestones, not promised dates.

- [x] v01: functional virtual GPU, guest task scheduling, self-test and stress workload.
- [x] v02: exact byte-sized affine reference, mutation search, simple fusion/direct controls, pre-commit discard model and generated C benchmark.
- [ ] v03: define versioned machine state and 32/64-bit ISA semantics; independent scalar reference; generic instruction loading and exact differential tests.
- [ ] v04: shared RAM, address translation/permissions, atomic operations, concurrent updates and rollback-safe commit semantics; tests for isolation, conflicting writes, deadlocks and faults.
- [ ] v05: dynamic resource allocation and batch scheduling under mixed load; measure latency distributions, throughput and memory footprint against direct controls; constrain starvation and interrupt response.
- [ ] v06: automatically optimize legal instruction groups, layout and dispatch; compare evolutionary search with random search and simple fusion under equal search budgets; unseen workloads and retained counterexamples.
- [ ] v07: generate GPU executable code, validate on simulator and real GPU; compare physical performance and energy.
- [ ] v1: independent boot, devices, persistent storage, application execution, permissions and recovery; evaluate minimal-CPU and hostless architectures.

First development priority: v03 ISA/state contract. Do not change reference semantics during a performance experiment. Each milestone requires documented scope, repeatable tests, and a negative-result policy.
