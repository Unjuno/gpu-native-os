#!/usr/bin/env python3
"""Instruction-level, functional virtual GPU. Python 3.10+, no dependencies.
The host interprets instructions; boot, job queue, dispatch and termination
are implemented in the guest assembly, not in this emulator.
"""
import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

ISA = dict(SET=2, ADD=3, MUL=3, LOAD=2, STORE=2, ATOM_ADD=3, CAS=4,
           BEQ=3, BNE=3, BLT=3, JMP=1, VID=1, VADD_S=3, VMUL=3,
           VREDUCE_SUM=2, HALT=0, TRAP=0)
BRANCH = {'BEQ', 'BNE', 'BLT', 'JMP'}
READONLY = {'wid', 'lanes', 'roots'}


def assemble(path):
    code, labels, consts = [], {}, {}
    for number, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        if line.startswith('.const '):
            _, name, value = line.split()
            if name in consts or name in labels:
                raise ValueError(f'duplicate symbol: {name}')
            consts[name] = int(value, 0)
        elif line.endswith(':'):
            name = line[:-1]
            if name in labels or name in consts:
                raise ValueError(f'duplicate symbol: {name}')
            labels[name] = len(code)
        else:
            op, *args = line.split()
            if op not in ISA or len(args) != ISA[op]:
                raise ValueError(f'line {number}: invalid instruction {line}')
            code.append((op, args, number))
    rom = []
    for op, args, line in code:
        resolved = [consts.get(a, a) for a in args]
        if op in BRANCH:
            resolved[-1] = labels[args[-1]]
        rom.append((op, tuple(resolved), line))
    return tuple(rom)


@dataclass
class Wave:
    regs: dict
    vectors: dict = field(default_factory=dict)
    pc: int = 0
    halted: bool = False
    instructions: int = 0


class VirtualGPU:
    def __init__(self, rom, waves, lanes, roots, ram_words=65536):
        if not (1 <= waves <= 128 and 1 <= lanes <= 128 and 0 <= roots <= 4096):
            raise ValueError('require 1..128 waves/lanes and 0..4096 roots')
        self.rom = rom
        self.lanes = lanes
        if not 1 <= ram_words <= 2**24:
            raise ValueError("require 1..2**24 RAM words")
        self.ram = [0] * ram_words  # device reset; word-addressed
        self.waves = [Wave({'wid': i, 'lanes': lanes, 'roots': roots}) for i in range(waves)]
        self.steps = 0
        self.vector_lane_ops = 0

    def step(self, w):
        if w.halted:
            raise RuntimeError('cannot issue a halted wave')
        if not 0 <= w.pc < len(self.rom):
            raise RuntimeError(f'PC out of range: {w.pc}')
        op, a, line = self.rom[w.pc]
        w.pc += 1
        w.instructions += 1
        self.steps += 1

        def val(x):
            if isinstance(x, int):
                return x
            if x in w.regs:
                return w.regs[x]
            return int(x, 0)  # undefined register or symbol fails loudly

        def put(r, n):
            if r in READONLY or not isinstance(r, str) or not r.startswith('s_'):
                raise RuntimeError(f'invalid scalar destination: {r}')
            w.regs[r] = n  # mathematical integers, no hardware overflow model

        def addr(x):
            n = val(x)
            if not 0 <= n < len(self.ram):
                raise RuntimeError(f'wave {w.regs["wid"]}, line {line}: bad address {n}')
            return n

        def vector(r):
            return w.vectors[r]

        def vput(r, values):
            if not isinstance(r, str) or not r.startswith('v_'):
                raise RuntimeError(f'invalid vector destination: {r}')
            w.vectors[r] = tuple(values)
            self.vector_lane_ops += self.lanes

        if op == 'SET':
            put(a[0], val(a[1]))
        elif op in ('ADD', 'MUL'):
            x, y = val(a[1]), val(a[2])
            put(a[0], x + y if op == 'ADD' else x * y)
        elif op == 'LOAD':
            put(a[0], self.ram[addr(a[1])])
        elif op == 'STORE':
            self.ram[addr(a[0])] = val(a[1])
        elif op == 'ATOM_ADD':
            p, n = addr(a[1]), val(a[2])
            old = self.ram[p]
            self.ram[p] = old + n
            put(a[0], old)
        elif op == 'CAS':
            p, expected, new = addr(a[1]), val(a[2]), val(a[3])
            old = self.ram[p]
            if old == expected:
                self.ram[p] = new
            put(a[0], old)
        elif op in ('BEQ', 'BNE', 'BLT'):
            x, y = val(a[0]), val(a[1])
            if {'BEQ': x == y, 'BNE': x != y, 'BLT': x < y}[op]:
                w.pc = a[2]
        elif op == 'JMP':
            w.pc = a[0]
        elif op == 'VID':
            vput(a[0], range(self.lanes))
        elif op == 'VADD_S':
            n = val(a[2])
            vput(a[0], (x + n for x in vector(a[1])))
        elif op == 'VMUL':
            vput(a[0], (x * y for x, y in zip(vector(a[1]), vector(a[2]), strict=True)))
        elif op == 'VREDUCE_SUM':
            put(a[0], sum(vector(a[1])))
            self.vector_lane_ops += max(0, self.lanes - 1)
        elif op == 'HALT':
            w.halted = True
        elif op == 'TRAP':
            raise RuntimeError(f'guest trap: wave {w.regs["wid"]}, line {line}')
        else:
            raise RuntimeError(f'unknown instruction {op}')

    def run(self, seed=0, max_steps=1_000_000, issue="round"):
        # Hardware-like instruction issue; random fair round ordering exposes
        # interleavings. This is not the guest OS's job scheduling algorithm.
        rng = random.Random(seed)
        active = list(self.waves)
        while active:
            if issue == "round":
                rng.shuffle(active)
                issued = active
            elif issue == "random":
                issued = [rng.choice(active)]
            else:
                raise ValueError(f"invalid issue policy: {issue}")
            for w in issued:
                if self.steps >= max_steps:
                    raise TimeoutError(f'instruction budget exhausted: {self.steps}')
                self.step(w)
            active = [w for w in active if not w.halted]


def check(rom, waves, lanes, roots, seed, max_steps=1_000_000, issue="round", ram_words=65536):
    gpu = VirtualGPU(rom, waves, lanes, roots, ram_words)
    gpu.run(seed, max_steps, issue)
    # Readback only after all waves halt. Host oracle is NOT used during execution.
    scratch = 16 + 6 * roots
    output = scratch + roots
    count1, count2 = output + roots, output + 2 * roots
    actual = gpu.ram[output:output + roots]
    expected = [3 * sum(x*x for x in range(i*lanes + 1, (i+1)*lanes + 1)) + i
                for i in range(roots)]
    per_wave = [w.regs["s_executed"] for w in gpu.waves]
    checks = {
        'boot_published': gpu.ram[0] == 1,
        'queue_drained': gpu.ram[1] == gpu.ram[2] == 2 * roots,
        'no_live_jobs': gpu.ram[3] == 0,
        'all_waves_halted': all(w.halted for w in gpu.waves),
        'root_exactly_once': gpu.ram[count1:count1 + roots] == [1] * roots,
        'child_exactly_once': gpu.ram[count2:count2 + roots] == [1] * roots,
        'results_match': actual == expected,
        'per_wave_total_matches': sum(per_wave) == 2 * roots,
        'all_slots_published': all(gpu.ram[16+3*i+2] == 1 for i in range(2*roots)),
    }
    return {'waves': waves, 'lanes': lanes, 'roots': roots, 'seed': seed, 'issue': issue, 'ram_words': ram_words,
            'jobs_completed_per_wave': per_wave,
            'guest_instructions': gpu.steps, 'vector_lane_ops': gpu.vector_lane_ops,
            'queue_head': gpu.ram[1], 'queue_tail': gpu.ram[2],
            'live_jobs': gpu.ram[3], 'results_first_8': actual[:8],
            'checks': checks, 'status': 'PASS' if all(checks.values()) else 'FAIL'}


def self_test(rom):
    results = []
    for waves in (1, 2, 4, 8, 16):
        for lanes in (1, 3, 8, 32):
            for roots in (0, 1, 7, 32):
                for seed in (0, 1, 2):
                    for issue in ("round", "random"):
                        results.append(check(rom, waves, lanes, roots, seed, issue=issue))
    # Negative test: a too-small instruction budget must report timeout.
    try:
        check(rom, 4, 8, 7, 0, max_steps=1)
    except TimeoutError:
        timeout_detected = True
    else:
        timeout_detected = False
    # Negative test: insufficient physical memory must fault.
    try:
        check(rom, 4, 8, 7, 0, ram_words=16)
    except RuntimeError as e:
        bounds_fault_detected = "bad address" in str(e)
    else:
        bounds_fault_detected = False
    passed = sum(r['status'] == 'PASS' for r in results)
    return {'status': 'PASS' if passed == len(results) and timeout_detected and bounds_fault_detected else 'FAIL',
            'passed': passed, 'total': len(results),
            'timeout_detected': timeout_detected,
            'bounds_fault_detected': bounds_fault_detected,
            'total_guest_instructions': sum(r['guest_instructions'] for r in results),
            'failures': [r for r in results if r['status'] != 'PASS']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rom', type=Path, default=Path(__file__).with_name('gpu_os.asm'))
    p.add_argument('--waves', type=int, default=4)
    p.add_argument('--lanes', type=int, default=8)
    p.add_argument('--roots', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-steps', type=int, default=1_000_000)
    p.add_argument('--issue', choices=("round", "random"), default="round")
    p.add_argument('--ram-words', type=int, default=65536)
    p.add_argument('--self-test', action='store_true')
    a = p.parse_args()
    rom = assemble(a.rom)
    result = self_test(rom) if a.self_test else check(rom, a.waves, a.lanes, a.roots, a.seed, a.max_steps, a.issue, a.ram_words)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
