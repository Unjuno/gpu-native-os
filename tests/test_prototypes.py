"""CPU functional regressions; no timing assertions."""
import json
import random
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "v02"))
import evolve as m
from compare import Direct


class PrototypeTests(unittest.TestCase):
    def test_v01(self):
        out = subprocess.check_output([sys.executable, str(ROOT / "experiments/v01/vgpu.py"), "--self-test"], text=True)
        result = json.loads(out)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["passed"], result["total"])
        self.assertTrue(result["timeout_detected"] and result["bounds_fault_detected"])

    def test_v02_controls(self):
        plans = [m.compile_genome(m.BASE)] + [m.compile_genome(m.full(b)) for b in m.BACKENDS]
        plans.append(tuple(tuple(Direct(op) for op in ops) for ops in m.PROGRAM))
        for n, seed in [(1, 1), (17, 202), (128, 303)]:
            data = m.make_input(n, seed)
            gold = m.oracle(data, True)
            for plan in plans:
                self.assertTrue(m.equivalent(m.run(plan, data, trace=True), gold, True))

    def test_v02_mutations(self):
        rng = random.Random(20261128)
        genome = m.BASE
        data = m.make_input(31, 8001)
        gold = m.oracle(data, True)
        for _ in range(24):
            genome = m.mutate(genome, rng)
            plan = m.compile_genome(genome)
            self.assertTrue(m.equivalent(m.run(plan, data, trace=True), gold, True))

    def test_v02_abort(self):
        data = m.make_input(256, 107)
        gold = m.oracle(data)
        for backend in m.BACKENDS:
            plan = m.compile_genome(m.full(backend))
            for block in (2, 3):
                result = m.run(plan, data, abort=(3, block))
                self.assertGreater(result[3]["discarded_blocks"], 0)
                self.assertTrue(m.equivalent(result, gold))

    def test_v02_xor(self):
        self.assertTrue(all(int(m.XOR[x, y]) == x ^ y for x in range(256) for y in range(256)))


if __name__ == "__main__":
    unittest.main()
