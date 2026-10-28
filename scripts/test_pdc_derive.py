"""R1 gate instances against the R0 kernel statements."""
import math
import random
import unittest

from pdc_derive import Body, evaluate


class FiniteDeriveTests(unittest.TestCase):
    def test_kernel_disagreement_witness(self):
        result = evaluate({"A": [Body("a1", 0), Body("a2", 0)],
                           "B": [Body("b1", 0.5)]}, 1)
        v, m = result["queries"]
        self.assertEqual((v["readout"], v["winners"]), ("viterbi", ["B"]))
        self.assertEqual((m["readout"], m["winners"]), ("marginal", ["A"]))
        self.assertAlmostEqual(m["scores"]["A"], math.log(2))
        self.assertTrue(result["fragile"])

    def test_generated_pic_argmax_and_temperature_invariance(self):
        rng = random.Random(20261024)
        for _ in range(256):
            # PIC logits: full source coalition plus a bias, both T-free.
            logits = {str(a): sum(rng.randint(-20, 20) / 8 for _ in range(4))
                      + rng.randint(-8, 8) / 8 for a in range(9)}
            table = {a: [Body("full-coalition", s)] for a, s in logits.items()}
            table["absent"] = []
            expected = sorted(a for a, s in logits.items() if s == max(logits.values()))
            for temperature in [1e-6, 0.125, 1, 8, 1e6]:
                result = evaluate(table, temperature)
                self.assertFalse(result["fragile"])
                for query in result["queries"]:
                    self.assertEqual(query["winners"], expected)
                    self.assertEqual({a: s for a, s in query["scores"].items()
                                      if s is not None}, logits)

    def test_ties_and_absence_do_not_commit(self):
        for query in evaluate({"a": [], "b": []}, 1)["queries"]:
            self.assertEqual(query["status"], "absent")
            self.assertEqual(query["winners"], [])
        for query in evaluate({"a": [Body("x", -5)], "b": [Body("x", -5)]}, 1)["queries"]:
            self.assertEqual(query["status"], "tie")
            self.assertEqual(query["margin"], 0)

    def test_invalid_inputs_refuse(self):
        for temperature in [0, -1, math.inf, math.nan]:
            with self.assertRaises(ValueError):
                evaluate({"a": []}, temperature)
        for table in [{}, {"a": [Body("x", math.nan)]},
                      {"a": [Body("x", 0), Body("x", 1)]}]:
            with self.assertRaises(ValueError):
                evaluate(table, 1)


if __name__ == "__main__":
    unittest.main()
