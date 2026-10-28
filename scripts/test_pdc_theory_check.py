import copy
import unittest

from pdc_theory_check import grade


def fixture():
    domain = {"rivals": [str(i) for i in range(9)], "theory": "bounded integer addition",
        "gamma": .5, "hallucination_bar": .1, "wide_miss_bar": .25,
        "token_map": {str(i): 100+i for i in range(9)}, "cases": []}
    bundle = {"rivals": domain["rivals"], "records": []}
    for i in range(75):
        domain["cases"].append({"id": str(i), "a": 0, "b": 0, "theory_atom": "0",
            "ids": [10, i], "split": "r5"})
        bundle["records"].append({"id": str(i), "a": 0, "b": 0, "context_ids": [10, i],
            "body_scores": {"full": [2.]+[0.]*8}, "host_token": 100,
            "chart": {"queries": [{"winners": ["1"], "readout": "viterbi"},
                                  {"winners": ["0"], "readout": "marginal"}], "fragile": True},
            "commitments": [{"readout": "viterbi", "status": "abstain", "atom": None, "reason": "guard"},
                {"readout": "marginal", "status": "committed", "atom": "0", "body": "full"}]})
    return bundle, domain


class TheoryTests(unittest.TestCase):
    def test_abstention_is_not_a_miss_or_a_passing_zero_denominator(self):
        bundle, domain = fixture()
        result = grade(bundle, domain)["summary"]
        self.assertEqual(result["misses"], 0)
        self.assertEqual(result["fragile_truth"], 75)
        self.assertEqual(result["viterbi"]["bar_result"], "untested")
        self.assertIsNone(result["viterbi"]["hallucination_rate"])
        self.assertEqual(result["marginal"]["affirmed"], 75)

    def test_truth_does_not_launder_a_missing_body_decision(self):
        bundle, domain = fixture()
        corrupt = copy.deepcopy(bundle)
        corrupt["records"][0]["body_scores"]["full"] = [0., 2.]+[0.]*7
        with self.assertRaisesRegex(ValueError, "body decision"):
            grade(corrupt, domain)

    def test_theory_is_checked_from_operands_not_host_or_manifest_label(self):
        bundle, domain = fixture()
        domain["cases"][0]["theory_atom"] = "1"
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            grade(bundle, domain)


if __name__ == "__main__":
    unittest.main()
