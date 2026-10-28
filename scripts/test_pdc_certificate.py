import unittest

from pdc_certificate import certify


def query(readout="viterbi"):
    return {"readout": readout, "status": "selected", "winners": ["a"],
        "margin": 2., "winning_bodies": {"a": ["full"]}}


class CertificateTests(unittest.TestCase):
    def test_common_offsets_cancel_but_target_drift_is_covered(self):
        good = certify(query(), {"full": [2., 0.]}, [102., 100.], ["a", "b"], ["a", "b"], .5)
        self.assertEqual(good["status"], "committed")
        bad = certify(query(), {"full": [2., 0.]}, [-1., 0.], ["a", "b"], ["a", "b"], .5)
        self.assertEqual(bad["reason"], "pairwise_fidelity_guard")
        self.assertIsNone(bad["atom"])

    def test_coverage_ties_and_threshold_equality_abstain(self):
        self.assertEqual(certify(query(), {"full": [2., 0.]}, [2., 0.],
            ["a", "b"], ["a", "c"], .5)["reason"], "candidate_coverage_mismatch")
        tied = {**query(), "status": "tie", "winners": ["a", "b"]}
        self.assertEqual(certify(tied, {"full": [2., 0.]}, [2., 0.],
            ["a", "b"], ["a", "b"], .5)["status"], "abstain")
        self.assertEqual(certify(query(), {"full": [.5, 0.]}, [.5, 0.],
            ["a", "b"], ["a", "b"], .5)["reason"], "body_margin_guard")

    def test_each_rival_uses_its_own_guard(self):
        result = certify(query("marginal"), {"full": [10., 8., 0.]}, [10., 8., 8.],
            ["a", "b", "c"], ["a", "b", "c"], .5)
        self.assertEqual(result["status"], "committed")
        self.assertEqual(result["readout"], "marginal")
        # A pooled min-gap/max-drift radius would incorrectly reject this case.
        self.assertEqual(result["minimum_pairwise_slack"], 2.)


if __name__ == "__main__":
    unittest.main()
