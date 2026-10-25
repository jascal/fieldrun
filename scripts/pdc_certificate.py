"""Candidate-wise PDC commitment guard, independent of any external truth theory."""
import math


def certify(query, body_scores, host_scores, rivals, covered_rivals, gamma):
    readout = query["readout"]
    if readout not in ["viterbi", "marginal"]:
        raise ValueError("a supported readout name is required")
    result = {"readout": readout, "status": "abstain", "atom": None,
        "gamma": gamma, "readout_margin": query["margin"], "memory": "none"}

    def abstain(reason):
        return {**result, "reason": reason}

    if rivals != covered_rivals or len(rivals) < 2 or len(set(rivals)) != len(rivals):
        return abstain("candidate_coverage_mismatch")
    if query["status"] != "selected" or len(query["winners"]) != 1:
        return abstain("no_unique_readout_winner")
    if query["margin"] is None or not math.isfinite(query["margin"]) or query["margin"] <= gamma:
        return abstain("readout_margin_guard")
    atom = query["winners"][0]
    if atom not in rivals:
        return abstain("winner_outside_coverage")
    best_bodies = sorted(query["winning_bodies"].get(atom, []))
    if not best_bodies or best_bodies[0] not in body_scores:
        return abstain("missing_body_witness")
    body = best_bodies[0]
    scores = body_scores[body]
    if len(scores) != len(rivals) or len(host_scores) != len(rivals):
        return abstain("score_coverage_mismatch")
    if not math.isfinite(gamma) or gamma <= 0 or not all(math.isfinite(v) for v in [*scores, *host_scores]):
        return abstain("nonfinite_or_invalid_certificate")
    t = rivals.index(atom)
    pairs = []
    for v in range(len(rivals)):
        if v == t:
            continue
        gap = scores[t] - scores[v]
        drift = (host_scores[t] - scores[t]) - (host_scores[v] - scores[v])
        pairs.append({"rival": rivals[v], "body_gap": gap, "gap_drift": drift,
            "delta": abs(drift)/2, "slack": gap - abs(drift)})
    result.update({"body": body, "attempted_atom": atom, "margin": min(p["body_gap"] for p in pairs),
        "covered_rivals": list(covered_rivals), "pairs": pairs,
        "minimum_pairwise_slack": min(p["slack"] for p in pairs)})
    if result["margin"] <= gamma:
        return abstain("body_margin_guard")
    if any(p["slack"] <= 0 for p in pairs):
        return abstain("pairwise_fidelity_guard")
    return {**result, "status": "committed", "atom": atom, "reason": None,
        "kernel_statement": "PDC_Admission.pairwise_source_preserves_decision",
        "scope": "recorded_candidate_set_only"}
