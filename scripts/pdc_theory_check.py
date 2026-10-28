"""Grade an exported bundle against bounded addition, without choosing any atom."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def judge_commitment(commitment, body_scores, rivals, truth):
    """External theory judgment; it neither chooses an atom nor alters its guard."""
    if commitment["status"] == "abstain":
        verdict = "abstain"
    else:
        body = body_scores[commitment["body"]]
        index = rivals.index(commitment["atom"])
        if not all(body[index] > s for j, s in enumerate(body) if j != index):
            raise ValueError("a committed atom has no recorded body decision")
        verdict = "affirmed" if commitment["atom"] == truth else "hallucination"
    return {**commitment, "theory_verdict": verdict,
        "theory_affirms": None if commitment["atom"] is None else commitment["atom"] == truth}


def grade(bundle, domain):
    by_id = {c["id"]: c for c in domain["cases"] if c["split"] == "r5"}
    if len(bundle["records"]) != 75 or set(by_id) != {r["id"] for r in bundle["records"]}:
        raise ValueError("theory domain and exported bundle do not match")
    rows = []
    for record in bundle["records"]:
        expected = by_id[record["id"]]
        if record["context_ids"] != expected["ids"] or (record["a"], record["b"]) != (expected["a"], expected["b"]):
            raise ValueError("theory operands or prompt mismatch")
        truth = str(record["a"] + record["b"])
        if truth != expected["theory_atom"] or truth not in bundle["rivals"]:
            raise ValueError("theory file is inconsistent with bounded integer addition")
        truth_index = bundle["rivals"].index(truth)
        deciding_bodies = [name for name, scores in record["body_scores"].items()
            if all(scores[truth_index] > s for j, s in enumerate(scores) if j != truth_index)]
        vit, mar = record["chart"]["queries"]
        fragile_truth = mar["winners"] == [truth] and vit["winners"] != [truth]
        verdicts = []
        for commitment in record["commitments"]:
            verdicts.append(judge_commitment(commitment, record["body_scores"], bundle["rivals"], truth))
        rows.append({"id": record["id"], "theory_atom": truth, "deciding_bodies": deciding_bodies,
            "miss": not deciding_bodies, "fragile_truth": fragile_truth, "fragile": record["chart"]["fragile"],
            "judgments": verdicts, "host_token": record["host_token"],
            "readout_winners": {q["readout"]: q["winners"] for q in record["chart"]["queries"]}})
    summary = {"n": len(rows), "misses": sum(r["miss"] for r in rows),
        "fragile_truth": sum(r["fragile_truth"] for r in rows), "fragile": sum(r["fragile"] for r in rows)}
    for readout in ["viterbi", "marginal"]:
        judgments = [next(j for j in row["judgments"] if j["readout"] == readout) for row in rows]
        counts = Counter(j["theory_verdict"] for j in judgments)
        committed = counts["affirmed"] + counts["hallucination"]
        rate = counts["hallucination"] / committed if committed else None
        summary[readout] = {"committed": committed, "abstained": counts["abstain"],
            "affirmed": counts["affirmed"], "hallucination": counts["hallucination"], "hallucination_rate": rate,
            "bar_result": "untested" if rate is None else "passed" if rate < domain["hallucination_bar"] else "retired",
            "wide_miss": rate is not None and rate >= domain["wide_miss_bar"],
            "abstention_reasons": dict(Counter(j["reason"] for j in judgments if j["status"] == "abstain")),
            "raw_theory_agreement": sum(r["readout_winners"][readout] == [r["theory_atom"]] for r in rows)/len(rows),
            "committed_full_host_agreement": sum(j["status"] == "committed" and
                domain["token_map"][j["atom"]] == row["host_token"] for j, row in zip(judgments, rows, strict=True)) / committed
                if committed else None}
    return {"schema": "pdc.theory-grade.v1", "theory": domain["theory"], "gamma": domain["gamma"],
        "hallucination_bar": domain["hallucination_bar"], "summary": summary, "records": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["bundle", "domain", "out"]:
        p.add_argument("--"+name, type=Path, required=True)
    args = p.parse_args()
    result = grade(json.loads(args.bundle.read_text()), json.loads(args.domain.read_text()))
    result["sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.bundle, args.domain]}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
