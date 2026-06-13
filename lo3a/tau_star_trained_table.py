#!/usr/bin/env python3
"""Regenerate the R2 frozen-vs-trained decode-head table from tau_star_trained.json (+ optional step-budget
control files tau_star_trained_gpt2_*.json). Prints markdown for PROVABLE_OPT §7."""
import glob, json, os
HERE = os.path.dirname(os.path.abspath(__file__))


def rows(recs):
    out = []
    for r in recs:
        if "error" in r: continue
        steps = r.get("steps", "?")
        for rk, v in r["ranks"].items():
            f, ti = v["frozen"], v["trained_tied"]
            fr = v.get("trained_free", {})
            out.append((r["model"], steps, int(rk), r["PR"], r["span90"],
                        f["R32_open"], fr.get("R32_open", float("nan")), ti["R32_open"],
                        f["R32_closed"], ti["R32_closed"], f["R1"], ti["R1"]))
    return out


def main():
    recs = json.load(open(os.path.join(HERE, "tau_star_trained.json")))
    print("### R2 — frozen SVD lens vs matched-capacity trained projection (open-class R@32)\n")
    print("| model | steps | rank r | PR | span90 | frozen | trained-free | trained-tied | tied−frozen | closed froz→tied |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m, st, rk, PR, sp, fo, fro, tio, fc, tc, f1, t1 in rows(recs):
        tag = "**r≈PR**" if abs(rk - PR) <= 2 else ("r≈span90" if abs(rk - sp) <= 2 else f"{rk}")
        print(f"| {m} | {st} | {tag} ({rk}) | {PR} | {sp} | {100*fo:.0f}% | "
              f"{100*fro:.0f}% | {100*tio:.0f}% | {100*(tio-fo):+.0f}pp | {100*fc:.0f}→{100*tc:.0f}% |")
    # step-budget control
    ctrl = sorted(glob.glob(os.path.join(HERE, "tau_star_trained_gpt2_*.json")))
    if ctrl:
        print("\n### Step-budget control (gpt2, tied head): does the high-rank 'frozen advantage' shrink with steps?\n")
        print("| steps | r≈PR open froz→tied | r≈span90 open froz→tied |")
        print("|---:|---:|---:|")
        base = json.load(open(os.path.join(HERE, "tau_star_trained.json")))
        g150 = next((r for r in base if r["model"] == "gpt2"), None)
        series = ([("150", g150)] if g150 else []) + [(os.path.basename(c).split("_")[-1].split(".")[0], json.load(open(c))[0]) for c in ctrl]
        for st, r in series:
            if not r: continue
            ks = sorted(int(k) for k in r["ranks"].keys())
            lo, hi = str(ks[0]), str(ks[-1])
            vl, vh = r["ranks"][lo], r["ranks"][hi]
            print(f"| {st} | {100*vl['frozen']['R32_open']:.0f}→{100*vl['trained_tied']['R32_open']:.0f}% "
                  f"| {100*vh['frozen']['R32_open']:.0f}→{100*vh['trained_tied']['R32_open']:.0f}% |")


if __name__ == "__main__":
    main()
