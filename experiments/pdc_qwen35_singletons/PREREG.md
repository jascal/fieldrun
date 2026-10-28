# Qwen3.5-4B native-singleton diagnostic, frozen 2026-10-25

The completed comparison has correct full-program witnesses on all100
contexts. This new OFFLINE diagnostic asks whether any individual native
source supplies a sufficient covered-score certificate. It is not an
extension of that comparison's success gate, fresh heldout confirmation,
learning, or a source/body/margin sweep. The original comparison and all
older negative results stay unchanged. No new model forward is allowed.

Inputs: all100 original Qwen native captures, their domain/probe manifests,
original run manifest, and the pinned converted frame. Require the prior
100/100 reference-and-certificate audit to have passed. Score all65 native
sources (embedding +32 attention +32 MLP) in fixed capture order for every
case. Do not try source pairs, empty/bias, ranked subsets, model changes,
threshold variants, learned frames or extra prompt templates. Gates and
full-host RMS scale stay frozen; numerical source closure is present in
host-minus-partial drift and is not a new source. Coverage is the same2261
probe tokens. Target is the original global host winner, never chosen using
theory; if missing from coverage record out-of-coverage rather than expand it.

Primary guard: exact pairwise sufficient rule, fixed margin floor0.5.
For source scores P and actual native host scores H, require for EVERY
covered rival v != t: P(t)-P(v)>0.5 AND
P(t)-P(v)>abs((H(t)-P(t))-(H(v)-P(v))). This is the existing kernel
pairwise_source_preserves_decision guard, not a truth condition.

Comparator: same bodies/targets/candidates, original stricter centered-tail
rule m>max(0.5,2delta), 2delta=range(H-P) across target AND every covered
rival. Any passing centered witness stores gamma between max(0.5,2delta)
and m, satisfying strict m>gamma>2delta. Do not relax the floor. Exact
pairwise passage should contain centered passage up to numeric tolerance;
any disagreement inconsistent with this inclusion fails the audit.

Report both original wordings splits and totals, but label both as saved-
data diagnostics: their outputs have already been observed. For each guard:
contexts with >=1 admitted singleton, first passing witness in native order,
all passing sources, raw source-target matches without a certificate,
per-source margin, minimum pairwise slack and centered-tail bound. Conditional
covered-host agreement is undefined at zero admissions. Report full-host
winner unchanged as a fixed-frame invariant, not intervention evidence.
External addition grades certified atoms only AFTER selection, with no
certificate-to-truth inference. Every certificate must retain source name,
covered scope, fixed host target and memory=none; no replay/crystallization.

A local positive requires exact pairwise admission in >=10% of the100
contexts, with100% covered agreement; zero admissions is negative for these
procedures on these frames; 1–9% is inconclusive. No population inference,
new heldout success, causal necessity or impossibility claim is licensed.
If all singletons fail, it says nothing about size-2 or larger coalitions.
No LP free-frame optimization is performed; this cannot fill the original
intended Qwen negative-domain Farkas/impossibility gate.

Freeze this protocol, executable, fixed inputs and full frame hashes before
scoring any singleton. Independently reconstruct all reported margins,
bounds, pairwise witnesses and per-case counts from saved source vectors
using a distinct computation. Save all6500 source verdicts. Stop after
publishing these two fixed procedures, without additional body shopping.
