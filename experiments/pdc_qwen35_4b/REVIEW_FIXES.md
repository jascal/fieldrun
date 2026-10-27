# Post-review saved-capture checks

The 100-case grader now applies the original engineering bounds on every case:
source sum and RMS fold <1e-4*(1+max(abs(reference))), and full candidate scores
<1e-3*(1+max(abs(reference))). Missing, malformed or nonfinite vectors fail.
No grade or derivation bundle is written if any case fails. The bundle checker
recomputes these identities from captured writes, residuals and hidden states.

The saved 100 cases pass. Error maxima are 3.62098e-6 (source sum), 2.44279e-5
(RMS fold), and 6.22932e-5 (full body scores). All four original split/readout
classifications are unchanged. The independent bundle check verifies all 100
retained reference predictions and 200 commitments, with maximum independent
body-score error 1.06581e-14. These are saved-data checks: zero new model calls.

`review_recheck.json` and `review_verification.json` retain the new results.
`review_sources.json` hashes the repaired checkers. The original capture bytes,
policies, thresholds, source manifests and published results remain unchanged.
For historical source hashes, the bundle checker accepts an explicit
`--frozen-source-root` containing the original e8e8707 source snapshot and its
pinned binary; every original hash must still match. This does not execute that
snapshot. The current checker reads and validates the saved numerical captures.

Covered candidate agreement is now counted from committed witness/host-probe
argmax matches. Singleton covered-host agreement is counted from each witness's
actual source argmax versus the host token. The saved singleton diagnosis remains
negative: zero exact/centered admissions over 100 cases, hence null agreement;
see `review_singleton_summary.json`.

Validation: 19 Python tests, including corruptions to the final (100th) case that
must prevent grading, and a deliberately mismatching singleton witness that must
report 1/2 agreement rather than 1.0. No parameters were retuned.
