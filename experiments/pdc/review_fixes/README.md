# Post-review checker corrections

The historical capture bytes, run policies, source snapshots, timings and results
remain unchanged. These reports are saved-artifact rechecks, not new model runs.

The held-out checker now pins the full baseline to the SHA-256 published in
PDC_ARTIFACTS.md, rather than comparing that file with itself. A mutually
consistent altered baseline/service pair fails this pin. Probe and source-write
discrepancies are computed from the two captures. No earlier R5 capture exists
for the 25 reserved contexts: the recheck explicitly reports zero earlier
independent host labels checked. The archive pin establishes artifact identity,
not an independent model execution.

The service and both paired auditors require every read and source write to be
at the final decision position, with finite, correctly shaped vectors whose
sum reconstructs that residual at 1e-4*(1+max(abs(residual))). The frozen 75-case
R5 pair and 25-case held-out pair pass these checks, with zero paired differences.

New GPT-2 dumps count and time the observed forward and scorer only. Observer
validation runs after the timer, fails on nonfinite or different hidden bits,
and infers identical predictions from exact hidden identity without a second
unembed. New captures identify this measurement scope explicitly. Historical
observer-on R5 capture counts/timings include the old duplicate and must not be
used as single-forward work measurements. The published R8-H timing pair has
observer checking off and is unaffected.

Validation: 16 Python tests; Rust 14 library and 179 binary tests, including
observer scorer counts and fail-closed hidden identity. No capture or threshold
was retuned. Model weights remain external; reports use the pinned saved data.
