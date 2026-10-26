# DeltaNet normalization correction before any transfer study

The first replay engineering run failed its predeclared tiny hidden-state
threshold1e-4: even the baseline without replay differs from Transformers
by0.0017129183. Default60-position global predictions and all nine replay
engineering predictions still agree with the reference. Zero replay is
identical to baseline. The failed absolute gate remains recorded in
initial_engineering_record.json and the original tiny capture/reference;
it is not relabeled as a pass and its tolerance is not relaxed.

Inspection finds a concrete implementation mismatch in src/deltanet.rs:

- prior denominator: sqrt(sum(q*q)) +1e-6, likewise for k;
- installed official Transformers5.10.2: sqrt(sum(q*q)+1e-6).

The distinction is material for low-norm queries/keys. The golden fixture
normalization_golden.json is generated directly from the official installed
torch_recurrent_gated_delta_rule, with its source hash and library version.
It includes small and zero query rows. A regression test added BEFORE the
fix fails on the original native kernel with maximum error0.08847154
(legacy_normalization_regression.log). The correction moves epsilon inside
the root and updates the two closed-form unit checks consistently. The
external golden supplies an independent test rather than relying on those
closed forms.

This is an intentional default-kernel behavior correction on a NEW review
branch, separate from the old observer-only PR. No old scientific capture,
checkpoint, run manifest, outcome or archive is overwritten or retuned.
The original source/coalition results concern the archived native runtime,
whose reported global reference agreements remain observations on their
fixed samples; they do not establish equal hidden-state arithmetic. Do not
pool old source vectors with a corrected-runtime first-hop capture or infer
that the old29/100 admission count holds after the correction.

The old experimental runtime binary is preserved separately. Build the
corrected executable in the dedicated replay target directory; keep its
new hash. Reuse the SAME engineering selections and independent reference,
saving new captures under distinct corrected names. The original engineering
thresholds and input/site/vector choices stay fixed. Require the same gate,
all Qwen/DeltaNet unit checks, and default60-position reference before any
fresh scientific transfer inputs may run. Any future study must freeze
its corrected runtime and recertify its own carriers.
