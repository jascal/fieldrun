# R3 preregistration: competing trigram-copy routes, 2026-10-24

Written before capturing or inspecting host tokens for this task. R2's bundle
and its 12-head set are retained; no heads are selected on R3 outcomes.

Generate 16 contexts using Python `random.Random(20261024)`, each drawing five
distinct IDs A,B,C,X,Y from `range(1000,1008)`. The exact context is
`[A,B,C,X,A,B,C,Y,A,B,C,X,Y,A,B,C]`. Query position is 15. Rivals are all eight
IDs 1000–1007, fixed before any forward call. Trigram continuation lookup says
the latest previously observed continuation of A,B,C is X. We report this task
label separately from host agreement. These are token-ID copy tasks.

The body generator enumerates every earlier exact trigram match of the final
trigram. Its continuation key j is a route. For each of the 12 fixed heads,
the route's source is the *actual* native weighted value write
`d_(h,j) = p_h(j|15) v_(h,j) W_O,h`. Assign this singleton coalition to the atom
equal to the token at j; there is no route for another atom. Do not discard
negative incidences or select routes by host logits. Every firing match and
head is retained, even when its attention weight underflows to zero.

Bracket: `inner(d_(h,j), Ueff_atom) + beff_atom`, where Ueff is the exact
per-position final-LayerNorm-folded unembedding, and beff is the final LN bias
read by that token. The frame and brackets are temperature independent. The
final LN scale is frozen at its measured value, as for PIC's DLA convention;
this is a readout decomposition, not a counterfactual intervention. Record
the source route, incidence, bias, and bracket separately. Store all bodies.

Evaluate Viterbi and marginal T=1 using the R1 executor. Preserve tied winner
sets as non-answers. Report exact winner agreement with the host's full-vocabulary
token (host outside rivals counts as disagreement), and separately with the host
argmax restricted to the eight preregistered rivals. Report fragility and task-label
agreement. No outcome bar is required by R3; zero fragility is retained as such.

Native capture must contain the final pre-LN residual and full-context native
hidden output, so the independent checker can fold the frame and compare full
and restricted predictions. Validate all 256 per-position full-vocabulary tokens
against the independent NumPy forward. The task names a known trigram-copy
operation. The measured paths are its candidate attention routes, not a claim
that these routes are sufficient or causally necessary for the host's decode.
Do not infer that claim from readout agreement alone.
