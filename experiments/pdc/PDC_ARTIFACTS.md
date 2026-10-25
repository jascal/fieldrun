# Full PDC evidence artifacts

Large captures and binary frames are hosted with the central PDC release.
The PR retains small result summaries; the archive retains the complete saved evidence.

- Release: https://github.com/jascal/pdc/releases/tag/pdc-v0.10-evidence
- Asset: https://github.com/jascal/pdc/releases/download/pdc-v0.10-evidence/pdc-review-v0.10.tar.gz
- Archive SHA-256: `4719765e54374771df2ea15cd736653c4896aa0fd27538eb7f82082e2e4b3f6c`
- Extraction preserves the `fieldrun/`, `pil/`, `i-orca/`, and `pdc/` paths.
- This is the original audited snapshot; the PR has subsequent publication documentation.

The archive has 448 verified members. Its `PDC_PACKAGE_MANIFEST.json` pins all
members. `fieldrun/experiments/pdc/release_inventory.json` pins the original
source/data/proof snapshot. Model weights and environments are external
prerequisites. Start with `PDC_REVIEW.md` after extraction.

## Files supplied by the archive

| Path | Bytes | SHA-256 |
|---|---:|---|
| `fieldrun/experiments/pdc/final_audit/baseline_bundle.json` | 40925789 | `3c299b57dcd160307793b40b90655fcc9951eb179ea336c2da66e200a2735797` |
| `fieldrun/experiments/pdc/final_audit/baseline_capture.json` | 29439084 | `89262975694d0d2e1d81ed11b8f8644cf5803dc2f80411ee8338cb686d9a688d` |
| `fieldrun/experiments/pdc/final_audit/service_bundle.json` | 40925804 | `859297708dfdfccfb3887ce174cedfd27d3bac79d69332ec6a9b6b9a80cb0260` |
| `fieldrun/experiments/pdc/final_audit/service_capture.json` | 29439084 | `46abfe17596ee6433f027a7c4eec8efd839e22949e5b8ec996d8bb8bbc09a5f9` |
| `fieldrun/experiments/pdc/r2_reads.json` | 4725630 | `d45bcca9ee0ed37f797ca9011934d0ce02fb3a7a3dd58ac69642cf757d4be64a` |
| `fieldrun/experiments/pdc/r3_charts.json` | 16996095 | `08081c1d1926d90a9b5857cf1e31867091ee3542b509c31372cacbffe19c1a26` |
| `fieldrun/experiments/pdc/r3_reads.json` | 15176813 | `092ff0e42b96e7fc16edc060bec31b6d494c0efa9eec09956e2c1c8638cbd50a` |
| `fieldrun/experiments/pdc/r4_qwen_sources.jsonl` | 33436322 | `e8d40075d713786a95ed652f48796908b51294e91455d3330cd41be8186bcb00` |
| `fieldrun/experiments/pdc/r5_bundle.json` | 46889256 | `eaa8571f5a5e9fed0e490cddff0c85c2142fa278ff6743f796d3582aac8d18b0` |
| `fieldrun/experiments/pdc/r5_reads.json` | 81080507 | `ee16c8aa33013dff0ca3dc9afdeaad584e9b32f3250961eb83a145f8444711fe` |
| `fieldrun/experiments/pdc/r6_capture.json` | 271063109 | `1ffe6d05436e9bf559de0776e3912a325238b592f47b8904d5a54485fea15fac` |
| `fieldrun/experiments/pdc/r6_chart_bundle.json` | 4068486 | `e4a42383a54539cdb8b432a30d0b506ebf83536198e57f653a547e3eb1823eba` |
| `fieldrun/experiments/pdc/r6_engineering_capture.json` | 6504287 | `7fce237637af9549fcab89b618178d79ce266e559d2fc771adb468c464915432` |
| `fieldrun/experiments/pdc/r6_requests.json` | 7604769 | `173d29eecb2b2e0926120df6165e6aad7b270e5ffbaf8a5f3580df28bfa1c6a7` |
| `fieldrun/experiments/pdc/r7_fixed_data.npz` | 15135795 | `ec7a855eaad8998c6eff6dac225e610b6dce641e283037399cfa93b0655ea195` |
| `fieldrun/experiments/pdc/r8h_baseline_bundle.json` | 82736673 | `bb934d48881fb439228c5d265908ccf9442c2a1706c405cfcd572c1d6f1b6e21` |
| `fieldrun/experiments/pdc/r8h_baseline_capture.json` | 84252897 | `284af9e976fdd6e220a36fc3206e3354f7d791a505f77ba8d8b1f16cf8406c44` |
| `fieldrun/experiments/pdc/r8h_service_bundle.json` | 82736738 | `225bbe36d6843588f91bc9a01f1a0df9a056521658e2d8e8d696b84ef9de4117` |
| `fieldrun/experiments/pdc/r8h_single_request_bundle.json` | 23215873 | `d440bda058f5ec9ff96379cc07b95782c22443d8364890dd9cd89a70af6a2ac7` |
| `fieldrun/experiments/pdc/r8h_timed_baseline_capture.json` | 84252897 | `13128218daa173bef284df7e9484b9b821d0883cbc87c9d483fe1c4549723dfb` |
| `fieldrun/experiments/pdc/r8h_timed_service_capture.json` | 84252897 | `963549f1abcc66a586fe28fd1280f435a52ec1705b22cb49f515141f0f4aac29` |
