"""Vendored LFM3 reference implementation (torch/transformers remote code) for the tiny parity fixture.

Source: huggingface.co/LiquidAI/LFM3-1.7B @ ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7 — the three .py files copied
verbatim (Apache-2.0, headers retained). transformers 5.22 ships lfm2* but not lfm3, and the fixture must build
offline, so `scripts/lfm3_ref.py build` imports the model class from here instead of `trust_remote_code`.
Only code is vendored: the LFM3 *weights* are under the LFM Open License and are never copied into this repo.
"""
