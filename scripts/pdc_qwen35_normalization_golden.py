"""Generate a low-norm DeltaNet regression from the installed official HF kernel."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import torch
import transformers
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    torch_recurrent_gated_delta_rule,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    assert not a.out.exists()
    q = [
        [0.0001, -0.0002, 0.0003, -0.0004],
        [0.001, 0.0002, -0.0004, 0.0003],
        [0.0, 0.0, 0.0, 0.0],
    ]
    k = [
        [0.0002, 0.0004, -0.0001, 0.0003],
        [-0.0003, 0.0001, 0.0002, 0.0005],
        [0.0001, -0.0003, 0.0002, 0.0004],
    ]
    v = [[0.5, -0.25], [0.1, 0.9], [-0.4, 0.6]]
    g = [-0.3, -0.1, -0.4]
    beta = [0.6, 0.7, 0.8]
    Q = torch.tensor(q, dtype=torch.float32)[None, :, None, :]
    K = torch.tensor(k, dtype=torch.float32)[None, :, None, :]
    V = torch.tensor(v, dtype=torch.float32)[None, :, None, :]
    G = torch.tensor(g, dtype=torch.float32)[None, :, None]
    B = torch.tensor(beta, dtype=torch.float32)[None, :, None]
    out, _ = torch_recurrent_gated_delta_rule(
        Q,
        K,
        V,
        G,
        B,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    source = Path(inspect.getfile(torch_recurrent_gated_delta_rule))
    report = dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        expected=out.flatten().tolist(),
        transformers_version=transformers.__version__,
        reference_source=str(source),
        reference_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        formula="x * rsqrt(sum(x*x) + 1e-6)",
    )
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
