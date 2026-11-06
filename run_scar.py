#!/usr/bin/env python3
"""scAR ambient RNA removal module (plan stage id: `cleaning`).

Reads the raw droplets of one pool + the called cell barcodes, estimates the
ambient profile from cell-free droplets (scar.setup_anndata), trains scAR and
writes the denoised native counts (integers, scAR's default stochastic
rounding) as h5ad.

API checked against scAR v0.7.0 (bioconda `scar`).
"""

import random
import sys
from pathlib import Path

import numpy as np
import torch
from scar import model, setup_anndata

sys.path.insert(0, str(Path(__file__).parent / "src"))
from ambient_io import read_inputs, write_counts  # noqa: E402
from cli import build_scar_parser, output_path  # noqa: E402


def main():
  args = build_scar_parser().parse_args()
  print(f"Full command: {' '.join(sys.argv)}")
  for k, v in vars(args).items():
    print(f"  {k}: {v}")

  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  raw, cells = read_inputs(args.rawdata_h5ad, args.called_cells_tsv)

  # setup_anndata reads var["feature_types"]; the raw h5ad (DropletUtils
  # read10xCounts) has `Type` instead, or nothing.
  for a in (raw, cells):
    if "feature_types" not in a.var.columns:
      a.var["feature_types"] = a.var["Type"].astype(str) if "Type" in a.var.columns else "Gene Expression"

  print("  estimating ambient profile from cell-free droplets ..")
  setup_anndata(adata=cells, raw_adata=raw, prob=args.prob, kneeplot=False)
  ambient_profile = cells.uns["ambient_profile_all"]

  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"  training scAR (epochs={args.epochs}, batch_size={args.batch_size}, device={device}) ..")
  scar_model = model(
    raw_count=cells,
    ambient_profile=ambient_profile,
    feature_type="mRNA",
    sparsity=args.sparsity,
    device=device,
  )
  scar_model.train(epochs=args.epochs, batch_size=args.batch_size, verbose=True)
  scar_model.inference()  # round_to_int="stochastic_rounding" -> int64 counts

  write_counts(
    cells,
    scar_model.native_counts,
    output_path(args, "filtered_h5ad", f"{args.name}.h5ad"),
  )


if __name__ == "__main__":
  main()
