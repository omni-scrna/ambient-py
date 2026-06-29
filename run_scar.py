#!/usr/bin/env python3
"""scAR ambient removal module.

Reads raw 10x H5 + filtered h5ad matrices, trains a scAR VAE model to estimate
native (non-ambient) counts, and writes a corrected h5ad matrix.
"""

import sys
from pathlib import Path

import scanpy as sc
from scar import model, setup_anndata

sys.path.insert(0, str(Path(__file__).parent / "src"))
from cli import build_scar_parser


def main():
  args = build_scar_parser().parse_args()
  print(f"Full command: {' '.join(sys.argv)}")
  for k in ("output_dir", "name", "rawdata_h5", "filtered_h5ad", "epochs", "batch_size"):
    print(f"  {k}: {getattr(args, k)}")

  out_dir = Path(args.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  print("  reading filtered matrix ..")
  adata = sc.read_h5ad(args.filtered_h5ad)
  adata.var_names_make_unique()
  print(f"  filtered: {adata.shape[0]} cells x {adata.shape[1]} genes")

  print("  reading raw matrix ..")
  adata_raw = sc.read_10x_h5(args.rawdata_h5)
  adata_raw.var_names_make_unique()
  print(f"  raw: {adata_raw.shape[0]} barcodes x {adata_raw.shape[1]} genes")

  print("  estimating ambient profile ..")
  setup_anndata(adata=adata, raw_adata=adata_raw, prob=0.995, kneeplot=False)

  ambient_key = None
  for key in adata.uns:
    if key.startswith("ambient_profile"):
      ambient_key = key
      break
  if ambient_key is None:
    raise RuntimeError("setup_anndata did not create an ambient_profile key in adata.uns")
  print(f"  ambient profile key: {ambient_key}")

  device = "cuda"
  try:
    import torch
    if not torch.cuda.is_available():
      device = "cpu"
      print("  CUDA not available, using CPU")
  except ImportError:
    device = "cpu"

  print(f"  training scAR model (epochs={args.epochs}, batch_size={args.batch_size}, device={device}) ..")
  scar_model = model(
    raw_count=adata,
    ambient_profile=adata.uns[ambient_key],
    feature_type="mRNA",
    sparsity=1,
    device=device,
  )
  scar_model.train(epochs=args.epochs, batch_size=args.batch_size, verbose=True)
  scar_model.inference()

  corrected = scar_model.native_counts
  print(f"  corrected matrix: {corrected.shape[0]} cells x {corrected.shape[1]} genes")

  adata_out = adata.copy()
  adata_out.X = corrected
  out_path = out_dir / f"{args.name}_corrected.h5ad"
  adata_out.write_h5ad(out_path)
  print(f"  wrote: {out_path}")


if __name__ == "__main__":
  main()
