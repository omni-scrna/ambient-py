#!/usr/bin/env python3
"""CellBender ambient removal + cell calling module.

Runs cellbender remove-background with default parameters (no --expected-cells
or --total-droplets-included). Assigns species from raw counts. Outputs a
corrected h5ad matrix with species labels in obs.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, str(Path(__file__).parent / "src"))
from cli import build_cellbender_parser


def assign_species(adata, raw_adata, quantile=5):
  common = adata.obs_names.intersection(raw_adata.obs_names)
  raw_sub = raw_adata[common]

  genomes = raw_sub.var["genome"]
  unique_genomes = genomes.unique()
  human_genome = [g for g in unique_genomes if g.startswith(("hg", "GRCh"))][0]
  mouse_genome = [g for g in unique_genomes if g.startswith(("mm", "GRCm"))][0]

  human_counts = np.asarray(raw_sub[:, genomes == human_genome].X.sum(axis=1)).flatten()
  mouse_counts = np.asarray(raw_sub[:, genomes == mouse_genome].X.sum(axis=1)).flatten()
  total = human_counts + mouse_counts

  human_ratio = human_counts / total
  mouse_ratio = mouse_counts / total

  species_raw = np.where(human_ratio > mouse_ratio, "human", "mouse")
  majority_ratio = np.maximum(human_ratio, mouse_ratio)

  thresh = np.percentile(majority_ratio, quantile)
  species = np.where(majority_ratio > thresh, species_raw, "ambiguous")

  adata.obs["species"] = pd.Categorical(species, categories=["human", "mouse", "ambiguous"])
  adata.obs["majority_ratio"] = majority_ratio


def main():
  args = build_cellbender_parser().parse_args()
  print(f"Full command: {' '.join(sys.argv)}")
  for k in ("output_dir", "name", "rawdata_h5", "epochs", "species_quantile"):
    print(f"  {k}: {getattr(args, k)}")

  out_dir = Path(args.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  cb_output = out_dir / f"{args.name}_cellbender.h5"
  cb_filtered = out_dir / f"{args.name}_cellbender_filtered.h5"

  cmd = [
    "cellbender", "remove-background",
    "--input", args.rawdata_h5,
    "--output", str(cb_output),
    "--epochs", str(args.epochs),
    "--cuda",
  ]
  print(f"  running: {' '.join(cmd)}")
  subprocess.run(cmd, check=True)

  print("  reading CellBender output")
  adata = sc.read_10x_h5(str(cb_filtered))
  print(f"  corrected matrix: {adata.shape[0]} cells x {adata.shape[1]} genes")

  print("  reading raw matrix for species assignment")
  adata_raw = sc.read_10x_h5(args.rawdata_h5)

  print("  assigning species")
  assign_species(adata, adata_raw, quantile=args.species_quantile)
  counts = adata.obs["species"].value_counts()
  print(f"  species counts: {', '.join(f'{k}={v}' for k, v in counts.items())}")

  out_path = out_dir / f"{args.name}_corrected.h5ad"
  adata.write_h5ad(out_path)
  print(f"  wrote: {out_path}")

  for tmp in [cb_output, cb_filtered]:
    if tmp.exists():
      tmp.unlink()


if __name__ == "__main__":
  main()
