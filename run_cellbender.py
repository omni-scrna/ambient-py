#!/usr/bin/env python3
"""CellBender ambient removal + cell calling module.

Runs cellbender remove-background with default parameters (no --expected-cells
or --total-droplets-included). Species assignment happens later, in the
demultiplexing stage, on this corrected output.
"""

import subprocess
import sys
from pathlib import Path

import scanpy as sc

sys.path.insert(0, str(Path(__file__).parent / "src"))
from cli import build_cellbender_parser


def main():
  # EXPERIMENTAL (api_version >= 0.8.0): the engine also passes
  # `--output <id>=<path>` unconditionally; parse_known_args() tolerates it
  # rather than erroring on an unrecognized flag.
  args = build_cellbender_parser().parse_known_args()[0]
  print(f"Full command: {' '.join(sys.argv)}")
  for k in ("output_dir", "name", "rawdata_h5ad", "epochs"):
    print(f"  {k}: {getattr(args, k)}")

  out_dir = Path(args.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  cb_output = out_dir / f"{args.name}_cellbender.h5"
  cb_filtered = out_dir / f"{args.name}_cellbender_filtered.h5"

  cmd = [
    "cellbender", "remove-background",
    "--input", args.rawdata_h5ad,
    "--output", str(cb_output),
    "--epochs", str(args.epochs),
    "--cuda",
  ]
  print(f"  running: {' '.join(cmd)}")
  subprocess.run(cmd, check=True)

  print("  reading CellBender output")
  adata = sc.read_10x_h5(str(cb_filtered))
  print(f"  corrected matrix: {adata.shape[0]} cells x {adata.shape[1]} genes")

  # CellBender's own output format carries only barcodes/genes -- re-attach
  # the original obs columns (pool_id etc.) by barcode so downstream stages
  # (demultiplexing, gather) can still identify which pool each cell is from.
  print("  re-attaching obs columns from the raw input")
  adata_raw = sc.read_h5ad(args.rawdata_h5ad)
  common = adata.obs_names.intersection(adata_raw.obs_names)
  if len(common) != adata.n_obs:
    raise ValueError(
      f"{adata.n_obs - len(common)} CellBender-called barcode(s) not found in rawdata_h5ad"
    )
  adata.obs = adata.obs.join(adata_raw.obs.loc[common])

  out_path = out_dir / f"{args.name}.h5ad"
  adata.write_h5ad(out_path)
  print(f"  wrote: {out_path}")

  for tmp in [cb_output, cb_filtered]:
    if tmp.exists():
      tmp.unlink()


if __name__ == "__main__":
  main()
