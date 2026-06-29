#!/usr/bin/env python3
"""CellClear ambient removal module.

Runs CellClear correct_expression CLI with raw + filtered 10x H5 matrices,
reads the MTX output, and writes a corrected h5ad matrix.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import scanpy as sc
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent / "src"))
from cli import build_cellclear_parser


def write_10x_h5(adata, path):
  """Write an AnnData object as a 10x-compatible H5 file.

  CellClear CLI expects 10x H5 input, so this is used to create a
  temporary file from the h5ad-sourced filtered data.
  """
  X = sp.csc_matrix(adata.X)
  with h5py.File(path, "w") as f:
    g = f.create_group("matrix")
    g.create_dataset("data", data=X.data)
    g.create_dataset("indices", data=X.indices)
    g.create_dataset("indptr", data=X.indptr)
    g.create_dataset("shape", data=np.array(X.shape[::-1]))
    g.create_dataset("barcodes", data=np.array(adata.obs_names, dtype="S"))
    gene_names = adata.var_names.values
    g.create_dataset("genes", data=np.array(gene_names, dtype="S"))
    if "gene_ids" in adata.var.columns:
      g.create_dataset("gene_ids", data=np.array(adata.var["gene_ids"], dtype="S"))


def main():
  args = build_cellclear_parser().parse_args()
  print(f"Full command: {' '.join(sys.argv)}")
  for k in ("output_dir", "name", "rawdata_h5", "filtered_h5ad"):
    print(f"  {k}: {getattr(args, k)}")

  out_dir = Path(args.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  print("  reading filtered h5ad matrix ..")
  adata_filt = sc.read_h5ad(args.filtered_h5ad)

  with tempfile.TemporaryDirectory() as tmpdir:
    tmp_filtered_h5 = str(Path(tmpdir) / "filtered.h5")
    print("  writing temporary 10x H5 for CellClear ..")
    write_10x_h5(adata_filt, tmp_filtered_h5)

    cmd = [
      "CellClear", "correct_expression",
      "--filtered_mtx_path", tmp_filtered_h5,
      "--raw_mtx_path", args.rawdata_h5,
      "--prefix", args.name,
      "--output", tmpdir,
    ]
    print(f"  running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    mtx_dir = Path(tmpdir) / "matrix"
    if not mtx_dir.exists():
      mtx_dir = Path(tmpdir)

    print("  reading CellClear MTX output ..")
    adata = sc.read_10x_mtx(str(mtx_dir))
    print(f"  corrected matrix: {adata.shape[0]} cells x {adata.shape[1]} genes")

  out_path = out_dir / f"{args.name}_corrected.h5ad"
  adata.write_h5ad(out_path)
  print(f"  wrote: {out_path}")


if __name__ == "__main__":
  main()
