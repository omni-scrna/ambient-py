#!/usr/bin/env python3
"""CellClear ambient RNA removal module (plan stage id: `cleaning`).

Reads the raw droplets of one pool + the called cell barcodes, runs the
CellClear `correct_expression` CLI and writes the corrected counts as h5ad.

Behaviour of CellClear 0.0.3 this wrapper accounts for (read in its source):
- Inputs are read with `sc.read_10x_mtx` when given a directory, so we write
  10x v3 MTX directories (matrix/features/barcodes .gz).
- Gene names have `_` replaced by `-` and are made unique. We pass sanitized,
  unique names ourselves and map the result back to the original genes.
- It clusters the cells itself (Leiden, `--resolution`) and DROPS cells in
  clusters of <= 50 cells. Those cells are kept here with their uncorrected
  counts and flagged in obs["cellclear_corrected"] = False, so that every
  ambient removal method returns the same set of called cells.
- Only "ambient genes" are corrected; the other genes keep their raw counts.
  Corrected values are clipped at 0 and truncated to int by CellClear.
"""

import gzip
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent / "src"))
from ambient_io import read_inputs, write_counts  # noqa: E402
from cli import build_cellclear_parser, output_path  # noqa: E402


def write_10x_mtx(matrix, barcodes, gene_ids, gene_names, out_dir):
  """Write a cells x genes matrix as a 10x v3 MTX directory (genes x cells)."""
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  with gzip.open(out_dir / "matrix.mtx.gz", "wb") as f:
    scipy.io.mmwrite(f, sp.coo_matrix(matrix.T))
  pd.DataFrame(
    {"id": gene_ids, "name": gene_names, "type": "Gene Expression"}
  ).to_csv(out_dir / "features.tsv.gz", sep="\t", header=False, index=False)
  pd.Series(barcodes).to_csv(out_dir / "barcodes.tsv.gz", sep="\t", header=False, index=False)


def sanitized_gene_names(var):
  """Unique gene names without `_` (CellClear rewrites `_` to `-`)."""
  names = var["Symbol"].astype(str) if "Symbol" in var.columns else pd.Series(var.index, index=var.index)
  names = names.str.replace("_", "-", regex=False)
  # make unique the same way for everyone: name, name-1, name-2, ...
  seen = {}
  out = []
  for n in names:
    k = seen.get(n, 0)
    out.append(n if k == 0 else f"{n}-{k}")
    seen[n] = k + 1
  if len(set(out)) != len(out):
    raise ValueError("could not make gene names unique")
  return out


def main():
  args = build_cellclear_parser().parse_args()
  print(f"Full command: {' '.join(sys.argv)}")
  for k, v in vars(args).items():
    print(f"  {k}: {v}")

  raw, cells = read_inputs(args.rawdata_h5ad, args.called_cells_tsv)
  gene_names = sanitized_gene_names(cells.var)
  gene_ids = [str(g).replace("_", "-") for g in cells.var_names]

  # Background droplets for CellClear: drop all-zero barcodes to keep the file small.
  nonzero = np.asarray(raw.X.sum(axis=1)).ravel() > 0
  raw_nz = raw[nonzero]
  print(f"  raw barcodes with >= 1 count: {raw_nz.n_obs}")

  with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    write_10x_mtx(cells.X, cells.obs_names, gene_ids, gene_names, tmp / "filtered")
    write_10x_mtx(raw_nz.X, raw_nz.obs_names, gene_ids, gene_names, tmp / "raw")

    cmd = [
      "CellClear", "correct_expression",
      "--filtered_mtx_path", str(tmp / "filtered"),
      "--raw_mtx_path", str(tmp / "raw"),
      "--resolution", str(args.resolution),
      "--roc_threshold", str(args.roc_threshold),
      "--prefix", args.name,
      "--output", str(tmp / "out"),
    ]
    (tmp / "out").mkdir()
    print(f"  running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    result = sc.read_10x_mtx(tmp / "out" / "matrix")  # var_names = our sanitized names
  print(f"  CellClear output: {result.n_obs} cells x {result.n_vars} genes")

  # Map back onto the full called-cells x genes matrix (start from raw counts).
  gene_pos = pd.Index(gene_names).get_indexer(result.var_names)
  cell_pos = cells.obs_names.get_indexer(result.obs_names)
  if (gene_pos < 0).any() or (cell_pos < 0).any():
    raise ValueError(
      f"CellClear returned {int((gene_pos < 0).sum())} unknown gene(s) and "
      f"{int((cell_pos < 0).sum())} unknown barcode(s)"
    )
  n_cells, n_genes = cells.shape
  n_res = result.n_obs
  corrected_flag = np.zeros(n_cells, dtype=bool)
  corrected_flag[cell_pos] = True
  gene_returned = np.zeros(n_genes, dtype=bool)
  gene_returned[gene_pos] = True
  print(f"  cells corrected by CellClear: {corrected_flag.sum()} of {n_cells}")
  print(f"  genes returned by CellClear: {gene_returned.sum()} of {n_genes}")

  # result rows/cols -> positions in `cells` (sparse permutation/selection matrices)
  rows = sp.csr_matrix((np.ones(n_res), (cell_pos, np.arange(n_res))), shape=(n_cells, n_res))
  cols = sp.csr_matrix(
    (np.ones(len(gene_pos)), (np.arange(len(gene_pos)), gene_pos)), shape=(len(gene_pos), n_genes)
  )
  X = sp.csr_matrix(cells.X, dtype=np.float64)
  keep_raw = (
    sp.diags((~corrected_flag).astype(float)) @ X                                  # cells CellClear dropped
    + sp.diags(corrected_flag.astype(float)) @ X @ sp.diags((~gene_returned).astype(float))  # genes it didn't return
  )
  corrected = keep_raw + rows @ sp.csr_matrix(result.X, dtype=np.float64) @ cols

  write_counts(
    cells,
    corrected,
    output_path(args, "filtered_h5ad", f"{args.name}.h5ad"),
    extra_obs={"cellclear_corrected": corrected_flag},
  )


if __name__ == "__main__":
  main()
