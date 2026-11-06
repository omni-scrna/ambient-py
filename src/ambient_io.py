"""Shared input/output for the ambient RNA removal entrypoints (scAR, CellClear).

Same behaviour as ambient-r's decontx/passthrough modules:
- read the raw droplets of one pool (`rawdata_h5ad`, X = counts),
- subset them to the called barcodes (`called_cells_tsv`, column `cell_id`),
- write the corrected matrix as h5ad with X = integer counts (CSR), keeping
  the called cells' obs (e.g. `pool_id`, needed downstream) and var.
"""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def read_inputs(rawdata_h5ad, called_cells_tsv):
  """Return (raw, cells): raw droplets, and a copy subset to called cells."""
  raw = ad.read_h5ad(rawdata_h5ad)
  raw.X = sp.csr_matrix(raw.X)
  print(f"  raw: {raw.n_obs} barcodes x {raw.n_vars} genes")

  cell_ids = pd.read_csv(called_cells_tsv, sep="\t")["cell_id"].astype(str)
  missing = ~cell_ids.isin(raw.obs_names)
  if missing.any():
    raise ValueError(
      f"{missing.sum()} called barcode(s) not found in rawdata_h5ad, "
      f"e.g. {cell_ids[missing].iloc[0]}"
    )
  cells = raw[cell_ids.values].copy()
  print(f"  called cells: {cells.n_obs} cells x {cells.n_vars} genes")
  return raw, cells


def write_counts(cells, corrected, out_path, extra_obs=None):
  """Write `corrected` (cells x genes, same order as `cells`) as h5ad."""
  corrected = sp.csr_matrix(corrected)
  if corrected.shape != cells.shape:
    raise ValueError(f"corrected shape {corrected.shape} != input shape {cells.shape}")
  if corrected.dtype.kind == "f":
    if not np.allclose(corrected.data, np.round(corrected.data)):
      raise ValueError("corrected counts are not integers")
    corrected.data = np.round(corrected.data)
  corrected = corrected.astype(np.int64)
  corrected.eliminate_zeros()

  out = ad.AnnData(X=corrected, obs=cells.obs.copy(), var=cells.var.copy())
  if extra_obs:
    for key, values in extra_obs.items():
      out.obs[key] = values
  Path(out_path).parent.mkdir(parents=True, exist_ok=True)
  out.write_h5ad(out_path)
  print(f"  wrote: {out_path} ({out.n_obs} cells x {out.n_vars} genes)")
