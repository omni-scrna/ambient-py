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


# var column naming the feature type: `feature_type` (ambient-data-download
# >= 8064b49), `Type` (DropletUtils read10xCounts), `feature_types` (scanpy).
FEATURE_TYPE_COLUMNS = ("feature_type", "Type", "feature_types")


def keep_gene_expression(adata):
  """Drop non-RNA features (e.g. HTO `Antibody Capture`), like ambient-r.

  Without a feature-type column all features are kept.
  """
  col = next((c for c in FEATURE_TYPE_COLUMNS if c in adata.var.columns), None)
  if col is None:
    print("  no feature-type column in var: keeping all features")
    return adata
  is_ge = (adata.var[col].astype(str) == "Gene Expression").values
  if not is_ge.any():
    raise ValueError(f"no 'Gene Expression' features in var['{col}']")
  if is_ge.all():
    return adata
  print(f"  keeping {is_ge.sum()} 'Gene Expression' of {adata.n_vars} features")
  return adata[:, is_ge].copy()


def read_inputs(rawdata_h5ad, called_cells_tsv):
  """Return (raw, cells): raw droplets, and a copy subset to called cells."""
  raw = ad.read_h5ad(rawdata_h5ad)
  raw.X = sp.csr_matrix(raw.X)
  print(f"  raw: {raw.n_obs} barcodes x {raw.n_vars} features")
  raw = keep_gene_expression(raw)

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
  # anndata < 0.9 (CellClear env: 0.8.0) casts X to float32 in the constructor;
  # assigning afterwards keeps int64 in old and new versions.
  out.X = corrected
  if out.X.dtype != np.int64:
    raise TypeError(f"output X has dtype {out.X.dtype}, expected int64")
  if extra_obs:
    for key, values in extra_obs.items():
      out.obs[key] = values
  Path(out_path).parent.mkdir(parents=True, exist_ok=True)
  out.write_h5ad(out_path)
  print(f"  wrote: {out_path} ({out.n_obs} cells x {out.n_vars} genes)")
