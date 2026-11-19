"""Shared input/output for the ambient RNA removal entrypoints (scAR, CellClear).

Same behaviour as ambient-r's decontx/passthrough modules:
- read the raw droplets of one pool (`rawdata_h5ad`, X = counts),
- subset them to the called barcodes (`called_cells_tsv`, column `cell_id`),
- write the corrected matrix as h5ad with X = integer counts (CSR), keeping
  the called cells' obs (e.g. `pool_id`, needed downstream) and var.
"""

from pathlib import Path

import anndata as ad
import h5py
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


def _attr_str(value):
  return value.decode() if isinstance(value, bytes) else str(value)


def _strings(values):
  return np.array([_attr_str(v) for v in np.atleast_1d(values)], dtype=object)


def _read_column(elem):
  """One obs/var column or index, for every encoding anndataR/anndata write."""
  if isinstance(elem, h5py.Dataset):
    values = elem[()]
    return _strings(values) if values.dtype.kind in "OSU" else values
  enc = _attr_str(elem.attrs.get("encoding-type", ""))
  if enc == "categorical":
    return pd.Categorical.from_codes(
      elem["codes"][()],
      _read_column(elem["categories"]),
      ordered=bool(elem.attrs.get("ordered", False)),
    )
  if enc in ("nullable-integer", "nullable-boolean", "nullable-string-array"):
    values = _read_column(elem["values"])
    mask = elem["mask"][()].astype(bool)
    if enc == "nullable-integer":
      return pd.arrays.IntegerArray(values, mask)
    if enc == "nullable-boolean":
      return pd.arrays.BooleanArray(values.astype(bool), mask)
    if not mask.any():
      return values
    # missing strings: categorical with NaN, which anndata 0.8 can write
    return pd.Categorical(np.where(mask, None, values))
  raise ValueError(f"unsupported encoding '{enc}' for {elem.name}")


def _read_dataframe(group):
  index = _read_column(group[_attr_str(group.attrs["_index"])])
  order = group.attrs.get("column-order", [])
  columns = [_attr_str(c) for c in np.atleast_1d(order)] if np.size(order) else []
  return pd.DataFrame(
    {c: _read_column(group[c]) for c in columns},
    index=pd.Index(index.astype(str)),
  )


def _read_h5ad_h5py(path):
  """Read X, obs and var of an h5ad with h5py only (uns/layers are skipped).

  Used when the installed anndata is too old for the file's encodings: CellClear
  pins anndata 0.8.0, which cannot read `nullable-string-array` columns written
  by anndataR (ambient-data-download).
  """
  with h5py.File(path, "r") as f:
    x = f["X"]
    if isinstance(x, h5py.Dataset):
      X = x[()]
    else:
      enc = _attr_str(x.attrs["encoding-type"])
      cls = {"csr_matrix": sp.csr_matrix, "csc_matrix": sp.csc_matrix}[enc]
      X = cls((x["data"][()], x["indices"][()], x["indptr"][()]), shape=tuple(x.attrs["shape"]))
    obs = _read_dataframe(f["obs"])
    var = _read_dataframe(f["var"])
  adata = ad.AnnData(obs=obs, var=var)
  adata.X = X  # after the constructor: anndata 0.8 would cast X to float32
  return adata


def read_h5ad(path):
  """`ad.read_h5ad`, falling back to h5py if anndata lacks an encoding."""
  try:
    return ad.read_h5ad(path)
  except KeyError as e:
    if "IOSpec" not in repr(e):
      raise
    print(f"  anndata {ad.__version__} cannot read {e.args[0][1]}: reading X/obs/var with h5py")
    return _read_h5ad_h5py(path)


def read_inputs(rawdata_h5ad, called_cells_tsv):
  """Return (raw, cells): raw droplets, and a copy subset to called cells."""
  raw = read_h5ad(rawdata_h5ad)
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
