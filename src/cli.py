"""CLI for the ambient-py entrypoints.

Mirrors the input contract of the ambient-plan stage for ambient RNA removal
(stage id `cleaning`; see ambient-r: src/common/schema/{_base,cleaning}.json):

  --output_dir DIR             (base)
  --name NAME                  (base; the module id under api >= 0.5)
  --rawdata_h5ad PATH          raw droplets of one pool, X = counts
  --called_cells_tsv PATH      TSV with a `cell_id` column (called barcodes)
  --output ID=PATH             api >= 0.8: once per declared output (repeatable)

`cell_calling_and_cleaning` (CellBender) takes only the raw matrix.
"""

import argparse


def _base_parser(description):
  # allow_abbrev=False: otherwise argparse could treat `--output` as an
  # abbreviation of `--output_dir`.
  parser = argparse.ArgumentParser(description=description, allow_abbrev=False)
  parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
  parser.add_argument("--name", type=str, required=True, help="Module name/identifier")
  parser.add_argument(
    "--output",
    action="append",
    default=[],
    metavar="ID=PATH",
    help="api >= 0.8: path of a declared output, once per output",
  )
  parser.add_argument(
    "--rawdata_h5ad",
    type=str,
    required=True,
    help="Raw h5ad count matrix (unfiltered, one pool per dataset)",
  )
  return parser


def build_ambient_removal_parser(description):
  parser = _base_parser(description)
  parser.add_argument(
    "--called_cells_tsv",
    type=str,
    required=True,
    help="TSV with called cell barcodes (column `cell_id`)",
  )
  return parser


def build_scar_parser():
  parser = build_ambient_removal_parser("scAR ambient removal module")
  parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
  parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
  parser.add_argument(
    "--prob",
    type=float,
    default=0.995,
    help="setup_anndata: probability cutoff for calling cell-free droplets",
  )
  parser.add_argument(
    "--sparsity",
    type=float,
    default=1.0,
    help="Expected sparsity of native signals (scAR: 1 for scRNA-seq mRNA)",
  )
  parser.add_argument("--seed", type=int, default=0, help="Random seed")
  return parser


def build_cellclear_parser():
  parser = build_ambient_removal_parser("CellClear ambient removal module")
  parser.add_argument(
    "--resolution", type=float, default=1.2, help="CellClear: Leiden clustering resolution"
  )
  parser.add_argument(
    "--roc_threshold",
    type=float,
    default=0.6,
    help="CellClear: ROC threshold separating ambient and clean clusters",
  )
  return parser


def build_cellbender_parser():
  parser = _base_parser("CellBender ambient removal + cell calling")
  parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
  return parser


def output_path(args, output_id, default_filename):
  """Path for a declared output.

  api >= 0.8 passes `--output <id>=<path>`; older api versions don't, so fall
  back to <output_dir>/<default_filename> (which must match the plan's path
  template, e.g. `{name}.h5ad`).
  """
  for item in args.output:
    key, sep, path = item.partition("=")
    if sep and key == output_id:
      return path
  return f"{args.output_dir}/{default_filename}"
