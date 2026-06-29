import argparse


def _base_parser(description="Ambient removal module"):
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument(
    "--output_dir", type=str, required=True, help="Output directory"
  )
  parser.add_argument(
    "--name", type=str, required=True, help="Module name/identifier"
  )
  parser.add_argument(
    "--rawdata.h5",
    dest="rawdata_h5",
    type=str,
    required=True,
    help="Raw 10x H5 count matrix (unfiltered)",
  )
  return parser


def build_parser(description="Ambient removal module"):
  parser = _base_parser(description)
  parser.add_argument(
    "--filtered.h5ad",
    dest="filtered_h5ad",
    type=str,
    required=True,
    help="Filtered h5ad count matrix (cells only)",
  )
  return parser


def build_scar_parser():
  parser = build_parser("scAR ambient removal module")
  parser.add_argument(
    "--epochs", type=int, default=200, help="Number of training epochs"
  )
  parser.add_argument(
    "--batch_size", type=int, default=64, help="Training batch size"
  )
  return parser


def build_cellclear_parser():
  return build_parser("CellClear ambient removal module")


def build_cellbender_parser():
  parser = _base_parser("CellBender ambient removal + cell calling")
  parser.add_argument(
    "--epochs", type=int, default=150, help="Number of training epochs"
  )
  parser.add_argument(
    "--species_quantile", type=float, default=5,
    help="Percentile of majority_ratio for ambiguous threshold (default: 5)",
  )
  return parser
