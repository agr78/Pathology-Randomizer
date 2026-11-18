#!/usr/bin/env bash
# Creates and activates the `pathodraw` environment and records the data paths: `source ./install.sh`
# (a CUDA build of PyTorch is picked by pip for the machine's driver; see https://pytorch.org for other builds)
conda create -y -n pathodraw python=3.10
conda activate pathodraw
pip install -r requirements.txt

# Data paths, asked once here and exported on every activation of the environment (Enter keeps the default)
ask() { local v=$1 d=$2 a; read -r -e -p "$3 [${!v:-$d}]: " a; export "$v=${a:-${!v:-$d}}"; }
ask PR_DATA_DIR "$PWD/label_maps" "Healthy label maps, PR_DATA_DIR"
ask PR_OUT_DIR "$PWD/runs" "Runs and checkpoints, PR_OUT_DIR"
ask PR_EVAL_DATA_ROOT "$PWD/datasets" "Evaluation datasets, PR_EVAL_DATA_ROOT"
ask PR_DERIV_ROOT "$PWD/derivatives" "Inference maps, PR_DERIV_ROOT"
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
printf 'export PR_DATA_DIR=%q PR_OUT_DIR=%q PR_EVAL_DATA_ROOT=%q PR_DERIV_ROOT=%q\n' \
  "$PR_DATA_DIR" "$PR_OUT_DIR" "$PR_EVAL_DATA_ROOT" "$PR_DERIV_ROOT" > "$CONDA_PREFIX/etc/conda/activate.d/pathodraw_paths.sh"
echo "Paths saved to $CONDA_PREFIX/etc/conda/activate.d/pathodraw_paths.sh (edit that file to change them)."
