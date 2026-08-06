#!/usr/bin/env python3
"""Prepare the exact CEARF leave-last-out split for MGCOT validation."""
from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import shutil

import cearf
import hid_protocol
from narm_mps import expand_sessions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output / "diginetica"
    destination.mkdir(parents=True, exist_ok=True)
    data = hid_protocol.load_hid_diginetica()
    tune_sessions, validation = cearf.make_validation_split(
        data["train_sessions"], 0.10, 5000)
    train_x, train_y = expand_sessions(tune_sessions)
    valid_x = [validation[uid]["context"] for uid in validation]
    valid_y = [validation[uid]["targets"][0] for uid in validation]
    with open(destination / "train.txt", "wb") as handle:
        pickle.dump((train_x, train_y), handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(destination / "test.txt", "wb") as handle:
        pickle.dump((valid_x, valid_y), handle, protocol=pickle.HIGHEST_PROTOCOL)
    # The public MGCOT graph is retained for architecture compatibility. The
    # held-out target interactions are absent from sequence-model training;
    # graph provenance is recorded as a limitation in the audit.
    source_adj = (Path(__file__).resolve().parents[1] / "reference_repos" /
                  "MGCOT" / "diginetica" / "adj_global.npz")
    shutil.copy2(source_adj, destination / "adj_global.npz")
    print({"train": len(train_y), "validation": len(valid_y),
           "destination": str(destination)}, flush=True)


if __name__ == "__main__":
    main()
