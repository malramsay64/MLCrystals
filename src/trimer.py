#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Utilities for handling the trimer molecule."""

import glob
import logging
from pathlib import Path
from typing import List, Tuple

import gsd.hoomd
import numpy as np
from scipy.sparse import coo_matrix
from sdanalysis import HoomdFrame
from sdanalysis.order import compute_neighbours
from sdanalysis.util import Variables, get_filename_vars

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def read_all_files(
    pathname: Path, index: int = 0, pattern: str = "dump-Trimer-*.gsd"
) -> List[Tuple[Variables, HoomdFrame]]:
    """Read all the gsd files from a directory given a pattern.

    A utility function for getting reading all the gsd files from a given directory,
    with the ability to use a pattern to match a subset.

    Args:
        pathname: The directory from which the files will be read
        index: The index of the snapshot to read for each trajectory.
        pattern: The pattern passed to the glob function to match.

    Returns:
        A list of tuples containing the variables for a configuration, established from the filename,
        along with the frame.

    """
    pathname = Path(pathname)
    snapshots = []
    for filename in glob.glob(str(pathname / pattern)):
        logger.debug("Reading %s", Path(filename).stem)
        with gsd.hoomd.open(str(filename)) as trj:
            try:
                snapshots.append((get_filename_vars(filename), HoomdFrame(trj[index])))
            except IndexError:
                continue
    if not snapshots:
        logger.warning(
            "There were no files found with a configuration at index %s", index
        )
    return snapshots


def read_file(
    index: int = 0,
    pressure: float = 1.00,
    temperature: float = 0.40,
    crystal: str = "p2",
    prefix: str = "dump",
) -> HoomdFrame:

    data_dir = Path("../data/simulation/dataset/output")
    fname = f"{prefix}-Trimer-P{pressure:.2f}-T{temperature:.2f}-{crystal}.gsd"
    filename = data_dir / fname
    if not filename.exists():
        raise FileNotFoundError(read_file)
    with gsd.hoomd.open(str(filename)) as trj:
        try:
            return HoomdFrame(trj[index])
        except IndexError:
            raise IndexError(f"Index {index} not found in trajectory.")


def classify_mols(snapshot, crystal, boundary_buffer=3.5, is_2d: bool = True):
    """Classify molecules as crystalline, amorphous or boundary."""
    mapping = {"liq": 0, "p2": 1, "p2gg": 2, "pg": 3, "None": 4}
    position = snapshot.position
    # This gets the details of the box from the simulation
    box = snapshot.box[:3]

    # All axes have to be True, True == 1, use product for logical and operation
    position_mat = np.abs(position) < box[:3] / 3
    if is_2d:
        is_crystal = np.product(position_mat[:, :2], axis=1).astype(bool)
    else:
        is_crystal = np.product(position_mat, axis=1).astype(bool)
    boundary = np.logical_and(
        np.product(np.abs(position) < box[:3] / 3 + boundary_buffer, axis=1),
        np.product(np.abs(position) > box[:3] / 3 - boundary_buffer, axis=1),
    )

    # Create classification array
    classification = np.zeros(len(snapshot), dtype=int)
    classification[is_crystal] = mapping[crystal]
    classification[boundary] = 4
    return classification


def neighbour_connectivity(snapshot, max_neighbours=6, max_radius=5):
    neighbours = compute_neighbours(
        snapshot.box, snapshot.position, max_neighbours, max_radius
    )
    sparse_values = np.ones(neighbours.shape[0] * neighbours.shape[1])
    sparse_coordinates = (
        np.repeat(np.arange(neighbours.shape[0]), neighbours.shape[1]),
        neighbours.flatten(),
    )
    connectivity = coo_matrix((sparse_values, sparse_coordinates))
    return connectivity.toarray()
