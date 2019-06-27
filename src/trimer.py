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
from itertools import count, product
from pathlib import Path
from typing import List, Tuple

import gsd.hoomd
import matplotlib.pyplot as plt
import numpy as np
from bokeh import palettes
from bokeh.plotting import gridplot
from scipy.sparse import coo_matrix
from sdanalysis import HoomdFrame
from sdanalysis.figures import plot_frame
from sdanalysis.order import compute_neighbours
from sdanalysis.util import Variables, get_filename_vars
from sklearn.metrics import confusion_matrix

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


def plot_grid(frames):
    for frame in frames:
        frame.plot_height = frame.plot_height // 3
        frame.plot_width = frame.plot_width // 3
    return gridplot(frames, ncols=3)


def plot_configuration_grid(snapshots, categories, max_frames=3):
    if len(np.unique(categories)) < 10:
        colormap = palettes.Category10_10
    else:
        colormap = palettes.Category20_20
    cluster_assignment = np.split(categories, len(snapshots))
    return plot_grid(
        [
            plot_frame(
                snap, order_list=cluster, categorical_colour=True, colormap=colormap
            )
            for snap, cluster, i in zip(snapshots, cluster_assignment, count())
            if i < max_frames
        ]
    )


def plot_clustering(algorithm, X, snapshots, fit=True, max_frames=3):
    if fit:
        clusters = algorithm.fit_predict(X)
    else:
        clusters = algorithm.predict(X)
    return plot_configuration_grid(snapshots, clusters, max_frames)


def plot_snapshots(snapshots):
    return plot_grid([plot_frame(snap) for snap in snapshots])


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


def plot_confusion_matrix(
    y_true, y_pred, classes, normalize=True, title="Confusion matrix", cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
