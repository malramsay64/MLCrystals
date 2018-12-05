#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Utilities for handling the trimer molecule."""

from itertools import count, product
from pathlib import Path
from typing import List

import gsd.hoomd
import matplotlib.pyplot as plt
import numpy as np
from bokeh.plotting import gridplot
from scipy.sparse import coo_matrix
from sdanalysis import HoomdFrame
from sdanalysis.figures import plot_frame
from sdanalysis.order import compute_neighbours


def read_files(
    index: int = 0,
    pressure: List[float] = 1.00,
    temperature: List[float] = 0.40,
    crystals: List[str] = ["p2", "p2gg", "pg"],
) -> List[HoomdFrame]:
    if isinstance(pressure, float):
        pressure = [pressure]
    if isinstance(temperature, float):
        temperature = [temperature]
    if isinstance(crystals, str):
        crystals = [crystals]

    data_dir = Path("../data/simulation/trimer")
    snapshots = []
    for press, temp, crys in product(pressure, temperature, crystals):
        fname = f"dump-Trimer-P{press:.2f}-T{temp:.2f}-{crys}.gsd"
        with gsd.hoomd.open(str(data_dir / fname)) as trj:
            snapshots.append(HoomdFrame(trj[index]))

    return snapshots


def plot_grid(frames):
    for frame in frames:
        frame.plot_height = frame.plot_height // 3
        frame.plot_width = frame.plot_width // 3
    return gridplot(frames, ncols=3)


def plot_clustering(algorithm, X, snapshots, fit=True, max_frames=3):
    if fit:
        clusters = algorithm.fit_predict(X)
    else:
        clusters = algorithm.predict(X)
    cluster_assignment = np.split(clusters, len(snapshots))
    fig = plot_grid(
        [
            plot_frame(snap, order_list=cluster, categorical_colour=True)
            for snap, cluster, i in zip(snapshots, cluster_assignment, count())
            if i < max_frames
        ]
    )
    return fig


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
    classification = np.zeros(len(snapshot))
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
    cm, classes, normalize=True, title="Confusion matrix", cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
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
