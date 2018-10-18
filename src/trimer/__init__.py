#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""Utilities for handling the trimer molecule."""

import numpy as np
from pathlib import Path
from itertools import product
from typing import List

from bokeh.plotting import gridplot
from sdanalysis.figures import plot_frame
from sdanalysis import HoomdFrame
import gsd.hoomd


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


def plot_clustering(algorithm, X, snapshots):
    cluster_assignment = np.split(algorithm.fit_predict(X), len(snapshots))
    fig = plot_grid(
        [
            plot_frame(snap, order_list=cluster, categorical_colour=True)
            for snap, cluster in zip(snapshots, cluster_assignment)
        ]
    )
    return fig


def plot_snapshots(snapshots):
    return plot_grid([plot_frame(snap) for snap in snapshots])


def classify_mols(snapshot, crystal, boundary_buffer=3.5):
    """Classify molecules as crystalline, amorphous or boundary."""
    mapping = {"liq": 0, "p2": 1, "p2gg": 2, "pg": 3, "None": 4}
    position = snapshot.position
    # This gets the details of the box from the simulation
    box = snapshot.box

    # All axes have to be True, True == 1, use product for logical and operation
    is_crystal = np.product(np.abs(position) < box[:3] / 3, axis=1).astype(bool)
    boundary = np.logical_and(
        np.product(np.abs(position) < box[:3] / 3 + boundary_buffer, axis=1),
        np.product(np.abs(position) > box[:3] / 3 - boundary_buffer, axis=1),
    )

    # Create classification array
    classification = np.zeros(len(snapshot))
    classification[is_crystal] = mapping[crystal]
    classification[boundary] = 4
    return classification
