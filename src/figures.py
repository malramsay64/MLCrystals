#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""A collection of tools to generate figures.

This module contains functions for the creation of figures, both through a jupyter
notebook interface or on the command line.

"""

from typing import List, Tuple

import click
import gsd.hoomd
from bokeh.io import export_png
from bokeh.plotting import Figure
from sdanalysis import HoomdFrame
from sdanalysis.figures import plot_frame


def cell_regions(
    x_len: float, y_len: float, factor: float = 2 / 3, buffer: float = 3.5
) -> Tuple[List[List[float]], ...]:
    """Calculate the boundary of the different regions used for classification.

    When running an interface simulation there is a region in which the molecules are
    not integrated, being the internal crystal region, while all the surrounding
    material is melted. This calculates the boundaries of that region from the length of
    the x and y coordinates of the simulation cell.

    Args:
        x_len: The length of the x component of the simulation cell
        y_len: The length of the y component of the simulations cell
        factor: The fraction of each side of the box which remains crystalline
        buffer: Size of the buffer region around the interface. This is to help
            the machine learning algorithm by not introducing molecular
            configurations which are confusing and ill defined.

    Returns:
        cell: The coordinates of the complete simulation cell
        liq: The inner boundary of the liquid region
        crys: The outer boundary of the crystal region

    """
    x_min, x_max = x_len / 2, -x_len / 2
    y_min, y_max = y_len / 2, -y_len / 2

    cell = [[x_min, x_max, x_max, x_min], [y_min, y_min, y_max, y_max]]

    liq = [
        [
            x_min * factor + buffer,
            x_max * factor - buffer,
            x_max * factor - buffer,
            x_min * factor + buffer,
        ],
        [
            y_min * factor + buffer,
            y_min * factor + buffer,
            y_max * factor - buffer,
            y_max * factor - buffer,
        ],
    ]

    crys = [
        [
            x_min * factor - buffer,
            x_max * factor + buffer,
            x_max * factor + buffer,
            x_min * factor - buffer,
        ],
        [
            y_min * factor - buffer,
            y_min * factor - buffer,
            y_max * factor + buffer,
            y_max * factor + buffer,
        ],
    ]

    return cell, liq, crys


def style_snapshot(figure: Figure) -> Figure:
    """Style a bokeh figure as a configuration snapshot.

    This is collection of style changes to make the output of a snapshot consistent and
    nice. Primarily it removes all the extra stuff which isn't helpful in defining the
    configuration like the axes, and the interactive tools.

    """
    figure.axis.visible = False
    figure.xgrid.visible = False
    figure.ygrid.visible = False
    figure.toolbar_location = None
    figure.toolbar.logo = None

    return figure


@click.group()
def main():
    pass


def plot_labelled_config(snapshot: HoomdFrame) -> Figure:
    """Plot an input configuration indicating the labelling scheme.

    This plots the configuration with an overlay indicating the regions in which
    particles are classified as liquid (blue) and the regions where they are classified
    as crystal (red).

    Args:
        snapshot: The snapshot configuration to display

    """
    x_len, y_len = snapshot.box[:2]
    cell, liq, crys = cell_regions(x_len, y_len)

    fig = plot_frame(snapshot)
    # use the canvas backend which supports transparency and other effects.
    fig.output_backend = "canvas"

    # Plot the liquid region, being the cell boundary with a hole in the middle
    fig.multi_polygons(
        xs=[[[cell[0], liq[0]]]],
        ys=[[[cell[1], liq[1]]]],
        fill_alpha=0.3,
        fill_color="blue",
        line_color=None,
    )

    # Plot the crystal region being a central rectangle
    fig.multi_polygons(
        xs=[[[crys[0]]]],
        ys=[[[crys[1]]]],
        fill_alpha=0.3,
        fill_color="red",
        line_color=None,
    )

    return style_snapshot(fig)


@main.command()
@click.argument("infile", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-i",
    "--index",
    default=0,
    help="The index of the frame within the input file to plot",
    type=int,
)
def labelled_config(infile, index):
    """Plot an input configuration indicating the labelling scheme.

    This plots the configuration with an overlay indicating the regions in which
    particles are classified as liquid (blue) and the regions where they are classified
    as crystal (red).

    """
    with gsd.hoomd.open(infile) as trj:
        snap = HoomdFrame(trj[index])

    fig = plot_labelled_config(snap)

    export_png(fig, "figures/labelled_config.png", height=1600, width=3200)


if __name__ == "__main__":
    main()
