#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""

import click
import gsd.hoomd
from bokeh.io import export_png
from sdanalysis import HoomdFrame
from sdanalysis.figures import plot_frame


@click.group()
def main():
    pass


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

    x_len, y_len = snap.box[:2]
    x_min, x_max = x_len / 2, -x_len / 2
    y_min, y_max = y_len / 2, -y_len / 2

    factor = 2 / 3
    buffer = 3.5

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

    fig = plot_frame(snap)
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

    fig.axis.visible = False
    fig.xgrid.visible = False
    fig.ygrid.visible = False
    fig.toolbar_location = None
    fig.toolbar.logo = None

    export_png(fig, "figures/labelled_config.png", height=1600, width=3200)


if __name__ == "__main__":
    main()
