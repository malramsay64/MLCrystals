#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Malcolm Ramsay <malramsay64@gmail.com>
#
# Distributed under terms of the MIT license.

import requests
from itertools import product
from pathlib import Path
import click


@click.command()
@click.argument("outdir")
def main(outdir):
    """Download crystal melting dataset from Zenodo."""
    crystals = ["p2", "p2gg", "pg"]
    temperatures = ["0.30", "0.35", "0.40", "0.45"]
    pressures = ["1.00"]

    base_url = "https://zenodo.org/record/1315097/files/"

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for crys, temp, press in product(crystals, temperatures, pressures):
        fname = f"dump-Trimer-P{press}-T{temp}-{crys}.gsd"
        print(base_url + fname)
        res = requests.get(base_url + fname, params={"download": 1})

        with open(outdir / fname, "wb") as dst:
            dst.write(res.content)


if __name__ == "__main__":
    main()
