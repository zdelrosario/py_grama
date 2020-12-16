"""Datasets

Built-in datasets.

Datasets:
    df_diamonds: Diamond characteristics and prices. Columns:
        carat:

        cut:

        color:

        clarity:

        depth:

        table:

        price:

        x:

        y:

        z:

    df_stang: Aluminum alloy data from Stang et al. (1946). Columns:
        thick (inches): Nominal thickness

        alloy: Alloy designation

        E (Kips/inch^2): Young's modulus

        mu (-): Poisson's ratio

        ang (degrees): Angle of test to alloy roll direction

    df_ruff: Metal data from Ruff (1984). Columns:
        part: Part identifier

        TYS: Tensile Yield Stress (ksi)

        TUS: Tensile Ultimate Stress (ksi)

        thickness: Part thickness (in)

    df_trajectory_full: Simulated trajectory data. Columns:

        t: Time since projectile launch (seconds)

        x: Projectile range (meters)

        y: Projectile height (meters)

References:
    Stang, Greenspan, and Newman, "Poisson's ratio of some structural alloys for
    large strains" (1946) U.S. Department of Commerce National Bureau of Standards

    Ruff, Paul E. An Overview of the MIL-HDBK-5 Program. BATTELLE COLUMBUS DIV
    OH, 1984.
"""

from .datasets import *
