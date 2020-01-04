"""Datasets

Built-in datasets.

Datasets:
    df_stang: Aluminum alloy data from Stang et al. (1946). Columns:
        thick (inches): Nominal thickness

        alloy: Alloy designation

        E (Kips/inch^2): Young's modulus

        mu (-): Poisson's ratio

        ang (degrees): Angle of test to alloy roll direction

References:
    Stang, Greenspan, and Newman, "Poisson's ratio of some structural alloys for
    large strains" (1946) U.S. Department of Commerce National Bureau of Standards
"""

from .datasets import df_stang
