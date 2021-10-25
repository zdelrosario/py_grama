__all__ = [
    "df_diamonds",
    "df_ruff",
    "df_stang",
    "df_stang_wide",
    "df_trajectory_full",
    "df_trajectory_windowed",
    "df_shewhart",
]

import os
from pandas import read_csv
from pathlib import Path


path_this = Path(__file__)
path_grama = path_this.parents[1]

# Stang (wdie form)
df_stang_wide = read_csv(Path(path_grama / "data/stang.csv"))

# Stang (tidy form)
df_stang = read_csv(Path(path_grama / "data/stang_long.csv"))

# Stang (tidy form)
df_diamonds = read_csv(Path(path_grama / "data/diamonds.csv"))

# Ruff (tidy form)
df_ruff = read_csv(Path(path_grama / "data/ruff.csv"))

# Trajectories
df_trajectory_full = read_csv(Path(path_grama / "data/trajectory_full.csv"))
df_trajectory_windowed = read_csv(Path(path_grama / "data/trajectory_windowed.csv"))

# Shewhart
df_shewhart = read_csv(Path(path_grama / "data/shewhart1931-table3.csv"))
