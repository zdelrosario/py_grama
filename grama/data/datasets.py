import os
from pandas import read_csv
from pathlib import Path

path_this = Path(__file__)
path_grama = path_this.parents[1]

# Stang (tidy form)
df_stang = read_csv(Path(path_grama / "data/stang_long.csv"))

# Stang (tidy form)
df_diamonds = read_csv(Path(path_grama / "data/diamonds.csv"))
