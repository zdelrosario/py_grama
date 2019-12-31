__all__ = [
    "df_stang"
]

import os
import pandas as pd
from pathlib import Path

path_this = Path(__file__)
path_grama = path_this.parents[1]

# Stang (tidy form)
df_stang = pd.read_csv(Path(path_grama / "data/stang_long.csv"))
