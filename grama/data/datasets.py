__all__ = [
    "df_channel",
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

# PSAAP II IJMF (2020) data
df_channel = read_csv(Path(path_grama / "data/psaap.csv"))
df_channel.rename(
    columns=dict(
        # xi="x",
        x="xi",
        # H="W",
        W="H",
        # U="U_0",
        U_0="U",
        # T_0="T_f",
        T_f="T_0",
        # cp_p="C_pv",
        C_pv="cp_p",
        # cp_f="C_fp",
        C_fp="cp_f",
        # nu_f="mu_f",
        mu_f="nu_f",
        # Q_abs="eps_p",
        eps_p="Q_abs",
        # h_p="h",
        h="h_p",
    ),
    inplace=True,
)
# Compute thermal diffusivity from initial temperature
df_channel["alpha_f"] = (
      1.862e1
    + 1.327e-1 * df_channel.T_0
    + 1.026e-4 * df_channel.T_0**2
    - 5.270e-9 * df_channel.T_0**3
) * 1e-6
# Back-calculate the proper number density
df_channel["n"] = df_channel.N_p / df_channel.H**2 / df_channel.L
