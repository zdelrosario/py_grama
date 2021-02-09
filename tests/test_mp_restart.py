from grama.models import make_trajectory_linear
from grama.data import df_trajectory_full
import grama as gr
import os 

n_process = int(os.cpu_count()/2)

md_trajectory = make_trajectory_linear()

df_fit = (
    md_trajectory
    >> gr.ev_nls(df_data=df_trajectory_full, n_restart=10, n_process=n_process)
)
