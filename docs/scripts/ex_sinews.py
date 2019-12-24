import grama as gr
import pandas as pd
import matplotlib.pyplot as plt

from grama.models import make_cantilever_beam
md_beam = make_cantilever_beam()

md_beam >> \
    gr.ev_sinews(n_density=50, n_sweeps=10, df_det="nom", skip=True) >> \
    gr.pt_auto()
plt.savefig("../images/ex_beam_sinews_doe.png")

md_beam >> \
    gr.ev_sinews(n_density=50, n_sweeps=10, df_det="nom", skip=False) >> \
    gr.pt_auto()
plt.savefig("../images/ex_beam_sinews_res.png")
