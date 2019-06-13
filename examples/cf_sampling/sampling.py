## Compare sampling plans
import numpy as np
import pandas as pd
import grama.core as gr
import time

from grama.core import pi # Import pipe
from grama.evals import ev_monte_carlo, ev_lhs
from scipy.stats import multivariate_normal

np.random.seed(101) # Set for reproducibility
np.set_printoptions(precision = 3)

SAMP_ALL = [10, 50, 100, 500, 1000]
n_samp   = len(SAMP_ALL)
n_repl   = int(50)

## Define simple Linear-Normal model
v   = np.array([1., 1.])
mu  = np.array([0, 0])
Sig = np.array([
    [1.0, 0.3],
    [0.3, 1.0]
])
mu_f   = np.dot(v, mu)
sig2_f = np.dot(v, np.dot(Sig, v))

model = gr.model_(
    name = "Linear-Normal",
    function = lambda x: np.dot(v, x),
    outputs  = ["f"],
    domain   = gr.domain_(
        hypercube = True,
        inputs    = ["X1", "X2"],
        bounds    = {
            "X1": [-np.Inf, +np.Inf],
            "X2": [-np.Inf, +np.Inf]
        }
    ),
    density  = gr.density_(
        pdf = lambda X: multivariate_normal.pdf(X, mean = mu, cov = Sig),
        pdf_factors = ["norm", "norm"],
        pdf_param = [
            {"loc": mu[0], "scale": Sig[0, 0]},
            {"loc": mu[1], "scale": Sig[1, 1]}
        ],
        pdf_corr = [Sig[0, 1]]
    )
)

## Replication study
mean_smc_all = np.zeros(n_repl); std_smc_all = np.zeros(n_repl)
mean_lhs_all = np.zeros(n_repl); std_lhs_all = np.zeros(n_repl)

mse_mean_smc = np.zeros(n_samp); mse_std_smc = np.zeros(n_samp)
mse_mean_lhs = np.zeros(n_samp); mse_std_lhs = np.zeros(n_samp)

t0 = time.time()
for jnd in range(n_samp):

    for ind in range(n_repl):
        ## Simple monte carlo
        df_res_smc = \
            model |pi| \
            ev_monte_carlo(n_samples = SAMP_ALL[jnd])

        ## Latin Hypercube Sample
        df_res_lhs = \
            model |pi| \
            ev_lhs(n_samples = SAMP_ALL[jnd])

        mean_smc_all[ind] = np.mean(df_res_smc["f"].values)
        mean_lhs_all[ind] = np.mean(df_res_lhs["f"].values)

        std_smc_all[ind] = np.std(df_res_smc["f"].values)
        std_lhs_all[ind] = np.std(df_res_lhs["f"].values)

    mse_mean_smc[jnd] = np.mean((mean_smc_all - mu_f)**2)
    mse_mean_lhs[jnd] = np.mean((mean_lhs_all - mu_f)**2)

    mse_std_smc[jnd] = np.mean((std_smc_all - np.sqrt(sig2_f))**2)
    mse_std_lhs[jnd] = np.mean((std_lhs_all - np.sqrt(sig2_f))**2)
t1 = time.time()

print("Execution time: {0:2.1f} sec".format(t1 - t0))
print("Samples        {}".format(SAMP_ALL))
print("")
print("mse_mean_smc = {}".format(mse_mean_smc))
print("mse_mean_lhs = {}".format(mse_mean_lhs))
print("")
print("mse_std_smc = {}".format(mse_std_smc))
print("mse_std_lhs = {}".format(mse_std_lhs))
