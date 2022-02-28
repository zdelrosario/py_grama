# Statistical Process Control tools
__all__ = ["c_sd", "B3", "B4"]

from scipy.special import gamma
from grama import make_symbolic
from numpy import sqrt

def c_sd(n):
    r"""Anti-biasing constant for aggregate standard deviation

    Returns the anti-biasing constant for aggregated standard deviation
    estimates. If the average of $k$ samples each size $n$ are averaged to
    produce $\overline{S} = (1/k) \sum_{i=1}^k S_i$, then the de-biased standard
    deviation is:

        $$\hat{\sigma} = \overline{S} / c(n)$$

    Arguments:
        n (int): Sample (batch) size

    Returns:
        float: anti-biasing constant

    References:
        Kenett and Zacks, Modern Industrial Statistics (2014) 2nd Ed

    """
    return gamma(n/2) / gamma( (n-1)/2 ) * sqrt( 2 / (n-1) )

def B3(n):
    r"""Lower Control Limit constant for standard deviation

    Returns the Lower Control Limit (LCL) constant for aggregated standard
    deviation estimates. If the average of $k$ samples each size $n$ are
    averaged to produce $\overline{S} = (1/k) \sum_{i=1}^k S_i$, then the LCL
    is:

        $$LCL = B_3 \overline{S}$$

    Arguments:
        n (int): Sample (batch) size

    Returns:
        float: LCL constant

    References:
        Kenett and Zacks, Modern Industrial Statistics (2014) 2nd Ed, Equation (8.22)

    """
    return max( 1 - 3 / c_sd(n) * sqrt(1 - c_sd(n)**2), 0 )

def B4(n):
    r"""Upper Control Limit constant for standard deviation

    Returns the Upper Control Limit (UCL) constant for aggregated standard
    deviation estimates. If the average of $k$ samples each size $n$ are
    averaged to produce $\overline{S} = (1/k) \sum_{i=1}^k S_i$, then the UCL
    is:

        $$UCL = B_4 \overline{S}$$

    Arguments:
        n (int): Sample (batch) size

    Returns:
        float: UCL constant

    References:
        Kenett and Zacks, Modern Industrial Statistics (2014) 2nd Ed, Equation (8.22)

    """
    return 1 + 3 / c_sd(n) * sqrt(1 - c_sd(n)**2)
