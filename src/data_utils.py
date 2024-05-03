import numpy as np
import pandas as pd
from scipy.stats import norm
import logging

# Utilitary functions to process a generic options pandas DataFrame
NB_MARGINALS = 2
LOGGER = logging.getLogger(__name__)

def get_common_strikes(df: pd.DataFrame) -> set:
    expis = df.expiration.unique().tolist()
    common_strikes = set(df.query("expiration==@expis[0]").strike)
    for i in range(1, len(expis)):
        curr_expi = expis[i]
        common_strikes = common_strikes.intersection(set(df.query("expiration==@curr_expi").strike))
    LOGGER.info(f"Number of common strikes at all expirations: {len(common_strikes)}")
    return common_strikes
    

def clean_df(df: pd.DataFrame, min_oi: int=1000, target_expis: list[float]=None) -> pd.DataFrame:
    """
    From an options dataframe, returns a simpler dataframe with the most liquid expirations,
    only rows for which we have data at the same strikes for all maturities.
    """
    df = df.query("open_interest > @min_oi")
    df["mid"] = (df["bid_1545"] + df["ask_1545"]) / 2
    # df = df.query("option_type=='C'") # Keep calls only

    if target_expis is None:
        target_expis = sorted(df["maturity"].value_counts().iloc[:NB_MARGINALS].index.tolist())
    tolerance = 0.01
    df = df[np.any([np.isclose(df['maturity'], target, atol=tolerance) for target in target_expis], axis=0)]

    # The below is not actually needed.
    # common_strikes = get_common_strikes(df)
    # df = df[df["strike"].isin(common_strikes)]

    return df[["strike", "implied_volatility_1545", "mid", "maturity", "expiration", "option_type"]]

# TODO: correct things here that are a bit messy in the first two terms.
def smile_to_density(strikes: np.ndarray | list[float], prices: np.ndarray | list[float]):
    """
    Breeden-Litzenberger formula: we derive (through finite differences) the options prices
    with regards to the strike twice in order to get the implied probability density of the asset at expiry.

    We compute $\frac{[C(x+h) - C(x)] - [C(x) - C(x-h)]}{h**2}.$
    """
    first_order = np.divide(np.diff(prices, prepend=0), np.diff(strikes, prepend=strikes[0]))
    second_order = np.divide(np.diff(first_order, prepend=0), np.diff(strikes, prepend=strikes[-1]))
    return second_order


def black_scholes_call_price(S, K, T, r, sigma):
    """
    Calculate the Black-Scholes call option price given the implied volatility.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# TODO: add a way to check for the convex ordering of a set of measures