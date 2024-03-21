import numpy as np
import pandas as pd
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
    

def clean_df(df: pd.DataFrame, min_oi: int=3000) -> pd.DataFrame:
    """
    From an options dataframe, returns a simpler dataframe with the most liquid expirations,
    only rows for which we have data at the same strikes for all maturities.
    """
    df = df.query("open_interest > @min_oi")
    df["mid"] = (df["bid_1545"] + df["ask_1545"]) / 2
    df = df.query("option_type=='C'") # Keep calls only

    target_expis = sorted(df["maturity"].value_counts().iloc[:NB_MARGINALS].index.tolist())
    tolerance = 0.01
    df = df[np.any([np.isclose(df['maturity'], target, atol=tolerance) for target in target_expis], axis=0)]

    common_strikes = get_common_strikes(df)
    df = df[df["strike"].isin(common_strikes)]

    return df[["strike", "implied_volatility_1545", "mid", "maturity", "expiration"]]

def smile_to_density(strikes: np.ndarray | list[float], prices: np.ndarray | list[float]):
    """
    Breeden-Litzenberger formula: we derive (through finite differences) the options prices
    with regards to the strike twice in order to get the implied probability density of the asset at expiry.

    We compute $\frac{[C(x+h) - C(x)] - [C(x) - C(x-h)]}{h**2}.$
    """
    first_order = np.divide(np.diff(prices, prepend=0), np.diff(strikes, prepend=strikes[0]))
    second_order = np.divide(np.diff(first_order, prepend=0), np.diff(strikes, prepend=strikes[0]))
    return second_order
