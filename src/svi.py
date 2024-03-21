import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class SVICalibrator:
    """
    A class to calibrate a volatility smile to the stochastic volatility inspired parametrization.
    Notations and process follows Jim Gatheral and Antoine Jacquier's “Arbitrage-free SVI volatility surfaces" (2014).
    """
    def __init__(self, strikes: list[float] | np.ndarray, ivs: list[float] | np.ndarray, forward: float=100) -> None:
        self.strikes = np.array(strikes)
        self.ivs = np.array(ivs)
        self.forward = forward
        self._calibrated_params = None

    def compute_svi_variance(self, params: list[float], log_moneyness: float):
        """
        Given log-moneyness $k$, returns $w(k, \Theta) = a + b \left(\rho (k-m) + \sqrt{(k-m)^2 + \sigma^2} \right)$.
        """
        a, b, rho, m, sigma = params
        return a + b * (rho * (log_moneyness - m) + np.sqrt((log_moneyness - m)**2 + sigma**2))

    def fit(self):
        def objective_function(params, log_moneyness, implied_vols):
            implied_variances = implied_vols**2
            a, b, rho, m, sigma = params
            return np.sum((implied_variances - self.compute_svi_variance(params, log_moneyness))**2)

        initial_guess = [0.1, 0.1, 0.1, 0.1, 0.1]  # Initial guess for the parameters
        log_moneyness = np.log(self.strikes / self.forward)

        result = minimize(objective_function, initial_guess, args=(log_moneyness, self.ivs))

        self._calibrated_params = result.x
    
    def get_calibrated_params(self):
        if self._calibrated_params is None:
            raise ValueError("Calibration has not been performed yet. Call fit() method first.")
        return self._calibrated_params

