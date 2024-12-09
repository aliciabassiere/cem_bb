import pandas as pd
from simulation_parameters import *

simu_parameters = SimulationParameters()

class CapacityFactor:
    """Initialize the CapacityFactors class.
        This class is responsible for loading and processing capacity factor data for wind and solar 
        energy systems. It calculates statistical metrics such as quantiles, means, and applies 
        shock multipliers to simulate low and high capacity scenarios. The processed data is stored 
        and exported to CSV files for further analysis and simulation.
        Attributes:
            input_path (Path): Path to the input data files.
            ref_year_index (int): Reference year index for shocks.
            wind_cf (np.ndarray): Wind capacity factor data.
            wind_cf_low (np.ndarray): 25th percentile of wind capacity factors.
            wind_cf_high (np.ndarray): 75th percentile of wind capacity factors.
            wind_cf_min (np.ndarray): Minimum wind capacity factors.
            wind_cf_max (np.ndarray): Maximum wind capacity factors.
            wind_cf_mean (np.ndarray): Mean wind capacity factors.
            wind_cf_multiplier (np.ndarray): Multipliers for capacity factor shocks.
            wind_cf_low_shock (np.ndarray): Wind capacity factors after low shock.
            wind_cf_high_shock (np.ndarray): Wind capacity factors after high shock.
            wind_cf_deterministic (np.ndarray): Deterministic wind capacity factors.
            wind_cf_mean_deterministic (np.ndarray): Mean of deterministic wind capacity factors.
            wind_cf_mimic (np.ndarray): Mimicked wind capacity factors with shocks.
            wind_cf_mean_mimic (np.ndarray): Mean of mimicked wind capacity factors.
            pv_cf (np.ndarray): Solar PV capacity factor data.
            pv_cf_mean (np.ndarray): Mean solar capacity factors.
            pv_cf_median (np.ndarray): Median solar capacity factors.
            pv_cf_low (np.ndarray): 25th percentile of solar capacity factors.
            pv_cf_high (np.ndarray): 75th percentile of solar capacity factors.
            pv_cf_min (np.ndarray): Minimum solar capacity factors.
            pv_cf_max (np.ndarray): Maximum solar capacity factors.
            pv_multiplier (np.ndarray): Multipliers for solar capacity factor shocks.
            pv_low_shock (np.ndarray): Solar capacity factors after low shock.
            pv_high_shock (np.ndarray): Solar capacity factors after high shock.
            pv_cf_deterministic (np.ndarray): Deterministic solar capacity factors.
            pv_cf_mean_deterministic (np.ndarray): Mean of deterministic solar capacity factors.
            pv_cf_mimic (np.ndarray): Mimicked solar capacity factors with shocks.
            pv_cf_mean_mimic (np.ndarray): Mean of mimicked solar capacity factors.
        """
    def __init__(self):
        self.cap_factor = np.array(pd.read_csv(simu_parameters.path_inputs + 
                                               r"/capacity_factor_inputs.csv", index_col=0).T)
        self.cap_factor_low = np.quantile(self.cap_factor, 0.25, axis=0)
        self.cap_factor_high = np.quantile(self.cap_factor, 0.75, axis=0)
        self.cap_factor_min = self.cap_factor.min(axis=0)
        self.cap_factor_max = self.cap_factor.max(axis=0)
        self.wind_cf_mean = self.cap_factor.mean(axis=0)
        
        self.cap_factor_multiplier = np.array([self.cap_factor_high / self.wind_cf_mean, 
                                               self.cap_factor_low / self.wind_cf_mean]).mean(axis=1)
        
        self.cap_factor_low_shock = self.cap_factor[4] * self.cap_factor_multiplier[0]
        self.cap_factor_high_shock = self.cap_factor[4] * self.cap_factor_multiplier[1]
        self.cap_factor_low_shock[self.cap_factor_low_shock > 1] = 1
        self.cap_factor = np.vstack([self.cap_factor[:simu_parameters.n_d-2], 
                                     self.cap_factor_low_shock, self.cap_factor_high_shock])
        self.cap_factor_deterministic = self.cap_factor[:simu_parameters.n_d-2]

        self.wind_cf_mean_deterministic = self.cap_factor_deterministic.mean(axis=0)
        self.cap_factor_mimic = np.vstack([self.cap_factor, self.cap_factor_low_shock, 
                                           self.cap_factor_low_shock, self.cap_factor_low_shock, 
                                           self.cap_factor_high_shock, self.cap_factor_high_shock, 
                                           self.cap_factor_high_shock])
        self.cap_factor_mean_mimic = self.cap_factor_mimic.mean(axis=0)

        self.pv_cf = np.array(pd.read_csv(simu_parameters.path_inputs + r"/capacity_factor_inputs_pv.csv", 
                                          index_col=0).T)
        self.pv_cf_mean = self.pv_cf.mean(axis=0) + 0.000000000000000000000000000000000001  # Prevents error message
        self.pv_cf_median =  np.quantile(self.pv_cf, 0.5, axis=0) + 0.000000000000000000000000000000000001
        self.pv_cf_low = np.quantile(self.pv_cf, 0.25, axis=0)
        self.pv_cf_high = np.quantile(self.pv_cf, 0.75, axis=0)
        self.pv_cf_min = self.pv_cf.min(axis=0)
        self.pv_cf_max = self.pv_cf.max(axis=0)
        
        self.pv_multiplier = np.array([self.pv_cf_high / self.pv_cf_mean, self.pv_cf_low / self.pv_cf_mean])
        self.pv_low_shock = self.pv_cf[4] * self.pv_multiplier[0]
        self.pv_high_shock = self.pv_cf[4] * self.pv_multiplier[1]
        self.pv_low_shock[self.pv_low_shock > 1] = 1

        self.pv_cf = np.vstack([self.pv_cf[:simu_parameters.n_d-2], self.pv_low_shock, self.pv_high_shock])        
        self.pv_cf_deterministic = self.pv_cf[:simu_parameters.n_d-2]
        self.pv_cf_mean_deterministic = self.pv_cf_deterministic.mean(axis=0) + 0.000000000000000000000000000000000001  
        self.pv_cf_mimic = np.vstack([self.pv_cf, self.pv_low_shock, self.pv_low_shock, 
                                      self.pv_low_shock, self.pv_high_shock, self.pv_high_shock, 
                                      self.pv_high_shock])
        self.pv_cf_mean_mimic = self.pv_cf_mimic.mean(axis=0)
