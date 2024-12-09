import numpy as np
import pandas as pd
import os
import time
import math
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
from tqdm.contrib.itertools import product

from generate_scenario import generate_scenarios

############################## Initialization ##############################

name = str(input("Enter the name of Simulation Batch: "))
#name = "benchmark_new"
path = r""  # Update this with your actual path
path_inputs = path + r"inputs"
subdirectories = ["functions", "values", "perfect_foresight", "stochastic", "deterministic"]
path2 = os.path.join(path, "outputs", "batch_simulations_" + name)
paths_to_create = [os.path.join(path2, subdir) for subdir in subdirectories]
if not os.path.exists(path2):
    os.makedirs(path2)
    for dir_path in paths_to_create:
        os.makedirs(dir_path)
    print(f"Directories created at: {path2}")
else:
    print(f"Directory already exists at: {path2}")

coal_phase_out = str(input("Apply Coal Phase-out Plan? [y/n] ")) or "n"
carbon_tax = input("Apply carbon tax in 2030? (y/n): ") or "n"
nombres_simu = int(input("Enter the number of simulations: ")  or "2")
n_simu = max(8, int(math.pow(2, math.ceil(math.log2(nombres_simu)))))

lambda_weight = input("Carbon target constraining parameter (if none press Enter):") or 0
mu_weight = input("Gas investing constraining parameter (if none press Enter):") or 0
kappa_weight = input("Wind investing constraining parameter (if none press Enter):") or 0
nu_weight = input("Solar investing constraining parameter (if none press Enter):") or 0



if n_simu != nombres_simu:
    print("WARNING! Number of simulations: " + str(nombres_simu) + " replaced by " + str(n_simu) + " for Sobol stability")

# ############################## General Parameters ##############################

class SimulationParameters:
    """
    This class holds the parameters for running a simulation.

    Attributes:
        now (float): The current time.
        name (str): The name of the simulation.
        path (Path): The path to the current directory.
        path2 (Path): The path to the output directory.
        path_inputs (Path): The path to the input directory.
        path_functions (Path): The path to the functions directory.
        path_values (Path): The path to the values directory.
        path_deterministic (Path): The path to the deterministic directory.
        path_stochastic (Path): The path to the stochastic directory.
        seed (NoneType): Seed for random number generation.
        t (int): Time horizon.
        extension (int): Extension parameter.
        h (int): Number of hours.
        beta (float): Discount factor.
        load_growth (float): Load growth rate.
        lambda_weight (int): Weight for lambda constraint.
        mu_weight (int): Weight for mu constraint.
        kappa_weight (int): Weight for kappa constraint.
        nu_weight (int): Weight for nu constraint.
        is_coal_phase_out (str): Indicator for coal phase-out.
        carbon_tax (str): Indicator for carbon tax.
        kc (ndarray): Coal phase-out path.
        cpath (ndarray): Carbon tax path.
        n_d (int): Number of scenarios
    """
    def __init__(self):
        tech_parameters = TechnoParameters()
        self.now = time.time()

        ### Simulation parameters

        self.n_d = 7
        self.n_simu = int(math.pow(2, math.ceil(math.log2(nombres_simu))))
        self.seed = 42  # Set your seed value here
        self.t = 18
        self.extension = 5
        self.h = 8760
        self.beta = 0.98
        self.load_growth = 0.024
        self.actualisation = np.power(self.beta, np.arange(self.t + 1))
        self.n_simu = n_simu

        self.scenarios = generate_scenarios(self.n_simu, self.n_d, self.t + 1, seed=self.seed)

        ### Paths

        self.name = name
        self.path = path
        self.path_inputs = path_inputs
        self.path2 = path2
        self.path_functions = paths_to_create[0]
        self.path_values = paths_to_create[1]
        self.path_perfectforesight = paths_to_create[2]
        self.path_stochastic = paths_to_create[3]
        self.path_deterministic = paths_to_create[4]
        
        ### Policies

        self.coal_phase_out = coal_phase_out
        if self.coal_phase_out.lower() == "y":
            self.kc = [32900, 30000, 27300, 25800, 23900, 23500, 21800, 20200, 17300, 16800, 14900, 12700, 10600, 8700, 8100,
                  6400, 4000, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif self.coal_phase_out.lower() == "n":
            self.kc = np.repeat(tech_parameters.kc0, self.t + 2)
        else:
            print("ERROR COAL PHASE OUT")

        self.carbon_tax = carbon_tax
        self.carbon_2030 = "130"
        self.carbon_2050 = "250"
        if self.carbon_tax.lower() == "y":
            self.cpath = np.linspace(50, float(self.carbon_2050), 2050 - 2022)
        elif self.carbon_tax.lower() == "n":
            self.cpath = np.zeros(2050 - 2022)
        else:
            print("ERROR CARBON TAX")

        self.lambda_weight = lambda_weight
        self.mu_weight = mu_weight
        self.kappa_weight = kappa_weight
        self.nu_weight = nu_weight

class Scenario:
    """
    A class used to represent different investment scenarios for simulation purposes.
    Attributes
    ----------
    scenarios : list
        A list of generated scenarios based on the simulation parameters.
    reference_scenario : numpy.ndarray
        An array representing the reference scenario, repeated value of 3.
    worst_scenario : numpy.ndarray
        An array representing the worst-case scenario, repeated value of 6.
    best_scenario : numpy.ndarray
        An array representing the best-case scenario, repeated value of 5.
    average_scenario : numpy.ndarray
        An array representing the average scenario, repeated value of 7.
    realistic_scenario : numpy.ndarray
        An array representing a realistic scenario, repeated value of 8.
    mimic_scenario : numpy.ndarray
        An array representing a mimic scenario, repeated value of 9.
    Methods
    -------
    __init__():
        Initializes the Scenario class with various predefined scenarios.
    """

    def __init__(self):
        simu_parameters = SimulationParameters()
        self.scenarios = generate_scenarios(simu_parameters.n_simu, simu_parameters.n_d, simu_parameters.t + 1, simu_parameters.seed)[0]
        self.reference_scenario = np.repeat(3, simu_parameters.t+1)
        self.worst_scenario = np.repeat(6, simu_parameters.t+1)
        self.best_scenario = np.repeat(5, simu_parameters.t+1)
        self.average_scenario = np.repeat(7, simu_parameters.t+1)
        self.realistic_scenario = np.repeat(8, simu_parameters.t+1)
        self.mimic_scenario = np.repeat(9, simu_parameters.t+1)


class TechnoParameters:
    """
    A class to represent the technological parameters for a dynamic investment program.
    Attributes
    ----------
    kwbound : int
        Maximum capacity for wind energy.
    kgbound : int
        Maximum capacity for gas energy.
    ksbound : int
        Maximum capacity for solar energy.
    step_w : int
        Discretization size for wind energy.
    step_s : int
        Discretization size for solar energy.
    step_g : int
        Discretization size for gas energy.
    unit_w : int
        Unit size for wind energy.
    unit_s : int
        Unit size for solar energy.
    unit_g : int
        Unit size for gas energy.
    kw0 : int
        Initial capacity for wind energy.
    kg0 : int
        Initial capacity for gas energy.
    ks0 : int
        Initial capacity for solar energy.
    kc0 : int
        Initial capacity for another energy type (not specified).
    kwlow : int
        Minimum capacity for wind energy (default is initial capacity).
    kglow : int
        Minimum capacity for gas energy (default is initial capacity).
    kslow : int
        Minimum capacity for solar energy (default is initial capacity).
    kpeak : int
        Peak capacity (not specified which energy type).
    n_w : int
        Number of grid points for wind energy.
    n_g : int
        Number of grid points for gas energy.
    n_s : int
        Number of grid points for solar energy.
    kw : numpy.ndarray
        Array of wind energy capacities.
    kg : numpy.ndarray
        Array of gas energy capacities.
    ks : numpy.ndarray
        Array of solar energy capacities.
    Methods
    -------
    __init__():
        Initializes the TechnoParameters with default values.
    """

    def __init__(self):
        self.kwbound = 155000 
        self.kgbound = 46000  
        self.ksbound = 258000  
        self.step_w = 2000 
        self.step_s = 2000  
        self.step_g = 1000 
        self.unit_w = 1000
        self.unit_s = 1000 
        self.unit_g = 500 
        self.kw0 = 55000  
        self.kg0 = 30000  
        self.ks0 = 58000  
        self.kc0 = 35400
        self.kwlow = self.kw0 
        self.kglow = self.kg0  
        self.kslow = self.ks0  

        self.kpeak = 4180 

        self.n_w = int(((self.kwbound - self.kwlow) / self.step_w) + 1)  
        self.n_g = int(((self.kgbound - self.kglow) / self.step_g) + 1)  
        self.n_s = int(((self.ksbound - self.kslow) / self.step_s) + 1)  

        self.kw = np.linspace(self.kwlow, self.kwbound, self.n_w)
        self.kg = np.linspace(self.kglow, self.kgbound, self.n_g)
        self.ks = np.linspace(self.kslow, self.ksbound, self.n_s)

        
class CostParameters:
    """
    A class to represent the cost parameters for different energy sources and their associated prices.
    Attributes
    ----------
    a_s : float
        Fixed cost parameter for solar energy.
    b_s : float
        Variable cost parameter for solar energy.
    a_w : float
        Fixed cost parameter for wind energy.
    b_w : float
        Variable cost parameter for wind energy.
    a_g : float
        Fixed cost parameter for gas energy.
    b_g : float
        Variable cost parameter for gas energy.
    a_c : float
        Fixed cost parameter for coal energy.
    b_c : float
        Variable cost parameter for coal energy.
    a_peak : float
        Fixed cost parameter for peak energy.
    b_peak : float
        Variable cost parameter for peak energy.
    pc : numpy.ndarray
        Array of coal prices.
    conv_factor_c : float
        Conversion factor for coal to electricity.
    pg : numpy.ndarray
        Array of gas prices.
    pg_mean : numpy.ndarray
        Mean gas prices.
    pg_low : numpy.ndarray
        25th percentile of gas prices.
    pg_high : numpy.ndarray
        75th percentile of gas prices.
    pg_min : numpy.ndarray
        Minimum gas prices.
    pg_max : numpy.ndarray
        Maximum gas prices.
    pg_multiplier : numpy.ndarray
        Multiplier for gas prices based on low and high percentiles.
    pg_low_shock : numpy.ndarray
        Gas price shock for low scenario.
    pg_high_shock : numpy.ndarray
        Gas price shock for high scenario.
    pg_deterministic : numpy.ndarray
        Deterministic gas prices.
    pg_mean_deterministic : numpy.ndarray
        Mean deterministic gas prices.
    p_peak : float
        Peak oil price.
    fossil_evol : numpy.ndarray
        Evolution factor for fossil fuel prices.
    cintensity_g : float
        Carbon intensity for gas energy.
    cintensity_c : float
        Carbon intensity for coal energy.
    cintensity_peak : float
        Carbon intensity for peak energy.
    voll : float
        Value of lost load (Euros/MWh).
    penalty_default : float
        Penalty for defaulting (Euros/MWh).
    Methods
    -------
    __init__():
        Initializes the cost parameters with default values.
    """
    def __init__(self):
        simu_parameters = SimulationParameters()
        self.a_s = 15000 
        self.b_s = 0 
        self.a_w = 21000 
        self.b_w = 0.25 
        self.a_g = 20000 
        self.b_g = 2.31 
        self.a_c = 32500 
        self.b_c = 3 
        self.a_peak = 6960
        self.b_peak = 18 

        self.conv_factor_c = 0.456 
        self.conv_factor_g = 0.44

        self.pc = np.array(pd.read_csv(simu_parameters.path_inputs + r"/coal_price_inputs.csv", index_col=0, sep= ";")).mean(axis=1).T
        
        self.pg = np.array(pd.read_csv(simu_parameters.path_inputs + r"/gas_price_inputs.csv", index_col=0).T)
        self.pg_mean = self.pg.mean(axis=0)
        self.pg_low = np.quantile(self.pg, 0.25, axis=0)
        self.pg_high = np.quantile(self.pg, 0.75, axis=0)
        self.pg_min = self.pg.min(axis=0)
        self.pg_max = self.pg.max(axis=0)
        self.pg_multiplier = np.array([self.pg_low/self.pg_mean, self.pg_high/self.pg_mean]).mean(axis=1)
        self.pg_low_shock = self.pg[5] # 2020
        self.pg_high_shock = self.pg[7] # 2022
        self.pg = np.vstack([self.pg[:simu_parameters.n_d-2], self.pg_low_shock, self.pg_high_shock])
        self.pg_mean = self.pg.mean(axis=0)
        self.pg_deterministic = self.pg[:simu_parameters.n_d-2]
        self.pg_mean_deterministic = self.pg_deterministic.mean(axis=0)
        
        self.p_peak = 94*0.0019047619*1000
        self.fossil_evol = np.repeat(np.linspace(1.00, 1 + ((0*simu_parameters.t)/100), simu_parameters.t+1), simu_parameters.h).reshape(simu_parameters.t+1, simu_parameters.h)

        self.cintensity_g = 0.33
        self.cintensity_c = 0.96
        self.cintensity_peak = 0.55

        self.voll = 12240 
        self.penalty_default = 12240

class InvestmentParameters:
    """
    A class to represent the investment parameters for different energy sources over time.
    Attributes
    ----------
    gammaw_seq : numpy.ndarray
        Investment costs for wind energy.
    gammag_seq : numpy.ndarray
        Investment costs for gas energy (€/MW).
    gammas_seq : numpy.ndarray
        Investment costs for solar PV energy (€/MW).
    scrapr : numpy.ndarray
        Scrapping values for wind energy investments.
    scrapg : numpy.ndarray
        Scrapping values for gas energy investments.
    scraps : numpy.ndarray
        Scrapping values for solar PV energy investments.
    -------
    
    """

    def __init__(self):
        simu_parameters = SimulationParameters()
        self.gamma_w_seq = np.linspace(1000000, 950000, simu_parameters.t+1) 
        self.gamma_g_seq = np.linspace(550000, 536000, simu_parameters.t + 1) 
        self.gamma_s_seq = np.linspace(463000, 387000, simu_parameters.t + 1) 
        self.scrap_w = -0.2*self.gamma_w_seq 
        self.scrap_g = -0.2*self.gamma_g_seq 
        self.scrap_s = -0.2*self.gamma_s_seq 
        # self.scrapw = np.linspace(-1, -1, simu_parameters.T+1)
        # self.scrapg = np.linspace(-1, -1, simu_parameters.T+1)
        # self.scraps = np.linspace(-1, -1, simu_parameters.T+1)


class GradientParameters:
    """
    A class used to represent the parameters for gradient simulation.
    Attributes
    ----------
    n_sample : int
        The number of samples to be used in the simulation (default is 63000).
    precision_gradient : int
        The precision level of the gradient (default is 8).
    gradient_depth : int
        The depth of the gradient (default is 16).
    Methods
    -------
    __init__()
        Initializes the Gradientparameters with default values.
    """

    def __init__(self):
        self.n_sample = 63000  
        self.precision_gradient = 8 
        self.gradient_depth = 16



