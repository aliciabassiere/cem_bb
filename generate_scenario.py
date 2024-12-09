import numpy as np
from scipy.stats import qmc
import math

def generate_scenarios(n_simu, n_d, t, seed=None):
    """
    Generate quasi-random scenarios using Sobol sequences.
    Parameters:
    n_simu (int): Number of simulations to generate.
    n_d (int): The range to scale the sequences to.
    t (int): The length of each sequence.
    seed (int, optional): Seed for the random number generator. Defaults to None.
    Returns:
    tuple: A tuple containing:
        - scenarios (list of numpy arrays): List of generated scenario arrays.
        - n (int): The closest power of 2 larger than n_simu.
    """

    if seed is not None:
        np.random.seed(seed)
    else:
        seed = 42
        np.random.seed(seed)

    n = int(math.pow(2, math.ceil(math.log2(n_simu))))
    scenarios = []
    sequence = qmc.Sobol(t, seed=seed)
    sequences = sequence.random(n)
    
    for seq in sequences:
        scenario = np.floor(seq * n_d).astype(int)
        scenarios.append(scenario)

    return scenarios, n
