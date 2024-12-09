from simulation_parameters import *

simu_parameters = SimulationParameters()
tech_parameters = TechnoParameters()

class Constraints:
    
    def __init__(self, lambda_weight=0, mu_weight=0, kappa_weight=0, nu_weight=0):
        """
        CO2 Investment Constraints

        Search for the CO2 constraint:
        In 2021: 206Mt of CO2 for the Power sector (https://www.statista.com/statistics/1290543/power-sector-carbon-emissions-germany/) with 580.40 TWh, and an intensity of 349 g/kWh (approximately 203 estimated)
        In 1990: carbon intensity of 652g/kWh produced in Germany (https://www.eea.europa.eu/ims/greenhouse-gas-emission-intensity-of-1) for a demand of 546.82 TWh in 1990 (356.52 Mt estimated)
        => Approximately 48% reduction
        """
        self.lambda_weight = simu_parameters.lambda_weight
        self.lambda_matrix = np.ones(simu_parameters.t)
        n = len(self.lambda_matrix)
        indices = np.arange(n)
        self.c_objective = 206000000 * 0.92 ** indices

        """
        Gas Investment Constraints
        """
        self.mu_weight = mu_weight
        self.mu_matrix = np.ones(2040-2022)
        self.kg_objective = np.linspace(tech_parameters.kg0, tech_parameters.kg0, 2040-2022)

        """
        Wind Investment Constraints
        """
        self.kappa_weight = kappa_weight
        self.kappa_matrix = np.ones(2040-2022)
        self.kw_objective = np.linspace(tech_parameters.kw0, 160000, 2040-2022)

        """
        Solar Investment Constraints
        """
        self.nu_weight = nu_weight
        self.nu_matrix = np.ones(2040-2022)
        self.ks_objective = np.linspace(tech_parameters.ks0, 400000, 2040-2022)

    def compute_lambda_constraint(self, t, carbon_realised):
        return (self.lambda_weight + self.lambda_weight * (carbon_realised/1000000 - self.c_objective[t]/1000000 )**2) * ((carbon_realised - self.c_objective[t]) >= 0)

    def compute_mu_constraint(self, t, actual_kg):
        return self.mu_weight * (actual_kg - self.kg_objective[t])**2 * (actual_kg - self.kg_objective[t] >= 0)

    def compute_kappa_constraint(self, t, actual_kw):
        return self.kappa_weight * (self.kw_objective[t] - actual_kw)**2 * (self.kw_objective[t] - actual_kw >= 0)

    def compute_nu_constraint(self, t, actual_ks):
        return self.nu_weight * (self.ks_objective[t] - actual_ks)**2 * (self.ks_objective[t] - actual_ks >= 0)
