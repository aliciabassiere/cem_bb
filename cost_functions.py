import itertools
from load import *
from simulation_parameters import *
from capacity_factors import CapacityFactor

cost_parameters = CostParameters()
investment_parameters = InvestmentParameters()
simu_parameters = SimulationParameters()
tech_parameters = TechnoParameters()

class IterativeFunctions:
    """
    A class to represent various cost functions.
    Attributes
    ----------
    cost_params : CostParameters
        An instance of CostParameters containing cost-related parameters.
    simu_params : SimulationParameters
        An instance of SimulationParameters containing simulation-related parameters.
    tech_params : TechnoParameters
        An instance of TechnoParameters containing technology-related parameters.
    capacity_factors : CapacityFactors
        An instance of CapacityFactors containing capacity factor-related parameters.
    Methods
    -------
    merit_order(pct, conv_factor_c, cintensity_c, pgt, conv_factor_g, cintensity_g, ctax):
        Determines the merit order between coal and gas generation based on given parameters.
    cost_res(kpvt, kwt, pv_cf, wind_cf, load):
        Computes the cost and marginal cost of PV and wind generation.
    cost_conventional(kgt, kct, load, pgt, pct, ctax, qpvt, qwt):
        Computes the cost and marginal cost of gas, coal, and peak generation.
    cost_default(load, qpvt, qwt, qgt, qct, qpeakt):
        Computes the default cost and marginal cost when the load is not met by generation.
    cost(kwt, kgt, kct, kpvt, load, pv_cf, wind_cf, pgt, pct, ctax):
        Computes the total cost, carbon emissions, adequacy, and other metrics for the given parameters.
    cost_deterministic(kwt, kgt, kct, kpvt, dt, at, pv_cf, wind_cf, pct, pgt, ctax, evol_gas):
        Computes the total cost and other metrics for deterministic scenarios.
    cost_solar(kpvt, pv_cf, load):
        Computes the cost and marginal cost of PV generation.
    cost_wind(kpvt, kwt, pv_cf, wind_cf, load):
        Computes the cost and marginal cost of wind generation.
    cost_gas(kgt, load, pgt):
        Computes the cost and marginal cost of gas generation.
    cost_ccoal(kct, load, pct):
        Computes the cost and marginal cost of coal generation.
    cost_peak(kpeak, load, p_peak):
        Computes the cost and marginal cost of peak generation.
    total_cost(kpvt, kwt, kgt, kct, kpeak, pv_cf, wind_cf, load, pgt, pct, p_peak, ctax):
        Computes the total cost and marginal cost for all types of generation.
    
    Outputs costs:
    0: Total cost
    1: Carbon emissions
    2: Adequacy
    3: Load
    4: PV quantity
    5: Renewable quantity
    6: Gas quantity
    7: Coal quantity
    8: Peak quantity
    9: Number_default
    10 : PV cost
    11: Renewable cost
    12: Gas cost
    13: Coal cost
    14: Peak cost
    15: Default cost
    16: Marginal cost
    """

    def __init__(self):
        self.params = CostParameters()
        self.capacity_factors = CapacityFactor()

    def merit_order(self, pct, conv_factor_c, cintensity_c, pgt, conv_factor_g, cintensity_g, ctax):
        return self.params.b_c + pct * conv_factor_c + cintensity_c * ctax > self.params.b_g + pgt * conv_factor_g + cintensity_g * ctax

    ######## Global functions (optimised) ########
    
    def cost_res(self, kst, kwt, pv_cap, wind_cap, load):
        # Compute PV generation
        qpvt = np.minimum(load, kst * pv_cap)
        cost_pvt = self.params.a_s * kst /simu_parameters.h + self.params.b_s * qpvt
        mc_pvt = self.params.b_s * (qpvt > 0)

        # Compute Wind generation
        qrt = np.minimum(load - qpvt, kwt * wind_cap)
        cost_rt = self.params.a_w * kwt /simu_parameters.h + self.params.b_w * qrt
        mc_rt = self.params.b_w * (qrt > 0)

        return qpvt, cost_pvt, mc_pvt, qrt, cost_rt, mc_rt

    def cost_conventional(self, kgt, kct, load, pgt, pct, ctax, qpvt, qrt):
        # Compute merit order value (order between gas and coal)
        merit_order_val = self.merit_order(pct, self.params.conv_factor_c, self.params.cintensity_c, pgt,
                                           self.params.conv_factor_g, self.params.cintensity_g, ctax)
        residual_load = load - qpvt - qrt    
        # Conventional generation
        qgt = np.minimum(residual_load, kgt) * merit_order_val
        qct = np.minimum(residual_load - qgt, kct) * merit_order_val
        qct += np.minimum(residual_load, kct) * (np.invert(merit_order_val))
        qgt += np.minimum(residual_load - qct, kgt) * (np.invert(merit_order_val))
        
        # Compute costs and marginal costs for G
        cost_gt = self.params.a_g * kgt /simu_parameters.h + qgt * (
                    self.params.b_g + pgt * self.params.conv_factor_g + self.params.cintensity_g * ctax)
        mc_gt = (self.params.b_g + pgt * self.params.conv_factor_g + self.params.cintensity_g * ctax) * (qgt > 0)

        # Compute costs and marginal costs for C
        cost_ct = self.params.a_c * kct /simu_parameters.h + qct * (
                    self.params.b_c + pct * self.params.conv_factor_c + self.params.cintensity_c * ctax)
        mc_ct = (self.params.b_c + pct * self.params.conv_factor_c + self.params.cintensity_c * ctax) * (qct > 0)

        # Compute default costs
        qpeakt = np.minimum(residual_load - qgt - qct, tech_parameters.kpeak)
        cost_peak_t = self.params.a_peak * tech_parameters.kpeak / simu_parameters.h + qpeakt * (
                    self.params.b_peak + self.params.p_peak + self.params.cintensity_peak * ctax)
        mc_peak_t = (self.params.b_peak + self.params.p_peak + self.params.cintensity_peak * ctax) * (qpeakt > 0)

        default = residual_load - qgt - qct - qpeakt
        cost_default_t = cost_parameters.penalty_default * (default > 0) + default * cost_parameters.voll
        mc_default_t = default * cost_parameters.voll

        return qgt, cost_gt, mc_gt, qct, cost_ct, mc_ct, qpeakt, cost_peak_t, mc_peak_t, default, cost_default_t, mc_default_t

    def cost(self, kwt, kgt, kct, kst, load, pv_cap, wind_cap, pgt, pct, ctax):

        qpvt, cost_pvt, mc_pvt, qrt, cost_rt, mc_rt = self.cost_res(kst, kwt, pv_cap, wind_cap, load)
        qgt, cost_gt, mc_gt, qct, cost_ct, mc_ct, qPeakt, cost_peak_t, mc_peak_t, default, cost_default_t, mc_default_t = self.cost_conventional(kgt, kct, load, pgt, pct, ctax, qpvt, qrt)
        mc_t = np.maximum.reduce([mc_pvt, mc_rt, mc_gt, mc_ct, mc_peak_t, mc_peak_t, mc_default_t])
        nb_default = np.sum(default > 0)
 
        return (cost_pvt + cost_rt + cost_gt + cost_ct + cost_peak_t + cost_default_t,
                self.params.cintensity_g * qgt + self.params.cintensity_c * qct + self.params.cintensity_peak * qPeakt,
                default,
                load,
                qpvt,
                qrt,
                qgt,
                qct,
                qPeakt,
                nb_default,
                cost_pvt,
                cost_rt,
                cost_gt,
                cost_ct,
                cost_peak_t,
                cost_default_t,
                mc_t
                )

    def cost_naive(self, kwt, kgt, kct, kst, dt, at, pv_cap, wind_cap, pct, pgt, ctax, evol_gas):
        load = at + dt
        pgt = pgt * evol_gas

        qpvt, cost_pvt, mc_pvt, qrt, cost_rt, mc_rt = self.cost_res(kst, kwt, pv_cap, wind_cap, load)
        qgt, cost_gt, mc_gt, qct, cost_ct, mc_ct, qpeakt, cost_peak_t, mc_peak_t, default, cost_default_t, mc_default_t = self.cost_conventional(
            kgt, kct, load, pgt, pct, ctax, qpvt, qrt)
        print("res_cost", cost_rt)
        print("conventional_cost", cost_gt)
        mc_t = np.maximum.reduce([mc_pvt, mc_rt, mc_gt, mc_ct, mc_peak_t, mc_peak_t, mc_default_t])
        nb_default = np.sum(default > 0)

        return (
            cost_pvt
            + cost_rt
            + cost_gt
            + cost_ct
            + cost_peak_t
            + cost_default_t,
            self.params.cintensity_g * qgt + self.params.cintensity_c * qct + self.params.cintensity_peak * qpeakt,
            load - qpvt - qrt - qgt - qct - qpeakt,
            load,
            qpvt,
            qrt,
            qgt,
            qct,
            qpeakt,
            default,
            cost_rt,
            cost_gt,
            cost_ct,
            cost_peak_t,
            cost_default_t,
            mc_t,
            cost_pvt,
        )

    def cost_solar(self, kst, pv_cap, load):
        qpvt = np.minimum(load, kst * pv_cap)
        cost_pvt = self.params.a_s * kst / simu_parameters.h + self.params.b_s * qpvt
        mc_pvt = self.params.b_s * (qpvt > 0)
        return qpvt, cost_pvt, mc_pvt

    def cost_wind(self, kst, kwt, pv_cap, wind_cap, load):
        qpvt = np.minimum(load, kst * pv_cap)
        qrt = np.minimum(load - qpvt, kwt * wind_cap)
        cost_rt = self.params.a_w * kwt / simu_parameters.h + self.params.b_w * qrt
        mc_rt = self.params.b_w * (qrt > 0)
        return qrt, cost_rt, mc_rt
    
    def cost_gas(self, kgt, kct, load, pgt, pct, ctax, qpvt, qrt):
        merit_order_val = self.merit_order(pct, self.params.conv_factor_c, self.params.cintensity_c, pgt, self.params.conv_factor_g, self.params.cintensity_g, ctax)
        
        residual_load = load - qpvt - qrt
        qgt = np.minimum(residual_load, kgt) * merit_order_val
        qct = np.minimum(residual_load - qgt, kct) * merit_order_val
        qct += np.minimum(residual_load, kct) * (np.invert(merit_order_val))
        qgt += np.minimum(residual_load - qct, kgt) * (np.invert(merit_order_val))

        cost_gt = self.params.a_g * kgt / simu_parameters.h + qgt * (self.params.b_g + pgt * self.params.conv_factor_g + self.params.cintensity_g * ctax)
        mc_gt = (self.params.b_g + pgt * self.params.conv_factor_g + self.params.cintensity_g * ctax) * (qgt > 0)
        return qgt, cost_gt, mc_gt

    def cost_coal(self, kgt, kct, load, pgt, pct, ctax, qpvt, qrt):
        
        merit_order_val = self.merit_order(pct, self.params.conv_factor_c, self.params.cintensity_c, pgt, self.params.conv_factor_g, self.params.cintensity_g, ctax)
        residual_load = load - qpvt - qrt
        
        qgt = np.minimum(residual_load, kgt) * merit_order_val
        qct = np.minimum(residual_load - qgt, kct) * merit_order_val
        qct += np.minimum(residual_load, kct) * (np.invert(merit_order_val))
        qgt += np.minimum(residual_load - qct, kgt) * (np.invert(merit_order_val))

        cost_ct = self.params.a_c * kct /simu_parameters.h + qct * (self.params.b_c + pct * self.params.conv_factor_c + self.params.cintensity_c * ctax)
        mc_ct = (self.params.b_c + pct * self.params.conv_factor_c + self.params.cintensity_c * ctax) * (qct > 0)
        return qct, cost_ct, mc_ct

    def cost_default(self, load, ctax, qpvt, qrt, qgt, qct):

        qpeakt = np.minimum(load - qpvt - qrt - qgt - qct, tech_parameters.kpeak)
        cost_peak_t = self.params.a_peak * tech_parameters.kpeak / simu_parameters.h + qpeakt * (self.params.b_peak + self.params.p_peak + self.params.cintensity_peak * ctax)
        mc_peak_t = (self.params.b_peak + self.params.p_peak + self.params.cintensity_peak * ctax) * (qpeakt > 0)

        default = load - qpvt - qrt - qgt - qct - qpeakt
        cost_default_t = cost_parameters.penalty_default * (default > 0) + default * cost_parameters.voll
        mc_default_t = default * cost_parameters.voll
        
        return qpeakt, cost_peak_t, mc_peak_t, default, cost_default_t, mc_default_t


class InvestmentFunctions:
    def __init__(self):
        self.params = InvestmentParameters()

    def invest(self, r, s, g, t):
        invest_r = np.where(r >= 0, self.params.gamma_w_seq[t] * r, self.params.scrap_w[t] * r)
        invest_s = np.where(s >= 0, self.params.gamma_s_seq[t] * s, self.params.scrap_s[t] * s)
        invest_g = np.where(g >= 0, self.params.gamma_g_seq[t] * g, self.params.scrap_g[t] * g)
        return invest_r + invest_s + invest_g

