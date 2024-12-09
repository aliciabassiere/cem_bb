import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from simulation_parameters import *
from load import *
from cost_functions import IterativeFunctions
from constraints import Constraints
from capacity_factors import CapacityFactor

sns.set_style('darkgrid')
plt.rcParams["figure.dpi"] = 500
np.set_printoptions(suppress=True, precision=5) # threshold=np.inf
seed = 42


iterative_functions = IterativeFunctions()
cost_parameters = CostParameters()
investment_parameters = InvestmentParameters()
simu_parameters = SimulationParameters()
gen_scenario = Scenario()
constraints = Constraints(simu_parameters.lambda_weight, simu_parameters.mu_weight, simu_parameters.kappa_weight, simu_parameters.nu_weight)
capacity_factors = CapacityFactor()
pct = cost_parameters.pc


def save_results_to_csv(name, optimal_trajectory, number_simulations=simu_parameters.n_simu, column_names=None, offset=0):
    # Check the shape of optimal_trajectory
    if optimal_trajectory.shape == (simu_parameters.t-simu_parameters.extension, 3):
        optimal_trajectories = np.tile(optimal_trajectory, (number_simulations, 1, 1))
    elif optimal_trajectory.shape == (number_simulations, simu_parameters.t-simu_parameters.extension, 3):
        optimal_trajectories = optimal_trajectory
    else:
        raise ValueError("Invalid shape for optimal_trajectory")

    if column_names is None:
        column_names = range(number_simulations)

    costs_df = pd.DataFrame(index=range(simu_parameters.t))
    costs_carbon_free_df = pd.DataFrame(index=range(simu_parameters.t))
    carbon_emissions_df = pd.DataFrame(index=range(simu_parameters.t))
    adequacy_df = pd.DataFrame(index=range(simu_parameters.t))
    quantity_pv_df = pd.DataFrame(index=range(simu_parameters.t))
    quantity_wind_df = pd.DataFrame(index=range(simu_parameters.t))
    quantity_gas_df = pd.DataFrame(index=range(simu_parameters.t))
    quantity_coal_df = pd.DataFrame(index=range(simu_parameters.t))
    quantity_peak_df = pd.DataFrame(index=range(simu_parameters.t))
    defaults_df = pd.DataFrame(index=range(simu_parameters.t))
    cost_pv_df = pd.DataFrame(index=range(simu_parameters.t))
    cost_wind_df = pd.DataFrame(index=range(simu_parameters.t))
    cost_gas_df = pd.DataFrame(index=range(simu_parameters.t))
    cost_coal_df = pd.DataFrame(index=range(simu_parameters.t))
    cost_peak_df = pd.DataFrame(index=range(simu_parameters.t))
    cost_default_df = pd.DataFrame(index=range(simu_parameters.t))
    marginal_cost_df = pd.DataFrame(index=range(simu_parameters.t*simu_parameters.h))
    load_df = pd.DataFrame(index=range(simu_parameters.t))
    inv_cost_df = pd.DataFrame(index=range(simu_parameters.t))
    mu_constraint_df = pd.DataFrame(index=range(simu_parameters.t))
    nu_constraint_df = pd.DataFrame(index=range(simu_parameters.t))
    lambda_constraint_df = pd.DataFrame(index=range(simu_parameters.t))
    kappa_constraint_df = pd.DataFrame(index=range(simu_parameters.t))
    npv_df = pd.DataFrame(index=range(1))
    npv_constrained_df = pd.DataFrame(index=range(1))
    npv_carbon_free_df = pd.DataFrame(index=range(1))
    inv_cost_discounted_df = pd.DataFrame(index=range(1))
    cost_default_discounted_df = pd.DataFrame(index=range(1))
    cost_gas_discounted_df = pd.DataFrame(index=range(1))
    cost_coal_discounted_df = pd.DataFrame(index=range(1))
    cost_wind_discounted_df = pd.DataFrame(index=range(1))
    cost_pv_discounted_df = pd.DataFrame(index=range(1))
    cost_peak_discounted_df = pd.DataFrame(index=range(1))
    cost_pv_discounted_df = pd.DataFrame(index=range(1))
    lambda_discounted_df = pd.DataFrame(index=range(1))
    kappa_discounted_df = pd.DataFrame(index=range(1))
    nu_discounted_df = pd.DataFrame(index=range(1))
    
    invest_cost = np.array([investment_parameters.gamma_w_seq[:simu_parameters.t],
                            investment_parameters.gamma_s_seq[:simu_parameters.t],
                            investment_parameters.gamma_g_seq[:simu_parameters.t]]).T

    path_results = os.path.join(simu_parameters.path2, str(name))
    power_vector = np.power(simu_parameters.beta, np.arange(simu_parameters.t))

    ## Hourly df
    
    costs_hourly_df = pd.DataFrame(index=range(simu_parameters.t*simu_parameters.h))
    carbon_emissions_hourly_df = pd.DataFrame(index=range(simu_parameters.t*simu_parameters.h))
    adequacy_hourly_df = pd.DataFrame(index=range(simu_parameters.t*simu_parameters.h))
    load_hourly_df = pd.DataFrame(index=range(simu_parameters.t*simu_parameters.h))

    quantity_pv_hourly_df = pd.DataFrame(index=range(simu_parameters.t*simu_parameters.h))
    quantity_wind_hourly_df = pd.DataFrame(index=range(simu_parameters.t*simu_parameters.h))
    quantity_gas_hourly_df = pd.DataFrame(index=range(simu_parameters.t*simu_parameters.h))
    quantity_coal_hourly_df = pd.DataFrame(index=range(simu_parameters.t*simu_parameters.h))
    quantity_peak_hourly_df = pd.DataFrame(index=range(simu_parameters.t*simu_parameters.h))
    # cost_pv_hourly_df = pd.DataFrame(index=range(simu_parameters.T*simu_parameters.H))
    # cost_wind_hourly_df = pd.DataFrame(index=range(simu_parameters.T*simu_parameters.H))
    # cost_gas_hourly_df = pd.DataFrame(index=range(simu_parameters.T*simu_parameters.H))
    # cost_coal_hourly_df = pd.DataFrame(index=range(simu_parameters.T*simu_parameters.H))
    # cost_peak_hourly_df = pd.DataFrame(index=range(simu_parameters.T*simu_parameters.H))
    # marginal_cost_hourly_df = pd.DataFrame(index=range(simu_parameters.T*simu_parameters.H))

    for n in range(number_simulations):
        print('')
        print("Writing results of Simulation " + str(n+offset))
        costs = []
        costs_carbon_free = []
        carbon_emissions = []
        adequacies = []
        quantities_pv = []
        quantities_wind = []
        quantities_gas = []
        quantities_coal = []
        quantities_peak = []
        defaults = []
        costs_pv = []
        costs_wind = []
        costs_gas = []
        costs_coal = []
        costs_peak = []
        costs_default = []
        marginal_costs = []
        loads = []
        costs_hour = []
        carbon_emissions_hour = []
        adequacies_hour = []
        quantities_pv_hour = []
        quantities_wind_hour = []
        quantities_gas_hour = []
        quantities_coal_hour = []
        quantities_peak_hour = []
        costs_pv_hour = []
        costs_wind_hour = []
        costs_gas_hour = []
        costs_coal_hour = []
        costs_peak_hour = []
        loads_hour = []
        mu_constraints = []
        nu_constraints = []
        lambda_constraints = []
        kappa_constraints = []
        nu_constraints = []
        adequacies_hourly = []
        trajectory = np.vstack((optimal_trajectories[n], np.tile(optimal_trajectories[n, simu_parameters.t-simu_parameters.extension-1], (simu_parameters.extension, 1))))
        D_scenario = gen_scenario.scenarios[n+offset]
        
        # Iteration
        
        for t in range(0, simu_parameters.t):
            d = D_scenario[t]
            #print("Time " + str(t))
            #print("scenario " + str(d))
            ctax = simu_parameters.cpath[t]
            At = load_curve[t]
            KCt = simu_parameters.kc[t]
            pct = cost_parameters.pc[t]
            f_evol = cost_parameters.fossil_evol[t]
            load = At + d_load[d]
            epsval = capacity_factors.cap_factor[d]
            pv_cap = capacity_factors.pv_cf[d]
            pgt = cost_parameters.pg[d] * f_evol
            KR_t, KPV_t, KG_t = trajectory[t]
 
            # Calculate the outputs
            cost_output = iterative_functions.cost(KR_t, KG_t, KCt, KPV_t, load, pv_cap, epsval, pgt, pct, ctax)

            cost = cost_output[0]
            costs.append(cost.sum())
            costs_hour.extend(cost)
            
            carbon_emission = cost_output[1]
            carbon_emissions.append(carbon_emission.sum())
            carbon_emissions_hour.extend(carbon_emission)
            adequacy = cost_output[2]
            adequacies_hourly.append(adequacy)
            adequacies.append(adequacy.sum())
            adequacies_hour.extend(adequacy)
            
            load = cost_output[3]
            loads.append(load.sum())
            loads_hour.extend(load)
                
            q_pv = cost_output[4]
            quantities_pv.append(q_pv.sum())
            quantities_pv_hour.extend(q_pv)
                
            q_wind = cost_output[5]
            quantities_wind.append(q_wind.sum())
            quantities_wind_hour.extend(q_wind)
            
            q_gas = cost_output[6]
            quantities_gas.append(q_gas.sum())
            quantities_gas_hour.extend(q_gas)
            
            q_coal = cost_output[7]
            quantities_coal.append(q_coal.sum())
            quantities_coal_hour.extend(q_coal)
            
            q_peak = cost_output[8]
            quantities_peak.append(q_peak.sum())
            quantities_peak_hour.extend(q_peak)
            
            default = cost_output[9]
            defaults.append(default.sum())
            
            cost_pv = cost_output[10]
            costs_pv.append(cost_pv.sum())
            costs_pv_hour.extend(cost_pv)
            
            cost_wind = cost_output[11]
            costs_wind.append(cost_wind.sum())
            costs_wind_hour.extend(cost_wind)
            
            cost_gas = cost_output[12]
            costs_gas.append(cost_gas.sum())
            costs_gas_hour.extend(cost_gas)
            
            cost_coal = cost_output[13]
            costs_coal.append(cost_coal.sum())
            costs_coal_hour.extend(cost_coal)
            
            cost_peak = cost_output[14]
            costs_peak.append(cost_peak.sum())
            costs_peak_hour.extend(cost_peak)
            
            cost_default = cost_output[15]
            costs_default.append(cost_default.sum())
            
            marginal_cost = list(cost_output[16])
            marginal_costs.extend(marginal_cost)

            if (t>0) & (t < simu_parameters.t-simu_parameters.extension):
                mu_constraint = constraints.compute_mu_constraint(t-1, KG_t)
                mu_constraints.append(mu_constraint)
                kappa_constraint = constraints.compute_kappa_constraint(t-1, KR_t)
                kappa_constraints.append(kappa_constraint)
                nu_constraint = constraints.compute_nu_constraint(t-1, KPV_t)
                nu_constraints.append(nu_constraint)
            else:
                mu_constraints.append(0)
                nu_constraints.append(0)
                kappa_constraints.append(0)

            lambda_constraint = constraints.compute_lambda_constraint(t, carbon_emission.sum())
            lambda_constraints.append(lambda_constraint)

            cost_carbon_free = iterative_functions.cost(KR_t, KG_t, KCt, KPV_t, load, pv_cap, epsval, pgt, pct, ctax=0)[0]
            costs_carbon_free.append(cost_carbon_free.sum())
            
        #print("cost: ", costs)
        #print("lambda: ", lambda_constraints)
       
        investment_realised = trajectory[1:] - trajectory[:-1].copy()
        investment_realised = np.append(investment_realised, np.array([0,0,0])).reshape(simu_parameters.t, 3)
        inv_cost = list(np.sum(invest_cost*investment_realised, axis=1))
        
        npv = np.array(power_vector*(np.array(inv_cost) + np.array(costs))).sum()
        npv_constrained = np.array(power_vector*(np.array(inv_cost) + np.array(costs) + np.array(lambda_constraints))).sum()
        npv_carbon_free = np.array(power_vector*(np.array(inv_cost) + np.array(costs_carbon_free))).sum()
                           
        inv_cost_discounted = np.array(power_vector*(np.array(inv_cost))).sum()
        cost_default_discounted = np.array(power_vector*(np.array(costs_default))).sum()
        cost_gas_discounted = np.array(power_vector*(np.array(costs_gas))).sum()
        cost_coal_discounted = np.array(power_vector*(np.array(costs_coal))).sum()
        cost_wind_discounted = np.array(power_vector*(np.array(costs_wind))).sum()
        cost_pv_discounted = np.array(power_vector*(np.array(costs_pv))).sum()
        cost_peak_discounted = np.array(power_vector*(np.array(costs_peak))).sum()
        lambda_discounted = np.array(power_vector*(np.array(lambda_constraints))).sum()
        kappa_discounted = np.array(power_vector*(np.array(kappa_constraints))).sum()
        nu_discounted = np.array(power_vector*(np.array(nu_constraints))).sum()
        
        costs_df[n] = costs
        carbon_emissions_df[n] = carbon_emissions
        adequacy_df[n] = adequacies
        load_df[n] = loads
        quantity_pv_df[n] = quantities_pv
        quantity_wind_df[n] = quantities_wind
        quantity_gas_df[n] = quantities_gas
        quantity_coal_df[n] = quantities_coal
        quantity_peak_df[n] = quantities_peak
        defaults_df[n] = defaults
        cost_pv_df[n] = costs_pv
        cost_wind_df[n] = costs_wind
        cost_gas_df[n] = costs_gas
        cost_coal_df[n] = costs_coal
        cost_peak_df[n] = costs_peak
        cost_default_df[n] = costs_default
        
        marginal_cost_df[n] = marginal_costs
        inv_cost_df[n] = inv_cost
        mu_constraint_df[n] = mu_constraints
        nu_constraint_df[n] = nu_constraints
        kappa_constraint_df[n] = kappa_constraints
        lambda_constraint_df[n] = lambda_constraints
        
        npv_constrained_df[n] = npv_constrained        
        npv_df[n] = npv
        npv_carbon_free_df[n] = npv_carbon_free
        inv_cost_discounted_df[n] = inv_cost_discounted
        cost_gas_discounted_df[n] = cost_gas_discounted
        cost_coal_discounted_df[n] = cost_coal_discounted
        cost_wind_discounted_df[n] = cost_wind_discounted
        cost_pv_discounted_df[n] = cost_pv_discounted
        cost_peak_discounted_df[n] = cost_peak_discounted
        cost_default_discounted_df[n] = cost_default_discounted
        lambda_discounted_df[n] = lambda_discounted
        kappa_discounted_df[n] = kappa_discounted
        nu_discounted_df[n] = nu_discounted
        
        costs_hourly_df[n] = costs_hour
        carbon_emissions_hourly_df[n] = carbon_emissions_hour
        adequacy_hourly_df[n] = adequacies_hour
        load_hourly_df[n] = loads_hour
        quantity_pv_hourly_df[n] = quantities_pv_hour
        quantity_wind_hourly_df[n] = quantities_wind_hour
        quantity_gas_hourly_df[n] = quantities_gas_hour
        quantity_coal_hourly_df[n] = quantities_coal_hour
        quantity_peak_hourly_df[n] = quantities_peak_hour
        # cost_pv_hourly_df[n] = costs_pv_hour
        # cost_wind_hourly_df[n] = costs_wind_hour
        # cost_gas_hourly_df[n] = costs_gas_hour
        # cost_coal_hourly_df[n] = costs_coal_hour
        # cost_peak_hourly_df[n] = costs_peak_hour

    
    # Save DataFrames as CSV files
    
    costs_df.to_csv(os.path.join(path_results, name + "_costs.csv"), index=False)
    carbon_emissions_df.to_csv(os.path.join(path_results, name + "_carbon_emissions.csv"), index=False)
    adequacy_df.to_csv(os.path.join(path_results, name + "_adequacy.csv"), index=False)
    adequacy_hourly_df.to_csv(os.path.join(path_results, name + "_adequacy_hourly.csv"), index=False)
    quantity_pv_df.to_csv(os.path.join(path_results, name + "_quantity_pv.csv"), index=False)
    quantity_wind_df.to_csv(os.path.join(path_results, name + "_quantity_wind.csv"), index=False)
    quantity_gas_df.to_csv(os.path.join(path_results, name + "_quantity_gas.csv"), index=False)
    quantity_coal_df.to_csv(os.path.join(path_results, name + "_quantity_coal.csv"), index=False)
    quantity_peak_df.to_csv(os.path.join(path_results, name + "_quantity_peak.csv"), index=False)
    quantity_pv_hourly_df.to_csv(os.path.join(path_results, name + "_quantity_pv_hourly.csv"), index=False)
    quantity_wind_hourly_df.to_csv(os.path.join(path_results, name + "_quantity_wind_hourly.csv"), index=False)
    quantity_gas_hourly_df.to_csv(os.path.join(path_results, name + "_quantity_gas_hourly.csv"), index=False)
    quantity_coal_hourly_df.to_csv(os.path.join(path_results, name + "_quantity_coal_hourly.csv"), index=False)
    quantity_peak_hourly_df.to_csv(os.path.join(path_results, name + "_quantity_peak_hourly.csv"), index=False)
    defaults_df.to_csv(os.path.join(path_results, name + "_defaults.csv"), index=False)
    cost_pv_df.to_csv(os.path.join(path_results, name + "_cost_pv.csv"), index=False)
    cost_wind_df.to_csv(os.path.join(path_results, name + "_cost_wind.csv"), index=False)
    cost_gas_df.to_csv(os.path.join(path_results, name + "_cost_gas.csv"), index=False)
    cost_coal_df.to_csv(os.path.join(path_results, name + "_cost_coal.csv"), index=False)
    cost_peak_df.to_csv(os.path.join(path_results, name + "_cost_peak.csv"), index=False)
    cost_default_df.to_csv(os.path.join(path_results, name + "_cost_default.csv"), index=False)
    marginal_cost_df.to_csv(os.path.join(path_results, name + "_marginal_cost.csv"), index=False)
    load_df.to_csv(os.path.join(path_results, name + "_load.csv"), index=False)
    load_hourly_df.to_csv(os.path.join(path_results, name + "_load_hourly.csv"), index=False)
    inv_cost_df.to_csv(os.path.join(path_results, name + "_investments_cost.csv"), index=False)
    mu_constraint_df.to_csv(os.path.join(path_results, name + "_wind_constraint.csv"), index=False)
    nu_constraint_df.to_csv(os.path.join(path_results, name + "_solar_constraint.csv"), index=False)
    kappa_constraint_df.to_csv(os.path.join(path_results, name + "_gas_constraint.csv"), index=False)
    lambda_constraint_df.to_csv(os.path.join(path_results, name + "_carbons_constraint.csv"), index=False)
    npv_df.to_csv(os.path.join(path_results, name + "_npv.csv"), index=False)
    npv_carbon_free_df.to_csv(os.path.join(path_results, name + "_npv_carbon_free.csv"), index=False)
    npv_constrained_df.to_csv(os.path.join(path_results, name + "_npv_constrained.csv"), index=False)
    lambda_discounted_df.to_csv(os.path.join(path_results, name + "_lambda_discounted.csv"), index=False)
    kappa_discounted_df.to_csv(os.path.join(path_results, name + "_kappa_discounted.csv"), index=False)
    nu_discounted_df.to_csv(os.path.join(path_results, name + "_nu_discounted.csv"), index=False)
    cost_pv_discounted_df.to_csv(os.path.join(path_results, name + "_cost_pv_discounted.csv"), index=False)
    cost_wind_discounted_df.to_csv(os.path.join(path_results, name + "_cost_wind_discounted.csv"), index=False)
    cost_gas_discounted_df.to_csv(os.path.join(path_results, name + "_cost_gas_discounted.csv"), index=False)
    cost_coal_discounted_df.to_csv(os.path.join(path_results, name + "_cost_coal_discounted.csv"), index=False)
    cost_peak_discounted_df.to_csv(os.path.join(path_results, name + "_cost_peak_discounted.csv"), index=False)
    cost_default_discounted_df.to_csv(os.path.join(path_results, name + "_cost_default_discounted.csv"), index=False)
    inv_cost_discounted_df.to_csv(os.path.join(path_results, name + "_inv_cost_discounted.csv"), index=False) 

