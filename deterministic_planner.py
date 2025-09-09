# %%
########################################## Deterministic planning ##########################################

# This Jupyter Notebook performs a deterministic planning simulation for energy investment and cost optimization.
# It includes the following steps:

# 1. Initialization of parameters.
# 2. Calculation of terminal value function using a deterministic approach with a reference scenario.
# 3. Execution of a backward algorithm to optimize investment decisions over time.
# 4. Determination of initial investment values.
# 5. Saving the optimal investment path and exporting the results.

# %%
from results_writing import save_results_to_csv
from simulation_parameters import *
from load import *
from capacity_factors import CapacityFactor
from cost_functions import IterativeFunctions, InvestmentFunctions
from gradient_boost import GradientBoostingModel
from constraints import Constraints

sns.set_style('darkgrid')
plt.rcParams["figure.dpi"] = 500
np.set_printoptions(suppress=True, precision=5)
seed = 42

iterative_functions = IterativeFunctions()
cost_parameters = CostParameters()
investment_parameters = InvestmentParameters()
capacity_factors = CapacityFactor()
simu_parameters = SimulationParameters()
gradient_parameters = GradientParameters()
tech_parameters = TechnoParameters()
gen_scenario = Scenario()
#d_reference = gen_scenario.average_scenario
d_reference = gen_scenario.realistic_scenario
investment_functions = InvestmentFunctions()
constraints = Constraints(simu_parameters.lambda_weight, simu_parameters.mu_weight, simu_parameters.kappa_weight, simu_parameters.nu_weight)
pct = cost_parameters.pc

time_start = time.time() 

print("Simulation name: " + simu_parameters.name)
print("Coal phase-out: " + simu_parameters.coal_phase_out)
print("Carbon tax: " + simu_parameters.carbon_tax)


# %%
############################# 0.Initialization: Terminal Value ##############################

value_func_deterministic = np.zeros((simu_parameters.t, tech_parameters.n_w, tech_parameters.n_s, tech_parameters.n_g)) 
next_value = 0

for t in tqdm.tqdm(range(simu_parameters.t-1, simu_parameters.t-2-simu_parameters.extension,-1)):
    print("Year " + str(t) + ": Beginning...")
    ctax = simu_parameters.cpath[t]
    #at = load_curve[t]
    kct = simu_parameters.kc[t]
    pct = cost_parameters.pc[t]
    f_evol = cost_parameters.fossil_evol[t]
    d = d_reference[t]
    
    if d == 7:
        print('Reference: Expected value')
        #load = at + d_load_mean_deterministic
        load  = d_load_mean_deterministic*(1 + simu_parameters.load_growth * t)
        wind_cap = capacity_factors.wind_cf_mean_deterministic
        solar_cap = capacity_factors.pv_cf_mean_deterministic
        pgt = cost_parameters.pg_mean_deterministic * f_evol
    else:
        if d == 8:
            print('Reference: Realistic')
            #load = at + d_load_mean
            load  = d_load_mean*(1 + simu_parameters.load_growth * t)
            wind_cap = capacity_factors.wind_cf_mean
            solar_cap = capacity_factors.pv_cf_mean
            pgt = cost_parameters.pg_mean * f_evol
        else:
            print('Reference: Historical year ' + str(2015+d))
            #load = at + d_load[d]
            load = d_load[d]*(1 + simu_parameters.load_growth * t)
            wind_cap = capacity_factors.cap_factor[d]
            solar_cap = capacity_factors.pv_cf[d]
            pgt = cost_parameters.pg[d] * f_evol
        
    for w, s, g in product(range(len(tech_parameters.kw)), range(len(tech_parameters.ks)), 
                           range(len(tech_parameters.kg))):
        cost_output = iterative_functions.cost(tech_parameters.kw[w], tech_parameters.kg[g], 
                                               kct, tech_parameters.ks[s], load, solar_cap, wind_cap, pgt, pct, ctax)
        cost = cost_output[0].sum()
        carbon_realised = cost_output[1].sum()


        mu_constraint = constraints.compute_mu_constraint(t, tech_parameters.kg[g])
        kappa_constraint = constraints.compute_kappa_constraint(t, tech_parameters.kw[w])
        nu_constraint = constraints.compute_nu_constraint(t, tech_parameters.ks[s])
        lambda_constraint = constraints.compute_lambda_constraint(t, carbon_realised)

        value_func_deterministic[t, w, s, g] = cost + lambda_constraint + mu_constraint + kappa_constraint + nu_constraint

    value_func_deterministic[t] += simu_parameters.beta*next_value
    next_value = value_func_deterministic[t]
    print("Year " + str(t) + ": completed")

value_t = value_func_deterministic[simu_parameters.t-simu_parameters.extension-1]
finalvalue = GradientBoostingModel(tech_parameters.kw, tech_parameters.ks, tech_parameters.kg)
finalvalue.train_data = value_t
mse_in, mse_out, X_train, X_test, y_train, y_test, X_mean, X_std, y_mean, y_std = finalvalue.train_deterministic()

finalvalue.save_model(simu_parameters.path_functions + "\\value_func_deterministic" + 
                      str(simu_parameters.t-simu_parameters.extension) + ".pkl")

time_elapsed = (time.time() - time_start)
print(time_elapsed / 60, "min")

# %%
############################## 1.Backward algorithm ##############################

next_value_func = GradientBoostingModel(tech_parameters.kw, tech_parameters.ks, tech_parameters.kg)
next_value_func.train_data = value_t
model, scaler_X, scaler_y, train_data_mean, train_data_std = next_value_func.load_model(simu_parameters.path_functions + "\\value_func_deterministic" + str(simu_parameters.t-simu_parameters.extension) + ".pkl")
next_value_func.model = model
next_value_func.scaler_X = scaler_X
next_value_func.scaler_y = scaler_y

for t in tqdm.tqdm(range(simu_parameters.t-simu_parameters.extension-2, -1, -1)):

    print("Year " + str(t) + ": Beginning...")
    ctax = simu_parameters.cpath[t]
    #at = load_curve[t]
    kct = simu_parameters.kc[t]
    pct = cost_parameters.pc[t]
    f_evol = cost_parameters.fossil_evol[t]
    d = d_reference[t]
    if d == 7:
        print('Reference: Expected value')
        load  = d_load_mean_deterministic*(1 + simu_parameters.load_growth * t)
        wind_cap = capacity_factors.wind_cf_mean_deterministic
        solar_cap = capacity_factors.pv_cf_mean_deterministic
        pgt = cost_parameters.pg_mean_deterministic * f_evol
    else:
        if d == 8:
            print('Reference: Realistic')
            load  = d_load_mean*(1 + simu_parameters.load_growth * t)
            wind_cap = capacity_factors.wind_cf_mean
            solar_cap = capacity_factors.pv_cf_mean
            pgt = cost_parameters.pg_mean * f_evol
        else:
            print('Reference: Historical year ' + str(2015+d))
            #load = at + d_load[d]
            load  = d_load[d]*(1 + simu_parameters.load_growth * t)
            wind_cap = capacity_factors.cap_factor[d]
            solar_cap = capacity_factors.pv_cf[d]
            pgt = cost_parameters.pg[d] * f_evol
        
    for w, s, g in product(range(len(tech_parameters.kw)), range(len(tech_parameters.ks)), range(len(tech_parameters.kg))):
        
        X, Y, Z = np.meshgrid(np.linspace(tech_parameters.kwlow - tech_parameters.kw[w], tech_parameters.kwbound - tech_parameters.kw[w], 
                                          tech_parameters.n_w),
                              np.linspace(tech_parameters.kslow - tech_parameters.ks[s], tech_parameters.ksbound - tech_parameters.ks[s], 
                                          tech_parameters.n_s),
                              np.linspace(tech_parameters.kglow - tech_parameters.kg[g], tech_parameters.kgbound - tech_parameters.kg[g], 
                                          tech_parameters.n_g), indexing='ij')

        grid = investment_functions.invest(X, Y, Z, t) + (simu_parameters.beta)*(value_func_deterministic[t+1])
        grid_minimum = np.unravel_index(np.argmin(grid), grid.shape)
        cost_output = iterative_functions.cost(tech_parameters.kw[w], tech_parameters.kg[g], kct, tech_parameters.ks[s], 
                                               load, solar_cap, wind_cap, pgt, pct, ctax)
        cost = cost_output[0].sum()
        carbon_realised = cost_output[1].sum()

        mu_constraint = constraints.compute_mu_constraint(t, tech_parameters.kg[g])
        kappa_constraint = constraints.compute_kappa_constraint(t, tech_parameters.kw[w])
        nu_constraint = constraints.compute_nu_constraint(t, tech_parameters.ks[s])
        lambda_constraint = constraints.compute_lambda_constraint(t, carbon_realised)

        next_value = next_value_func.minimize_quantity(tech_parameters.kw[w], tech_parameters.ks[s], tech_parameters.kg[g], 
                                                       t, grid_minimum, grid[grid_minimum])[3]
        value_func_deterministic[t, w, s, g] = cost + lambda_constraint + mu_constraint + kappa_constraint + nu_constraint + next_value

    value_t = value_func_deterministic[t]
    model = GradientBoostingModel(tech_parameters.kw, tech_parameters.ks, tech_parameters.kg)
    model.train_data = value_t
    mse_in, mse_out, X_train, X_test, y_train, y_test, X_mean, X_std, y_mean, y_std = model.train_deterministic()

    model.save_model(simu_parameters.path_functions + "\\value_func_deterministic" + str(t) + ".pkl")

    next_value_func = GradientBoostingModel(tech_parameters.kw, tech_parameters.ks, tech_parameters.kg)
    next_value_func.train_data = value_t
    model, scaler_X, scaler_y, train_data_mean, train_data_std = next_value_func.load_model(simu_parameters.path_functions 
                                                                                            + "\\value_func_deterministic" + str(t) + ".pkl")
    next_value_func.model = model
    next_value_func.scaler_X = scaler_X
    next_value_func.scaler_y = scaler_y

print("Mean Squared Error in sample:", mse_in)
print("Mean Squared Error out sample:", mse_out)


# %%
############################## 2. Initial Investment ##############################

X0, Y0, Z0 = np.meshgrid(np.linspace(tech_parameters.kwlow - tech_parameters.kw0, tech_parameters.kwbound - tech_parameters.kw0, tech_parameters.n_w),
                              np.linspace(tech_parameters.kslow - tech_parameters.ks0, tech_parameters.ksbound - tech_parameters.ks0, tech_parameters.n_s),
                              np.linspace(tech_parameters.kglow - tech_parameters.kg0, tech_parameters.kgbound - tech_parameters.kg0, tech_parameters.n_g), indexing='ij')

invest_initial = investment_functions.invest(X0, Y0, Z0, 0)

value_func_deterministic[0] = value_func_deterministic[0] + invest_initial

model = GradientBoostingModel(tech_parameters.kw, tech_parameters.ks, tech_parameters.kg)
model.train_data = value_func_deterministic[0]
mse_in, mse_out, X_train, X_test, y_train, y_test, X_mean, X_std, y_mean, y_std = model.train_deterministic(sample_size=gradient_parameters.n_sample)

time_elapsed = (time.time() - time_start)
print(time_elapsed/60, "min")

# %%
############################## 3. Save Optimal Path and export of the results ##############################

deterministic_optimal_trajectory = np.zeros((simu_parameters.t-simu_parameters.extension, 3))
kw_t, ks_t, kg_t = tech_parameters.kw0, tech_parameters.ks0, tech_parameters.kg0
final_model = GradientBoostingModel(kw_t, ks_t, kg_t)
deterministic_optimal_trajectory[0] = [tech_parameters.kw0, tech_parameters.ks0, tech_parameters.kg0]

for t in range(0, simu_parameters.t-simu_parameters.extension-1):
    model, scaler_X, scaler_y, train_data_mean, train_data_std = final_model.load_model(simu_parameters.path_functions + "\\value_func_deterministic" + str(t) + ".pkl")
    final_model.model = model
    final_model.scaler_X = scaler_X
    final_model.scaler_y = scaler_y
    X, Y, Z = np.meshgrid(np.linspace(tech_parameters.kwlow - kw_t, tech_parameters.kwbound - kw_t, tech_parameters.n_w),
                                  np.linspace(tech_parameters.kslow - ks_t, tech_parameters.ksbound - ks_t, tech_parameters.n_s),
                                  np.linspace(tech_parameters.kglow - kg_t, tech_parameters.kgbound - kg_t, tech_parameters.n_g), indexing='ij')

    grid = investment_functions.invest(X, Y, Z, t) + simu_parameters.beta*(value_func_deterministic[t+1])
    grid_minimum = np.unravel_index(np.argmin(grid), grid.shape)
    kw_t, ks_t, kg_t, value = final_model.minimize_quantity(kw_t, ks_t, kg_t, t, grid_minimum, grid[grid_minimum])
    deterministic_optimal_trajectory[t+1] = [kw_t, ks_t, kg_t]

deterministic_optimal_df = pd.DataFrame(deterministic_optimal_trajectory, columns=['KW', 'KPV', 'KG'])
deterministic_optimal_df.to_csv(os.path.join(simu_parameters.path_deterministic, 'deterministic_optimal_trajectory.csv'), index=False)

print("Optimal deterministic trajectory", deterministic_optimal_trajectory)
save_results_to_csv('deterministic', deterministic_optimal_trajectory)

time_elapsed = (time.time() - time_start)
print(time_elapsed/60, "min")


