# %%
########################################## Perfect Foresight planning ##########################################

# This Jupyter Notebook performs a deterministic planning simulation for each scenario of the simulation on energy investment and cost optimization.
# It includes the following steps:

# 1. Initialization of parameters, indication of the realised scenario.
# 2. Calculation of terminal value function using a deterministic approach.
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
d_reference = gen_scenario.average_scenario
investment_functions = InvestmentFunctions()
constraints = Constraints(simu_parameters.lambda_weight, simu_parameters.mu_weight, simu_parameters.kappa_weight, simu_parameters.nu_weight)
pct = cost_parameters.pc

time_start = time.time() 

print("Simulation name: " + simu_parameters.name)
print("Coal phase-out: " + simu_parameters.coal_phase_out)
print("Carbon tax: " + simu_parameters.carbon_tax)

time_start = time.time()

# %%
############################## 0.Initialization: Terminal Value ##############################

value_func_pf = np.zeros((simu_parameters.n_simu, simu_parameters.t, tech_parameters.n_w, tech_parameters.n_s, 
                          tech_parameters.n_g))
value_terminal = np.zeros((simu_parameters.n_simu, tech_parameters.n_w, tech_parameters.n_s, tech_parameters.n_g))

# Adjust for simulating a sample of the scenarios
low_bound = int(0)
high_bound = int(simu_parameters.n_simu)

for n in tqdm.tqdm(range(low_bound, high_bound)):
    print("Simulation: Terminal Value " + str(n) + ": Beginning...")
    d_scenario = gen_scenario.scenarios[n]
    next_value = 0

    for t in tqdm.tqdm(range(simu_parameters.t-1, simu_parameters.t-2-simu_parameters.extension, -1)):
        ctax = simu_parameters.cpath[t]
        #at = load_curve[t]
        pct = cost_parameters.pc[t]
        kct = simu_parameters.kc[t]
        f_evol = cost_parameters.fossil_evol[t]
        d = d_scenario[t]
        #load = at + d_load[d]
        load = d_load[d]*(1 + simu_parameters.load_growth * t)
        epsval = capacity_factors.cap_factor[d]
        pv_cap = capacity_factors.pv_cf[d]
        pgt = cost_parameters.pg[d] * f_evol
        
        for w, s, g in product(range(len(tech_parameters.kw)), range(len(tech_parameters.ks)), 
                                         range(len(tech_parameters.kg))):
            cost_output = iterative_functions.cost(tech_parameters.kw[w], tech_parameters.kg[g], kct, 
                                                   tech_parameters.ks[s], load, pv_cap, epsval, pgt, pct, ctax)
            cost = cost_output[0].sum()
            carbon_realised = cost_output[1].sum()

            mu_constraint = constraints.compute_mu_constraint(t, tech_parameters.kg[g])
            kappa_constraint = constraints.compute_kappa_constraint(t, tech_parameters.kw[w])
            nu_constraint = constraints.compute_nu_constraint(t, tech_parameters.ks[s])
            lambda_constraint = constraints.compute_lambda_constraint(t, carbon_realised)

            value_func_pf[n, t, w, s, g] = cost + lambda_constraint + mu_constraint + kappa_constraint + nu_constraint

        value_func_pf[n, t] += simu_parameters.beta*next_value
        next_value = value_func_pf[n, t]

    # Gradient Boost approximation
    value_terminal[n] = value_func_pf[n, simu_parameters.t-simu_parameters.extension-1]

    finalvalue = GradientBoostingModel(tech_parameters.kw, tech_parameters.ks, tech_parameters.kg, d)
    finalvalue.train_data = value_terminal[n]
    mse_in, mse_out, X_train, X_test, y_train, y_test, X_mean, X_std, y_mean, y_std = finalvalue.train_deterministic()
    finalvalue.save_model(simu_parameters.path_functions + "\\value_func_pf_numero_" + str(n) + "_time_" 
                          + str(simu_parameters.t-simu_parameters.extension) + ".pkl")

time_elapsed = (time.time() - time_start)
print(time_elapsed/60, "min")

# %%
############################## 1.Backward algorithm ##############################

perfect_foresight_optimal_trajectory = np.zeros((simu_parameters.n_simu, simu_parameters.t-simu_parameters.extension, 3))

for n in tqdm.tqdm(range(low_bound, high_bound)):
    
    print("Simulation " + str(n))

    d_scenario = gen_scenario.scenarios[n]
    next_value_func_pf = GradientBoostingModel(tech_parameters.kw, tech_parameters.ks, tech_parameters.kg)
    next_value_func_pf.train_data = value_terminal[n]
    next_value_func_pf.model, next_value_func_pf.scaler_X, next_value_func_pf.scaler_y, next_value_func_pf.train_data_mean, next_value_func_pf.train_data_std = next_value_func_pf.load_model(simu_parameters.path_functions + "\\value_func_pf_numero_" + str(n) + "_time_" + str(simu_parameters.t-simu_parameters.extension) + ".pkl")

    for t in tqdm.tqdm(range(simu_parameters.t-simu_parameters.extension-2, -1, -1)):
        ctax = simu_parameters.cpath[t]
        #at = load_curve[t]
        pct = cost_parameters.pc[t]
        kct = simu_parameters.kc[t]
        f_evol = cost_parameters.fossil_evol[t]
        d = d_scenario[t]
        #load = at + d_load[d]
        load = d_load[d]*(1 + simu_parameters.load_growth * t)
        epsval = capacity_factors.cap_factor[d]
        pv_cap = capacity_factors.pv_cf[d]
        pgt = cost_parameters.pg[d] * f_evol
        for w, s, g in product(range(len(tech_parameters.kw)), range(len(tech_parameters.ks)), 
                               range(len(tech_parameters.kg))):
            X, Y, Z = np.meshgrid(np.linspace(tech_parameters.kwlow - tech_parameters.kw[w], 
                                              tech_parameters.kwbound - tech_parameters.kw[w], tech_parameters.n_w),
                              np.linspace(tech_parameters.kslow - tech_parameters.ks[s], 
                                          tech_parameters.ksbound - tech_parameters.ks[s], tech_parameters.n_s),
                              np.linspace(tech_parameters.kglow - tech_parameters.kg[g], 
                                          tech_parameters.kgbound - tech_parameters.kg[g], tech_parameters.n_g), indexing='ij')

            grid = investment_functions.invest(X, Y, Z, t) + (simu_parameters.beta)*(value_func_pf[n, t+1])
            grid_minimum = np.unravel_index(np.argmin(grid), grid.shape)

            cost_output = iterative_functions.cost(tech_parameters.kw[w], tech_parameters.kg[g], kct, tech_parameters.ks[s], 
                                                   load, pv_cap, epsval, pgt, pct, ctax)
            cost = cost_output[0].sum()
            carbon_realised = cost_output[1].sum()

            mu_constraint = constraints.compute_mu_constraint(t, tech_parameters.kg[g])
            kappa_constraint = constraints.compute_kappa_constraint(t, tech_parameters.kw[w])
            nu_constraint = constraints.compute_nu_constraint(t, tech_parameters.ks[s])
            lambda_constraint = constraints.compute_lambda_constraint(t, carbon_realised)

            next_value = next_value_func_pf.minimize_quantity(tech_parameters.kw[w], tech_parameters.ks[s], 
                                                              tech_parameters.kg[g], t, grid_minimum, grid[grid_minimum])[3]
            
            value_func_pf[n, t, w, s, g] = cost + lambda_constraint + mu_constraint + kappa_constraint + nu_constraint + simu_parameters.beta * next_value

        value_t = value_func_pf[n, t]
        model = GradientBoostingModel(tech_parameters.kw, tech_parameters.ks, tech_parameters.kg)
        model.train_data = value_t
        mse_in, mse_out, X_train, X_test, y_train, y_test, X_mean, X_std, y_mean, y_std = model.train_deterministic()

        model.save_model(simu_parameters.path_functions + "\\value_func_pf_numero_" + str(n) + "_time_" + str(t) + ".pkl")
        next_value_func_pf = GradientBoostingModel(tech_parameters.kw, tech_parameters.ks, tech_parameters.kg)
        next_value_func_pf.train_data = value_t
        model, scaler_X, scaler_y, train_data_mean, train_data_std = next_value_func_pf.load_model(simu_parameters.path_functions + "\\value_func_pf_numero_" + str(n) + "_time_" + str(t) + ".pkl")
        next_value_func_pf.model = model
        next_value_func_pf.scaler_X = scaler_X
        next_value_func_pf.scaler_y = scaler_y

    X0, Y0, Z0 = np.meshgrid(np.linspace(tech_parameters.kwlow - tech_parameters.kw0, 
                                         tech_parameters.kwbound - tech_parameters.kw0, tech_parameters.n_w),
                             np.linspace(tech_parameters.kslow - tech_parameters.ks0, 
                                         tech_parameters.ksbound - tech_parameters.ks0, tech_parameters.n_s),
                             np.linspace(tech_parameters.kglow - tech_parameters.kg0, 
                                         tech_parameters.kgbound - tech_parameters.kg0, tech_parameters.n_g), indexing='ij')

    invest_initial = investment_functions.invest(X0, Y0, Z0, t)
    value_func_pf[n, 0] = value_func_pf[n, 0] + invest_initial

    final_model = GradientBoostingModel(tech_parameters.kw, tech_parameters.ks, tech_parameters.kg)

    final_model.train_data = value_func_pf[n, 0]
    mse_in, mse_out, X_train, X_test, y_train, y_test, X_mean, X_std, y_mean, y_std = final_model.train_deterministic()

    final_model.save_model(simu_parameters.path_functions + "\\value_pf_numero_" + str(n) + "_0.pkl")

    perfect_foresight_optimal_trajectory[n, 0] = [tech_parameters.kw0, tech_parameters.ks0, tech_parameters.kg0]
    kw_t, ks_t, kg_t = tech_parameters.kw0, tech_parameters.ks0, tech_parameters.kg0

    for t in range(0, simu_parameters.t-simu_parameters.extension-1):
        model, scaler_X, scaler_y, train_data_mean, train_data_std = final_model.load_model(simu_parameters.path_functions + "\\value_func_pf_numero_" + str(n) + "_time_" + str(t) + ".pkl")
        final_model.model = model
        final_model.scaler_X = scaler_X
        final_model.scaler_y = scaler_y
        X, Y, Z = np.meshgrid(np.linspace(tech_parameters.kwlow - kw_t, 
                                          tech_parameters.kwbound - kw_t, tech_parameters.n_w),
                                  np.linspace(tech_parameters.kslow - ks_t, 
                                              tech_parameters.ksbound - ks_t, tech_parameters.n_s),
                                  np.linspace(tech_parameters.kglow - kg_t, 
                                              tech_parameters.kgbound - kg_t, tech_parameters.n_g), indexing='ij')

        grid = investment_functions.invest(X, Y, Z, t) + (simu_parameters.beta)*(value_func_pf[n, t+1])
        grid_minimum = np.unravel_index(np.argmin(grid), grid.shape)
        kw_t, ks_t, kg_t, value = final_model.minimize_quantity(kw_t, ks_t, kg_t, t, grid_minimum, grid[grid_minimum])
        perfect_foresight_optimal_trajectory[n, t+1] = [kw_t, ks_t, kg_t]
        
    print("Perfect foresight trajectory n°", n, perfect_foresight_optimal_trajectory[n])
    
    perfect_foresight_optimal_df = pd.DataFrame(perfect_foresight_optimal_trajectory[n], columns=['KW', 'KPV', 'KG'])
    perfect_foresight_optimal_df.to_csv(os.path.join(simu_parameters.path_perfectforesight, 'perfect_foresight_optimal_trajectory_'+str(n)+'.csv'), index=False)

col_names = np.arange(low_bound, high_bound, 1)

save_results_to_csv('perfect_foresight_'+str(low_bound), perfect_foresight_optimal_trajectory[low_bound:high_bound], 
                    int(high_bound-low_bound), column_names = col_names, offset=low_bound)

for n in tqdm.tqdm(range(low_bound, high_bound)):
    perfect_foresight_optimal_df = pd.DataFrame(perfect_foresight_optimal_trajectory[n], columns=['KW', 'KPV', 'KG'])
    perfect_foresight_optimal_df.to_csv(os.path.join(simu_parameters.path_perfectforesight, 
                                                     'perfect_foresight_optimal_trajectory_'+str(n)+'.csv'), index=False)
    
time_elapsed = (time.time() - time_start)
print(time_elapsed/60, "min")


