# %%
########################################## Stochastic planning ##########################################

# This Jupyter Notebook performs a probabilistic planning simulation for energy investment and cost optimization.
# It includes the following steps:

# 1. Initialization of parameters.
# 2. Calculation of terminal value function using a stochastic approach.
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
np.set_printoptions(suppress=True, precision=5) # threshold=np.inf
seed = 42

iterative_functions = IterativeFunctions()
cost_parameters = CostParameters()
investment_parameters = InvestmentParameters()
capacity_factors = CapacityFactor()
simu_parameters = SimulationParameters()
gradient_parameters = GradientParameters()
tech_parameters = TechnoParameters()
gen_scenario = Scenario()
investment_functions = InvestmentFunctions()
constraints = Constraints(simu_parameters.lambda_weight, simu_parameters.mu_weight, simu_parameters.kappa_weight, simu_parameters.nu_weight)

time_start = time.time()

print("Simulation name: " + simu_parameters.name)
print("Coal phase-out: " + simu_parameters.coal_phase_out)
print("Carbon tax: " + simu_parameters.carbon_tax)



# %%
############################## 0.Initialization: Terminal Value ##############################

value_func = np.zeros((simu_parameters.t, tech_parameters.n_w, tech_parameters.n_s, tech_parameters.n_g, simu_parameters.n_d))  # Cost for each capacity step
next_value = 0

for t in tqdm.tqdm(range(simu_parameters.t-1, simu_parameters.t-2-simu_parameters.extension, -1)):

    print("Year " + str(t) + ": Beginning...")
    ctax = simu_parameters.cpath[t]
    #at = load_curve[t]
    kct = simu_parameters.kc[t]
    f_evol = cost_parameters.fossil_evol[t]
    pct = cost_parameters.pc[t]
    
    for d in tqdm.tqdm(range(simu_parameters.n_d)):
        #load = at + d_load[d]
        load = d_load[d]*(1 + simu_parameters.load_growth * t)
        epsval = capacity_factors.cap_factor[d]
        pv_cap = capacity_factors.pv_cf[d]
        pgt = cost_parameters.pg[d] * f_evol
        for w, s, g in product(range(0, len(tech_parameters.kw), 1), range(0, len(tech_parameters.ks), 1), 
                               range(0, len(tech_parameters.kg), 1)):
            cost_output = iterative_functions.cost(tech_parameters.kw[w], tech_parameters.kg[g], kct, 
                                                   tech_parameters.ks[s], load, pv_cap, epsval, pgt, pct, ctax)
            cost = cost_output[0].sum()
            carbon_realised = cost_output[1].sum()

            mu_constraint = constraints.compute_mu_constraint(t, tech_parameters.kg[g])
            kappa_constraint = constraints.compute_kappa_constraint(t, tech_parameters.kw[w])
            nu_constraint = constraints.compute_nu_constraint(t, tech_parameters.ks[s])
            lambda_constraint = constraints.compute_lambda_constraint(t, carbon_realised)

            value_func[t, w, s, g, d] = cost + lambda_constraint + mu_constraint + kappa_constraint + nu_constraint

        value_func[t,:,:,:,d] += simu_parameters.beta*next_value

    next_value = value_func[t].mean(axis=3)
    print("Year " + str(t) + ": completed")

# Gradient Boost approximation

finalvalue = GradientBoostingModel(tech_parameters.kw, tech_parameters.ks, tech_parameters.kg)
finalvalue.train_data = value_func[simu_parameters.t-1]

mse_in, mse_out, X_train, X_test, y_train, y_test, X_mean, X_std, y_mean, y_std = finalvalue.train(sample_size=gradient_parameters.n_sample)
print("Mean Squared Error in sample:", mse_in)
print("Mean Squared Error out sample:", mse_out)

finalvalue.save_model(simu_parameters.path_functions + "\\value_function_stochastic_" + 
                      str(simu_parameters.t-simu_parameters.extension) + ".pkl")

next_value_func = GradientBoostingModel(tech_parameters.kw, tech_parameters.ks, tech_parameters.kg)
next_value_func.train_data = value_func[simu_parameters.t-1]
model, scaler_X, scaler_y, train_data_mean, train_data_std = next_value_func.load_model(simu_parameters.path_functions 
                                                                                        + "\\value_function_stochastic_" + str(simu_parameters.t-simu_parameters.extension) + ".pkl")
next_value_func.model = model
next_value_func.scaler_X = scaler_X
next_value_func.scaler_y = scaler_y

time_elapsed = (time.time() - time_start)
print(time_elapsed/60, "min")


# %%
############################## 1.Backward algorithm ##############################

for t in tqdm.tqdm(range(simu_parameters.t-simu_parameters.extension-2, -1, -1)):

    print("Year " + str(t) + ": Beginning...")

    # Precompute constant values outside the loop

    ctax = simu_parameters.cpath[t]
    kct = simu_parameters.kc[t]
    pct = cost_parameters.pc[t]
    f_evol = cost_parameters.fossil_evol[t]
    pct = cost_parameters.pc[t]
    
    for d in tqdm.tqdm(range(simu_parameters.n_d)): 
        load = d_load[d]*(1 + simu_parameters.load_growth * t)
        epsval = capacity_factors.cap_factor[d]
        pv_cap = capacity_factors.pv_cf[d]
        pgt = cost_parameters.pg[d] * f_evol

        for w, s, g in product(range(0, len(tech_parameters.kw), 1), range(0, len(tech_parameters.ks), 1), 
                               range(0, len(tech_parameters.kg), 1)):

            # print("Loop: ", KR[r], KPV[s], KG[g])
            X, Y, Z = np.meshgrid(np.linspace(tech_parameters.kwlow - tech_parameters.kw[w], 
                                              tech_parameters.kwbound - tech_parameters.kw[w], tech_parameters.n_w),
                              np.linspace(tech_parameters.kslow - tech_parameters.ks[s], 
                                          tech_parameters.ksbound - tech_parameters.ks[s], tech_parameters.n_s),
                              np.linspace(tech_parameters.kglow - tech_parameters.kg[g], 
                                          tech_parameters.kgbound - tech_parameters.kg[g], tech_parameters.n_g), indexing='ij')

            grid = investment_functions.invest(X, Y, Z, t) + (simu_parameters.beta)*(value_func[t+1].mean(axis=3))
            grid_minimum = np.unravel_index(np.argmin(grid), grid.shape)

            cost_output = iterative_functions.cost(tech_parameters.kw[w], tech_parameters.kg[g], kct, 
                                                   tech_parameters.ks[s], load, pv_cap, epsval, pgt, pct, ctax)
            cost = cost_output[0].sum()
            carbon_realised = cost_output[1].sum()

            mu_constraint = constraints.compute_mu_constraint(t, tech_parameters.kg[g])
            kappa_constraint = constraints.compute_kappa_constraint(t, tech_parameters.kw[w])
            nu_constraint = constraints.compute_nu_constraint(t, tech_parameters.ks[s])
            lambda_constraint = constraints.compute_lambda_constraint(t, carbon_realised)

            next_value = next_value_func.minimize_expected_quantity(tech_parameters.kw[w], tech_parameters.ks[s], tech_parameters.kg[g], d, t, grid_minimum, grid[grid_minimum])[3]
            value_func[t, w, s, g, d] = cost + lambda_constraint + mu_constraint + kappa_constraint + nu_constraint + next_value

    print("Year " + str(t) + ": completed")

    # Gradient Boost approximation

    value_t = value_func[t]
    model = GradientBoostingModel(tech_parameters.kw, tech_parameters.ks, tech_parameters.kg)
    model.train_data = value_t
    mse_in, mse_out, X_train, X_test, y_train, y_test, X_mean, X_std, y_mean, y_std = model.train(sample_size=gradient_parameters.n_sample)
    print("Mean Squared Error in sample:", mse_in)
    print("Mean Squared Error out sample:", mse_out)

    model.save_model(simu_parameters.path_functions + "\\value_function_stochastic_" + str(t) + ".pkl")

    next_value_func = GradientBoostingModel(tech_parameters.kw, tech_parameters.ks, tech_parameters.kg)
    next_value_func.train_data = value_t
    model, scaler_X, scaler_y, train_data_mean, train_data_std = next_value_func.load_model(simu_parameters.path_functions 
                                                                                            + "\\value_function_stochastic_" + str(t) + ".pkl")
    next_value_func.model = model
    next_value_func.scaler_X = scaler_X
    next_value_func.scaler_y = scaler_y


# %%
############################## 2. Initial Investment ##############################

X0, Y0, Z0 = np.meshgrid(np.linspace(tech_parameters.kwlow - tech_parameters.kw0, 
                                     tech_parameters.kwbound - tech_parameters.kw0, tech_parameters.n_w),
                              np.linspace(tech_parameters.kslow - tech_parameters.ks0, 
                                          tech_parameters.ksbound - tech_parameters.ks0, tech_parameters.n_s),
                              np.linspace(tech_parameters.kglow - tech_parameters.kg0, 
                                          tech_parameters.kgbound - tech_parameters.kg0, tech_parameters.n_g), indexing='ij')


invest_initial = np.repeat(investment_functions.invest(X0, Y0, Z0, 0), simu_parameters.n_d).reshape(value_func[0].shape)

value_func[0] = value_func[0] + invest_initial

model = GradientBoostingModel(tech_parameters.kw, tech_parameters.ks, tech_parameters.kg)
model.train_data = value_func[0]
mse_in, mse_out, X_train, X_test, y_train, y_test, X_mean, X_std, y_mean, y_std = model.train(sample_size=gradient_parameters.n_sample)

model.save_model(simu_parameters.path_functions + "\\value_stochastic_0.pkl")

time_elapsed = (time.time() - time_start)
print(time_elapsed/60, "min")

# %%
############################## 3. Save Optimal Path and export of the results ##############################

d_seq = np.arange(0, simu_parameters.n_d,1)

stochastic_optimal_trajectory = np.zeros((simu_parameters.t-simu_parameters.extension, 3))
kw_t, ks_t, kg_t = tech_parameters.kw0, tech_parameters.ks0, tech_parameters.kg0
value_func_final = GradientBoostingModel(kw_t, ks_t, kg_t)
stochastic_optimal_trajectory[0] = [tech_parameters.kw0, tech_parameters.ks0, tech_parameters.kg0]

for t in range(0, simu_parameters.t-1-simu_parameters.extension):
    model, scaler_X, scaler_y, train_data_mean, train_data_std = value_func_final.load_model(
        simu_parameters.path_functions + "\\value_function_stochastic_" + str(t) + ".pkl")
    value_func_final.model = model
    value_func_final.scaler_X = scaler_X
    value_func_final.scaler_y = scaler_y
    X, Y, Z = np.meshgrid(np.linspace(tech_parameters.kwlow - kw_t, 
                                      tech_parameters.kwbound - kw_t, tech_parameters.n_w),
                                  np.linspace(tech_parameters.kslow - ks_t, 
                                              tech_parameters.ksbound - ks_t, tech_parameters.n_s),
                                  np.linspace(tech_parameters.kglow - kg_t, 
                                              tech_parameters.kgbound - kg_t, tech_parameters.n_g), indexing='ij')

    grid = investment_functions.invest(X, Y, Z, t) + (simu_parameters.beta)*(value_func[t+1].mean(axis=3))
    grid_minimum = np.unravel_index(np.argmin(grid), grid.shape)
    kw_t, ks_t, kg_t, value = value_func_final.minimize_expected_quantity(kw_t, ks_t, kg_t, d_seq, t, grid_minimum, grid[grid_minimum])
    stochastic_optimal_trajectory[t+1] = [kw_t, ks_t, kg_t]

stochastic_optimal_df = pd.DataFrame(stochastic_optimal_trajectory, columns=['KW', 'KS', 'KG'])
stochastic_optimal_df.to_csv(os.path.join(simu_parameters.path_stochastic, 'stochastic_optimal_trajectory.csv'), 
                             index=False)

print("Optimal stochastic trajectory: ", stochastic_optimal_trajectory)

save_results_to_csv('stochastic', stochastic_optimal_trajectory)

time_elapsed = (time.time() - time_start)
print(time_elapsed/60, "min")


