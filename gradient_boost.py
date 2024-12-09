import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import qmc
from scipy.optimize import minimize
from simulation_parameters import *
from cost_functions import InvestmentFunctions

xgb.set_config(use_rmm = True)

simu_parameters = SimulationParameters()
gradient_parameters = GradientParameters()
investment_functions = InvestmentFunctions()
tech_parameters = TechnoParameters()

class GradientBoostingModel:
    """
    A class used to represent a Gradient Boosting Model for dynamic investment programming.
    Attributes
    ----------
    kr : array-like
        Wind power capacity.
    ks : array-like
        Solar power capacity.
    kg : array-like
        Gas power capacity.
    scenario : int
        Scenario number.
    train_data : DataFrame or None
        Training data for the model.
    model : XGBRegressor or None
        The gradient boosting model.
    X_train : array-like or None
        Training features.
    X_test : array-like or None
        Testing features.
    y_train : array-like or None
        Training labels.
    y_test : array-like or None
        Testing labels.
    predictions : array-like or None
        Predictions made by the model on the test set.
    random_s : int
        Random seed for reproducibility.
    dimensions : int or None
        Dimensions of the data.
    n_jobs : int
        Number of parallel threads used to run XGBoost.
    Methods
    -------
    data_preprocessing(data, kw, ks, kg)
        Preprocesses the data for the model.
    data_preprocessing_deterministic(data, kw, ks, kg)
        Preprocesses the data for the deterministic model.
    data_preprocessing_with_sampling(data, kw, ks, kg, sample_size)
        Preprocesses the data with sampling for the model.
    train(sample_size, test_size)
        Trains the gradient boosting model.
    train_deterministic(sample_size, test_size)
        Trains the deterministic gradient boosting model.
    save_model(filename)
        Saves the trained model to a file.
    load_model(filename)
        Loads a trained model from a file.
    predicted_value(kr, kpv, kg, scenario)
        Predicts the value for given inputs and scenario.
    predicted_value_deterministic(kr, kpv, kg)
        Predicts the value for given inputs in a deterministic setting.
    plot_histogram_test(title)
        Plots a histogram of the test set predictions.
    plot_histogram_train(title)
        Plots a histogram of the training set predictions.
    plot_scatter(title)
        Plots a scatter plot of true values vs predictions.
    minimize_quantity(kw_initial, kpv_initial, kg_initial, t, grid_minimum)
        Minimizes the quantity for given initial values and grid minimum.
    minimize_expected_quantity(kw_initial, ks_initial, kg_initial, D, t, grid_minimum)
        Minimizes the expected quantity for given initial values, scenarios, and grid minimum.
    """

    def __init__(self, kw, ks, kg, scenario = 3):
        self.kr = kw
        self.ks = ks
        self.kg = kg
        self.train_data = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.predictions = None
        self.random_s = 42 # seed
        self.dimensions = None
        self.n_jobs = -1
        self.scenario = scenario

    def data_preprocessing(self, data, kw, ks, kg):
        flattened = data.flatten()
        wind_indices, pv_indices, gas_indices, scenarios_indices = np.indices(data.shape)
        wind_indices = wind_indices.flatten()
        pv_indices = pv_indices.flatten()
        gas_indices = gas_indices.flatten()
        scenarios_indices = scenarios_indices.flatten()

        scenario_matrix = np.eye(simu_parameters.n_d, dtype=int)  # Generate a scenario matrix

        values = flattened
        winds = kw[wind_indices]
        pvs = ks[pv_indices]
        gases = kg[gas_indices]
        scenarios = scenario_matrix[scenarios_indices]

        df = pd.DataFrame({
            "Value": values,
            "Wind": winds,
            "PV": pvs,
            "Gas": gases
        })

        scenario_columns = [f"Scenario_{i}" for i in range(simu_parameters.n_d)]
        scenario_df = pd.DataFrame(scenarios, columns=scenario_columns)
        df = pd.concat([df, scenario_df], axis=1)
        df.drop(columns=df.columns[-1], inplace=True)

        return df

    def data_preprocessing_deterministic(self, data, kw, ks, kg):
        flattened = data.flatten()
        wind_indices, pv_indices, gas_indices = np.indices(data.shape)

        wind_indices = wind_indices.flatten()
        pv_indices = pv_indices.flatten()
        gas_indices = gas_indices.flatten()

        values = flattened
        winds = kw[wind_indices]
        pvs = ks[pv_indices]
        gases = kg[gas_indices]

        df = pd.DataFrame({
            "Value": values,
            "Wind": winds,
            "PV": pvs,
            "Gas": gases
        })

        return df

    def data_preprocessing_with_sampling(self, data, kw, ks, kg, sample_size=gradient_parameters.n_sample):

        flattened = data.flatten()
        base_columns = ["Value", "Wind", "PV", "Gas"]

        scenario_columns = [f"Scenario_{i}" for i in range(simu_parameters.n_d)]

        columns = base_columns + scenario_columns

        df = pd.DataFrame(columns=columns)

        sobol_sequence = qmc.Sobol(d=1, scramble=True)
        random_points = flattened.min() + sobol_sequence.random(
            min(2 ** int(np.floor(np.log2(flattened.size))), sample_size)) * (
                                    flattened.max() - flattened.min())

        for point in random_points:
            indices = np.unravel_index(np.abs(flattened - point).argmin(), data.shape)
            wind, pv, gas, scenario = indices
            row = [point.item(), kw[wind], ks[pv], kg[gas]] + [int(j == scenario) for j in range(simu_parameters.n_d)]
            df.loc[len(df)] = row

        df.drop(columns=df.columns[-1], inplace=True)

        return df

    def train(self, sample_size=gradient_parameters.n_sample, test_size=0.2):
        random_state = self.random_s
        df = self.data_preprocessing(self.train_data, self.kr, self.ks, self.kg)

        sample_size = min(len(df), sample_size) 
        y_data = np.array(df.iloc[:sample_size, 0]) 
        X_data = np.array(df.iloc[:sample_size, 1:])
        self.scaler_X = StandardScaler()
        X_data[:, :3] = self.scaler_X.fit_transform(X_data[:, :3])
        self.scaler_y = StandardScaler()
        y_data = self.scaler_y.fit_transform(y_data.reshape(-1, 1)).flatten()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_data, y_data, test_size=test_size, random_state=random_state, shuffle=True)
        self.model = xgb.XGBRegressor(tree_method = "hist", n_estimators=gradient_parameters.precision_gradient, max_depth=gradient_parameters.gradient_depth, n_jobs=self.n_jobs)  # Gradient Boost
        self.model.fit(self.X_train, self.y_train)
        self.predictions = self.model.predict(self.X_test)
        self.predictions_train = self.model.predict(self.X_train)

        mse_in = mean_squared_error(self.y_train, self.predictions_train)
        mse_out = mean_squared_error(self.y_test, self.predictions)

        X_mean = self.scaler_X.mean_
        X_std = self.scaler_X.scale_
        y_mean = self.scaler_y.mean_[0]
        y_std = self.scaler_y.scale_[0]

        return mse_in, mse_out, self.X_train, self.X_test, self.y_train, self.y_test, X_mean, X_std, y_mean, y_std

    def train_deterministic(self, sample_size=gradient_parameters.n_sample, test_size=0.2):
        random_state = self.random_s
        df = self.data_preprocessing_deterministic(self.train_data, self.kr, self.ks, self.kg)
        sample_size = min(len(df), sample_size)  
        y_data = np.array(df.iloc[:sample_size, 0])  
        X_data = np.array(df.iloc[:sample_size, 1:])  

        self.scaler_X = StandardScaler()
        X_data = self.scaler_X.fit_transform(X_data)
        self.scaler_y = StandardScaler()
        y_data = self.scaler_y.fit_transform(y_data.reshape(-1, 1)).flatten()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_data, y_data, test_size=test_size,
                                                                                random_state=random_state, shuffle=True)
        self.model = xgb.XGBRegressor(tree_method = "hist", n_estimators=gradient_parameters.precision_gradient, max_depth=gradient_parameters.gradient_depth,
                                      n_jobs=self.n_jobs)
        self.model.fit(self.X_train, self.y_train)
        self.predictions = self.model.predict(self.X_test)
        self.predictions_train = self.model.predict(self.X_train)

        mse_in = mean_squared_error(self.y_train, self.predictions_train)
        mse_out = mean_squared_error(self.y_test, self.predictions)

        X_mean = self.scaler_X.mean_
        X_std = self.scaler_X.scale_
        y_mean = self.scaler_y.mean_[0]
        y_std = self.scaler_y.scale_[0]

        return mse_in, mse_out, self.X_train, self.X_test, self.y_train, self.y_test, X_mean, X_std, y_mean, y_std

    def save_model(self, filename):
        if self.model is not None:
            model_data = {
                'model': self.model,
                'scaler_X_mean': self.scaler_X.mean_,
                'scaler_X_std': self.scaler_X.scale_,
                'scaler_y_mean': self.scaler_y.mean_[0],
                'scaler_y_std': self.scaler_y.scale_[0],
                'train_data_mean': self.train_data.mean(),
                'train_data_std': self.train_data.std()
            }
            with open(filename, 'wb') as file: 
                pickle.dump(model_data, file)

    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as file:
            model_data = pickle.load(file)

        model = model_data['model']
        scaler_X_mean = model_data['scaler_X_mean']
        scaler_X_std = model_data['scaler_X_std']
        scaler_y_mean = model_data['scaler_y_mean']
        scaler_y_std = model_data['scaler_y_std']
        train_data_mean = model_data['train_data_mean']
        train_data_std = model_data['train_data_std']

        scaler_X = StandardScaler()
        scaler_X.mean_ = scaler_X_mean
        scaler_X.scale_ = scaler_X_std

        scaler_y = StandardScaler()
        scaler_y.mean_ = np.array([scaler_y_mean])
        scaler_y.scale_ = np.array([scaler_y_std])

        return model, scaler_X, scaler_y, train_data_mean, train_data_std

    def predicted_value(self, kr, kpv, kg, scenario):
        transformed_scenario = [int(j == scenario) for j in
                                range(simu_parameters.n_d)] 

        input_data = np.array([[kr, kpv, kg, *transformed_scenario]])
        input_data = input_data[:, :-1]
        input_data[:, :3] = self.scaler_X.transform(input_data[:, :3])
        prediction = self.model.predict(input_data)
        prediction = self.scaler_y.inverse_transform(prediction.reshape(-1, 1)).flatten()

        return prediction.item()

    def predicted_value_deterministic(self, kr, kpv, kg):
        input_data = np.array([[kr, kpv, kg]])
        input_data = self.scaler_X.transform(input_data)
        prediction = self.model.predict(input_data)
        prediction = self.scaler_y.inverse_transform(prediction.reshape(-1, 1)).flatten()

        return prediction.item()

    def plot_histogram_test(self, title):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(self.y_test, kde=True, ax=ax, color='blue', alpha=0.5, label='True values')
        sns.histplot(self.predictions, kde=True, ax=ax, color='pink', alpha=0.5, label='Predictions')
        ax.set_xlabel('Values')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.legend()
        plt.show()

    def plot_histogram_train(self, title):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(self.y_train, kde=True, ax=ax, color='blue', alpha=0.5, label='True values')
        sns.histplot(self.predictions_train, kde=True, ax=ax, color='pink', alpha=0.5, label='Predictions')
        ax.set_xlabel('Values')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.legend()
        plt.show()

    def plot_scatter(self, title):
        plt.scatter(self.y_test, self.predictions, color='green', alpha=0.5)
        plt.xlabel('True values')
        plt.ylabel('Predictions')
        plt.title(title)
        plt.show()
    
    def minimize_quantity(self, kw_initial, kpv_initial, kg_initial, t, grid_minimum):

        def objective_function(params):
            kw_obj, ks_obj, kg_obj = params
            value = self.predicted_value_deterministic(kw_obj, ks_obj, kg_obj)
            invest = investment_functions.invest(kw_obj-kw_initial, ks_obj-kpv_initial, kg_obj-kg_initial, t)
            total_investment = np.sum(invest)
            return value + total_investment

        kw_min, ks_min, kg_min = tech_parameters.kw[grid_minimum[0]], tech_parameters.ks[grid_minimum[1]], tech_parameters.kg[grid_minimum[2]]
        initial_guesses = np.array([kw_min, ks_min, kg_min])
        bounds = [(tech_parameters.kwlow, tech_parameters.kwbound), (tech_parameters.kslow, tech_parameters.ksbound), 
                  (tech_parameters.kglow-tech_parameters.step_g, tech_parameters.kgbound+tech_parameters.step_g)]
        result = minimize(objective_function, initial_guesses, method="Nelder-Mead", bounds=bounds)
        kw_optimized, ks_optimized, kg_optimized = result.x

        kw_optimized = tech_parameters.unit_w * np.floor(kw_optimized / tech_parameters.unit_w)
        ks_optimized = tech_parameters.unit_s * np.floor(ks_optimized / tech_parameters.unit_s)
        kg_optimized = tech_parameters.unit_g * np.floor(kg_optimized / tech_parameters.unit_g)   
        return kw_optimized, ks_optimized, kg_optimized, result.fun

    def minimize_expected_quantity(self, kw_initial, ks_initial, kg_initial, D, t, grid_minimum):
        X, Y, Z = np.meshgrid(np.linspace(tech_parameters.kwlow - kw_initial, 
                                          tech_parameters.kwbound - kw_initial, tech_parameters.n_w),
                              np.linspace(tech_parameters.kslow - ks_initial, 
                                          tech_parameters.ksbound - ks_initial, tech_parameters.n_s),
                              np.linspace(tech_parameters.kglow - kg_initial, 
                                          tech_parameters.kgbound - kg_initial, tech_parameters.n_g), indexing='ij')

        def objective_function(params):
            kw, kpv, kg = params
            mean_value = np.mean([self.predicted_value(kw, kpv, kg, d) for d in D])
            invest = investment_functions.invest(X, Y, Z, t)
            total_investment = np.sum(invest)

            return mean_value + total_investment

        kw_min, kpv_min, kg_min = tech_parameters.kw[grid_minimum[0]], tech_parameters.ks[grid_minimum[1]], tech_parameters.kg[grid_minimum[2]]

        initial_guesses = np.array([kw_min, kpv_min, kg_min])

        bounds = [(tech_parameters.kwlow-100, tech_parameters.kwbound+100), 
                  (tech_parameters.kslow-100, tech_parameters.ksbound+100), 
                  (tech_parameters.kglow-100, tech_parameters.kgbound+100)]
        
        result = minimize(objective_function, initial_guesses, method="L-BFGS-B", bounds=bounds)
        kw_optimized, kpv_optimized, kg_optimized = result.x
        kw_optimized = tech_parameters.unit_w * np.floor(kw_optimized / tech_parameters.unit_w)
        kpv_optimized = tech_parameters.unit_s * np.floor(kpv_optimized / tech_parameters.unit_s)
        kg_optimized = tech_parameters.unit_g * np.floor(kg_optimized / tech_parameters.unit_g)
        
        return kw_optimized, kpv_optimized, kg_optimized, result.fun


    
