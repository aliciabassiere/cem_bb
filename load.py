import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from functools import reduce
import holidays
from stargazer.stargazer import Stargazer
from simulation_parameters import *

myFmt = mdates.DateFormatter('%m-%d')
plt.rcParams["figure.figsize"] = (11, 5)
np.set_printoptions(suppress=True)

simu_parameters = SimulationParameters()

"""
This module loads and processes load data for a dynamic investment programming simulation.

The module performs the following tasks:
1. Defines the input path for loading data files.
2. Loads load data and residual load data from CSV files.
3. Constructs deterministic and stochastic demand trajectories.
4. Computes anticipated deterministic demand evolution scenarios.
5. Calculates various deterministic scenarios including mean, quantiles, and min/max of the stochastic trajectories.
6. Prepares deterministic and mimic demand trajectories for further simulation.

Variables:
    simu_parameters (SimulationParameters): An instance of the SimulationParameters class.
    input_path (Path): The path to the input data directory.
    load_data (DataFrame): The loaded load data.
    residual_load_data (DataFrame): The loaded residual load data.
    a_load (ndarray): The deterministic demand trajectory.
    d_load (ndarray): The stochastic demand trajectory.
    residual_load (ndarray): The residual load data.
    aevol (ndarray): The anticipated deterministic demand evolution scenario.
    load_curve (ndarray): The load curve after applying the anticipated evolution.
    d_load_mean (ndarray): The mean of the stochastic demand trajectory.
    d_load_low (ndarray): The 25th percentile of the stochastic demand trajectory.
    d_load_high (ndarray): The 75th percentile of the stochastic demand trajectory.
    d_load_min (ndarray): The minimum of the stochastic demand trajectory.
    d_load_max (ndarray): The maximum of the stochastic demand trajectory.
    d_load_deterministic (ndarray): The deterministic part of the stochastic demand trajectory.
    d_load_mean_deterministic (ndarray): The mean of the deterministic part of the stochastic demand trajectory.
"""

# Load data
load_data = pd.read_csv(simu_parameters.path_inputs + r"/load.csv", index_col=0)
residual_load_data = pd.read_csv(simu_parameters.path_inputs + r"/residual_load_inputs.csv", index_col=0)

a_load = np.tile(np.array(load_data['fitted_2019']), simu_parameters.t + 1).reshape(simu_parameters.t + 1, simu_parameters.h)  # Trajectoire déterministe
d_load = np.array(load_data[['residuals_2015', 'residuals_2016', 'residuals_2017', 'residuals_2018', 'residuals_2019', 'low_shock', 'high_shock']].T)  # Trajectoire stochastique
residual_load = np.array(residual_load_data[['2015', '2016', '2017', '2018', '2019', 'low_shock', 'high_shock']].T, dtype=object)
d_load = d_load - residual_load 
aevol = np.repeat(np.linspace(1 + simu_parameters.load_growth, (1 + simu_parameters.load_growth * simu_parameters.t), simu_parameters.t + 1), simu_parameters.h).reshape(simu_parameters.t + 1, simu_parameters.h)  # Scénario d'évolution de la demande déterministe anticipé
load_curve = a_load * aevol

# Deterministic scenario scenarios
d_load_mean = d_load.mean(axis=0)
d_load_low = np.quantile(d_load, 0.25, axis=0)
d_load_high = np.quantile(d_load, 0.75, axis=0)
d_load_min = d_load.min(axis=0)
d_load_max = d_load.max(axis=0)
d_load_naive = d_load[:simu_parameters.n_d-2]
d_load_mean_deterministic = d_load_naive.mean(axis=0)

d_draw = np.array(pd.read_csv(simu_parameters.path_inputs + r"/sobol_draws.csv", header=None))

d = np.array(np.linspace(0, simu_parameters.n_d-1, simu_parameters.n_d).astype(int))


class DemandEstimator:
    """
    A class used to estimate and analyze demand based on historical data.
    Attributes
    ----------
    demand_df : pandas.DataFrame
        A DataFrame containing the demand data.
    Methods
    -------
    prepare_data():
        Prepares the data by renaming columns, resampling, and creating new features.
    fit_model():
        Fits an Ordinary Least Squares (OLS) regression model to the prepared data.
    calculate_residuals():
        Calculates the residuals from the fitted model.
    smooth_data():
        Smooths the demand and fitted values using a rolling window.
    save_results():
        Saves the results of the demand estimation to a CSV file.
    plot_results():
        Plots the smoothed demand and fitted values and saves the plot as a PDF.
    """

    def __init__(self, demand_df):
        self.demand_df = demand_df
        self.prepare_data()
        self.fit_model()
        self.calculate_residuals()
        self.smooth_data()
        self.plot_results()
        self.save_results()

    def prepare_data(self):
        self.demand_df = self.demand_df.rename(columns={'Unnamed: 0': 'date', 'Actual Load': 'load'})
        self.demand_df['date'] = pd.to_datetime(self.demand_df['date'], utc=True) + pd.DateOffset(hours=1)
        date = pd.to_datetime(self.demand_df['date'], utc=True) + pd.DateOffset(hours=1)
        self.demand_df = pd.DataFrame(self.demand_df.set_index('date')['load'].resample('1h').mean())
        self.demand_df['hour'] = self.demand_df.index.hour.values
        self.demand_df['year'] = self.demand_df.index.year.values
        self.demand_df['month'] = self.demand_df.index.month.values
        self.demand_df['day'] = self.demand_df.index.day_name()
        self.demand_df['day_number'] = self.demand_df.index.day.values
        self.demand_df['lockdown'] = (((self.demand_df.index >= '2020-03-22') & (self.demand_df.index <= '2020-04-20')) | ((self.demand_df.index >= '2020-12-16') & (self.demand_df.index <= '2020-12-31'))).astype(int)
        self.demand_df['date_offset'] = (self.demand_df.index.month.values*100 + self.demand_df.index.day.values - 320) % 1300
        self.demand_df['season'] = pd.cut(self.demand_df['date_offset'], [0, 300, 602, 900, 1300], labels=['spring', 'summer', 'autumn', 'winter'])
        self.demand_df['htype'] = pd.cut(self.demand_df['hour'], [0, 8, 12, 18, 22, 23], labels=['Opeak', 'Peak', 'Opeak', 'Peak', 'Opeak'], ordered=False, include_lowest=True)
        self.demand_df['wday'] = self.demand_df.index.weekday.values
        self.demand_df['weekday'] = pd.cut(self.demand_df['wday'], [0, 4, 6], labels=['Wday', 'Wend'], ordered=False, include_lowest=True)
        self.demand_df['wh'] = self.demand_df['weekday'].astype(str) + "_" + self.demand_df['htype'].astype(str)
        self.demand_df = pd.merge(self.demand_df, pd.get_dummies(self.demand_df, columns=['season']))
        self.demand_df = pd.merge(self.demand_df, pd.get_dummies(self.demand_df, columns=['htype']))
        self.demand_df = pd.merge(self.demand_df, pd.get_dummies(self.demand_df, columns=['weekday']))
        self.demand_df = pd.merge(self.demand_df, pd.get_dummies(self.demand_df, columns=['year']))
        self.demand_df = pd.merge(self.demand_df, pd.get_dummies(self.demand_df, columns=['wh']))
        self.demand_df['date'] = date.dt.date

        holidays_date = pd.DataFrame(holidays.DE(years=self.demand_df['year'].unique()).keys()).rename(columns={0: 'date'})
        holidays_date['holiday'] = 1
        self.demand_df = pd.merge(self.demand_df, holidays_date, how='left')
        self.demand_df.holiday = self.demand_df.holiday.fillna(0)

    def fit_model(self):
        X = self.demand_df[['holiday', 'lockdown', 'season_winter', 'season_summer', 'season_autumn', 'wh_Wday_Opeak', 
                            'wh_Wday_Peak', 'wh_Wend_Peak', 'year_2016', 'year_2017', 'year_2018', 'year_2019', 'year_2020', 'year_2021']]
        X = sm.add_constant(X.astype(int))
        Y = self.demand_df[['load']] / 1000

        self.model = sm.OLS(Y, X)
        self.results = self.model.fit()
        print(self.results.summary())

    def calculate_residuals(self):
        self.demand_df['fitted'] = self.results.fittedvalues
        self.demand_df['residuals'] = self.demand_df['load'] - self.demand_df['fitted']

    def smooth_data(self):
        self.load_smoothed = self.demand_df['load'][:8759].rolling(window=24).mean()/1000
        self.fit_smoothed = self.demand_df['fitted'][:8759].rolling(window=24).mean()

    def save_results(self):
        print_results = Stargazer([self.results])
        d_year = {}
        for i in range(self.demand_df['year'].nunique()):
            year_aim = self.demand_df['year'].min() + i
            d_year[year_aim] = self.demand_df[self.demand_df['year'] == year_aim][['hour', 'day', 'month', 'residuals', 'fitted']]
            d_year[year_aim] = d_year[year_aim].rename(columns={'residuals': 'residuals_' + str(year_aim), 'fitted': 'fitted_' + str(year_aim)})
        df_list = [v[['day', 'month', 'hour', 'residuals_' + str(year_aim), 'fitted_' + str(year_aim)]] for year_aim, v in d_year.items()]
        df = df_list[0]
        for next_df in df_list[1:]:
            df = pd.merge(df, next_df, on=['day', 'month', 'hour'])
        df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        df = df.set_index('datetime').drop(columns=['year', 'month', 'day', 'hour'])
        df.to_csv(r"load_estimated.csv", chunksize=10000)

    def plot_results(self):
        fig, ax = plt.subplots()
        ax.plot(self.load_smoothed, color='blue', linestyle="dotted", linewidth=0.5)
        ax.plot(self.fit_smoothed, color='red', linewidth=0.5)
        ax.set_xlabel('Date')
        ax.set_ylabel('MWh')
        ax.xaxis.set_major_formatter(myFmt)
        plt.legend(["Realised demand", "Fitted demand"], frameon=True, facecolor="white", loc='upper right')
        plt.grid("black")
        plt.title("Demand estimation", fontsize=15)
        fig.patch.set_facecolor('xkcd:white')
        plt.savefig(r"demand_estimation.pdf", dpi=300)
        plt.show()