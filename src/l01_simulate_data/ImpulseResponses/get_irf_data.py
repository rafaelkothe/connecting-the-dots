import numpy as np
import pandas as pd
from datetime import date
from typing import List, Tuple
from pathlib import Path
#from l00_package.l00_parameters.parameters import Params
from l00_package import multiple_simulation_runner

import datetime
import os

from scipy.stats import norm

import matplotlib.pyplot as plt

from l00_package import Params 





def simulate_benchmark(persuasion_values):
    """
    Simulate multiple parameter combinations.

    Args:
        persuasion_values (list): List of persuasion values to simulate.
        num_simulations (int): Number of simulations per parameter combination.
        capT (int): Value for the capT parameter.
        agent_number (int): Value for the agent_number parameter.
        degroot (bool): Value for the degroot parameter.
        topology_m (int): Value for the topology_m parameter.

    Returns:
        dict: Dictionary with parameter combinations as keys and simulation results as values.
    """

    simulation_results = [multiple_simulation_runner(1000, Params(capT=100, agent_number=100, degroot=True, tgt_agt_centr=0, intervention=False, persuasion=persuasion, topology='albert-barabasi')) for persuasion in persuasion_values]  
        


    return simulation_results

def simulate_shock(persuasion_values):
    """
    Simulate multiple parameter combinations.

    Args:
        persuasion_values (list): List of persuasion values to simulate.
        num_simulations (int): Number of simulations per parameter combination.
        capT (int): Value for the capT parameter.
        agent_number (int): Value for the agent_number parameter.
        degroot (bool): Value for the degroot parameter.
        topology_m (int): Value for the topology_m parameter.

    Returns:
        dict: Dictionary with parameter combinations as keys and simulation results as values.
    """


    simulation_results = [multiple_simulation_runner(1000, Params(capT=100, agent_number=100, degroot=True, tgt_agt_centr=0, intervention=False, is_shock='Inflation', persuasion=persuasion, topology='albert-barabasi')) for persuasion in persuasion_values]
    

    return simulation_results

"""

first: persuasion

second: seed

third: variables

"""



################################################################################################################
#                                           Simulating Data w/o shock                                          #
################################################################################################################

persuasion_values = [0, 0.3, 0.5, 0.7, 1]
simulation_results_benchmark = simulate_benchmark(persuasion_values)


################################################################################################################
#                                           Simulating Data w/ shock                                           #
################################################################################################################

simulation_results_shock = simulate_shock(persuasion_values)



################################################################################################################
#                                           Calculate IRF and Save IRF DF                                      #
################################################################################################################

def calculate_irf(simulation_results_benchmark, simulation_results_shock):
    # Define predefined persuasion values
    persuasion_values = [0, 0.3, 0.5, 0.7, 1]
    
    # Initialize dictionary to store results
    data_dict = {f'Persuasion {val}': None for val in persuasion_values}
    
    # Initialize dictionaries to store IRF results
    irf_results = {}
    mean_irf = {}
    ci_95 = {}

    # Loop through each predefined persuasion value
    for idx, persuasion_value in enumerate(persuasion_values):
        irf_results[persuasion_value] = {}
        
        # Loop through each seed
        for seed in range(len(simulation_results_benchmark[idx])):
            # Extract dataframes
            df_benchmark = simulation_results_benchmark[idx][seed][0]
            df_shock = simulation_results_shock[idx][seed][0]
            
            # Calculate IRF
            irf = df_shock - df_benchmark
            
            # Store IRF
            irf_results[persuasion_value][seed] = irf
        
        # Convert IRF results to a list of dataframes
        irf_list = [irf_results[persuasion_value][seed] for seed in irf_results[persuasion_value].keys()]
        
        # Calculate mean IRF over all seeds
        mean_irf[persuasion_value] = pd.concat(irf_list).groupby(level=0).mean()
        
        # Calculate 95% confidence interval
        std_error = pd.concat(irf_list).groupby(level=0).sem()
        ci_95[persuasion_value] = {
            'lower': mean_irf[persuasion_value] - norm.ppf(0.975) * std_error,
            'upper': mean_irf[persuasion_value] + norm.ppf(0.975) * std_error
        }
        
        # Store results in data_dict
        data_dict[f'Persuasion {persuasion_value}'] = {
            'mean_irf': mean_irf[persuasion_value],
            'ci_lower': ci_95[persuasion_value]['lower'],
            'ci_upper': ci_95[persuasion_value]['upper']
        }
    
    return data_dict

def save_data_dict_to_csv(data_dict):
    today = date.today().strftime("%Y-%m-%d")
    folder_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "output_data" / "impulse_responses" / "degroot" / today
    folder_path.mkdir(parents=True, exist_ok=True)
    
    for key, value in data_dict.items():
        
        csv_filename = f"{key}_mean_irf.csv"
        csv_filepath = folder_path / csv_filename
        
        mean_irf_df = pd.DataFrame(value['mean_irf'])
        mean_irf_df.to_csv(csv_filepath, index=False)
        
        csv_filename = f"{key}_ci_lower.csv"
        csv_filepath = folder_path / csv_filename
        
        ci_lower_df = pd.DataFrame(value['ci_lower'])
        ci_lower_df.to_csv(csv_filepath, index=False)
        
        csv_filename = f"{key}_ci_upper.csv"
        csv_filepath = folder_path / csv_filename
        
        ci_upper_df = pd.DataFrame(value['ci_upper'])
        ci_upper_df.to_csv(csv_filepath, index=False)

################################################################################################################
#                                           Calculate IRF and Save IRF DF                                      #
################################################################################################################
data_dict = calculate_irf(simulation_results_benchmark, simulation_results_shock)
save_data_dict_to_csv(data_dict)



def calculate_benchmark_irf(capT: int, a1: float, a2: float, b1: float, b2: float, c1: float, c2: float, c3: float, sigma_inflation: float, shock_period: int = 45) -> pd.DataFrame:
    """
    Calculate the Impulse Response Function (IRF) for inflation and output.

    Args:
        capT (int): The number of time steps.
        a1 (float): Coefficient a1.
        a2 (float): Coefficient a2.
        b1 (float): Coefficient b1.
        b2 (float): Coefficient b2.
        c1 (float): Coefficient c1.
        c2 (float): Coefficient c2.
        c3 (float): Coefficient c3.
        sigma_inflation (float): Standard deviation of the shock.
        shock_period (int, optional): The time step at which the shock occurs. Defaults to 45.

    Returns:
        pd.DataFrame: DataFrame containing the IRF for inflation and output.
    """
    # Initialize the shock
    shock = 2 * sigma_inflation
    
    # Initialize the time series lists
    pi_series: List[float] = []
    x_series: List[float] = []
    
    # Initial values
    pi_0 = 0.0
    x_0 = 0.0
    
    pi_series.append(pi_0)
    x_series.append(x_0)
    
    # Recursive computation
    for t in range(1, capT):
        if t == shock_period:
            pi_t = shock / (1 + a2 * c1 * b2)
            x_t = -a2 * c1 * shock / (1 + a2 * c1 * b2)
        else:
            pi_t = b1 * pi_series[t-1] + b2 * x_series[t-1]
            x_t = -a2 * (c1 * pi_series[t-1] + c2 * x_series[t-1]) / (1 + a2 * c2)
        
        pi_series.append(pi_t)
        x_series.append(x_t)
    
    # Create a DataFrame to store the results
    irf_df = pd.DataFrame({
        'Time': range(capT),
        'Inflation': pi_series,
        'Output': x_series
    })
    
    return irf_df


################################################################################################################
#                                  Simulate Benchmark IRF and Save IRF DF                                      #
################################################################################################################
# Example usage
capT = 100
a1 = 0.5
a2 = 0.2
b1 = 0.5
b2 = 0.05
c1 = 1.5
c2 = 0.5
c3 = 0.5
sigma_inflation = 0.5

benchmark_irf_df = calculate_benchmark_irf(capT, a1, a2, b1, b2, c1, c2, c3, sigma_inflation)

folder_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "output_data" / "impulse_responses" / "benchmark"
folder_path.mkdir(parents=True, exist_ok=True)

csv_filename = "benchmark_irf.csv"
csv_filepath = os.path.join(folder_path, csv_filename)

benchmark_irf_df.to_csv(csv_filepath, index=False)



