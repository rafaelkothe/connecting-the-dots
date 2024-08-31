import numpy as np
import pandas as pd
from datetime import date
from typing import List, Tuple
from pathlib import Path
from l00_package import Model
from l00_package import Params
from l00_package import Grids
from l00_package import multiple_simulation_runner

import datetime
import os


import matplotlib.pyplot as plt



def simulate_parameter_combinations(persuasion_values, tgt_agt_centr_values):
    """
    Simulate multiple parameter combinations.

    Args:
        persuasion_values (list): List of persuasion values to simulate.
        tgt_agt_centr_values (list): List of tgt_agt_centr values to simulate.
        num_simulations (int): Number of simulations per parameter combination.
        capT (int): Value for the capT parameter.
        agent_number (int): Value for the agent_number parameter.
        degroot (bool): Value for the degroot parameter.
        topology_m (int): Value for the topology_m parameter.

    Returns:
        dict: Dictionary with parameter combinations as keys and simulation results as values.
    """
    tgt_agt_centr_mesh, persuasion_mesh = np.meshgrid(tgt_agt_centr_values, persuasion_values)
    
    # Flatten the mesh grids
    persuasion_values_flat = persuasion_mesh.flatten()
    tgt_agt_centr_values_flat = tgt_agt_centr_mesh.flatten()

    simulation_results = {}

    for tgt_agt_centr, persuasion in zip(tgt_agt_centr_values_flat, persuasion_values_flat):
        results = multiple_simulation_runner(500, Params(capT=200, agent_number=100, tgt_agt_centr=tgt_agt_centr, degroot=True, intervention=True, target=True, persuasion=persuasion, topology='albert-barabasi'))
        simulation_results[(persuasion, tgt_agt_centr)] = results


    return simulation_results

def simulate_parameter_combinations_benchmark(persuasion_values):
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

    simulation_results = {}

    for persuasion in persuasion_values:
        results = multiple_simulation_runner(500, Params(capT=200, agent_number=100, degroot=True, tgt_agt_centr=0, intervention=False, persuasion=persuasion, topology='albert-barabasi'))
        simulation_results[persuasion] = results


    return simulation_results



################################################################################################################
#                                           Simulating Data                                                    #
################################################################################################################


persuasion_values = [0, 0.3, 0.5, 0.7, 1]
tgt_agt_centr_values = [0, 4, 9, 24]
simulation_results = simulate_parameter_combinations(persuasion_values, tgt_agt_centr_values)


################################################################################################################
#                               Simulating Data (Benchmark)                                                    #
################################################################################################################

simulation_results_benchmark = simulate_parameter_combinations_benchmark(persuasion_values)

################################################################################################################
#                               Creating Dataset                                                               #
################################################################################################################

def create_data_dict(persuasion_values, tgt_agt_centr_values, variable, stat_method='mean'):
    data_dict = {
        'Benchmark': {
            'Persuasion 0': None,
            'Persuasion 0.3': None,
            'Persuasion 0.5': None,
            'Persuasion 0.7': None,
            'Persuasion 1': None,
        },
        '0': {
            'Persuasion 0': None,
            'Persuasion 0.3': None,
            'Persuasion 0.5': None,
            'Persuasion 0.7': None,
            'Persuasion 1': None,
        },
        '4': {
            'Persuasion 0': None,
            'Persuasion 0.3': None,
            'Persuasion 0.5': None,
            'Persuasion 0.7': None,
            'Persuasion 1': None,
        },
        '9': {
            'Persuasion 0': None,
            'Persuasion 0.3': None,
            'Persuasion 0.5': None,
            'Persuasion 0.7': None,
            'Persuasion 1': None,
        },
        '24': {
            'Persuasion 0': None,
            'Persuasion 0.3': None,
            'Persuasion 0.5': None,
            'Persuasion 0.7': None,
            'Persuasion 1': None,
        }
    }

    for persuasion in persuasion_values:
            if stat_method == 'mean':
                data_benchmark = [simulation_results_benchmark[persuasion][i][0][variable][25:].mean() for i in range(len(simulation_results_benchmark[persuasion]))]  # Example data
            elif stat_method == 'std':
                data_benchmark = [simulation_results_benchmark[persuasion][i][0][variable][25:].std() for i in range(len(simulation_results_benchmark[persuasion]))]  # Example data
                
            data_dict['Benchmark'][f'Persuasion {persuasion}'] = np.array(data_benchmark)

            for tgt_agt_centr in tgt_agt_centr_values:
                if stat_method == 'mean':
                    data = [simulation_results[(persuasion, tgt_agt_centr)][i][0][variable][25:].mean() for i in range(len(simulation_results[(persuasion, tgt_agt_centr)]))]  # Example data
                elif stat_method == 'std':
                    data = [simulation_results[(persuasion, tgt_agt_centr)][i][0][variable][25:].std() for i in range(len(simulation_results[(persuasion, tgt_agt_centr)]))]  # Example data
                data_dict[str(tgt_agt_centr)][f'Persuasion {persuasion}'] = np.array(data)

    return data_dict

def concat_dict_to_df_melted(data_dict):
    dict_of_df = {k: pd.DataFrame(v) for k,v in data_dict.items()}
    df_concat = pd.concat(dict_of_df, axis=1)
    df_melted = df_concat.stack(level=1).reset_index().rename(columns={"level_0": "value", "level_1": "Persuasion"})
    return df_melted

################################################################################################################
#                               Saving Dataset                                                                 #
################################################################################################################


# Define the variables to iterate over
variables = ['Output', 'Inflation', 'Interest', 'RealInterest', 'OutTarWe', 'OutNaiveWe', 'InfTarWe', 'InfNaiveWe', 'OutMarketExp', 'InfMarketExp', 'OutAniSp', 'InfAniSp']

today = date.today().strftime("%Y-%m-%d")
folder_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "output_data" / "target_interventions" / today
folder_path.mkdir(parents=True, exist_ok=True)

# Create and save dictionaries and DataFrames for each variable
for variable in variables:
    simulated_data_mean_dict = create_data_dict(persuasion_values, tgt_agt_centr_values, variable, stat_method='mean')
    simulated_data_mean_df = concat_dict_to_df_melted(simulated_data_mean_dict)

    # Save the "mean" DataFrame as a .csv file in the "mean" folder
    csv_filename = f'{variable}_mean_data.csv'
    csv_filepath = folder_path / csv_filename
    simulated_data_mean_df.to_csv(csv_filepath, index=False)

    simulated_data_std_dict = create_data_dict(persuasion_values, tgt_agt_centr_values, variable, stat_method='std')
    simulated_data_std_df = concat_dict_to_df_melted(simulated_data_std_dict)

    # Save the "std" DataFrame as a .csv file in the "std" folder
    csv_filename = f'{variable}_std_data.csv'
    csv_filepath = folder_path / csv_filename
    simulated_data_std_df.to_csv(csv_filepath, index=False)

    # Assign the "mean" DataFrame to a variable with the variable name
    globals()[f'simulated_data_{variable}_mean_df'] = simulated_data_mean_df

    # Assign the "std" DataFrame to a variable with the variable name
    globals()[f'simulated_data_{variable}_std_df'] = simulated_data_std_df

    # Assign the "mean" dictionary to a variable with the variable name
    globals()[f'simulated_data_{variable}_mean_dict'] = simulated_data_mean_dict

    # Assign the "std" dictionary to a variable with the variable name
    globals()[f'simulated_data_{variable}_std_dict'] = simulated_data_std_dict