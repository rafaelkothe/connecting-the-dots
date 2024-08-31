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



def simulate_robustnesscheck_data(persuasion_values, tgt_agt_centr_values, network_types):
    """
    Simulate multiple parameter combinations.

    Args:
        persuasion_values (list): List of persuasion values to simulate.
        tgt_agt_centr_values (list): List of tgt_agt_centr values to simulate.
        network_types (list): List of network types to simulate.

    Returns:
        dict: Nested dictionary with network type, tgt_agt_centr, and persuasion as keys and simulation results as values.
    """
    simulation_results_Inflation_mean = {}
    simulation_results_Inflation_std = {}

    simulation_results_InfMarketExp_mean = {}
    simulation_results_InfMarketExp_std = {}


    variables = ['Output', 'Inflation', 'Interest', 'RealInterest', 'OutTarWe', 'OutNaiveWe', 'InfTarWe', 'InfNaiveWe', 'OutMarketExp', 'InfMarketExp', 'OutAniSp', 'InfAniSp']

    # to do: Data retrieving and dictionary creation split: One dictionary per variable each + Focus on persuasion 8 
    for network_type in network_types:
        simulation_results_Inflation_mean[network_type] = {}
        simulation_results_Inflation_std[network_type] = {}

        simulation_results_InfMarketExp_mean[network_type] = {}
        simulation_results_InfMarketExp_std[network_type] = {}

        for tgt_agt_centr in tgt_agt_centr_values:
            simulation_results_Inflation_mean[network_type][tgt_agt_centr] = {}
            simulation_results_Inflation_std[network_type][tgt_agt_centr] = {}

            simulation_results_InfMarketExp_mean[network_type][tgt_agt_centr] = {}
            simulation_results_InfMarketExp_std[network_type][tgt_agt_centr] = {}

            for persuasion in persuasion_values:
                results = multiple_simulation_runner(250, Params(capT=200, agent_number=100, intervention=True, tgt_agt_centr=tgt_agt_centr, degroot=True, target=True, persuasion=persuasion, topology=network_type))
                    
                results_Inflation_mean_lst = []
                results_Inflation_std_lst = []

                results_InfMarketExp_mean_lst = []
                results_InfMarketExp_std_lst = []

                for i in range(len(results)):
                    results_Inflation_mean_lst.append(results[i][0]['Inflation'].mean())
                    results_Inflation_std_lst.append(results[i][0]['Inflation'].std())

                    results_InfMarketExp_mean_lst.append(results[i][0]['InfMarketExp'].mean())
                    results_InfMarketExp_std_lst.append(results[i][0]['InfMarketExp'].std())
                    

                simulation_results_Inflation_mean[network_type][tgt_agt_centr][persuasion] = np.array(results_Inflation_mean_lst)
                simulation_results_Inflation_std[network_type][tgt_agt_centr][persuasion] = np.array(results_Inflation_std_lst)
                simulation_results_InfMarketExp_mean[network_type][tgt_agt_centr][persuasion] = np.array(results_InfMarketExp_mean_lst)
                simulation_results_InfMarketExp_std[network_type][tgt_agt_centr][persuasion] = np.array(results_InfMarketExp_std_lst)

    return simulation_results_Inflation_mean, simulation_results_Inflation_std, simulation_results_InfMarketExp_mean, simulation_results_InfMarketExp_std
# Simulate the data

################################################################################################################
#                                           Simulating Data                                                    #
################################################################################################################

network_types = ["albert-barabasi", 'random-graph', 'watts-strogatz', 'regular-graph']
persuasion_values = np.linspace(0, 1, num=11)
tgt_agt_centr_values = [0, 4, 9, 24]

robustness_check_results_Inflation_mean,robustness_check_results_Inflation_std, robustness_check_results_InfMarketExp_mean, robustness_check_results_InfMarketExp_std = simulate_robustnesscheck_data(persuasion_values, tgt_agt_centr_values, network_types)


def save_robust_check_data(dfs):
    # Create a folder with the current date within the directory

    today = date.today().strftime("%Y-%m-%d")
    folder_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "output_data" / "sensitivity_analysis" / today
    folder_path.mkdir(parents=True, exist_ok=True)


    for i, df in enumerate(dfs):
        variable_names = ['Inflation_mean', 'Inflation_std', 'InfMarketExp_mean', 'InfMarketExp_std']
        df = pd.concat({k: pd.DataFrame(v).T for k, v in df.items()}, axis=0)

        # set the names of the row and column indexes
        df.index.names = ['network_type', 'tgt_agt_centr']
        df.columns.names = ['Persuasion']

        # use stack() to convert the dataframe from wide to long format
        df_long = df.stack(level=0).reset_index().rename(columns={"level_2": "Persuasion", 0: "Value"})

        # Assign the appropriate variable name based on the index
        if i < len(variable_names):
            globals()[f"{variable_names[i]}_long"] = df_long
            
            csv_filename = f'{variable_names[i]}_robust_check_data.csv'


            csv_filepath = folder_path / csv_filename
            df_long.to_csv(csv_filepath, index=False)

################################################################################################################
#                               Saving Dataset                                                                 #
################################################################################################################

dfs = [robustness_check_results_Inflation_mean,
       robustness_check_results_Inflation_std, 
       robustness_check_results_InfMarketExp_mean, 
       robustness_check_results_InfMarketExp_std]

save_robust_check_data(dfs)