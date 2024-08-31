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

import seaborn as sns
sns.set_theme(style="whitegrid")
sns.set_context("paper")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import ast



folder_path = Path(__file__).resolve().parent.parent.parent / "data" / "output_data" / "sensitivity_analysis" / "2024-08-07"

variable_names = ['Inflation_mean', 'Inflation_std']

# Loop through the variables and read the corresponding CSV files
for variable in variable_names:
    # Read the "mean" CSV file and assign it to a variable with the variable name
    csv_filename = f'{variable}_robust_check_data.csv'

    csv_filepath = os.path.join(folder_path, csv_filename)
    
    df = pd.read_csv(csv_filepath)


    # Remove the square brackets from the beginning and end of the string
    df['Value'] = df['Value'].str.strip('[]')
    # Split the string on whitespace and convert each element to a float
    df['Value'] = df['Value'].str.split().apply(lambda x: [float(i) for i in x])

    # Convert the "Persuasion" column to integer values
    df['Persuasion'] = df['Persuasion'].astype(float).round(1)

    globals()[f'{variable}_long'] = df



def plot_robustness_check_results(dfs_long):
    network_types = ["albert-barabasi", 'random-graph', 'watts-strogatz', 'regular-graph']
    colors = sns.color_palette(["#1E3924",  "#9A7265", "#2F6C48", "#C7BFB1", "#2F6C48"]) 
        
    linestyles = [":", "-", "-.", "--"]
    markers = ["o", "^", "D", "v"]
    
    # Create a dictionary to map tgt_agt_centr values to colors
    tgt_agt_centr_colors = {label: color for label, color in zip(['1', '5', '10', '25'], colors)}

    # Create custom legend handles
    legend_handles = [Line2D([0], [0], marker=marker, color='w', label=label, 
                             markerfacecolor=tgt_agt_centr_colors[label], markeredgecolor='black', markersize=10)
                      for marker, label in zip(markers, ['1', '5', '10', '25'])]



    #sns.color_palette(["#6BAED6",  "#2171B5", "#08519C"])
    for network in network_types: 
        for i, df_long in enumerate(dfs_long):
            # Explode the arrays in the 'Value' column
            df_exploded = df_long.explode('Value')

            # Filter the DataFrame
            data = df_exploded[df_exploded['network_type'] == f'{network}'].copy()

            # Convert the 'Value' column to numeric using .loc[row_indexer,col_indexer] = value
            data.loc[:, 'Value'] = pd.to_numeric(data['Value'])

            #ax = sns.stripplot(
            #    data=data, x="Persuasion", y="Value", hue="tgt_agt_centr", alpha=.6, jitter=True,edgecolor='gray', dodge=True, 
            #)

            # Draw the pointplot
            #sns.pointplot(x="Persuasion", y="Value", hue="tgt_agt_centr", data=data, errorbar="ci", capsize=.2,
            #            dodge=True, ax=ax)
            # Draw the pointplot
            # Create a figure and axis
            plt.figure(figsize=(12, 6)) 
            #ax = sns.pointplot(x="Persuasion", y="Value", hue="tgt_agt_centr", data=data, errorbar="ci", capsize=.1,
            #            markers=markers, linestyles=linestyles, palette=colors, dodge=.4)
            
            ax = sns.catplot(x="Persuasion", y="Value", hue="tgt_agt_centr", kind="point", errorbar="ci", capsize=.15, 
                             data=data, aspect=1.5, errwidth=1.5,markers=markers, linestyles=linestyles, palette=colors, dodge=.4, legend=False)
            if i == 0:
                ax.set(ylim=(-0.2, 0.2))
                plt.legend(handles=legend_handles, loc="lower right", frameon=True, title='Centrality Rank', prop={'size': 14}, title_fontsize=16)
            else:
                ax.set(ylim=(0.4, 1.6))
                plt.legend(handles=legend_handles, loc="lower left", frameon=True, title='Centrality Rank', prop={'size': 14}, title_fontsize=16)
            
                # Increase the size of axis descriptions
            ax.tick_params(axis='both', which='both', labelsize=14)

            plt.xlabel("Persuasion Value", fontsize=16, fontweight='bold')
            
            if i == 0:
                plt.ylabel("Mean Inflation", fontsize=16, fontweight='bold')
            else:
                plt.ylabel("Std. Dev. of Inflation", fontsize=16, fontweight='bold')  
            #plt.legend(loc="best", frameon=False, title='Targeted Agent', prop={'size': 14})



            # Save the figure
            folder_path = Path(__file__).resolve().parent.parent.parent / "reports" / "figures" / "Submission" / "sensitivity_analysis"
            folder_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
            
            if i == 0:
                figure_path = folder_path / f"{network}_inf_mean.png"
            else:
                figure_path = folder_path / f"{network}_inf_std.png"

            print(f"Figure successfully saved to {figure_path}")
            
            # Use bbox_inches='tight' to ensure the entire figure is saved
            plt.savefig(figure_path, bbox_inches='tight')
            # Show the plot
            plt.show()

dfs_long = [
    Inflation_mean_long,
    Inflation_std_long]

plot_robustness_check_results(dfs_long)

#https://stackoverflow.com/questions/56203420/how-to-use-custom-error-bar-in-seaborn-lineplot