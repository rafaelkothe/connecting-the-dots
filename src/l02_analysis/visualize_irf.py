import os
from datetime import date
from pathlib import Path
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

import matplotlib.pyplot as plt



def load_data_dict_from_csv(keys):
    today = date.today().strftime("%Y-%m-%d")
    folder_path = Path(__file__).resolve().parent.parent.parent / "data" / "output_data" / "impulse_responses" / "degroot" / "2024-08-07"
    
    data_dict = {}
    for key in keys:
        mean_irf_df = pd.read_csv(os.path.join(folder_path, f"{key}_mean_irf.csv"))
        ci_lower_df = pd.read_csv(os.path.join(folder_path, f"{key}_ci_lower.csv"))
        ci_upper_df = pd.read_csv(os.path.join(folder_path, f"{key}_ci_upper.csv"))
        
        data_dict[key] = {
            'mean_irf': mean_irf_df,
            'ci_lower': ci_lower_df,
            'ci_upper': ci_upper_df 
        }
        
    folder_path = Path(__file__).resolve().parent.parent.parent / "data" / "output_data" / "impulse_responses" / "benchmark" 
    
    benchmark_irf_df = pd.read_csv(os.path.join(folder_path, "benchmark_irf.csv"))
    
    return data_dict, benchmark_irf_df
# To load the data back
keys = ['Persuasion 0', 'Persuasion 0.3', 'Persuasion 0.5', 'Persuasion 0.7', 'Persuasion 1']
loaded_data_dict = load_data_dict_from_csv(keys)[0]
benchmark_irf_df = load_data_dict_from_csv(keys)[1]

def plot_inflation_irf(data_dict, benchmark_irf_df):
    # Get the persuasion values
    persuasion_values = list(data_dict.keys())

    # Set the color palette
    colors = sns.color_palette(["#1E3924",  "#2F6C48", "#87A898", "#C7BFB1", "#9A7265"])
    sns.set_palette(colors)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(6, 4))
    
    
        # Plot the benchmark IRF for inflation
    benchmark_inflation = benchmark_irf_df['Inflation']
    plt.plot(benchmark_inflation.index, benchmark_inflation, label='RE', color='red', linewidth=3, linestyle='--') 
    
    for persuasion_value in persuasion_values:
        mean_inflation = data_dict[persuasion_value]['mean_irf']['Inflation']
        plt.plot(mean_inflation.index, mean_inflation, label=f"$\chi$ = {persuasion_value[11:]}", linewidth=3)

    # Add labels and legend
    plt.xlabel('Period', fontsize=16, fontweight='bold')
    plt.ylabel('Impulse Response', fontsize=16, fontweight='bold')
    legend = ax.legend(loc="best", bbox_to_anchor=(0.75,0.45), frameon=True)
    
    # legend.get_title().set_fontsize(14)
    for text in legend.get_texts():
        text.set_fontsize(14)  # Adjust the font size as needed
        #text.set_fontweight('bold')  # Adjust the font weight as needed
    
    # Increase the size of axis descriptions
    ax.tick_params(axis='both', which='both', labelsize=14)
    # Set x-axis limits
    plt.xlim(40, 70)
    #sns.despine(offset={'bottom': 15}, trim=False)
    

    # Save the figure
    folder_path = Path(__file__).resolve().parent.parent.parent / "reports" / "figures" / "Submission" / "impulse_responses"
    folder_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    
    figure_path = folder_path / "inflation_irf.png"
    
    print(f"Figure successfully saved to {figure_path}")
            
    # Use bbox_inches='tight' to ensure the entire figure is saved
    plt.savefig(figure_path, bbox_inches='tight')
    # Show plot
    plt.show()

# Call the function with the loaded data
plot_inflation_irf(loaded_data_dict, benchmark_irf_df)

def plot_marketexp_irf(data_dict, benchmark_irf_df):
    # Get the persuasion values
    persuasion_values = list(data_dict.keys())

    # Set the color palette
    colors = sns.color_palette(["#1E3924",  "#2F6C48", "#87A898", "#C7BFB1", "#9A7265"])
    sns.set_palette(colors)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(6, 4))
    
    
        # Plot the benchmark IRF for inflation
    plt.axhline(y=0, color='red', linewidth= 3, linestyle='--', label='RE')
    
    for persuasion_value in persuasion_values:
        mean_inflation = data_dict[persuasion_value]['mean_irf']['InfMarketExp']
        plt.plot(mean_inflation.index, mean_inflation, label=f"$\chi$ = {persuasion_value[11:]}", linewidth=3)

    # Add labels and legend
    plt.xlabel('Period', fontsize=18, fontweight='bold')
    plt.ylabel('Impulse Response', fontsize=18, fontweight='bold')
    legend = ax.legend(loc="lower center", bbox_to_anchor=(0.9,0.45), frameon=True)
    
    # legend.get_title().set_fontsize(14)
    for text in legend.get_texts():
        text.set_fontsize(14)  # Adjust the font size as needed
        #text.set_fontweight('bold')  # Adjust the font weight as needed
    
    # Increase the size of axis descriptions
    ax.tick_params(axis='both', which='both', labelsize=14)
    # Set x-axis limits
    plt.xlim(40, 70)
    #sns.despine(offset=20, trim=False)

    # Save the figure
    folder_path = Path(__file__).resolve().parent.parent.parent / "reports" / "figures" / "Submission" / "impulse_responses"
    folder_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    
    figure_path = folder_path / "exp_irf.png"
    
    print(f"Figure successfully saved to {figure_path}")
            
    # Use bbox_inches='tight' to ensure the entire figure is saved
    plt.savefig(figure_path, bbox_inches='tight')
    # Show plot
    plt.show()
    
# Call the function with the loaded data
plot_marketexp_irf(loaded_data_dict, benchmark_irf_df)