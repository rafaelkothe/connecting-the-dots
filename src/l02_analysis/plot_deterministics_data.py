import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import date

import seaborn as sns

sns.set_theme(style="whitegrid")
sns.set_context("paper")
class Deterministics_Plotter:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)

    def plot_results(self, ax, x, y, label, title):
        ax.plot(x, y, 'b.', ms=0.6, label=label)
        ax.set_xlabel('$Chi$', fontsize=16)
        ax.set_ylabel(title, fontsize=16)
        ax.set_title(title)

    def plot_figures(self):
        for network_type in self.df['network_type'].unique():
            fig_inflation, ax_inflation = plt.subplots()
            fig_output, ax_output = plt.subplots()
            fig_target_inflation, ax_target_inflation = plt.subplots()
            fig_target_output, ax_target_output = plt.subplots()

            network_df = self.df[self.df['network_type'] == network_type]

            for persuasion in np.arange(0, 1, 0.1):
                persuasion_df = network_df[network_df['persuasion'] == persuasion]

                inflation = persuasion_df['inflation']
                output = persuasion_df['output']
                target_inflation = persuasion_df['target_inflation']
                target_output = persuasion_df['target_output']

                self.plot_results(ax_inflation, [persuasion] * len(inflation), inflation, f'Chi={persuasion}', f'{network_type} - Inflation')
                self.plot_results(ax_output, [persuasion] * len(output), output, f'b={persuasion}', f'{network_type} - Output')
                self.plot_results(ax_target_inflation, [persuasion] * len(target_inflation), target_inflation, f'Chi={persuasion}', f'{network_type} - Inflation')
                self.plot_results(ax_target_output, [persuasion] * len(target_output), target_output, f'Chi={persuasion}', f'{network_type} - Output')

            today = date.today().strftime("%Y-%m-%d")
            folder_path = Path(__file__).resolve().parent.parent.parent / "reports" / "figures" / "Submission" / "Deterministics"
            folder_path.mkdir(parents=True, exist_ok=True)

            file_path_inflation = folder_path / f"deterministics_inflation_{network_type}.png"
            file_path_output = folder_path / f"deterministics_output_{network_type}.png"
            file_path_target_inflation = folder_path / f"deterministics_target_inflation_{network_type}.png"
            file_path_target_output = folder_path / f"deterministics_target_output_{network_type}.png"

            fig_inflation.savefig(file_path_inflation)
            fig_output.savefig(file_path_output)
            fig_target_inflation.savefig(file_path_target_inflation)
            fig_target_output.savefig(file_path_target_output)

        
# Specify the file path
#file_path = Path("C:/Users/ba1bc7/OneDrive/0 - Promotion/Projects/01_Agent-Based Models/02_Running/connecting_the_dots/data/output_data/Deterministics/2024-05-04")

file_path = Path(__file__).resolve().parent.parent.parent / "data" / "output_data" / "Deterministics" / "2024-08-07" / "deterministics_data.csv"

# Create a Deterministics_Plotter object
plotter = Deterministics_Plotter(file_path)

plotter.plot_figures()