import numpy as np
import pandas as pd
from datetime import date
from typing import List, Tuple
from pathlib import Path
from l00_package import Model
from l00_package import Params
from l00_package import Grids
from l00_package import multiple_simulation_runner
import seaborn as sns
from pathlib import Path

import matplotlib.pyplot as plt

targets = [0]

class TimeseriesSimRunner:
    def run_simulation(self, target: int, num_simulations: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        model_parameters = Params(capT=250,
                                  agent_number=100,
                                  intervention=False,
                                  tgt_agt_centr=target,
                                  sigma_inflation=0.5,
                                  sigma_output=0.5,
                                  sigma_taylor=0.5,
                                  randomness=True,
                                  inflation_target=0,
                                  persuasion=0.5,
                                  degroot=True,
                                  topology="albert-barabasi")

        results = multiple_simulation_runner(num_simulations, model_parameters)[1][0]

        inflation = np.round(results.Inflation, 3)
        output = np.round(results.Output, 3)
        market_inflation = np.round(results.InfMarketExp, 3)
        market_output = np.round(results.OutMarketExp, 3)

        return inflation, output, market_inflation, market_output


    def run_multiple_simulations(self):
        df = pd.DataFrame()

        
        for target in targets:
                inflation, output, market_inflation, market_output = self.run_simulation(target, 2)
                # Create the DataFrame
                temp_df = pd.DataFrame({
                    'inflation': inflation,
                    'output': output,
                    'market_inflation': market_inflation,
                    'market_output': market_output,
                    'fe_inflation': market_inflation - inflation,
                    'fe_output': market_output - output,
                    'target': target
                })

                df = pd.concat([df, temp_df], ignore_index=True)

        today = date.today().strftime("%Y-%m-%d")
        folder_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "output_data" / "time_series" / today
        folder_path.mkdir(parents=True, exist_ok=True)
        file_path = folder_path / "time_series_data.csv"

        df.to_csv(file_path, index=False)
        
        return df
        
sim_runner = TimeseriesSimRunner()
test = sim_runner.run_multiple_simulations()
def plot_time_series(test, option):
    for i, target in enumerate(targets):
        
        if option == "inflation":
            price = test[test['target'] == target]['inflation'].values
            predictions = test[test['target'] == target]['market_inflation'].values
            inset_data = test[test['target'] == target]['fe_inflation'].values
            
            # Create figure and subplots with shared x-axis
            fig, axs = plt.subplots(2, 1, figsize=(14, 6), gridspec_kw={'height_ratios': [2, 2]}, sharex=True)
            # Plot price
            axs[0].plot(price, 'r-', linewidth=2, color='#1E3924')
            #axs[0].set_title('Group 4')
            axs[0].set_ylabel('Inflation', fontsize=16, fontweight='bold')
            axs[0].grid(True)

            # Plot predictions
            axs[1].plot(predictions, linewidth=1, color='#9A7265')
            axs[1].set_ylabel('Market Expectations', fontsize=16, fontweight='bold')
            axs[1].grid(True)
            # Set y limits
            axs[1].set_ylim(-8, None)
            axs[1].set_xlabel('Period', fontsize=16, fontweight='bold')
        else: # option == "output"
            price = test[test['target'] == target]['output'].values
            predictions = test[test['target'] == target]['market_output'].values
            inset_data = test[test['target'] == target]['fe_output'].values
            
            # Create figure and subplots with shared x-axis
            fig, axs = plt.subplots(2, 1, figsize=(14, 6), gridspec_kw={'height_ratios': [2, 2]}, sharex=True)
            # Plot price
            axs[0].plot(price, 'r-', linewidth=2, color='#1E3924')
            #axs[0].set_title('Group 4')
            axs[0].set_ylabel('Output', fontsize=16, fontweight='bold')
            axs[0].grid(True)

            # Plot predictions
            axs[1].plot(predictions, linewidth=1, color='#9A7265')
            axs[1].set_ylabel('Market Expectations', fontsize=16, fontweight='bold')
            axs[1].grid(True)
            # Set y limits
            axs[1].set_ylim(-8, None)
            axs[1].set_xlabel('Period', fontsize=16, fontweight='bold')

        # Create inset plot in the lower right corner
        inset_ax = fig.add_axes([0.38, 0.14, 0.6, 0.15])  # [left, bottom, width, height] in normalized figure coordinates
        inset_ax.plot(inset_data, 'k-', linewidth=1)
        #inset_ax.set_ylim(-4, 4)
        inset_ax.set_xlabel('')
        inset_ax.set_ylabel('Forecast Error', color='black', fontweight='bold')
        inset_ax.grid(True)
        inset_ax.set_xticks([])  # Exclude x ticks

        # Hide x-axis labels for the price plot
        plt.setp(axs[0].get_xticklabels(), visible=False)

        # Adjust layout
        plt.tight_layout()

    
        # Save the figure
        folder_path = Path(__file__).resolve().parent.parent.parent.parent / "reports" / "figures" / "Submission" / "time_series"
        print(folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

        if option == "inflation":
            figure_path = folder_path / "time_series_inflation.png"
        else: # option == "output"
            figure_path = folder_path / "time_series_output.png"

        print(f"Figure successfully saved to {figure_path}")

        # Use bbox_inches='tight' to ensure the entire figure is saved
        plt.savefig(figure_path, bbox_inches='tight')

        # Show plot
        plt.show()

plot_time_series(test, "inflation")
plot_time_series(test, "output")
