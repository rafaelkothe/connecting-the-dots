import pandas as pd
import os
from pathlib import Path
import numpy as np 
import scipy.stats as stats
from pingouin import ttest
import seaborn as sns
import matplotlib.pyplot as plt
# Set a dark background style
sns.set_theme(style="whitegrid")
sns.set_context("paper")

# Define the variables to iterate over
variables = ['Output', 'Inflation', 'Interest', 'RealInterest', 'OutTarWe', 'OutNaiveWe', 'InfTarWe', 'InfNaiveWe', 'OutMarketExp', 'InfMarketExp', 'OutAniSp', 'InfAniSp']

# Loop through the variables and read the corresponding CSV files
for variable in variables:
    # Read the "mean" CSV file and assign it to a variable with the variable name
    csv_filename = f'{variable}_mean_data.csv'
    csv_filepath = Path(__file__).resolve().parent.parent.parent / "data" / "output_data" / "target_interventions" / "2024-08-06" / csv_filename
    globals()[f'simulated_data_{variable}_mean_df'] = pd.read_csv(csv_filepath)

    # Read the "std" CSV file and assign it to a variable with the variable name
    csv_filename = f'{variable}_std_data.csv'
    csv_filepath = Path(__file__).resolve().parent.parent.parent / "data" / "output_data" / "target_interventions" / "2024-08-06" / csv_filename
    globals()[f'simulated_data_{variable}_std_df'] = pd.read_csv(csv_filepath)
    
    
################################################################################################################
#                                    Process simulated Data                                                    #
################################################################################################################

def process_simulated_data(persuasion_values, tgt_agt_centr_values, error_method='t-statistic', variable='InfTarWe', stat_method='mean'):
    results = []
    
    for persuasion in persuasion_values:
        # Retrieve data_benchmark for the current persuasion
        if stat_method == 'mean':
            data_benchmark =  globals()[f'simulated_data_{variable}_mean_df']  # Example data
            data_benchmark = data_benchmark[data_benchmark['Persuasion'] == f'Persuasion {persuasion}']['Benchmark'] 

        elif stat_method == 'std':
            data_benchmark =  globals()[f'simulated_data_{variable}_std_df']  # Example data
            data_benchmark = data_benchmark[data_benchmark['Persuasion'] == f'Persuasion {persuasion}']['Benchmark'] 

        #data_benchmark_kurtosis = np.mean([stats.kurtosis(simulation_results_benchmark[persuasion][i][0][variable][25:]) for i in range(len(simulation_results_benchmark[persuasion]))])
        #data_benchmark_skewness = np.mean([stats.skew(simulation_results_benchmark[persuasion][i][0][variable][25:]) for i in range(len(simulation_results_benchmark[persuasion]))])
        n = len(data_benchmark)  # Sample size should be calculated for all cases
        column_name_1 = 'Mean'
        column_name_2 = 'Lower_CI'
        column_name_3 = 'Upper_CI'
        
        # Calculate statistics for data_benchmark
        if stat_method == 'mean':
            mean = np.mean(data_benchmark) #adjust for sdt()
            std = np.std(data_benchmark) 
        elif stat_method == 'std':
            mean = np.sqrt(np.mean(np.power(data_benchmark, 2), axis=0))
            variance = sum([((x - mean) ** 2) for x in data_benchmark]) / len(data_benchmark)
            std = variance ** 0.5

        if error_method == "t-statistic":
            # Calculate statistics using t-statistic
            t_critical = stats.t.ppf(0.975, df=n-1)  # 95% confidence level, two-tailed
            margin_of_error = t_critical * (std / np.sqrt(n))
            lower_ci = mean - margin_of_error
            upper_ci = mean + margin_of_error
        
        elif error_method == "std-dev":
            lower_ci = mean - std
            upper_ci = mean + std

        elif error_method == "std-error":
            # Calculate statistics using standard error
            margin_of_error = std / np.sqrt(n)
            lower_ci = mean - margin_of_error
            upper_ci = mean + margin_of_error

        elif error_method == "quartiles":
            # Calculate statistics using quartiles
            lower_perc = np.percentile(data_benchmark, 25)
            median = np.percentile(data_benchmark, 50)
            upper_perc = np.percentile(data_benchmark, 75)
        
            column_name_1 = 'Median'
            column_name_2 = 'Lower_Perc'
            column_name_3 = 'Upper_Perc'

        results.append({
                'Persuasion': persuasion,
                'Tgt_agt_centr': "Benchmark",
                column_name_1: mean if error_method != "quartiles" else median,
                column_name_2: lower_ci if error_method != "quartiles" else lower_perc,
                column_name_3: upper_ci if error_method != "quartiles" else upper_perc,
                'n': n,
                'SD': std,
                'CV': (std / mean) * 100 if mean != 0 else 0,
                't-statistic': None,
                'p-Value': None,
                'CI Width': None, 
                'Effect Size': None,
                'Power': None
            })

        for tgt_agt_centr in tgt_agt_centr_values:
            # Simulate data (replace with your data generation logic)

            if stat_method == 'mean':
                data =  globals()[f'simulated_data_{variable}_mean_df']  # Example data
                data = data[data['Persuasion'] == f'Persuasion {persuasion}'][f'{tgt_agt_centr}']

            elif stat_method == 'std':
                data =  globals()[f'simulated_data_{variable}_std_df']  # Example data
                data = data[data['Persuasion'] == f'Persuasion {persuasion}'][f'{tgt_agt_centr}']
            
            #data_kurtosis = np.mean([stats.kurtosis(simulation_results[(persuasion, tgt_agt_centr)][i][0][variable][25:]) for i in range(len(simulation_results_benchmark[persuasion]))])
            #data_skewness = np.mean([stats.skew(simulation_results[(persuasion, tgt_agt_centr)][i][0][variable][25:]) for i in range(len(simulation_results_benchmark[persuasion]))])
            n = len(data)  # Sample size should be calculated for all cases
            column_name_1 = 'Mean'
            column_name_2 = 'Lower_CI'
            column_name_3 = 'Upper_CI'

            if stat_method == 'mean':
                mean = np.mean(data) #adjust for sdt()
                std = np.std(data) 
            elif stat_method == 'std':
                mean = np.sqrt(np.mean(np.power(data, 2), axis=0))
                variance = sum([((x - mean) ** 2) for x in data]) / len(data)
                std = variance ** 0.5

            if error_method == "t-statistic":
                # Calculate statistics using t-statistic
                t_critical = stats.t.ppf(0.975, df=n-1)  # 95% confidence level, two-tailed
                margin_of_error = t_critical * (std / np.sqrt(n))
                lower_ci = mean - margin_of_error
                upper_ci = mean + margin_of_error
            
            elif error_method == "std-dev":
                lower_ci = mean - std
                upper_ci = mean + std

            elif error_method == "std-error":
                # Calculate statistics using standard error
                margin_of_error = std / np.sqrt(n)
                lower_ci = mean - margin_of_error
                upper_ci = mean + margin_of_error

            elif error_method == "quartiles":
                # Calculate statistics using quartiles
                lower_perc = np.percentile(data, 25)
                median = np.percentile(data, 50)
                upper_perc = np.percentile(data, 75)
            
                column_name_1 = 'Median'
                column_name_2 = 'Lower_Perc'
                column_name_3 = 'Upper_Perc'

            difference_test = ttest(data_benchmark, data, correction = False)
      
            results.append({
                'Persuasion': persuasion,
                'Tgt_agt_centr': tgt_agt_centr,
                column_name_1: mean if error_method != "quartiles" else median,
                column_name_2: lower_ci if error_method != "quartiles" else lower_perc,
                column_name_3: upper_ci if error_method != "quartiles" else upper_perc,
                'n': n,
                'SD': std,
                'CV': (std / mean) * 100 if mean != 0 else 0,
                't-statistic': difference_test['T'],
                'p-Value': difference_test['p-val'],
                'CI Width': difference_test['CI95%'][0][1] - difference_test['CI95%'][0][0], 
                'Effect Size': difference_test['cohen-d'],
                'Power': difference_test['power']
            })

    
    return pd.DataFrame(results)


# Define values for persuasion and tgt_agt_centr
persuasion_values_grid = [0, 0.3, 0.5, 0.7, 1]
tgt_agt_centr_values_grid = [0, 4, 9, 24]

# Define error calculation method ("t-statistic", "std-dev", "std-error", or "quartiles")

"""

['Output', 'Inflation', 'Interest', 'RealInterest', 'OutTarWe',
       'OutNaiveWe', 'InfTarWe', 'InfNaiveWe', 'OutMarketExp', 'InfMarketExp',
       'OutAniSp', 'InfAniSp']

"""

error_method = "t-statistic"  # Uncomment the desired method
variable = 'InfTarWe'
stat_method = 'mean'

# Generate simulated data
simulated_data_InfTarWe_mean = process_simulated_data(persuasion_values_grid, tgt_agt_centr_values_grid, error_method, variable, stat_method)



################################################################################################################

error_method = "t-statistic"  # Uncomment the desired method
variable = 'Inflation'
stat_method = 'std'

# Generate simulated data
simulated_data_Inflation_std = process_simulated_data(persuasion_values_grid, tgt_agt_centr_values_grid, error_method, variable, stat_method)


################################################################################################################
#                                      Error Bar Plots Data                                                #
################################################################################################################

class ErrorBarPlot:
    def __init__(self, simulated_data, persuasion_values_grid, error_method, option):
        self.simulated_data = simulated_data
        self.persuasion_values_grid = persuasion_values_grid
        self.error_method = error_method
        self.option = option    
    
    def create_summary_bar_plot(self):
        tgt_agt_centr_values_grid = ["Benchmark", 0, 4, 9, 24]
        # Define a Seaborn muted color palette
        colors = sns.color_palette(["#1E3924",  "#2F6C48", "#87A898", "#C7BFB1", "#9A7265"]) 
        
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))

        # Set up positions for bars
        positions = np.arange(len(tgt_agt_centr_values_grid))

        # Set up width for bars
        width = 0.15

        # Iterate through persuasion values and create barplots
        for i, persuasion in enumerate(self.persuasion_values_grid):
            # Create an empty list to store bar heights for each tgt_agt_centr
            bar_heights = []
            error_bands = []  # Store error bands for each bar
            
            for tgt_agt_centr in tgt_agt_centr_values_grid:
                # Extract statistics from simulated data
                data_subset = self.simulated_data[(self.simulated_data['Persuasion'] == persuasion) & (self.simulated_data['Tgt_agt_centr'] == tgt_agt_centr)]
                if data_subset.shape[0] > 0:  # Check if there is data in data_subset
                    if self.error_method == "quartiles":
                        median = data_subset['Median'].values[0]
                        lower_ci = data_subset['Lower_Perc'].values[0]
                        upper_ci = data_subset['Upper_Perc'].values[0]
                        # Append the mean to bar_heights
                        bar_heights.append(median)
                    else:
                        mean = data_subset['Mean'].values[0]
                        lower_ci = data_subset['Lower_CI'].values[0]
                        upper_ci = data_subset['Upper_CI'].values[0]
                        # Append the mean to bar_heights
                        bar_heights.append(mean)

                    # Calculate error (confidence interval)
                    error = (abs(upper_ci) - abs(lower_ci)) / 2
                    error_bands.append(error)
                else:
                    # Handle the case where there is no data for the specified conditions
                    bar_heights.append(0)
                    error_bands.append(0)
            
            # Calculate the position for the current persuasion
            x = positions + i * width
            
            # Create the bar plot
            bars = ax.bar(x, bar_heights, width=width, label=f'$\\chi = {persuasion}$', color=colors[i])
            
            
            # Add dots at the end of the bar plots
            dots = ax.scatter(x, bar_heights, color="grey", marker='.', s=200)
            
            # Connect the dots by a line
            for j in range(len(x) - 1):
                ax.plot([x[j], x[j+1]], [bar_heights[j], bar_heights[j+1]], linestyle='--', color=colors[i])
            
            # Add error bars (boxplot-like whiskers)
            for j, (output, error) in enumerate(zip(bar_heights, error_bands)):
                ax.errorbar(x[j], output, yerr=error, c='black', capsize=10)
            
        # Set x-axis labels as tgt_agt_centr values
        ax.set_xticks(positions + (len(self.persuasion_values_grid) - 1) * width / 2)
        ax.set_xticklabels(["Benchmark", "1st", "5th", "10th", "25th"])

        # Increase the size of axis descriptions
        ax.tick_params(axis='both', which='both', labelsize=14)

        # Set x and y labels
        ax.set_xlabel('Target Centrality', fontsize=16, fontweight='bold')
        #ax.xaxis.set_label_coords(0.5, -0.08)
        

        if self.option == 'targeters':
            ax.set_ylabel('% Share of Targeters', fontsize=16, fontweight='bold')
        else:
            ax.set_ylabel('Std. Dev. of Inflation', fontsize=16, fontweight='bold')



        # Add legend on the upper center of the figure
        legend = ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.12), fancybox=True, shadow=True, ncol=5)
        for text in legend.get_texts():
            text.set_fontsize(16)  # Adjust the font size as needed
            
        # Remove the top and right spines and offset only the x-axis
        sns.despine(offset={'bottom': 60}, trim=True)  # Offset the x-axis by 10

        # Save the figure
        folder_path = Path(__file__).resolve().parent.parent.parent / "reports" / "figures" / "Submission" / "target_interventions"
        folder_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
            
        if self.option == "targeters":
            figure_path = folder_path / "errorbar_targeters_tgt.png"
        else:
            figure_path = folder_path / "errorbar_std_tgt.png"

        print(f"Figure successfully saved to {figure_path}")
            
        # Use bbox_inches='tight' to ensure the entire figure is saved
        plt.savefig(figure_path, bbox_inches='tight')

        # Show the plot
        plt.show()


# Example usage: InfTarWe
summary_bar_plot = ErrorBarPlot(simulated_data_InfTarWe_mean, persuasion_values_grid, option="targeters", error_method="t-statistic")
summary_bar_plot.create_summary_bar_plot()

# Example usage: OutTarWe
#summary_bar_plot = SummaryBarPlot(simulated_data_OutTarWe, persuasion_values_grid, error_method)
#summary_bar_plot.create_summary_bar_plot()

# Example usage: Inflation
summary_bar_plot = ErrorBarPlot(simulated_data_Inflation_std, persuasion_values_grid, error_method="t-statistic", option="volatility")
summary_bar_plot.create_summary_bar_plot()

# Example usage: Output
#summary_bar_plot = SummaryBarPlot(simulated_data_Output, persuasion_values_grid, error_method)
#summary_bar_plot.create_summary_bar_plot()



################################################################################################################
#                                      Boxplots Data                                                #
################################################################################################################

class Custom_Boxplots:
    def __init__(self, data_df, palette=None, option=None):
        """
        Initialize a CustomRaincloudPlot object.

        Parameters:
        - data_dict (dict): A nested dictionary containing data organized by groups and subgroups.
        - palette (str or dict): Optional. Color palette to use for plotting. If None, the default Seaborn palette is used.
        """
        self.data_df = data_df
        self.palette = sns.color_palette(["#1E3924",  "#2F6C48", "#87A898", "#C7BFB1", "#9A7265"]) 
        self.option = option
        

    def create_plot(self):
        """
        Create and display a Boxplot.

        """

        fig, ax = plt.subplots(figsize=(12, 8))  # Create a figure and axis
        
        dataframes = []
        # Prepare the data for plotting by creating individual DataFrames for each subgroup within groups: 
        for group in self.data_df.columns[2:]:
            for subgroup in self.data_df['Persuasion'].unique():
                data = self.data_df[self.data_df['Persuasion'] == f'{subgroup}'][f'{group}']
                df = pd.DataFrame({'Group': [group] * len(data), 'Value': data, 'Subgroup': [subgroup] * len(data)})
                dataframes.append(df)

        combined_df = pd.concat(dataframes)

        # Create the box plot to show quartiles and median of the data
        sns.boxplot(
            y="Value", x="Group", hue='Subgroup', data=combined_df, orient='v', saturation=1, showfliers=True, 
            width =0.8, boxprops={'alpha' : .9}, dodge=True,  palette=self.palette, ax=ax
        )
        sns.despine(offset=10, trim=True)

        # Adjust the x-ticks for the groups
        unique_groups = combined_df['Group'].unique()
        ax.set_xticks(np.arange(len(unique_groups)))
        ax.set_xticklabels(["Benchmark", "1st", "5th", "10th", "25th"])

        # Increase the size of axis descriptions
        ax.tick_params(axis='both', which='both', labelsize=14)

        # Add labels to the boxplots
        handles, labels = ax.get_legend_handles_labels()
        labels = [label[10:] for label in labels]
        new_labels = [f'$\\chi = {label.split("=")[-1].strip()}$' for label in labels]
        legend = ax.legend(handles, new_labels, loc='upper center', bbox_to_anchor=(0.5, 0.13), fancybox=True, shadow=True, ncol=len(new_labels))


        for text in legend.get_texts():
            text.set_fontsize(16)  # Adjust the font size as needed
        ax.set_xlabel("Target Centrality", fontsize=16, fontweight='bold')
        if self.option == "mean":
            ax.set_ylabel("Mean Market Sentiment", fontsize=16, fontweight='bold')
        else:
            ax.set_ylabel("Std. Dev. of Expectations", fontsize=16, fontweight='bold')


        plt.margins(y=0.2)
        
        # Save the figure
        folder_path = Path(__file__).resolve().parent.parent.parent / "reports" / "figures" / "Submission" / "target_interventions"
        folder_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
            
        if self.option == "mean":
            figure_path = folder_path / "boxplots_targeters_tgt.png"
        else:
            figure_path = folder_path / "boxplots_std_tgt.png"

        print(f"Figure successfully saved to {figure_path}")
            
        # Use bbox_inches='tight' to ensure the entire figure is saved
        plt.savefig(figure_path, bbox_inches='tight')
        plt.show()

        return
    
    ################################################################################################################
#                                           Boxplots                                                           #
################################################################################################################


# Create and use the CustomRaincloudPlot class
#custom_palette = sns.color_palette(["#6BAED6",  "#2171B5", "#08519C"])
boxplot_1 = Custom_Boxplots(data_df=simulated_data_InfAniSp_mean_df, palette=None, option="mean")
boxplot_2 = Custom_Boxplots(data_df=simulated_data_InfMarketExp_std_df, palette=None)

boxplot_1.create_plot() 
boxplot_2.create_plot() 
#simulated_data_InfMarketExp_std_df