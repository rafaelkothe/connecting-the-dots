import pandas as pd
import os
from pathlib import Path
import numpy as np 
import scipy.stats as stats
from pingouin import ttest

import matplotlib.pyplot as plt
from tabulate import tabulate



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


################################################################################################################
#                        Process simulated Data (Target Intervention)                                          #
################################################################################################################

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
InfTarWe_mean_sim_data_tgt = process_simulated_data(persuasion_values_grid, tgt_agt_centr_values_grid, error_method, variable, stat_method)


################################################################################################################

error_method = "t-statistic"  # Uncomment the desired method
variable = 'Inflation'
stat_method = 'std'

# Generate simulated data
InfTarWe_std_sim_data_tgt = process_simulated_data(persuasion_values_grid, tgt_agt_centr_values_grid, error_method, variable, stat_method)

################################################################################################################
#                        Process simulated Data (Naive Intervention)                                           #
################################################################################################################

# Define the variables to iterate over
variables = ['Output', 'Inflation', 'Interest', 'RealInterest', 'OutTarWe', 'OutNaiveWe', 'InfTarWe', 'InfNaiveWe', 'OutMarketExp', 'InfMarketExp', 'OutAniSp', 'InfAniSp']

# Loop through the variables and read the corresponding CSV files
for variable in variables:
    # Read the "mean" CSV file and assign it to a variable with the variable name
    csv_filename = f'{variable}_mean_data.csv'
    csv_filepath = Path(__file__).resolve().parent.parent.parent / "data" / "output_data" / "naive_interventions" / "2024-08-06" / csv_filename
    globals()[f'simulated_data_{variable}_mean_df'] = pd.read_csv(csv_filepath)

    # Read the "std" CSV file and assign it to a variable with the variable name
    csv_filename = f'{variable}_std_data.csv'
    csv_filepath = Path(__file__).resolve().parent.parent.parent / "data" / "output_data" / "naive_interventions" / "2024-08-06" / csv_filename
    globals()[f'simulated_data_{variable}_std_df'] = pd.read_csv(csv_filepath)
    

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
InfTarWe_mean_sim_data_naive = process_simulated_data(persuasion_values_grid, tgt_agt_centr_values_grid, error_method, variable, stat_method)


################################################################################################################

error_method = "t-statistic"  # Uncomment the desired method
variable = 'Inflation'
stat_method = 'std'

# Generate simulated data
InfTarWe_std_sim_data_naive = process_simulated_data(persuasion_values_grid, tgt_agt_centr_values_grid, error_method, variable, stat_method)


################################################################################################################
#                                      Summary Tables                                                          #
################################################################################################################


class SummaryTableGenerator:
    def __init__(self, simulated_data, error_method):
        self.simulated_data = simulated_data
        self.error_method = error_method

    def generate_summary_table(self):
        if self.error_method == "quartiles":
            self._generate_quartiles_summary_table()
        else:
            self._generate_default_summary_table()

        self._round_summary_table()
        self._apply_custom_order()
        self._add_stars_to_effect_size()
        self._sort_summary_table()
        self._switch_columns()
        self._format_summary_table()

    def _generate_quartiles_summary_table(self):
        self.summary_table = self.simulated_data.groupby(['Persuasion', 'Tgt_agt_centr']).agg({
            'Median ': 'mean',
            'Lower_Perc': 'mean',
            'Upper_Perc': 'mean',
            'n': 'mean',
            'SD': 'mean',
            'CV': 'mean',
            't-statistic': 'mean',
            'p-Value': 'mean',
            'CI Width': 'mean',
            'Effect Size': 'mean',
            'Power': 'mean'
        }).reset_index()

    def _generate_default_summary_table(self):
        self.summary_table = self.simulated_data.groupby(['Persuasion', 'Tgt_agt_centr']).agg({
            'Mean': 'mean',
            'Lower_CI': 'mean',
            'Upper_CI': 'mean',
            'n': 'mean',
            'SD': 'mean',
            'CV': 'mean',
            't-statistic': 'mean',
            'p-Value': 'mean',
            'CI Width': 'mean',
            'Effect Size': 'mean',
            'Power': 'mean'
        }).reset_index()

    def _round_summary_table(self):
        self.summary_table = self.summary_table.round(4)

    def _apply_custom_order(self):
        custom_order = ["Benchmark", 0, 4, 9, 24]
        custom_order_2 = [0, 0.3, 0.5, 0.7, 1]

        self.summary_table['Tgt_agt_centr'] = pd.Categorical(self.summary_table['Tgt_agt_centr'], categories=custom_order, ordered=True)
        self.summary_table['Persuasion'] = pd.Categorical(self.summary_table['Persuasion'], categories=custom_order_2, ordered=True)

    def _add_stars_to_effect_size(self):
        # Convert 'Effect Size' column to string format
        self.summary_table['Effect Size'] = self.summary_table['Effect Size'].astype(str)

        self.summary_table['Effect Size'] = self.summary_table.apply(lambda row: row['Effect Size'] + ' ***' if row['p-Value'] < 0.001 else (row['Effect Size'] + ' **' if row['p-Value'] < 0.01 else (row['Effect Size'] + ' *' if row['p-Value'] < 0.05 else row['Effect Size'])), axis=1)

    def _sort_summary_table(self):
        self.summary_table = self.summary_table.sort_values(by=['Persuasion', 'Tgt_agt_centr'])

    def _switch_columns(self):
        self.summary_table = self.summary_table[['Tgt_agt_centr', 'Persuasion'] + [col for col in self.summary_table.columns if col not in ['Tgt_agt_centr', 'Persuasion']]]

    def _format_summary_table(self):
        self.summary_table_str = tabulate(self.summary_table, headers='keys', tablefmt='pretty')

    def print_summary_table(self):
        print("\nSummary Table:")
        print(self.summary_table_str)




################################################################################################################
#                        Summary Table (Target Intervention)                                                   #
################################################################################################################


summary_table_generator = SummaryTableGenerator(InfTarWe_std_sim_data_tgt, error_method="t-statistic")
summary_table_generator.generate_summary_table()
summary_table_generator.print_summary_table()

################################################################################################################
#                        Summary Table (Naive Intervention)                                                   #
################################################################################################################


summary_table_generator = SummaryTableGenerator(InfTarWe_std_sim_data_naive, error_method="t-statistic")
summary_table_generator.generate_summary_table()
summary_table_generator.print_summary_table()