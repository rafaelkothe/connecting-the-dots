import numpy as np
import pandas as pd
from datetime import date
from typing import List, Tuple
from pathlib import Path
from l00_package import Model
from l00_package import Params
from l00_package import Grids
from l00_package import multiple_simulation_runner

import matplotlib.pyplot as plt


class DeterministicsSimRunner:
    def __init__(self):
        self.network_types: List[str] = ["albert-barabasi", 'random-graph', 'watts-strogatz', 'regular-graph']

    def run_simulation(self, persuasion: float, network_type: str, num_simulations: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        model_parameters = Params(capT=1000,
                                  agent_number=100,
                                  tgt_agt_centr=0,
                                  sigma_inflation=0,
                                  sigma_output=0,
                                  sigma_taylor=0,
                                  randomness=True,
                                  inflation_target=0.5,
                                  persuasion=persuasion,
                                  degroot=True,
                                  target=False,
                                  topology=network_type)

        results = multiple_simulation_runner(num_simulations, model_parameters)[0][0]

        inflation = np.round(results.Inflation[-10:], 3)
        output = np.round(results.Output[-10:], 3)
        target_inflation = np.round(results.InfTarWe[-10:], 3)
        target_output = np.round(results.OutTarWe[-10:], 3)

        return inflation, output, target_inflation, target_output

    def run_multiple_simulations(self):
        df = pd.DataFrame()

        for network_type in self.network_types:
            for persuasion in np.arange(0, 1, 0.1):
                inflation, output, target_inflation, target_output = self.run_simulation(persuasion, network_type, 1)

                temp_df = pd.DataFrame({
                    'inflation': inflation,
                    'output': output,
                    'target_inflation': target_inflation,
                    'target_output': target_output,
                    'persuasion': persuasion,
                    'network_type': network_type,
                })

                df = pd.concat([df, temp_df], ignore_index=True)

        today = date.today().strftime("%Y-%m-%d")
        folder_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "output_data" / "Deterministics" / today
        folder_path.mkdir(parents=True, exist_ok=True)
        file_path = folder_path / "deterministics_data.csv"

        df.to_csv(file_path, index=False)
        
sim_runner = DeterministicsSimRunner()
sim_runner.run_multiple_simulations()
