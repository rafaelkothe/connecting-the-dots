import time
from typing import List, Any

from l00_package.l00_parameters.parameters import Params
from l00_package.l01_grids.environment import Grids
from l00_package.l03_model.model import Model

def multiple_simulation_runner(num_simulations: int, model_parameters) -> List[Any]:
    """
    Run multiple simulations with different seeds and parameters.

    Args:
        num_simulations (int): Number of simulations to run.
        model (Any): Model class.
        environment (Any): Environment class.
        model_parameters (Optional[Dict[str, Any]]): Parameters for the model.

    Returns:
        List[Any]: List of simulation results and corresponding simulation times.
    """
    simulation_results = []

    for simulation_counter in range(num_simulations):
        # Record the start time
        start_time = time.time()
        model_instance = Model(p=model_parameters, g=Grids(simulation_params=model_parameters, random_seed=simulation_counter))
   
        
        results = model_instance.simulate(seed=simulation_counter)

        # Calculate simulation time
        simulation_time = time.time() - start_time
        print("Simulation time (in seconds):", simulation_time)

        simulation_results.append(results)

    return simulation_results

