import math
import numpy as np
import pandas as pd


import matplotlib.pyplot as plt

from l00_package.l01_grids.environment import Grids
from l00_package.l00_parameters.parameters import Params




#TODO : Define utility functions for the model using the decorator @staticmethod
class Model(object):

    def __init__(self, p=Params(), g=None):
        if g is None:
            g = Grids(p)

        self.p = p
        self.g = g

        # Expose the create_network and initialize_matrices functions
        self.create_network = self.g._create_network
        self.initialize_matrices = self.g._initialize_matrices

        if self.p.degroot:
            # node-specific parameters
            self.degree_centrality = self.g.degree_centrality(self.g.agent_network)


            self.clustering = self.g.clustering(self.g.agent_network)

            # network-specific parameters
            self.average_clustering = self.g.average_clustering(self.g.agent_network)

            if self.p.target:
                self.p.persuasion[int(self.degree_centrality.node)] = 0


    """ Expectation Formation Methods """

    def calculate_attractivity(self, lagged_value, lag2_expected_value, heuristic_attraction):
        """
        Calculate the attractivity based on lagged values, lag-2 expected values, and heuristic attraction.

        Parameters:
        lagged_value (float): The lagged variable value.
        lag2_expected_value (float): The expected variable value with a lag of 2.
        heuristic_attraction (float): The heuristic attraction factor.

        Returns:
        float: The calculated attractivity value.
        """
        # Calculate the squared difference between lagged_value and lag2_expected_value
        squared_difference = (lagged_value - lag2_expected_value) ** 2

        # Calculate the attractivity using the squared difference and the heuristic attraction
        attractivity = -squared_difference + self.p.zeta * heuristic_attraction
       
        return attractivity


    def calculate_switching_probabilities(self, attractiveness_i, attractiveness_j):
        """
        Calculate the switching probabilities between two options based on their attractiveness.

        Args:
            attractiveness_i (float): Attractiveness of option i.
            attractiveness_j (float): Attractiveness of option j.

        Returns:
            numpy.ndarray: An array containing the switching probabilities for option i and option j.
        """
        # Calculate the exponentiated attractiveness values
        exp_attractiveness_i = math.exp(self.p.theta * attractiveness_i)
        exp_attractiveness_j = math.exp(self.p.theta * attractiveness_j)
    
        
        # Calculate the probabilities
        probability_i = exp_attractiveness_i / (exp_attractiveness_i + exp_attractiveness_j)
        probability_j = 1 - probability_i

       
        
        # Return probabilities as a numpy array
        return np.array([probability_i, probability_j])

    def calculate_indicator_matrix(self, switching_probabilities_matrix):
        """
        Calculate the indicator matrix of agent choices based on switching probabilities.

        Args:
            switching_probabilities_matrix (numpy.ndarray): A matrix containing switching probabilities for each agent.

        Returns:
            numpy.ndarray: An indicator matrix where each agent chooses one column based on probabilities.
        """


        num_agents, num_columns = switching_probabilities_matrix.shape
        
        # Create an empty indicator matrix
        indicator_matrix = np.zeros((num_agents, num_columns))
        
        # Iterate over agents and select columns based on probabilities
        for i in range(num_agents):
            chosen_column = np.random.choice(num_columns, 1, replace=False, p=switching_probabilities_matrix[i])
            indicator_matrix[i, chosen_column] = 1
        
        return indicator_matrix

    def calculate_conformity_probabilities_matrix(self, indicator_matrix):
        """
        Calculate conformity probabilities matrix based on an indicator matrix.

        Args:
            indicator_matrix (numpy.ndarray): An indicator matrix representing agent choices.

        Returns:
            numpy.ndarray: A conformity probabilities matrix.
        """
        # Calculate conformity probabilities matrix using matrix multiplication

        conformity_probabilities_matrix = np.dot(self.g.adj_trust_matrix, indicator_matrix)

        return conformity_probabilities_matrix

    def calculate_weighted_probabilities_matrix(self, switching_probabilities, conformity_probabilities_matrix):
        """
        Calculate a weighted probabilities matrix based on switching probabilities and conformity probabilities.

        Args:
            switching_probabilities (numpy.ndarray): The switching probabilities matrix.
            conformity_probabilities_matrix (numpy.ndarray): The conformity probabilities matrix.

        Returns:
            numpy.ndarray: The weighted probabilities matrix.
        """
        # Calculate the weighted probabilities matrix as a weighted combination of conformity and switching probabilities
        weighted_probabilities_matrix = (
            self.p.persuasion * conformity_probabilities_matrix +
            (1 - self.p.persuasion) * switching_probabilities
        )

        return weighted_probabilities_matrix

    def calculate_weights(self, indicator_matrix):
        """
        Calculate the weight of each heuristic relative to their appearance in the population.

        Args:
            indicator_matrix (numpy.ndarray): The indicator matrix representing heuristic choices.

        Returns:
            numpy.ndarray: An array of heuristic weights.
        """
        # Calculate the sum of choices for each heuristic (axis=0 sums vertically)
        heuristic_counts = indicator_matrix.sum(axis=0)
        
        # Calculate the weights by dividing the counts by the total number of agents
        heuristic_weights = heuristic_counts / self.p.agent_number
        
        return heuristic_weights

    def calculate_quartal(self, previous_variable, previous_interest_rate, market_expectations, epsilon):
        """
        Calculate the current variable for the next time quartal.

        Args:
            previous_variable (numpy.ndarray): The variable from the previous quartal.
            previous_interest_rate (numpy.ndarray): The nominal interest rate from the previous quartal.
            market_expectations (numpy.ndarray): Market expectations for the current quartal.
            epsilon (numpy.ndarray): Random noise or error term.

        Returns:
            numpy.ndarray: The current variable for the next quartal.
        """
        # Simplify notation for matrices
        A, B, C, b, c = self.g.A, self.g.B, self.g.C, self.g.b, self.g.c

        # Calculate the inverse of matrix A
        Ainv = np.linalg.inv(A)

        # Calculate the components of the equation separately
        component_B = np.dot(B, previous_variable)
        component_C = np.dot(C, market_expectations)
        component_b = b * previous_interest_rate
        component_c = c * self.p.inflation_target

        # Calculate the current variable using the equation
        current_variable = np.dot(Ainv, (component_B + component_C + component_b + component_c + epsilon))

        return current_variable

    def calculate_market_expectations(self, weight_variable_a, weight_variable_b, expected_variable_a, expected_variable_b):
        """
        Calculate market expectations based on weighted expected variables.

        Args:
            weight_variable_a (float): Weight for expected_variable_a.
            weight_variable_b (float): Weight for expected_variable_b.
            expected_variable_a (float): Expected variable A.
            expected_variable_b (float): Expected variable B.

        Returns:
            float: Calculated market expectations.

        """
        # Calculate market expectations
        market_expectations = (weight_variable_a * expected_variable_a) + (weight_variable_b * expected_variable_b)
        
        return market_expectations


    def calculate_animal_spirits(self, lagged_variable, heuristic_weight_naive, heuristic_weight_target):
        """
        Calculate animal spirits based on lagged variable and heuristic weights.

        Args:
            lagged_variable (float): Lagged variable from the previous time step.
            heuristic_weight_naive (float): Heuristic weight for the naive case.
            heuristic_weight_target (float): Heuristic weight for the target case.

        Returns:
            float: Calculated animal spirits.

        """
        # Initialize AnimalSpirits
        animal_spirits = 0

        # Check the sign of the lagged variable
        if lagged_variable > 0:
            animal_spirits = heuristic_weight_naive - heuristic_weight_target
        elif lagged_variable < 0:
            animal_spirits = -heuristic_weight_naive + heuristic_weight_target

        return animal_spirits

    def calculate_current_interest(self, current_variable, lagged_interest, random_value):
        """
        Calculate the current interest rate using a forward-looking Taylor rule.

        Args:
            current_variable (float): Current economic variable.
            lagged_interest (float): Lagged interest rate.
            random_value (float): Random value used in the calculation.

        Returns:
            float: Calculated current interest rate.

        """
        # Coefficients for the Taylor rule
        inflation_coefficient = self.p.c1 * (current_variable[1] - self.p.inflation_target)
        output_coefficient = self.p.c2 * current_variable[0]
        lagged_interest_term = self.p.c3 * lagged_interest

        # Calculate the current interest rate
        current_interest = (1 - self.p.c3) * (inflation_coefficient + output_coefficient) + lagged_interest_term + random_value

        return current_interest

    def update(self, VarLag1, NomIntLag1, AttrOut, AttrInf, OutExpLag2, InfExpLag2, epsilon, rand, indicator_matrix_output, indicator_matrix_inflation):
            """
            Update the model for the 'degroot' case.

            Parameters:
                lagged_variables (np.array): Lagged economic variables.
                lagged_nominal_interest (float): Lagged nominal interest rate.
                attractivity_output (np.array): Attractivity of output expectations.
                attractivity_inflation (np.array): Attractivity of inflation expectations.
                lagged2_output_expectations (np.array): Lagged2 output expectations.
                lagged2_inflation_expectations (np.array): Lagged2 inflation expectations.
                epsilon (float): Random variable.
                rand (float): Random value.
                indicator_matrix_output (np.array): Indicator matrix for output expectations.
                indicator_matrix_inflation (np.array): Indicator matrix for inflation expectations.

            Returns:
                Tuple: Updated variables and matrices.
            """
    
            # Output: Calculating attractivity of target and naive expectations
            AttrOut[:2] = self.calculate_attractivity(VarLag1[0], OutExpLag2[:2], AttrOut[:2])

            # Inflation: Calculating attractivity of target and naive expectations
            AttrInf[:2] = self.calculate_attractivity(VarLag1[1], InfExpLag2[:2], AttrInf[:2])

            # Output: Calculating Switching Probabilities of output and inflation
            switching_probabilities_output = self.calculate_switching_probabilities(AttrOut[0], AttrOut[1])
            switching_probabilities_inflation = self.calculate_switching_probabilities(AttrInf[0], AttrInf[1])

            OutHeuWe, InfHeuWe = switching_probabilities_output, switching_probabilities_inflation  # Output (Target Naive)

            if self.p.degroot:
                # Update Conformity Probability Matrix (CPM) of output and inflation
                conformity_probabilities_matrix_output = self.calculate_conformity_probabilities_matrix(indicator_matrix_output)
                conformity_probabilities_matrix_inflation = self.calculate_conformity_probabilities_matrix(indicator_matrix_inflation)
       

                # Update weighted Probability Matrix (wPM) of output and inflation
                weighted_probabilities_matrix_output = self.calculate_weighted_probabilities_matrix(switching_probabilities_output, conformity_probabilities_matrix_output)
                weighted_probabilities_matrix_inflation = self.calculate_weighted_probabilities_matrix(switching_probabilities_inflation, conformity_probabilities_matrix_inflation)

                # Update Indicator Matrices (IM) of output and inflation
                indicator_matrix_output = self.calculate_indicator_matrix(weighted_probabilities_matrix_output)
                indicator_matrix_inflation = self.calculate_indicator_matrix(weighted_probabilities_matrix_inflation)
                
                if self.p.intervention:
                    if self.p.target:
                        central_node = int(self.degree_centrality.node)
                        indicator_matrix_inflation[central_node] = [1, 0]
                    else:
                        central_node = int(self.degree_centrality.node)
                        indicator_matrix_inflation[central_node] = [0, 1]


                # Update heuristics' weight of output and inflation
                OutHeuWe, InfHeuWe = self.calculate_weights(indicator_matrix_output), self.calculate_weights(indicator_matrix_inflation)

            # Update expected values of output and inflation
            OutExp, InfExp = [0, VarLag1[0]], [self.p.inflation_target, VarLag1[1]]

            # Update Animal Spirits for output and inflation
            OutAniSp = self.calculate_animal_spirits(VarLag1[0], OutHeuWe[1], OutHeuWe[0])
            InfAniSp = self.calculate_animal_spirits(VarLag1[1], InfHeuWe[1], InfHeuWe[0])

            # Calculate Market Expectations for output and inflation
            OutMarketExp = self.calculate_market_expectations(OutHeuWe[0], OutHeuWe[1], OutExp[0], OutExp[1])
            InfMarketExp = self.calculate_market_expectations(InfHeuWe[0], InfHeuWe[1], InfExp[0], InfExp[1])
            MarketExp = [OutMarketExp, InfMarketExp]

            # Update Variables
            VarCur = self.calculate_quartal(VarLag1, NomIntLag1, MarketExp, epsilon)

            # Update interest rate
            NomIntCur = self.calculate_current_interest(VarCur, NomIntLag1, rand)
            InfMarketExp = self.calculate_market_expectations(InfHeuWe[0], InfHeuWe[1], InfExp[0], InfExp[1])  # Recalculate for inflation
            RealIntCur = NomIntCur - InfMarketExp

            return VarCur, NomIntCur, RealIntCur, AttrOut, AttrInf, OutExp, InfExp, OutHeuWe, InfHeuWe, OutMarketExp, InfMarketExp, OutAniSp, InfAniSp, indicator_matrix_output, indicator_matrix_inflation


    def simulate(self, seed):
        """
        Simulate the model over time.

        Parameters:
            seed (int): Seed for random number generation.

        Returns:
            Tuple: Dataframes and simulation results.
        """
        capT = self.p.capT
        np.random.seed(seed)


        # Simulating a time-series solution
        #---------------------------------------
        # Initialize indicator matrices
        indicator_matrix_output = np.zeros((self.p.agent_number, 2))
        indicator_matrix_inflation = np.zeros((self.p.agent_number, 2)) 

        for i in range(self.p.agent_number):
            agent_choice_output = np.random.choice([0, 1], 1, replace=False,p = [1/2, 1/2])
            agent_choice_inflation = np.random.choice([0, 1], 1, replace=False,p = [1/2, 1/2])
            indicator_matrix_output[i, agent_choice_output] = 1
            indicator_matrix_inflation[i, agent_choice_inflation] = 1


        # Allocate the memory the 3 exogenous shocks and draw innovations
        #-----------------------------------------------------------------
        epsilon = np.zeros((capT, 2))
        rands = np.zeros((3,capT))

        # Generate the series 3 exogenous shocks of ...
        #-------------------------------
        rands[0] = np.random.normal(0,self.p.sigma_output, capT)      # ... EpsY(t)
        rands[1] = np.random.normal(0,self.p.sigma_inflation, capT)   # ... EpsPi(t)
        rands[2] = np.random.normal(0,self.p.sigma_taylor, capT)     # ... EpsR(t)

        
        if self.p.is_shock == 'Inflation':
            rands[1, 43] += 2 * self.p.sigma_inflation
        elif self.p.is_shock == 'Interest':
            rands[2, 43] += self.p.sigma_taylor

        for t in range(1, capT):
            epsilon[t, 0] = -self.p.a2 * rands[2, t - 1] + rands[0, t - 1]
            epsilon[t, 1] = rands[1, t - 1]

        # Allocate memory for time series of ...
        VarCur = np.zeros((capT,2))         # ... Y(t) & PI(t)
        if not self.p.randomness:
            VarCur[-1] = [4,4]    
        NomIntCur = np.zeros(capT)          # ... I(t)
        RealIntCur = np.zeros(capT)         # ... R(t)
        AttrOut = np.zeros((capT,2))        # ... AY(t)
        AttrInf = np.zeros((capT,2))        # ... AI(t)
        OutExp = np.zeros((capT,2))          # ... YExp(t)
        InfExp = np.zeros((capT,2))          # ... PIExp(t)
        OutHeuWe = np.zeros((capT,2))        # ... WExpY(t)
        InfHeuWe = np.zeros((capT,2))        # ... WExpPi(t)
        OutMarketExp = np.zeros(capT)       # ... MYExp(t)
        InfMarketExp = np.zeros(capT)       # ... MPIExp(t)
        OutAniSp = np.zeros(capT)           # ... SY(t)
        InfAniSp = np.zeros(capT)           # ... SPI(t)

        agents_type_output = []
        agents_type_inflation = []
        
        for t in range(capT):
            VarCur[t], NomIntCur[t], RealIntCur[t], AttrOut[t], AttrInf[t], OutExp[t], InfExp[t],\
            OutHeuWe[t], InfHeuWe[t], OutMarketExp[t], InfMarketExp[t], OutAniSp[t], InfAniSp[t],\
            indicator_matrix_output, indicator_matrix_inflation = self.update(
                VarCur[t - 1], NomIntCur[t - 1], AttrOut[t - 1], AttrInf[t - 1], OutExp[t - 2],
                InfExp[t - 2], epsilon[t - 1], rands[2, t - 1], indicator_matrix_output,
                indicator_matrix_inflation)
            agents_type_output.append(indicator_matrix_output[:, 0])
            agents_type_inflation.append(indicator_matrix_inflation[:, 0])


        # Create a dictionary to hold simulation results
        simulation_data = {
            'Output': VarCur[:, 0],
            'Inflation': VarCur[:, 1],
            'Interest': NomIntCur,
            'RealInterest': RealIntCur,
            'OutTarWe': OutHeuWe[:, 0],
            'OutNaiveWe': OutHeuWe[:, 1],
            'InfTarWe': InfHeuWe[:, 0],
            'InfNaiveWe': InfHeuWe[:, 1],
            'OutMarketExp': OutMarketExp,
            'InfMarketExp': InfMarketExp,
            'OutAniSp': OutAniSp,
            'InfAniSp': InfAniSp
        }

        # Create a DataFrame from the simulation data
        df_simulation = pd.DataFrame(simulation_data)

        # Create dataframes to hold network properties
        df_degree_centrality = self.degree_centrality if self.p.degroot else 0
        df_clustering = self.clustering if self.p.degroot else 0
        df_average_clustering = self.average_clustering if self.p.degroot else 0


        return df_simulation, df_degree_centrality, df_clustering, df_average_clustering, agents_type_output, agents_type_inflation

        
        