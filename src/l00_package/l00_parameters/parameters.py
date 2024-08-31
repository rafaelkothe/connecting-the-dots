import numpy as np

class Params:
    def __init__(self,
                 capT=250,
                 agent_number=100,
                 randomness=True,
                 persuasion=0.5,
                 degroot=False,
                 topology='albert-barabasi',
                 topology_m=25,
                 topology_seed=None,
                 tgt_agt_centr=None,
                 intervention=False,
                 is_shock=False,
                 a1=0.5,
                 a2=0.2,
                 b1=0.5,
                 b2=0.05,
                 inflation_target=0,
                 c1=1.5,
                 c2=0.5,
                 c3=0.5,
                 theta=2,
                 sigma_output=0.5,
                 sigma_inflation=0.5,
                 sigma_taylor=0.5,
                 zeta=0.5,
                 alpha=0.5,
                 target=False,
    ):
        
        #TODO : Use setattr to set the attributes
        """        
        
        check_params = ['ad','bd','az','bz','t']
        for i in check_params:
            try:
                params[i]
            except KeyError:
                raise ValueError("parameter " + i +" missing") 
        
        for key,value in params.items():
            setattr(self, key, value) # setattr() is a function which initializes data of a given object
        if self.ad < self.az:
            raise ValueError('Insufficient demand.')
            
            """
        # simulation parameters
        self.capT = capT
        self.agent_number = agent_number
        self.randomness = randomness
        self.persuasion = persuasion * np.ones((agent_number, 1))
        
        # network parameters
        self.degroot = degroot
        self.topology = topology
        self.topology_m = topology_m
        self.topology_seed = topology_seed
        self.tgt_agt_centr = tgt_agt_centr
        self.intervention = intervention
        self.is_shock = is_shock

        # structural parameters
        self.a1 = a1
        self.a2 = a2
        self.b1 = b1
        self.b2 = b2
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.inflation_target = inflation_target

        # standard deviations
        self.sigma_output = sigma_output if randomness else 0
        self.sigma_inflation = sigma_inflation if randomness else 0
        self.sigma_taylor = sigma_taylor if randomness else 0
        
        self.zeta = zeta
        self.theta = theta
        self.alpha = alpha
        
        self.target = target 


    def __str__(self):
        return f"Simulation Parameters:\n{str(vars(self))}"
