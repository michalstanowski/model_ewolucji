import math
import numpy as np
import json
from scipy.spatial import distance

with open("fisher_model_params.json") as f:
    try:
        [
            init_population, max_population, num_genes, mutation_probability,
            mutation_effect, fitness_coefficient, max_num_children, scenario,
            global_warming_scale, global_warming_var, meteor_impact_strategy,
            meteor_impact_every, meteor_impact_at, num_steps
        ] = json.load(f).values()
        
    except Exception as e:
        print(e)
        
class Specimen:
    def __init__(self, genotype, env, fit):
        self.genotype = genotype
        self.fit = fit if fit is not None else 0
        self.environment = env
        self.parent = None
        self.rank = 0
        self.generation = 0
        self.kids = []

    def mutate(self):
        index = np.random.randint(num_genes)
        x = np.random.uniform(0, 1)
        if index < int(num_genes/2):
            if x < mutation_probability:
                self.genotype[index] += np.random.normal(0, mutation_effect)
        else:
            if x*10 < mutation_probability:
              self.genotype[index] += np.random.normal(0, mutation_effect)

    def calc_fit(self):
        env_genotype = self.environment.opt_genotype
        self.fit = (math.exp(-distance.euclidean(self.genotype, env_genotype)/(2*fitness_coefficient**2)))
    
    



