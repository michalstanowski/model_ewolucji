from files.specimen_file import Specimen
import math
import numpy as np
import json
import copy


with open("files/fisher_model_params.json") as f:
    try:
        [
            init_population, max_population, num_genes, mutation_probability,
            mutation_effect, fitness_coefficient, max_num_children, scenario,
            global_warming_scale, global_warming_var, meteor_impact_strategy,
            meteor_impact_every, meteor_impact_at, num_steps
        ] = json.load(f).values()

    except Exception as e:
        print(e)

class Environment:
    def __init__(self):
        self.population = [Specimen(np.random.uniform(0, 1, num_genes), self, 0) for _ in range(init_population)]
        self.max_pop_num = max_population
        self.ancestral_population = []
        self.opt_genotype = np.random.uniform(0, 1, num_genes)
        self.mutation_effect = mutation_effect
        self.mutation_probability = mutation_probability
        self.old_genotype = None
        self.pop_num = [init_population]
        self.gen = [0]
        self.most_fitted_genotype = []
        self.std_dev_sum = []
        self.avg_genotypes = []
        self.how_big = 0
    
    def step(self, nr):

        self.gen.append(nr)
        self.old_genotype = copy.copy(self.opt_genotype)


        if nr%meteor_impact_every==0:
            sgn = np.random.choice([1, -1])
            self.how_big = np.random.randint(meteor_impact_at[0], meteor_impact_at[1])
            self.opt_genotype += sgn * global_warming_scale * self.how_big * 1.5
        else:
            self.opt_genotype += np.array(global_warming_scale)
                                           

        for spec in self.population:
            spec.mutate() 
            spec.calc_fit()
        
        self.std_dev_sum.append(sum([round(np.std([ind.genotype[i] for ind in self.population]), 3) for i in range(num_genes)]))

        rank_population = sorted(self.population, key=lambda spec: spec.fit, reverse=True)

        self.most_fitted_genotype.append(rank_population[0].genotype)

        avg_genotype = [0] * num_genes

        for i in range(num_genes):
            for spec in self.population:
                avg_genotype[i] += spec.genotype[i]/len(self.population)

        self.avg_genotypes.append(avg_genotype)

        for i in range(len(rank_population)):
            rank_population[i].generation = nr
            rank_population[i].rank = i+1

        if nr == 1:
            self.ancestral_population = copy.copy(rank_population)
        
        new_population = []

        for spec in self.population:
       
            i = 0
            for _ in range(math.floor(spec.fit * (max_num_children +1))):
                i+=1
                baby = Specimen(spec.genotype.copy(), copy.copy(self), spec.fit)
                spec.kids.append(baby)
                baby.parent = spec
                new_population.append(baby)

        if len(new_population) > 0:
            self.population = self.select_population(sorted(new_population, key=lambda spec: spec.fit, reverse=True), max_population)
            self.pop_num.append(len(self.population))
            return False
        
        else:
            self.pop_num.append(0)
            return True 
       
    def select_population(self, new_population, max_population):
                
        if not new_population or max_population <= 0:
            return []
    
        probabilities = np.linspace(1, 0.0001, len(new_population))

        max_population = min(max_population, len(new_population))

        selected_indices = np.random.choice(range(len(new_population)), size=max_population, replace=False, p=probabilities/np.sum(probabilities))

        selected_population = [new_population[i] for i in selected_indices]
        
        return selected_population