from environment_file import Environment
from population_visualiser import PopulationVisualizer
import json

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

def start_sim():
    env = Environment()
    pop_vis = PopulationVisualizer()
    is_dead = False
    j=1
    while not is_dead and j<=num_steps:
        is_dead = env.step(j)
        pop_vis.update_plot(env.population, env.opt_genotype, env.old_genotype, j, is_dead, env.how_big)
        j+=1
    else:
       pop_vis.the_end(env.population, env.ancestral_population,j, env.gen, env.pop_num, env.avg_genotypes, env.most_fitted_genotype, env.std_dev_sum)
 
start_sim()
 