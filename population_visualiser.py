import pygame
import math
from sklearn.decomposition import PCA
from scipy.spatial import distance
import json
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np

with open("fisher_model_params.json") as f:
        [
            init_population, max_population, num_genes, mutation_probability,
            mutation_effect, fitness_coefficient, max_num_children, scenario,
            global_warming_scale, global_warming_var, meteor_impact_strategy,
            meteor_impact_every, meteor_impact_at, num_steps
        ] = json.load(f).values()

class PopulationVisualizer:
    def __init__(self):
        self.fitness_coefficient = fitness_coefficient
        self.max_num_children = 7
        self.population_sizes = []
        self.clock = pygame.time.Clock()
        pygame.init()
        
        self.window_width, self.window_height = 1200, 600
        self.screen = pygame.display.set_mode((self.window_width, self.window_height), pygame.DOUBLEBUF)
        pygame.display.set_caption('Population Visualization')

        self.koala_img = pygame.image.load('koala.png')
        self.koala_sigma = pygame.image.load('sigma_koala.png')
        self.dead_koala_img = pygame.image.load('dead_koala.png')
        self.the_end_img = pygame.image.load('the_end.jpeg')
        self.background_image = pygame.image.load("background.jpeg").convert()
        self.background_end_image = pygame.image.load("tlo.jpeg").convert()
        self.background_phy_image = pygame.image.load("tlo_wykresy.jpeg").convert_alpha()
        self.meteor_image = pygame.image.load("meteor.webp").convert_alpha()

        koala_size = 70
        skull_size = 30
        self.koala_img = pygame.transform.scale(self.koala_img, (koala_size, koala_size))
        self.sigma_koala = pygame.transform.scale(self.koala_sigma, (koala_size+10, koala_size+10))
        self.dead_koala = pygame.transform.scale(self.dead_koala_img, (skull_size, skull_size))

        self.pca = PCA(n_components=2)
        self.euclidean_distances = []

        self.font = pygame.font.Font("czcionka.ttf", 13)
        self.play_again_button = pygame.Rect(0, 0, 400, 50)
        self.quit_button = pygame.Rect(0, 0, 400, 50)
        self.chart_button = pygame.Rect(0,0,400, 50)
        self.center_buttons()

        self.explosion = pygame.mixer.Sound("explosion.mp3")
        self.explosion.set_volume(0.5)

        self.highlight_color = (200, 200, 200)
        self.text_color = (255, 255, 255)
        self.last_fit = 0

    def center_buttons(self):
    # Ustawia przyciski na środku ekranu
        self.play_again_button.center = (1024 // 2, 1024 // 2 - 50)
        self.quit_button.center = (1024 // 2, 1024 // 2 + 150)
        self.chart_button.center = (1024 // 2, 1024 // 2 + 50)
        
    def update_plot(self, population, opt_genotype, old_genotype, nr, is_dead, how_big):
        def depict_pop(gen_img, opt_img):
            pca_result = self.pca.fit_transform(all_genotypes)
            pca_population = pca_result[:-2]
            pca_opt_genotype = pca_result[-2]
            pca_old_genotype = pca_result[-1]

            mean_x = pca_population[:, 0].mean()
            mean_y = pca_population[:, 1].mean()

            pca_population[:, 0] -= mean_x
            pca_population[:, 1] -= mean_y
            pca_opt_genotype[0] -= mean_x
            pca_opt_genotype[1] -= mean_y
            pca_old_genotype[0] -= mean_x 
            pca_old_genotype[1] -= mean_y

            old_optimum_x, old_optimum_y = pca_old_genotype[0] * 100 + self.window_width / 4, pca_old_genotype[1] * 100 + self.window_height / 4

            optimum_x, optimum_y = pca_opt_genotype[0] * 100 + self.window_width / 4, pca_opt_genotype[1] * 100 + self.window_height / 4
            sigma_rect = opt_img.get_rect(center=(optimum_x, optimum_y))


            for i in range(1, self.max_num_children):
                print(self.fitness_coefficient)
                i_children_radius = -2 * (self.fitness_coefficient ** 2) * math.log(i / (self.max_num_children + 1)) * 100
                print(i_children_radius)
                pygame.draw.circle(self.screen, (139, 0, 0), (int(optimum_x), int(optimum_y)), int(i_children_radius), 1)

            for pca_genotype in pca_population:
                koala_rect = gen_img.get_rect(center=(pca_genotype[0] * 100 + self.window_width / 4, pca_genotype[1] * 100 + self.window_height /4))
                self.screen.blit(gen_img, koala_rect)

            self.screen.blit(opt_img, sigma_rect)

            if nr % meteor_impact_every == 0:
                scale = 100 / meteor_impact_at[1]
                size = int(how_big * scale) 
                scaled_meteor = pygame.transform.scale(self.meteor_image, (size, size))  
                meteor_rect = scaled_meteor.get_rect(center=(old_optimum_x, old_optimum_y))
                self.screen.blit(scaled_meteor, meteor_rect)
                self.explosion.play()

            most_fitted = sorted(population, key=lambda spec: spec.fit, reverse=True)[0]
            best_genotype = most_fitted.genotype
            euclidean_diff = distance.euclidean(opt_genotype, best_genotype)
            self.euclidean_distances.append(euclidean_diff)

           

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # pygame.transform.scale(pygame.image.load("koala_charts.jpeg"), (512, 512))
        genotypes = [spec.genotype for spec in population]
        all_genotypes = genotypes + [opt_genotype] + [old_genotype]

        self.screen.blit(self.background_image, (0, 0))

        if not is_dead:
            depict_pop(self.koala_img, self.sigma_koala)
        else:
            depict_pop(self.dead_koala, self.sigma_koala)

        bar_width = self.window_width / (2 * num_steps) 

        self.screen.blit(pygame.transform.scale(pygame.image.load("koala_charts.jpeg"), (600, 600)), (600, 0))
        for i, diff in enumerate(self.euclidean_distances):
            bar_height = (diff / max(self.euclidean_distances)) * (self.window_height / 2.5)
            bar_x = self.window_width / 2 + i * bar_width
            bar_y = self.window_height / 2 - bar_height  
            if nr % 20 == 0:
                bar_color = (255, 0, 0)
            else:
                bar_color = (0, 0, 0)
            pygame.draw.rect(self.screen, bar_color, (bar_x, bar_y, bar_width, bar_height))


        std_devs = [round(np.std([ind.genotype[i] for ind in population]), 3) for i in range(num_genes)]
    
        if is_dead:
            population_label = "Liczba osobników:"
            population_value = "0"
        else:
            population_label = "Liczba osobników:"
            population_value = str(len(population))

        genetic_variability_label = "Zmienność genetyczna:"
        genetic_variability_value = str(std_devs)

        generations_label = "Generacja:"
        generations_value = str(nr)

        population_label_rendered = self.font.render(population_label, True, (0, 0, 0))
        population_value_rendered = self.font.render(population_value, True, (0, 0, 0))

        genetic_variability_label_rendered = self.font.render(genetic_variability_label, True, (0, 0, 0))
        genetic_variability_value_rendered = self.font.render(genetic_variability_value, True, (0, 0, 0))

        generations_label_rendered = self.font.render(generations_label, True, (0, 0, 0))
        generations_value_rendered = self.font.render(generations_value, True, (0, 0, 0))

        table_start_x = self.window_width * 3 / 4 - 150
        table_start_y = self.window_height - 250
        line_spacing = 30

        self.screen.blit(population_label_rendered, (table_start_x, table_start_y))
        self.screen.blit(population_value_rendered, (table_start_x + 200, table_start_y))

        self.screen.blit(genetic_variability_label_rendered, (table_start_x, table_start_y + line_spacing))
        self.screen.blit(genetic_variability_value_rendered, (table_start_x + 200, table_start_y + line_spacing))

        self.screen.blit(generations_label_rendered, (table_start_x, table_start_y + 2 * line_spacing))
        self.screen.blit(generations_value_rendered, (table_start_x + 200, table_start_y + 2 * line_spacing))
                    
                
        text = self.font.render('Odległość od optimum najbardziej przystosowanego osobnika', True, (0, 0, 0))
        text_rect = text.get_rect(center=(self.window_width * 3 / 4, 20))
        self.screen.blit(text, text_rect)
        print(f"{nr}, lalala")

        pygame.display.flip()
        self.clock.tick(60) 
        if is_dead:
            pygame.time.wait(4000)
        else:
            pygame.time.wait(400)
            
            
    def the_end(self, population, ancestral_population, nr, env_gen, env_pop_num, env_avg_gen, env_avg_fitted_gen, env_std):
        end_screen = pygame.display.set_mode((1024, 1024), pygame.DOUBLEBUF)
      

        phy = False
        population = sorted(population, key=lambda spec: spec.fit, reverse=True)
        
        def draw_button(rect, text):
            mouse_pos = pygame.mouse.get_pos()
            rounded_rect = rect.inflate(-6, -6)  
            if rect.collidepoint(mouse_pos):
                pygame.draw.rect(end_screen, self.highlight_color, rounded_rect, border_radius=5)
            else:
                pygame.draw.rect(end_screen, (135,151,151), rounded_rect, border_radius=8)
            text_surface = self.font.render(text, True, self.text_color)
            text_rect = text_surface.get_rect(center=rect.center)
            end_screen.blit(text_surface, text_rect) 

        def draw_evolutionary_history(generations, fitness_values, ancestral_ranking):
            fitness_values =fitness_values[::-1]
            plt.plot(generations, fitness_values)
            plt.xlabel('Generacja')
            plt.ylabel('Dostosowanie')
            plt.title('Historia ewolucyjna')
            for i in range(20, len(generations)+1, 20):
                plt.axvline(x=i, color='red', linestyle='--')

            plt.axhline(y=1/8, color='green', linestyle='--')
            plt.text(generations[0], fitness_values[0]+0.1, f'AR:{ancestral_ranking}', color='blue')
            

            plt.show()

        def display_genotype_history(genotype, generations=None, fitness_values=None):
            if generations is None:
                generations = []
                fitness_values = []

            generations.append(len(generations) + 1)
            fitness_values.append(genotype.fit)

            if genotype.parent is not None:
                display_genotype_history(genotype.parent, generations, fitness_values)
            else:
                ancestral_ranking = genotype.rank if hasattr(genotype, 'rank') else "N/A"
                draw_evolutionary_history(generations, fitness_values, ancestral_ranking)

        def add_nodes_edges(genotype, G=None):
            if G is None:
                G = nx.DiGraph()
                G.add_node(f"{genotype.generation}.{genotype.rank}", fit=genotype.fit)
            
            if genotype.kids:
                for kid in genotype.kids:
                    if kid.generation != 0 or kid.rank != 0:
                        G.add_node(f"{kid.generation}.{kid.rank}", fit=kid.fit)
                        G.add_edge(f"{genotype.generation}.{genotype.rank}", f"{kid.generation}.{kid.rank}")
                        add_nodes_edges(kid, G=G)
            return G

        
        def draw_population_size_chart(generations, population_sizes):
            plt.plot(generations, population_sizes)
            plt.xlabel('Pokolenie')
            plt.ylabel('Liczba osobników')
            plt.title('Liczebność populacji w czasie')
            for i in range(20, len(generations)+1, 20):
                plt.axvline(x=i, color='red', linestyle='--')

            plt.show()

        def draw_genotypes(generations, env_avg_gen, env_fitted_gen):
            for i in range(num_genes):
                plt.plot(generations, [gen[i] for gen in env_avg_gen], label=f'GP {i+1}')
            for i in range(num_genes):
                plt.plot(generations, [gen[i] for gen in env_fitted_gen], label=f'GF {i+1}')

            for i in range(20, len(generations)+1, 20):
                plt.axvline(x=i, color='red', linestyle='--')
            
            plt.legend(loc="upper right")  
            plt.xlabel('Pokolenie')

            plt.ylabel('Wartości poszczególnych genów')
            plt.title('Ewolucja genów w czasie (osobnik najlepiej przystosowany vs populacja)')

            plt.show()

        def depict_diversity(generations, diversity):
            plt.plot(generations, diversity)
            plt.xlabel('Pokolenie')
            for i in range(20, len(generations)+1, 20):
                plt.axvline(x=i, color='red', linestyle='--')

            plt.ylabel('Zmienność populacji')
            plt.title('Zmienność w czasie')
            plt.show()

        
        phy_screen = pygame.display.set_mode((1024, 1024), pygame.DOUBLEBUF)
        mouse_pos = (0,0)
        return_button_phy = pygame.Rect(1024 // 2 - 100, 1024 // 2 + 300, 200, 30)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = event.pos

            if phy:
                input_box_phy = pygame.Rect(1024 // 2 - 70, 1024 // 2 - 16 - 50, 140, 32) 
                input_box_ancestral = pygame.Rect(1024 // 2 - 70, 1024 // 2 - 16 + 50, 140, 32)  

                active_phy = False
                active_ancestral = False
                text_phy = '' 
                text_ancestral = ''  

                button_phy = pygame.Rect(1024 // 2 - 100, 1024 // 2 + 30 - 50, 200, 30)
                button_ancestral = pygame.Rect(1024 // 2 - 100, 1024 // 2 + 30 + 50, 200, 30)
           

                while phy:
                    for event in pygame.event.get():
                        if event.type == pygame.MOUSEBUTTONDOWN:

                            if input_box_phy.collidepoint(event.pos):
                                active_phy = not active_phy
                                active_ancestral = False 
                            else:
                                active_phy = False

                            if input_box_ancestral.collidepoint(event.pos):
                                active_ancestral = not active_ancestral
                                active_phy = False  
                            else:
                                active_ancestral = False


                            if button_phy.collidepoint(event.pos):
                                index = int(text_phy)
                                genotype = population[index-1]
                                display_genotype_history(genotype)

                            if button_ancestral.collidepoint(event.pos):
                                index = int(text_ancestral) - 1
                                ancestral_genotype = ancestral_population[index]
                                G = add_nodes_edges(ancestral_genotype)
                                nx.write_graphml(G, f"graph_{text_ancestral}.graphml")
                                

                            
                            elif return_button_phy.collidepoint(event.pos):
                                phy = False
                                mouse_pos = (0,0)


                        elif event.type == pygame.KEYDOWN:
                            if active_phy:
                                if event.key == pygame.K_RETURN:
                                    text_phy = ''
                                elif event.key == pygame.K_BACKSPACE:
                                    text_phy = text_phy[:-1]
                                else:
                                    text_phy += event.unicode
                            elif active_ancestral:
                                if event.key == pygame.K_RETURN:
                                    text_ancestral = ''
                                elif event.key == pygame.K_BACKSPACE:
                                    text_ancestral = text_ancestral[:-1]
                                else:
                                    text_ancestral += event.unicode
                        elif event.type == pygame.QUIT:
                            pygame.quit()
                            quit()

                    phy_screen.blit(self.background_phy_image, (0, 0))

                    text_surface = self.font.render(f"Wielkość populacji przed wymarciem: {len(population)}", True, (0, 0, 0))
                    text_rect = text_surface.get_rect(center=(1024 // 2, 1024 // 2 - 100))
                    phy_screen.blit(text_surface, text_rect)

                    txt_surface_phy = self.font.render(text_phy, True, (0, 0, 0))
                    phy_screen.blit(txt_surface_phy, (input_box_phy.x + 5, input_box_phy.y + 5))
                    pygame.draw.rect(phy_screen, (0, 0, 0), input_box_phy, 2)

                    txt_surface_ancestral = self.font.render(text_ancestral, True, (0,0,0))
                    phy_screen.blit(txt_surface_ancestral, (input_box_ancestral.x + 5, input_box_ancestral.y + 5))
                    pygame.draw.rect(phy_screen, (0, 0, 0), input_box_ancestral, 2)


                    draw_button(button_phy, "Wyświetl wykres")
            

                    draw_button(button_ancestral, "Zapisz drzewo")


                    draw_button(return_button_phy, "Powrót")
              
                    

                    # Wewnątrz pętli obsługującej ekran "phy"
                   

                    pygame.display.flip()


            if self.play_again_button.collidepoint(mouse_pos):
                phy = True
                phy_screen.blit(self.background_phy_image, (0, 0))

                # Rysowanie kwadratów z osobnikami po kliknięciu "Filogeneza"
                # draw_population()

                pygame.display.flip()

            elif self.quit_button.collidepoint(mouse_pos):
                the_end_img = pygame.transform.scale(self.the_end_img, (1024, 1024))
                end_screen.blit(the_end_img, (0, 0))

                text = self.font.render("To koniec", True, (240, 240, 240))
                end_screen.blit(text, (512 - text.get_width() // 2, 512 - text.get_height() // 2))
                pygame.display.flip()
                pygame.time.wait(5000)

                pygame.quit()
                quit()

            elif self.chart_button.collidepoint(mouse_pos):
                chart_screen = pygame.display.set_mode((1024, 1024), pygame.DOUBLEBUF)
                chart_img = pygame.transform.scale(pygame.image.load("wykresy.jpeg"), (1024, 1024))
                charts = True
                while charts:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            quit()
                        elif event.type == pygame.MOUSEBUTTONDOWN:
                            if button_count.collidepoint(event.pos):
                                draw_population_size_chart(env_gen, env_pop_num)

                            if button_genotypes.collidepoint(event.pos):
                                trimmed= env_gen[1:]
                                draw_genotypes(trimmed, env_avg_gen, env_avg_fitted_gen)

                            if button_diversity.collidepoint(event.pos):
                                trimmed= env_gen[1:]
                                depict_diversity(trimmed, env_std)
                            if return_button.collidepoint(event.pos):
                                charts = False
                                mouse_pos = (0,0)

                    chart_screen.blit(chart_img, (0, 0))

                    button_count = pygame.Rect(1024 // 2 - 100, 1024 // 2 + 50, 200, 30)
                    draw_button(button_count, "Liczebność")

                    button_genotypes = pygame.Rect(1024 // 2 - 100, 1024 // 2 + 100, 200, 30)
                    draw_button(button_genotypes, "Genotypy")

                    return_button = pygame.Rect(1024 // 2 - 100, 1024 // 2 + 200, 200, 30)
                    draw_button(return_button, "Powrót")

                    button_diversity = pygame.Rect(1024 // 2 - 100, 1024 // 2 + 150, 200, 30)
                    draw_button(button_diversity, "Zmienność")

                    pygame.display.flip()

            if not phy:
                # Wyświetlenie przycisków i tekstu
                self.screen.blit(self.background_end_image, (0, 0))
                draw_button(self.play_again_button, "Filogeneza")
                draw_button(self.quit_button, "Koniec")
                draw_button(self.chart_button, "Wykresy")

                pygame.display.flip()
                self.clock.tick(60)

if __name__ == "__main__":
    pop_vis = PopulationVisualizer()
