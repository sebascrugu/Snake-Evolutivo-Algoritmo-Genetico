#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Módulo Snake

import numpy as np
import matplotlib.pyplot as plt
import time
import random
import os
from matplotlib.ticker import MaxNLocator
from snakepy import GeneticAlgorithm, DecisionTable, Action, Direction, Point, SnakeGame
from snakepy import BLOCK_SIZE
from joblib import Parallel, delayed
import multiprocessing as mp
import seaborn as sns

# Colores
COLORS = [
    '#7EC8E3',  # Azul
    '#FF9999',  # Rosa
    '#90EE90',  # Verde
]

# Estilo
sns.set_style("whitegrid", {
    'axes.grid': True,
    'grid.color': '.8',
    'grid.linestyle': ':',
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.right': False,
    'axes.spines.top': False,
})

# Contexto
sns.set_context("notebook", rc={
    'figure.figsize': (20, 15),
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
})

# Matplotlib
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'axes.labelcolor': '#2c3e50',
    'text.color': '#2c3e50',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'legend.frameon': True,
    'legend.framealpha': 0.8,
    'legend.edgecolor': '#e0e0e0',
    'legend.facecolor': 'white',
    'axes.prop_cycle': plt.cycler('color', COLORS)
})

# Procesos
N_JOBS = 6

def evaluate_agent_fitness(agent, agent_index, seed):
    # Fitness
    agent_seed = seed + agent_index
    random.seed(agent_seed)
    np.random.seed(agent_seed)

    ga = GeneticAlgorithm()
    fitness_value = ga.fitness(agent, num_games=7, show_game=False, silent=True)
    return fitness_value


def run_experiment(config, shared_results, experiment_index):
    # Experimento
    pop_size = config.get('pop_size', 50)
    generations = config.get('generations', 50)
    mutation_rate = config.get('mutation_rate', 0.1)
    elitism = config.get('elitism', 5)
    base_seed = config.get('base_seed', int(time.time()))
    
    print(f"Iniciando experimento {experiment_index} con semilla base: {base_seed}")
    
    # Población
    population = [DecisionTable() for _ in range(pop_size)]
    
    # Historial
    history = {
        'best_fitness': [],
        'avg_fitness': [],
        'best_score': [],
        'best_survival_time': [],
        'diversity_metric': [],
        'best_agents_history': [],
        'food_eaten_avg': [],
        'survival_time_avg': [],
        'navigation_skill': [],
        'obstacle_avoidance': [],
        'food_std_dev': [],
        'previous_best': None
    }
    
    stagnation_counter = 0
    best_fitness_ever = 0
    best_agent_ever = None
    
    # Evolución
    for generation in range(generations):
        gen_seed = base_seed + (generation * 100)
        print(f"Generación {generation+1}/{generations}, semilla: {gen_seed}")
        
        # Evaluación
        fitness_results = Parallel(n_jobs=N_JOBS)(
            delayed(evaluate_agent_fitness)(agent, idx, gen_seed)
            for idx, agent in enumerate(population)
        )
        
        fitness_dict = {idx: fitness for idx, fitness in enumerate(fitness_results)}
        sorted_indices = sorted(fitness_dict.keys(), key=lambda idx: fitness_dict[idx], reverse=True)
        sorted_population = [population[idx] for idx in sorted_indices]
        
        best_fitness = fitness_dict[sorted_indices[0]]
        avg_fitness = sum(fitness_dict.values()) / len(fitness_dict)
        
        if best_fitness > best_fitness_ever:
            best_fitness_ever = best_fitness
            best_agent_ever = sorted_population[0]
        
        # Estancamiento
        if history['previous_best'] is not None:
            if best_fitness <= history['previous_best'] * 1.01:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
        
        history['previous_best'] = best_fitness
        history['best_fitness'].append(best_fitness)
        history['avg_fitness'].append(avg_fitness)
        
        # Métricas
        best_agent = sorted_population[0]
        scores = []
        survival_times = []
        food_reached_counts = []
        obstacle_avoidance_counts = []
        navigation_efficiency = []
        
        # Evaluar juegos
        for eval_game in range(5):
            eval_seed = gen_seed + 10000 + eval_game
            random.seed(eval_seed)
            np.random.seed(eval_seed)
            
            game = SnakeGame(ai_control=True, training_mode=False, headless=True)
            done = False
            score = 0
            steps = 0
            food_reached = 0
            wall_avoidance = 0
            efficient_moves = 0
            total_moves = 0
            
            prev_distance = None
            while not done and steps < 10000:
                state = game.get_state()
                action = best_agent.get_action(state)
                
                curr_distance_before = abs(game.head.x - game.food.x) + abs(game.head.y - game.food.y)
                if prev_distance is None:
                    prev_distance = curr_distance_before
                
                danger_ahead = state[0]
                if danger_ahead and action != 0:
                    wall_avoidance += 1
                
                done, new_score, _ = game.play_step(action)
                steps += 1
                total_moves += 1
                
                curr_distance_after = abs(game.head.x - game.food.x) + abs(game.head.y - game.food.y)
                if curr_distance_after < curr_distance_before:
                    efficient_moves += 1
                
                prev_distance = curr_distance_after
                
                if new_score > score:
                    food_reached += 1
                    score = new_score
            
            scores.append(score)
            survival_times.append(steps)
            food_reached_counts.append(food_reached)
            obstacle_avoidance_counts.append(wall_avoidance / max(1, steps))
            navigation_efficiency.append(efficient_moves / max(1, total_moves))
        
        # Guardar datos
        history['best_score'].append(max(scores))
        history['best_survival_time'].append(max(survival_times))
        history['food_eaten_avg'].append(sum(scores) / len(scores))
        history['survival_time_avg'].append(sum(survival_times) / len(survival_times))
        history['navigation_skill'].append(sum(navigation_efficiency) / len(navigation_efficiency))
        history['obstacle_avoidance'].append(sum(obstacle_avoidance_counts) / len(obstacle_avoidance_counts))
        
        # Desviación estándar
        sample_size = min(20, len(population))
        sample_agents = random.sample(population, sample_size)
        food_scores = []
        
        for agent_idx, agent in enumerate(sample_agents):
            # Test de juegos
            agent_food_scores = []
            for game_idx in range(3):
                eval_seed = gen_seed + 20000 + agent_idx * 100 + game_idx
                random.seed(eval_seed)
                np.random.seed(eval_seed)
                
                game = SnakeGame(ai_control=True, training_mode=False, headless=True)
                done = False
                food_eaten = 0
                steps = 0
                
                while not done and steps < 2000:
                    state = game.get_state()
                    action = agent.get_action(state)
                    done, new_score, _ = game.play_step(action)
                    steps += 1
                    
                    if new_score > food_eaten:
                        food_eaten = new_score
                
                agent_food_scores.append(food_eaten)
            
            food_scores.append(np.mean(agent_food_scores))
        
        # Calcular desviación
        food_std = np.std(food_scores) if food_scores else 0
        history['food_std_dev'].append(food_std)
        
        # Diversidad
        diversity = 0
        sample_size = min(20, len(population))
        if sample_size > 1:
            sample_agents = random.sample(population, sample_size)
            total_dist = 0
            comparisons = 0
            for i in range(sample_size):
                for j in range(i+1, sample_size):
                    dist = np.sum((sample_agents[i].weights - sample_agents[j].weights)**2)**0.5
                    total_dist += dist
                    comparisons += 1
            diversity = total_dist / max(1, comparisons)
        history['diversity_metric'].append(diversity)
        
        # Mejores agentes
        top_agents = [(sorted_population[i], fitness_dict[sorted_indices[i]]) 
                     for i in range(min(3, len(sorted_population)))]
        history['best_agents_history'].append(top_agents)
        
        # Progreso
        print(f"  Mejor fitness: {best_fitness:.2f}, Fitness promedio: {avg_fitness:.2f}")
        print(f"  Comida máxima: {max(scores)}, Supervivencia máx: {max(survival_times)}")
        print(f"  Diversidad: {diversity:.2f}, Nav: {history['navigation_skill'][-1]:.2f}, EvitObs: {history['obstacle_avoidance'][-1]:.2f}")
        
        if generation == generations - 1:
            break
        
        # Elitismo
        effective_elitism = elitism
        if stagnation_counter > 3:
            effective_elitism = min(pop_size // 4, elitism * 2)
            
        new_population = sorted_population[:effective_elitism]
        
        # Reproducción
        while len(new_population) < pop_size:
            tournament_size = min(8, pop_size // 4)
            
            candidates1 = random.sample(range(pop_size // 3), tournament_size)
            best_candidate1 = min(candidates1, key=lambda i: -fitness_dict[sorted_indices[i]])
            parent1 = sorted_population[best_candidate1]
            
            candidates2 = random.sample(range(pop_size // 2), tournament_size)
            best_candidate2 = min(candidates2, key=lambda i: -fitness_dict[sorted_indices[i]])
            parent2 = sorted_population[best_candidate2]
            
            child_table = parent1.crossover(parent2)
            
            base_mutation = mutation_rate * (1.0 - (generation / generations) * 0.4)
            
            if stagnation_counter > 5:
                stagnation_factor = min(0.5, stagnation_counter * 0.05)  
                adjusted_mutation = base_mutation * (1.0 + stagnation_factor)
            else:
                adjusted_mutation = base_mutation
                
            child_table.mutate(adjusted_mutation)
            new_population.append(child_table)
        
        population = new_population
    
    # Resultado
    shared_results[experiment_index] = {
        'best_agent': sorted_population[0],
        'best_fitness': history['best_fitness'][-1],
        'history': history
    }
    
    print(f"Experimento {experiment_index} completado. Mejor fitness: {history['best_fitness'][-1]:.2f}")
    return sorted_population[0], history

def compare_experiments(experiment_results):
    print("\n" + "="*100)
    print(f"{'RESUMEN DE EXPERIMENTOS':^100}")
    print("="*100)
    
    # Gráficos principales
    plt.figure(figsize=(20, 15))
    
    # 1. Fitness Máximo
    plt.subplot(2, 2, 1)
    for i, result in enumerate(experiment_results):
        fitness_data = np.array(result['avg_best_fitness'])
        generations = np.arange(len(fitness_data))
        color = COLORS[i % len(COLORS)]
        
        plt.plot(generations, fitness_data, 
                label=result['config']['name'], 
                color=color, linewidth=2.5)
        
        # Área de tendencia
        if len(fitness_data) > 5:
            z = np.polyfit(generations, fitness_data, 2)
            p = np.poly1d(z)
            trend = p(generations)
            plt.fill_between(generations, trend*0.95, trend*1.05, 
                           color=color, alpha=0.2)
    
    plt.title('Evolución del Fitness Máximo\nMejor desempeño por generación', pad=20)
    plt.xlabel('Número de Generación')
    plt.ylabel('Valor de Fitness')
    plt.legend(title='Configuración', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 2. Fitness Promedio
    plt.subplot(2, 2, 2)
    for i, result in enumerate(experiment_results):
        avg_fitness = np.array(result['avg_avg_fitness'])
        generations = np.arange(len(avg_fitness))
        color = COLORS[i % len(COLORS)]
        
        plt.plot(generations, avg_fitness, 
                label=result['config']['name'], 
                color=color, linewidth=2.5)
        
        # Área de tendencia
        if len(avg_fitness) > 5:
            z = np.polyfit(generations, avg_fitness, 2)
            p = np.poly1d(z)
            trend = p(generations)
            plt.fill_between(generations, trend*0.95, trend*1.05, 
                           color=color, alpha=0.2)
    
    plt.title('Evolución del Fitness Promedio\nDesempeño general de la población', pad=20)
    plt.xlabel('Número de Generación')
    plt.ylabel('Fitness Promedio')
    plt.legend(title='Configuración', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 3. Diversidad
    plt.subplot(2, 2, 3)
    for i, result in enumerate(experiment_results):
        if 'avg_diversity' in result:
            diversity = np.array(result['avg_diversity'])
            generations = np.arange(len(diversity))
            color = COLORS[i % len(COLORS)]
            
            plt.plot(generations, diversity, 
                    label=result['config']['name'],
                    color=color, linewidth=2.5)
            
            # Área de tendencia
            if len(diversity) > 5:
                z = np.polyfit(generations, diversity, 2)
                p = np.poly1d(z)
                trend = p(generations)
                plt.fill_between(generations, trend*0.95, trend*1.05, 
                               color=color, alpha=0.2)
    
    plt.title('Diversidad de la Población\nVariabilidad genética entre individuos', pad=20)
    plt.xlabel('Número de Generación')
    plt.ylabel('Índice de Diversidad')
    plt.legend(title='Configuración', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 4. Capacidad Alimentación
    plt.subplot(2, 2, 4)
    for i, result in enumerate(experiment_results):
        if 'avg_best_score' in result:
            food_data = np.array(result['avg_best_score'])
            generations = np.arange(len(food_data))
            color = COLORS[i % len(COLORS)]
            
            plt.plot(generations, food_data, 
                    label=result['config']['name'],
                    color=color, linewidth=2.5)
            
            # Área de tendencia
            if len(food_data) > 5:
                z = np.polyfit(generations, food_data, 2)
                p = np.poly1d(z)
                trend = p(generations)
                plt.fill_between(generations, trend*0.95, trend*1.05, 
                               color=color, alpha=0.2)
    
    plt.title('Evolución de la Capacidad de Alimentación\nMáxima puntuación alcanzada por generación', pad=20)
    plt.xlabel('Número de Generación')
    plt.ylabel('Cantidad de Alimento Obtenido')
    plt.legend(title='Configuración', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Guardar gráfico
    plt.tight_layout()
    plt.savefig('evolucion_snake.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Mejor configuración
    best_experiment_idx = np.argmax([result['best_fitness'] for result in experiment_results])
    best_result = experiment_results[best_experiment_idx]
    
    print("\n" + "="*50)
    print("MEJOR CONFIGURACIÓN ENCONTRADA:")
    print(f"Experimento: {best_result['config']['name']}")
    print(f"Fitness máximo: {best_result['best_fitness']:.2f}")
    
    if 'avg_best_score' in best_result and best_result['avg_best_score']:
        print(f"Puntuación máxima: {max(best_result['avg_best_score']):.2f}")
        
    print(f"\nParámetros:")
    print(f"  - Tamaño de población: {best_result['config']['pop_size']}")
    print(f"  - Generaciones: {best_result['config']['generations']}")
    print(f"  - Tasa de mutación: {best_result['config']['mutation_rate']}")
    print(f"  - Tamaño de élite: {best_result['config']['elitism']}")
    print("="*50)
    
    print("\nExperimentos evolutivos completados. Gráfica guardada como 'evolucion_snake.png'")

def run_experiments():
    # Experimentos
    print("\n" + "="*50)
    print("EXPERIMENTOS EVOLUTIVOS COMPARATIVOS PARA SNAKE")
    print("="*50)
    print("\nSe ejecutarán 3 experimentos con diferentes configuraciones.")
    print("Cada experimento ejecutará el proceso evolutivo completo.")
    print("Al finalizar, se compararán los resultados de los tres experimentos.\n")
    
    # Configuraciones
    experiment_configs = [
        {
            'name': 'Experimento 1',
            'pop_size': 40,
            'generations': 100,
            'mutation_rate': 0.25,
            'elitism': 10,
            'base_seed': (int(time.time()) % 10000) + 10000
        },
        {
            'name': 'Experimento 2',
            'pop_size': 50,
            'generations': 100,
            'mutation_rate': 0.35,
            'elitism': 6,
            'base_seed': int(time.time()) % 10000
        },
        {
            'name': 'Experimento 3',
            'pop_size': 60,
            'generations': 100,
            'mutation_rate': 0.45,
            'elitism': 4,
            'base_seed': (int(time.time()) % 10000) + 5000
        }
    ]
    
    # Inicialización
    results = []
    total_experiments = len(experiment_configs)
    shared_results = {}
    
    print("\nEjecutando experimentos secuencialmente...")
    
    for i, config in enumerate(experiment_configs):
        print(f"\nEJECUTANDO EXPERIMENTO {i+1}/{total_experiments}: {config['name']}")
        print("="*50)
        print(f"Configuración: Población={config['pop_size']}, Generaciones={config['generations']}, ")
        print(f"Mutación={config['mutation_rate']}, Élite={config['elitism']}")
        print("="*50)
        
        start_time = time.time()
        best_agent, history = run_experiment(config, shared_results, i)
        execution_time = time.time() - start_time
        
        # Recopilación datos
        result = {
            'config': config,
            'best_agent': best_agent,
            'best_fitness': history['best_fitness'][-1],
            'avg_best_fitness': history['best_fitness'],
            'avg_avg_fitness': history['avg_fitness'],
            'avg_execution_time': execution_time,
            'avg_best_score': history['best_score'] if 'best_score' in history else [],
            'avg_survival_time': history['best_survival_time'] if 'best_survival_time' in history else [],
            'avg_food_eaten': history['food_eaten_avg'] if 'food_eaten_avg' in history else [],
            'avg_survival_avg': history['survival_time_avg'] if 'survival_time_avg' in history else [],
            'avg_navigation': history['navigation_skill'] if 'navigation_skill' in history else [],
            'avg_obstacle_avoidance': history['obstacle_avoidance'] if 'obstacle_avoidance' in history else [],
            'avg_diversity': history['diversity_metric'] if 'diversity_metric' in history else [],
            'top_agents_history': history['best_agents_history'] if 'best_agents_history' in history else [],
            'food_std_dev': history['food_std_dev'] if 'food_std_dev' in history else []
        }
        results.append(result)
        
        # Resumen
        print(f"\nResumen del Experimento {i+1}: {config['name']}")
        print(f"Fitness máximo alcanzado: {history['best_fitness'][-1]:.2f}")
        
        if 'best_score' in history and history['best_score']:
            print(f"Puntuación máxima: {history['best_score'][-1]:.2f}")
        if 'best_survival_time' in history and history['best_survival_time']:
            print(f"Tiempo de supervivencia máximo: {history['best_survival_time'][-1]:.2f}")
        if 'navigation_skill' in history and history['navigation_skill']:
            print(f"Habilidad de navegación final: {history['navigation_skill'][-1]:.2f}")
        if 'obstacle_avoidance' in history and history['obstacle_avoidance']:
            print(f"Habilidad de evitación de obstáculos final: {history['obstacle_avoidance'][-1]:.2f}")
        if 'diversity_metric' in history and history['diversity_metric']:
            print(f"Diversidad genética final: {history['diversity_metric'][-1]:.2f}")
            
        print(f"Tiempo de ejecución: {execution_time:.2f} segundos")
        print("="*50)
    
    print("\nTodos los experimentos evolutivos han sido completados.")
    
    print("\nGenerando gráficas comparativas de los tres experimentos...")
    compare_experiments(results)

if __name__ == "__main__":
    run_experiments()
