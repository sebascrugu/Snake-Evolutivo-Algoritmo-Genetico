#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Módulo de experimentos con algoritmo genético para Snake

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

# Colores para gráficas
COLORS = [
    '#8FBC8F',  # Verde sage (Mutación)
    '#DDA0DD',  # Violeta plum (Reproducción)
    '#F0E68C',  # Khaki (Selección)
]

# Estilos para gráficas
STYLE = {
    'figure.figsize': (20, 15),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
}

plt.style.use('seaborn')
plt.rcParams.update(STYLE)

# Procesos paralelos para MacBook Air M1
N_JOBS = 6

def evaluate_agent_fitness(agent, agent_index, seed):
    # Evalúa fitness de un agente
    agent_seed = seed + agent_index
    random.seed(agent_seed)
    np.random.seed(agent_seed)

    ga = GeneticAlgorithm()
    fitness_value = ga.fitness(agent, num_games=7, show_game=False, silent=True)
    return fitness_value


def run_experiment(config, shared_results, experiment_index):
    # Ejecuta experimento evolutivo
    pop_size = config.get('pop_size', 50)
    generations = config.get('generations', 50)
    mutation_rate = config.get('mutation_rate', 0.1)
    elitism = config.get('elitism', 5)
    base_seed = config.get('base_seed', int(time.time()))
    
    print(f"Iniciando experimento {experiment_index} con semilla base: {base_seed}")
    
    # Inicializar población
    population = [DecisionTable() for _ in range(pop_size)]
    
    # Historial de métricas
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
    
    # Ciclo evolutivo
    for generation in range(generations):
        gen_seed = base_seed + (generation * 100)
        print(f"Generación {generation+1}/{generations}, semilla: {gen_seed}")
        
        # Evaluación paralela
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
        
        # Detección de estancamiento
        if history['previous_best'] is not None:
            if best_fitness <= history['previous_best'] * 1.01:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
        
        history['previous_best'] = best_fitness
        history['best_fitness'].append(best_fitness)
        history['avg_fitness'].append(avg_fitness)
        
        # Obtener métricas detalladas
        best_agent = sorted_population[0]
        scores = []
        survival_times = []
        food_reached_counts = []
        obstacle_avoidance_counts = []
        navigation_efficiency = []
        
        # Evaluar en 5 juegos
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
        
        # Guardar métricas
        history['best_score'].append(max(scores))
        history['best_survival_time'].append(max(survival_times))
        history['food_eaten_avg'].append(sum(scores) / len(scores))
        history['survival_time_avg'].append(sum(survival_times) / len(survival_times))
        history['navigation_skill'].append(sum(navigation_efficiency) / len(navigation_efficiency))
        history['obstacle_avoidance'].append(sum(obstacle_avoidance_counts) / len(obstacle_avoidance_counts))
        
        # Calcular desviación estándar de la comida obtenida por los agentes
        # Evaluar una muestra de agentes para medir la estabilidad
        sample_size = min(20, len(population))
        sample_agents = random.sample(population, sample_size)
        food_scores = []
        
        for agent_idx, agent in enumerate(sample_agents):
            # Ejecutar 3 juegos por agente para medir su rendimiento en comida
            agent_food_scores = []
            for game_idx in range(3):
                eval_seed = gen_seed + 20000 + agent_idx * 100 + game_idx
                random.seed(eval_seed)
                np.random.seed(eval_seed)
                
                game = SnakeGame(ai_control=True, training_mode=False, headless=True)
                done = False
                food_eaten = 0
                steps = 0
                
                while not done and steps < 2000:  # Limitar a 2000 pasos para eficiencia
                    state = game.get_state()
                    action = agent.get_action(state)
                    done, new_score, _ = game.play_step(action)
                    steps += 1
                    
                    if new_score > food_eaten:
                        food_eaten = new_score
                
                agent_food_scores.append(food_eaten)
            
            # Usar el promedio de comida para este agente
            food_scores.append(np.mean(agent_food_scores))
        
        # Calcular la desviación estándar
        food_std = np.std(food_scores) if food_scores else 0
        history['food_std_dev'].append(food_std)
        
        # Calcular diversidad
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
        
        # Guardar mejores agentes
        top_agents = [(sorted_population[i], fitness_dict[sorted_indices[i]]) 
                     for i in range(min(3, len(sorted_population)))]
        history['best_agents_history'].append(top_agents)
        
        # Mostrar progreso
        print(f"  Mejor fitness: {best_fitness:.2f}, Fitness promedio: {avg_fitness:.2f}")
        print(f"  Comida máxima: {max(scores)}, Supervivencia máx: {max(survival_times)}")
        print(f"  Diversidad: {diversity:.2f}, Nav: {history['navigation_skill'][-1]:.2f}, EvitObs: {history['obstacle_avoidance'][-1]:.2f}")
        
        if generation == generations - 1:
            break
        
        # Elitismo adaptativo
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
    
    # Guardar resultado
    shared_results[experiment_index] = {
        'best_agent': sorted_population[0],
        'best_fitness': history['best_fitness'][-1],
        'history': history
    }
    
    print(f"Experimento {experiment_index} completado. Mejor fitness: {history['best_fitness'][-1]:.2f}")
    return sorted_population[0], history

def compare_experiments(experiment_results):
    # Compara resultados de experimentos con gráficas
    print("\n" + "="*100)
    print(f"{'RESUMEN DE EXPERIMENTOS':^100}")
    print("="*100)
    print(f"{'Experimento':^15} | {'Fitness Máx':^12} | {'Comida Máx':^12} | {'Superv. Máx':^12} | {'Nav. Habilidad':^14} | {'Evit. Obstác.':^14} | {'Diversidad':^12}")
    print("-"*100)
    
    # Crear figura para el grupo 2: Comportamiento de la Serpiente (ahora primero)
    plt.figure(figsize=(20, 15))
    
    # Gráfico 1: Puntuación Máxima de Alimento
    plt.subplot(2, 2, 1)
    for i, result in enumerate(experiment_results):
        if 'avg_best_score' in result:
            food_data = np.array(result['avg_best_score'])
            generations = np.arange(len(food_data))
            color = COLORS[i % len(COLORS)]
            
            plt.plot(generations, food_data, 
                    label=result['config']['name'],
                    color=color, linewidth=2.5)
    
    plt.axhline(y=35, color='gray', linestyle='--', alpha=0.5)
    plt.text(len(generations) * 0.05, 36, 'Rendimiento Elite', fontsize=10)
    
    plt.title('Evolución de la Capacidad de Alimentación\nMáxima puntuación alcanzada por generación', pad=20)
    plt.xlabel('Número de Generación')
    plt.ylabel('Cantidad de Alimento Obtenido')
    plt.legend(title='Configuración', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Gráfico 2: Tiempo de Supervivencia
    plt.subplot(2, 2, 2)
    for i, result in enumerate(experiment_results):
        if 'avg_survival_time' in result:
            survival_data = np.array(result['avg_survival_time'])
            generations = np.arange(len(survival_data))
            color = COLORS[i % len(COLORS)]
            
            plt.plot(generations, survival_data, 
                    label=result['config']['name'],
                    color=color, linewidth=2.5)
    
    elite_threshold = 0
    for result in experiment_results:
        if 'avg_survival_time' in result:
            elite_threshold = max(elite_threshold, np.max(result['avg_survival_time']) * 0.8)
    
    plt.axhline(y=elite_threshold, color='gray', linestyle='--', alpha=0.5)
    plt.text(len(generations) * 0.05, elite_threshold * 1.02, 'Umbral Elite', fontsize=10)
    
    plt.title('Longevidad de los Agentes\nTiempo máximo de supervivencia por generación', pad=20)
    plt.xlabel('Número de Generación')
    plt.ylabel('Pasos de Supervivencia')
    plt.legend(title='Configuración', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Gráfico 3: Eficiencia de Adquisición
    plt.subplot(2, 2, 3)
    for i, result in enumerate(experiment_results):
        if 'avg_best_score' in result and 'avg_survival_time' in result:
            food_data = np.array(result['avg_best_score'])
            survival_data = np.array(result['avg_survival_time'])
            efficiency_data = food_data / (survival_data + 1)
            generations = np.arange(len(efficiency_data))
            color = COLORS[i % len(COLORS)]
            
            window = 5
            if len(efficiency_data) >= window:
                smoothed = np.convolve(efficiency_data, np.ones(window)/window, mode='valid')
                plt.plot(generations[window-1:], smoothed,
                        label=result['config']['name'], 
                        color=color, linewidth=2.5)
    
    plt.title('Eficiencia en la Búsqueda de Alimento\nRelación entre alimento obtenido y tiempo invertido', pad=20)
    plt.xlabel('Número de Generación')
    plt.ylabel('Índice de Eficiencia (Alimento/Tiempo)')
    plt.legend(title='Configuración', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Gráfico 4: Estabilidad de Rendimiento
    plt.subplot(2, 2, 4)
    for i, result in enumerate(experiment_results):
        if 'food_std_dev' in result and len(result['food_std_dev']) > 0:
            std_data = np.array(result['food_std_dev'])
            generations = np.arange(len(std_data))
            color = COLORS[i % len(COLORS)]
            
            plt.plot(generations, std_data, 
                    label=result['config']['name'],
                    color=color, linewidth=2.5)
            
            # Área sombreada para la tendencia
            if len(std_data) > 5:
                z = np.polyfit(generations, std_data, 2)
                p = np.poly1d(z)
                trend = p(generations)
                plt.fill_between(generations, trend*0.9, trend*1.1, 
                               color=color, alpha=0.2)
    
    plt.title('Consistencia del Comportamiento\nVariabilidad en la obtención de alimento', pad=20)
    plt.xlabel('Número de Generación')
    plt.ylabel('Desviación Estándar del Rendimiento')
    plt.legend(title='Configuración', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comportamiento_serpiente.png', dpi=300, bbox_inches='tight')
    
    # Crear figura para el grupo 1: Evolución del Fitness
    plt.figure(figsize=(20, 15))
    
    # Gráfico 1: Fitness Máximo
    plt.subplot(2, 2, 1)
    for i, result in enumerate(experiment_results):
        fitness_data = np.array(result['avg_best_fitness'])
        generations = np.arange(len(fitness_data))
        color = COLORS[i % len(COLORS)]
        
        plt.plot(generations, fitness_data, 
                label=result['config']['name'], 
                color=color, linewidth=2.5)
        
        # Puntos de inflexión importantes
        if len(fitness_data) > 10:
            diff = np.diff(fitness_data)
            threshold = np.std(diff) * 2
            peaks = np.where(diff > threshold)[0]
            
            if len(peaks) > 3:
                top_peaks = peaks[np.argsort(diff[peaks])[-3:]]
                for peak in top_peaks:
                    plt.scatter(peak, fitness_data[peak], 
                              color=color, s=100, zorder=5)
                    plt.annotate(f'+{diff[peak]:.1f}', 
                               xy=(peak, fitness_data[peak]),
                               xytext=(10, 10), 
                               textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.3', 
                                       fc='white', alpha=0.7))
    
    plt.title('Evolución del Fitness Máximo\nMejor desempeño por generación', pad=20)
    plt.xlabel('Número de Generación')
    plt.ylabel('Valor de Fitness')
    plt.legend(title='Configuración', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Gráfico 2: Fitness Promedio
    plt.subplot(2, 2, 2)
    for i, result in enumerate(experiment_results):
        avg_fitness = np.array(result['avg_avg_fitness'])
        generations = np.arange(len(avg_fitness))
        color = COLORS[i % len(COLORS)]
        
        plt.plot(generations, avg_fitness, 
                label=result['config']['name'], 
                color=color, linewidth=2.5)
        
        # Área sombreada para mostrar tendencia
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
    
    # Gráfico 3: Diversidad Genética
    plt.subplot(2, 2, 3)
    for i, result in enumerate(experiment_results):
        if 'avg_diversity' in result:
            diversity = np.array(result['avg_diversity'])
            generations = np.arange(len(diversity))
            color = COLORS[i % len(COLORS)]
            
            plt.plot(generations, diversity, 
                    label=result['config']['name'],
                    color=color, linewidth=2.5)
            
            # Suavizado y área de confianza
            if len(diversity) > 10:
                window = 5
                smoothed = np.convolve(diversity, np.ones(window)/window, mode='valid')
                smooth_gen = generations[window-1:]
                plt.plot(smooth_gen, smoothed, 
                        color=color, linestyle='--', alpha=0.8)
                plt.fill_between(smooth_gen, 
                               smoothed*0.9, smoothed*1.1,
                               color=color, alpha=0.2)
    
    plt.title('Diversidad de la Población\nVariabilidad genética entre individuos', pad=20)
    plt.xlabel('Número de Generación')
    plt.ylabel('Índice de Diversidad')
    plt.legend(title='Configuración', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Gráfico 4: Tasa de Mejora
    plt.subplot(2, 2, 4)
    window_size = 10
    
    for i, result in enumerate(experiment_results):
        fitness_data = np.array(result['avg_best_fitness'])
        color = COLORS[i % len(COLORS)]
        
        improvement_rates = []
        generations = []
        
        for j in range(window_size, len(fitness_data), window_size):
            start_val = fitness_data[j - window_size]
            end_val = fitness_data[j]
            improvement = ((end_val - start_val) / start_val) * 100 if start_val != 0 else 0
            improvement_rates.append(improvement)
            generations.append(j)
        
        plt.plot(generations, improvement_rates, 
                label=result['config']['name'],
                color=color, marker='o', linewidth=2.5)
        
        if len(improvement_rates) > 2:
            z = np.polyfit(generations, improvement_rates, 2)
            p = np.poly1d(z)
            plt.plot(generations, p(generations), 
                    color=color, linestyle='--', alpha=0.5)
    
    plt.title('Velocidad de Evolución\nTasa de mejora entre generaciones', pad=20)
    plt.xlabel('Número de Generación')
    plt.ylabel('Tasa de Mejora (%)')
    plt.legend(title='Configuración', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('evolucion_fitness.png', dpi=300, bbox_inches='tight')
    
    # Gráfico detallado de Estabilidad de Rendimiento
    plt.figure(figsize=(15, 10))
    
    for i, result in enumerate(experiment_results):
        if 'food_std_dev' in result and len(result['food_std_dev']) > 0:
            std_data = np.array(result['food_std_dev'])
            generations = np.arange(len(std_data))
            color = COLORS[i % len(COLORS)]
            
            plt.plot(generations, std_data, 
                   label=f"{result['config']['name']} - Desviación",
                   color=color, marker='o', markersize=4, linewidth=2.5)
            
            if 'avg_food_eaten' in result and len(result['avg_food_eaten']) > 0:
                food_avg = np.array(result['avg_food_eaten'])
                cv_data = std_data / np.maximum(food_avg, 0.001)
                
                plt.plot(generations, cv_data,
                       label=f"{result['config']['name']} - CV",
                       color=color, linestyle='--', linewidth=2)
                
                if len(cv_data) > 10:
                    cv_diff = np.diff(cv_data)
                    threshold = np.std(cv_diff) * 1.5
                    change_points = np.where(abs(cv_diff) > threshold)[0]
                    
                    if len(change_points) > 3:
                        top_indices = np.argsort(-abs(cv_diff[change_points]))[:3]
                        change_points = change_points[top_indices]
                    
                    for cp in change_points:
                        plt.scatter(generations[cp], cv_data[cp], 
                                  s=100, color=color, edgecolor='black', zorder=5)
                        
                        if cv_diff[cp] < 0:
                            change_type = "↓ Mejora"
                        else:
                            change_type = "↑ Variación"
                            
                        plt.annotate(change_type,
                                   xy=(generations[cp], cv_data[cp]),
                                   xytext=(10, 10), textcoords='offset points',
                                   bbox=dict(boxstyle='round,pad=0.3', 
                                           fc='white', alpha=0.7))
    
    plt.title('Análisis Detallado de la Estabilidad del Rendimiento\n' + 
              'Evolución de la consistencia en la obtención de alimento', 
              pad=20)
    plt.xlabel('Número de Generación')
    plt.ylabel('Métricas de Variabilidad')
    plt.legend(title='Configuración y Métrica', 
              bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Anotaciones explicativas
    plt.figtext(0.02, 0.97, 
               'Desviación Estándar: Dispersión absoluta en la cantidad de alimento obtenido',
               fontsize=10, ha='left', va='top',
               bbox=dict(boxstyle='round,pad=0.5', 
                        fc='white', ec='gray', alpha=0.8))
               
    plt.figtext(0.02, 0.93, 
               'Coeficiente de Variación (CV): Variabilidad relativa al rendimiento promedio\n' +
               'Un CV menor indica mayor consistencia proporcional',
               fontsize=10, ha='left', va='top',
               bbox=dict(boxstyle='round,pad=0.5', 
                        fc='white', ec='gray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('estabilidad_rendimiento.png', dpi=300, bbox_inches='tight')
    plt.show()

def run_experiments():
    # Ejecuta experimentos comparativos
    print("\n" + "="*50)
    print("EXPERIMENTOS EVOLUTIVOS COMPARATIVOS PARA SNAKE")
    print("="*50)
    print("\nSe ejecutarán 3 experimentos con diferentes configuraciones.")
    print("Cada experimento ejecutará el proceso evolutivo completo.")
    print("Al finalizar, se compararán los resultados de los tres experimentos.\n")
    
    # Configuraciones
    experiment_configs = [
        {
            'name': 'Configuración Óptima',
            'pop_size': 50,
            'generations': 100,
            'mutation_rate': 0.35,
            'elitism': 6,
            'base_seed': int(time.time()) % 10000
        },
        {
            'name': 'Alta Diversidad Balanceada',
            'pop_size': 60,
            'generations': 100,
            'mutation_rate': 0.45,
            'elitism': 4,
            'base_seed': (int(time.time()) % 10000) + 5000
        },
        {
            'name': 'Convergencia Optimizada',
            'pop_size': 40,
            'generations': 100,
            'mutation_rate': 0.25,
            'elitism': 10,
            'base_seed': (int(time.time()) % 10000) + 10000
        }
    ]
    
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
    
    best_experiment_idx = np.argmax([result['best_fitness'] for result in results])
    best_result = results[best_experiment_idx]
    
    print("\n" + "="*50)
    print("MEJOR CONFIGURACIÓN ENCONTRADA:")
    print(f"Experimento: {best_result['config']['name']}")
    print(f"Fitness máximo: {best_result['best_fitness']:.2f}")
    
    if 'avg_best_score' in best_result and best_result['avg_best_score']:
        print(f"Puntuación máxima: {max(best_result['avg_best_score']):.2f}")
    if 'avg_survival_time' in best_result and best_result['avg_survival_time']:
        print(f"Tiempo de supervivencia máximo: {max(best_result['avg_survival_time']):.2f}")
    if 'avg_navigation' in best_result and best_result['avg_navigation']:
        nav_end = best_result['avg_navigation'][-1]
        print(f"Habilidad de navegación final: {nav_end:.2f}")
    if 'avg_obstacle_avoidance' in best_result and best_result['avg_obstacle_avoidance']:
        avoid_end = best_result['avg_obstacle_avoidance'][-1]
        print(f"Habilidad de evitación final: {avoid_end:.2f}")
        
    print(f"\nParámetros:")
    print(f"  - Tamaño de población: {best_result['config']['pop_size']}")
    print(f"  - Generaciones: {best_result['config']['generations']}")
    print(f"  - Tasa de mutación: {best_result['config']['mutation_rate']}")
    print(f"  - Tamaño de élite: {best_result['config']['elitism']}")
    print("="*50)
    
    print("\nExperimentos evolutivos completados. Gráficas guardadas como 'evolucion_fitness.png', 'comportamiento_serpiente.png', 'eficiencia_adquisicion.png' y 'estabilidad_rendimiento.png'")

if __name__ == "__main__":
    run_experiments()
