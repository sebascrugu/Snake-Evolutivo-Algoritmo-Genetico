#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para realizar experimentos con el algoritmo genético del juego Snake.
Implementa diferentes configuraciones y comparaciones de parámetros.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import random
import os
from matplotlib.ticker import MaxNLocator
from snakepy import GeneticAlgorithm, DecisionTable, Action, Direction, Point, SnakeGame
from snakepy import BLOCK_SIZE  # Importar la constante BLOCK_SIZE desde snakepy
from joblib import Parallel, delayed
import multiprocessing as mp

# Colores para las gráficas
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# Número de procesos paralelos optimizado para MacBook Air M1
N_JOBS = 6

# Función auxiliar para evaluación paralela de fitness
def evaluate_agent_fitness(agent, agent_index, seed):
    """
    Evalúa el fitness de un agente en el juego Snake.
    
    Args:
        agent: Agente a evaluar
        agent_index: Índice del agente
        seed: Semilla para la aleatoriedad
    
    Returns:
        Fitness del agente
    """
    agent_seed = seed + agent_index
    random.seed(agent_seed)
    np.random.seed(agent_seed)

    #Crear semilla única para cada agente
    ga = GeneticAlgorithm()

    fitness_value = ga.fitness(agent, num_games=7, show_game=False, silent=True)

    return fitness_value


def run_experiment(config, shared_results, experiment_index):
    """
    Ejecuta un experimento de evolución de agentes Snake utilizando algoritmos genéticos.
    
    Args:
        config: Configuración del experimento 
        shared_results: Diccionario compartido para almacenar resultados
        experiment_index: Índice del experimento actual
    """
    # Extraer parámetros de configuración
    pop_size = config.get('pop_size', 50)  # Valor predeterminado aumentado a 50
    generations = config.get('generations', 50)  # Valor predeterminado aumentado a 50
    mutation_rate = config.get('mutation_rate', 0.1)
    elitism = config.get('elitism', 5)
    base_seed = config.get('base_seed', int(time.time()))
    
    print(f"Iniciando experimento {experiment_index} con semilla base: {base_seed}")
    
    # Inicializar población aleatoria - ahora directamente con DecisionTable
    population = [DecisionTable() for _ in range(pop_size)]
    
    # Historia para seguimiento de progreso
    history = {
        'best_fitness': [],          # Fitness real sin suavizado
        'avg_fitness': [],           # Fitness promedio real sin suavizado
        'best_score': [],            # Puntuación máxima (comida comida)
        'best_survival_time': [],    # Tiempo de supervivencia máximo
        'diversity_metric': [],      # Métrica de diversidad genética
        'best_agents_history': [],   # Mejores agentes históricos (Top 3)
        'food_eaten_avg': [],        # Comida comida promedio
        'survival_time_avg': [],     # Tiempo promedio de supervivencia
        'navigation_skill': [],      # Habilidad de navegación (0-1)
        'obstacle_avoidance': [],    # Habilidad de evitación de obstáculos (0-1)
        'previous_best': None        # Para seguimiento de mejoras
    }
    
    # Variables para suavizar mutación y mantener estabilidad
    stagnation_counter = 0
    best_fitness_ever = 0
    
    # Mejor agente histórico
    best_agent_ever = None
    
    # Ciclo de evolución
    for generation in range(generations):
        # Crear una semilla única para esta generación
        gen_seed = base_seed + (generation * 100)
        print(f"Generación {generation+1}/{generations}, semilla: {gen_seed}")
        
        # Evaluar fitness de toda la población en paralelo usando joblib
        fitness_results = Parallel(n_jobs=N_JOBS)(
            delayed(evaluate_agent_fitness)(agent, idx, gen_seed)
            for idx, agent in enumerate(population)
        )
        
        # Crear un diccionario de fitness para cada agente
        fitness_dict = {idx: fitness for idx, fitness in enumerate(fitness_results)}
        
        # Ordenar población por fitness (descendente)
        sorted_indices = sorted(fitness_dict.keys(), key=lambda idx: fitness_dict[idx], reverse=True)
        sorted_population = [population[idx] for idx in sorted_indices]
        
        # Calcular estadísticas
        best_fitness = fitness_dict[sorted_indices[0]]
        avg_fitness = sum(fitness_dict.values()) / len(fitness_dict)
        
        # Mantener mejor fitness y agente histórico
        if best_fitness > best_fitness_ever:
            best_fitness_ever = best_fitness
            best_agent_ever = sorted_population[0]
        
        # Detectar estancamiento
        if history['previous_best'] is not None:
            if best_fitness <= history['previous_best'] * 1.01:  # Menos de 1% de mejora
                stagnation_counter += 1
            else:
                stagnation_counter = 0  # Reiniciar si hay mejora significativa
        
        # Guardar el mejor fitness actual para la siguiente generación
        history['previous_best'] = best_fitness
        
        # ELIMINAMOS SUAVIZADO: Guardamos valores reales de fitness
        history['best_fitness'].append(best_fitness)
        history['avg_fitness'].append(avg_fitness)
        
        # Calcular y guardar métricas adicionales
        
        # 1. Obtener datos de rendimiento del mejor agente (simulando 5 juegos)
        best_agent = sorted_population[0]
        scores = []
        survival_times = []
        food_reached_counts = []
        obstacle_avoidance_counts = []
        navigation_efficiency = []
        
        # Evaluar el mejor agente en 5 juegos para métricas detalladas
        for eval_game in range(5):
            # Crear una semilla consistente para esta evaluación
            eval_seed = gen_seed + 10000 + eval_game
            random.seed(eval_seed)
            np.random.seed(eval_seed)
            
            # Crear juego para evaluación detallada
            game = SnakeGame(ai_control=True, training_mode=False, headless=True)
            done = False
            score = 0
            steps = 0
            food_reached = 0
            wall_avoidance = 0
            efficient_moves = 0
            total_moves = 0
            
            # Jugar hasta que termine
            prev_distance = None
            while not done and steps < 10000:
                state = game.get_state()
                action = best_agent.get_action(state)
                
                # Calcular distancia a la comida antes de moverse
                curr_distance_before = abs(game.head.x - game.food.x) + abs(game.head.y - game.food.y)
                if prev_distance is None:
                    prev_distance = curr_distance_before
                
                # Detectar si va a evitar una pared
                danger_ahead = state[0]
                if danger_ahead and action != 0:  # No va hacia adelante cuando hay peligro
                    wall_avoidance += 1
                
                # Ejecutar acción
                done, new_score, _ = game.play_step(action)
                steps += 1
                total_moves += 1
                
                # Verificar si mejoró acercamiento a la comida
                curr_distance_after = abs(game.head.x - game.food.x) + abs(game.head.y - game.food.y)
                if curr_distance_after < curr_distance_before:
                    efficient_moves += 1
                
                prev_distance = curr_distance_after
                
                # Verificar si consiguió comida
                if new_score > score:
                    food_reached += 1
                    score = new_score
            
            # Guardar métricas de este juego
            scores.append(score)
            survival_times.append(steps)
            food_reached_counts.append(food_reached)
            # Calcular tasa de evitación de obstáculos
            obstacle_avoidance_counts.append(wall_avoidance / max(1, steps))
            # Calcular eficiencia de navegación
            navigation_efficiency.append(efficient_moves / max(1, total_moves))
        
        # Guardar métricas promediadas en el historial
        history['best_score'].append(max(scores))
        history['best_survival_time'].append(max(survival_times))
        history['food_eaten_avg'].append(sum(scores) / len(scores))
        history['survival_time_avg'].append(sum(survival_times) / len(survival_times))
        history['navigation_skill'].append(sum(navigation_efficiency) / len(navigation_efficiency))
        history['obstacle_avoidance'].append(sum(obstacle_avoidance_counts) / len(obstacle_avoidance_counts))
        
        # 2. Calcular diversidad de la población (usando distancia promedio entre pesos)
        diversity = 0
        sample_size = min(20, len(population))  # Usar muestra para optimizar cálculo
        if sample_size > 1:
            sample_agents = random.sample(population, sample_size)
            total_dist = 0
            comparisons = 0
            for i in range(sample_size):
                for j in range(i+1, sample_size):
                    # Distancia euclidiana entre pesos
                    dist = np.sum((sample_agents[i].weights - sample_agents[j].weights)**2)**0.5
                    total_dist += dist
                    comparisons += 1
            diversity = total_dist / max(1, comparisons)
        history['diversity_metric'].append(diversity)
        
        # 3. Guardar los mejores agentes históricos (top 3)
        top_agents = [(sorted_population[i], fitness_dict[sorted_indices[i]]) 
                     for i in range(min(3, len(sorted_population)))]
        history['best_agents_history'].append(top_agents)
        
        # Mostrar progreso con las nuevas métricas
        print(f"  Mejor fitness: {best_fitness:.2f}, Fitness promedio: {avg_fitness:.2f}")
        print(f"  Comida máxima: {max(scores)}, Supervivencia máx: {max(survival_times)}")
        print(f"  Diversidad: {diversity:.2f}, Nav: {history['navigation_skill'][-1]:.2f}, EvitObs: {history['obstacle_avoidance'][-1]:.2f}")
        
        # Verificar si estamos en la última generación
        if generation == generations - 1:
            break
        
        # Crear nueva población con elitismo incrementado para mayor estabilidad
        # Aumentamos el elitismo cuando hay estancamiento para preservar lo bueno
        effective_elitism = elitism
        if stagnation_counter > 3:
            effective_elitism = min(pop_size // 4, elitism * 2)  # Duplicar hasta un 25% de la población
            
        new_population = sorted_population[:effective_elitism]  # Mantener mejores individuos
        
        # Completar población mediante reproducción más conservadora
        while len(new_population) < pop_size:
            # Selección mediante torneo más grande para mayor estabilidad
            tournament_size = min(8, pop_size // 4)
            
            # Primer torneo - enfocado en los mejores individuos
            candidates1 = random.sample(range(pop_size // 3), tournament_size)  # Del primer tercio
            best_candidate1 = min(candidates1, key=lambda i: -fitness_dict[sorted_indices[i]])
            parent1 = sorted_population[best_candidate1]
            
            # Segundo torneo - más amplio para mantener diversidad
            candidates2 = random.sample(range(pop_size // 2), tournament_size)  # De la primera mitad
            best_candidate2 = min(candidates2, key=lambda i: -fitness_dict[sorted_indices[i]])
            parent2 = sorted_population[best_candidate2]
            
            # Crear hijo mediante cruce (directamente con DecisionTable)
            child_table = parent1.crossover(parent2)
            
            # Mutación adaptativa con suavizado para evitar cambios bruscos
            # Reducimos gradualmente la mutación a medida que avanza la evolución
            base_mutation = mutation_rate * (1.0 - (generation / generations) * 0.4)
            
            # Aumentamos ligeramente si hay estancamiento, pero sin cambios bruscos
            if stagnation_counter > 5:
                # Incremento gradual basado en cuánto tiempo llevamos estancados
                stagnation_factor = min(0.5, stagnation_counter * 0.05)  
                adjusted_mutation = base_mutation * (1.0 + stagnation_factor)
            else:
                adjusted_mutation = base_mutation
                
            # Aplicar mutación con la tasa ajustada
            child_table.mutate(adjusted_mutation)
            
            # Añadir a nueva población
            new_population.append(child_table)
        
        # Actualizar población
        population = new_population
    
    # Guardar resultado en estructura compartida
    shared_results[experiment_index] = {
        'best_agent': sorted_population[0],  # El mejor agente de la última generación
        'best_fitness': history['best_fitness'][-1],
        'history': history
    }
    
    print(f"Experimento {experiment_index} completado. Mejor fitness: {history['best_fitness'][-1]:.2f}")
    return sorted_population[0], history

def compare_experiments(experiment_results):
    """
    Compara los resultados de diferentes experimentos generando gráficas.
    
    Args:
        experiment_results: Lista de resultados de experimentos
    """
    # Mostrar tabla de resumen antes de las gráficas
    print("\n" + "="*100)
    print(f"{'RESUMEN DE EXPERIMENTOS':^100}")
    print("="*100)
    print(f"{'Experimento':^15} | {'Fitness Máx':^12} | {'Comida Máx':^12} | {'Superv. Máx':^12} | {'Nav. Habilidad':^14} | {'Evit. Obstác.':^14} | {'Diversidad':^12}")
    print("-"*100)
    
    for result in experiment_results:
        name = result['config']['name']
        max_fitness = max(result['avg_best_fitness'])
        
        # Incluir métricas adicionales si están disponibles
        max_food = max(result['avg_best_score']) if 'avg_best_score' in result else 'N/A'
        max_survival = max(result['avg_survival_time']) if 'avg_survival_time' in result else 'N/A'
        avg_nav = np.mean(result['avg_navigation']) if 'avg_navigation' in result else 'N/A'
        avg_avoid = np.mean(result['avg_obstacle_avoidance']) if 'avg_obstacle_avoidance' in result else 'N/A'
        avg_diversity = np.mean(result['avg_diversity']) if 'avg_diversity' in result else 'N/A'
        
        # Formatear valores para visualización
        max_food_str = f"{max_food:.2f}" if isinstance(max_food, (int, float)) else max_food
        max_survival_str = f"{max_survival:.2f}" if isinstance(max_survival, (int, float)) else max_survival
        avg_nav_str = f"{avg_nav:.2f}" if isinstance(avg_nav, (int, float)) else avg_nav
        avg_avoid_str = f"{avg_avoid:.2f}" if isinstance(avg_avoid, (int, float)) else avg_avoid
        avg_diversity_str = f"{avg_diversity:.2f}" if isinstance(avg_diversity, (int, float)) else avg_diversity
        
        print(f"{name:^15} | {max_fitness:^12.2f} | {max_food_str:^12} | {max_survival_str:^12} | {avg_nav_str:^14} | {avg_avoid_str:^14} | {avg_diversity_str:^12}")
    
    print("="*100 + "\n")
    
    # Crear figura más grande para más gráficas
    plt.figure(figsize=(18, 14))
    
    # Gráfica 1: Comparación de fitness máximo (ahora sin suavizado)
    plt.subplot(3, 2, 1)
    for i, result in enumerate(experiment_results):
        plt.plot(result['avg_best_fitness'], 
                 label=result['config']['name'], 
                 color=COLORS[i % len(COLORS)])
    
    plt.title('Fitness Máximo por Generación (Real)')
    plt.xlabel('Generación')
    plt.ylabel('Fitness Máximo')
    plt.legend()
    plt.grid(True)
    
    # Gráfica 2: Comparación de fitness promedio (ahora sin suavizado)
    plt.subplot(3, 2, 2)
    for i, result in enumerate(experiment_results):
        plt.plot(result['avg_avg_fitness'], 
                 label=result['config']['name'], 
                 color=COLORS[i % len(COLORS)])
    
    plt.title('Fitness Promedio por Generación (Real)')
    plt.xlabel('Generación')
    plt.ylabel('Fitness Promedio')
    plt.legend()
    plt.grid(True)
    
    # Gráfica 3: Puntuación máxima (comida comida)
    plt.subplot(3, 2, 3)
    for i, result in enumerate(experiment_results):
        if 'avg_best_score' in result:
            plt.plot(result['avg_best_score'], 
                    label=result['config']['name'], 
                    color=COLORS[i % len(COLORS)])
    
    plt.title('Puntuación Máxima por Generación')
    plt.xlabel('Generación')
    plt.ylabel('Comida Comida')
    plt.legend()
    plt.grid(True)
    
    # Gráfica 4: Tiempo de supervivencia máximo
    plt.subplot(3, 2, 4)
    for i, result in enumerate(experiment_results):
        if 'avg_survival_time' in result:
            plt.plot(result['avg_survival_time'], 
                    label=result['config']['name'], 
                    color=COLORS[i % len(COLORS)])
    
    plt.title('Tiempo de Supervivencia Máximo')
    plt.xlabel('Generación')
    plt.ylabel('Pasos de Supervivencia')
    plt.legend()
    plt.grid(True)
    
    # Gráfica 5: Diversidad de la población
    plt.subplot(3, 2, 5)
    for i, result in enumerate(experiment_results):
        if 'avg_diversity' in result:
            plt.plot(result['avg_diversity'], 
                    label=result['config']['name'], 
                    color=COLORS[i % len(COLORS)])
    
    plt.title('Diversidad Genética por Generación')
    plt.xlabel('Generación')
    plt.ylabel('Diversidad (Distancia)')
    plt.legend()
    plt.grid(True)
    
    # Gráfica 6: Habilidades específicas (Navegación y Evitación)
    plt.subplot(3, 2, 6)
    
    # Usar líneas sólidas para navegación y punteadas para evitación
    for i, result in enumerate(experiment_results):
        color = COLORS[i % len(COLORS)]
        if 'avg_navigation' in result:
            plt.plot(result['avg_navigation'], 
                    label=f"{result['config']['name']} - Nav",
                    color=color,
                    linestyle='-')
        if 'avg_obstacle_avoidance' in result:
            plt.plot(result['avg_obstacle_avoidance'],
                    label=f"{result['config']['name']} - Evit",
                    color=color,
                    linestyle='--')
    
    plt.title('Habilidades Específicas por Generación')
    plt.xlabel('Generación')
    plt.ylabel('Nivel de Habilidad (0-1)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('comparacion_detallada.png')
    plt.show()

def run_experiments():
    """
    Ejecuta tres experimentos con diferentes configuraciones de algoritmos genéticos
    para el juego Snake y compara los resultados utilizando el proceso evolutivo completo.
    """
    print("\n" + "="*50)
    print("EXPERIMENTOS EVOLUTIVOS COMPARATIVOS PARA SNAKE")
    print("="*50)
    print("\nSe ejecutarán 3 experimentos con diferentes configuraciones.")
    print("Cada experimento ejecutará el proceso evolutivo completo.")
    print("Al finalizar, se compararán los resultados de los tres experimentos.\n")
    
    # Definir las tres configuraciones experimentales con parámetros optimizados para mejor aprendizaje
    experiment_configs = [
        {
            'name': 'Configuración Óptima',
            'pop_size': 50,             # Población sustancial para diversidad genética
            'generations': 100,         # Mantener 100 generaciones para convergencia completa
            'mutation_rate': 0.35,      # Ligeramente mayor que antes para mejor exploración
            'elitism': 6,               # Aumento moderado del elitismo para estabilidad
            'base_seed': int(time.time()) % 10000
        },
        {
            'name': 'Alta Diversidad Balanceada',
            'pop_size': 60,             # Mayor población para garantizar diversidad
            'generations': 100,         # Mismas generaciones para comparación justa
            'mutation_rate': 0.45,      # Alta mutación pero no extrema
            'elitism': 4,               # Elitismo moderado para equilibrar exploración/explotación
            'base_seed': (int(time.time()) % 10000) + 5000
        },
        {
            'name': 'Convergencia Optimizada',
            'pop_size': 40,             # Población moderada
            'generations': 100,         # Mismas generaciones para comparación justa
            'mutation_rate': 0.25,      # Mutación más baja para refinamiento
            'elitism': 10,              # Alto elitismo (25%) para rápida convergencia
            'base_seed': (int(time.time()) % 10000) + 10000
        }
    ]
    
    # Almacenar los resultados de cada experimento
    results = []
    total_experiments = len(experiment_configs)
    
    # Diccionario para compartir resultados
    shared_results = {}
    
    # Ejecutar cada experimento secuencialmente
    print("\nEjecutando experimentos secuencialmente...")
    
    for i, config in enumerate(experiment_configs):
        print(f"\nEJECUTANDO EXPERIMENTO {i+1}/{total_experiments}: {config['name']}")
        print("="*50)
        print(f"Configuración: Población={config['pop_size']}, Generaciones={config['generations']}, ")
        print(f"Mutación={config['mutation_rate']}, Élite={config['elitism']}")
        print("="*50)
        
        # Tiempo de inicio
        start_time = time.time()
        
        # Ejecutar el experimento
        best_agent, history = run_experiment(config, shared_results, i)
        
        # Tiempo total
        execution_time = time.time() - start_time
        
        # Almacenar resultados completos con nuevas métricas
        result = {
            'config': config,
            'best_agent': best_agent,
            'best_fitness': history['best_fitness'][-1],
            'avg_best_fitness': history['best_fitness'],
            'avg_avg_fitness': history['avg_fitness'],
            'avg_execution_time': execution_time,
            # Nuevas métricas
            'avg_best_score': history['best_score'] if 'best_score' in history else [],
            'avg_survival_time': history['best_survival_time'] if 'best_survival_time' in history else [],
            'avg_food_eaten': history['food_eaten_avg'] if 'food_eaten_avg' in history else [],
            'avg_survival_avg': history['survival_time_avg'] if 'survival_time_avg' in history else [],
            'avg_navigation': history['navigation_skill'] if 'navigation_skill' in history else [],
            'avg_obstacle_avoidance': history['obstacle_avoidance'] if 'obstacle_avoidance' in history else [],
            'avg_diversity': history['diversity_metric'] if 'diversity_metric' in history else [],
            'top_agents_history': history['best_agents_history'] if 'best_agents_history' in history else []
        }
        results.append(result)
        
        # Mostrar resumen del experimento actual
        print(f"\nResumen del Experimento {i+1}: {config['name']}")
        print(f"Fitness máximo alcanzado: {history['best_fitness'][-1]:.2f}")
        
        # Mostrar nuevas métricas si están disponibles
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
    
    # Comparar los resultados utilizando la función compare_experiments
    print("\nGenerando gráficas comparativas de los tres experimentos...")
    compare_experiments(results)
    
    # Encontrar la mejor configuración
    best_experiment_idx = np.argmax([result['best_fitness'] for result in results])
    best_result = results[best_experiment_idx]
    
    print("\n" + "="*50)
    print("MEJOR CONFIGURACIÓN ENCONTRADA:")
    print(f"Experimento: {best_result['config']['name']}")
    print(f"Fitness máximo: {best_result['best_fitness']:.2f}")
    
    # Mostrar métricas adicionales de la mejor configuración
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
        
    # Mostrar los parámetros de la mejor configuración
    print(f"\nParámetros:")
    print(f"  - Tamaño de población: {best_result['config']['pop_size']}")
    print(f"  - Generaciones: {best_result['config']['generations']}")
    print(f"  - Tasa de mutación: {best_result['config']['mutation_rate']}")
    print(f"  - Tamaño de élite: {best_result['config']['elitism']}")
    print("="*50)
    
    print("\nExperimentos evolutivos completados. Gráficas guardadas como 'comparacion_detallada.png'")

if __name__ == "__main__":
    run_experiments()
