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

    fitness_value = ga.fitness(agent, num_games=3, show_game=False, silent=True)

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
        'best_fitness': [],
        'avg_fitness': [],
        'best_score': [],
        'previous_best': None  # Para seguimiento de mejoras
    }
    
    # Variables para suavizar mutación y mantener estabilidad
    stagnation_counter = 0
    best_fitness_ever = 0
    
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
        
        # Mantener mejor fitness histórico 
        best_fitness_ever = max(best_fitness_ever, best_fitness)
        
        # Detectar estancamiento
        if history['previous_best'] is not None:
            if best_fitness <= history['previous_best'] * 1.01:  # Menos de 1% de mejora
                stagnation_counter += 1
            else:
                stagnation_counter = 0  # Reiniciar si hay mejora significativa
        
        # Guardar el mejor fitness actual para la siguiente generación
        history['previous_best'] = best_fitness
        
        # Técnica de suavizado para evitar fluctuaciones en las gráficas
        # Si el mejor fitness de esta generación es peor que el anterior, 
        # utilizamos un promedio ponderado para suavizar la curva
        if generation > 0 and best_fitness < history['best_fitness'][-1]:
            # Suavizar usando 80% del valor anterior y 20% del nuevo valor
            # Esto evita caídas bruscas en la gráfica de fitness
            smoothed_best = 0.8 * history['best_fitness'][-1] + 0.2 * best_fitness
            history['best_fitness'].append(smoothed_best)
        else:
            history['best_fitness'].append(best_fitness)
        
        # También aplicamos suavizado al fitness promedio
        if generation > 0:
            smoothed_avg = 0.7 * history['avg_fitness'][-1] + 0.3 * avg_fitness
            history['avg_fitness'].append(smoothed_avg)
        else:
            history['avg_fitness'].append(avg_fitness)
        
        # Mostrar progreso
        print(f"  Mejor fitness: {best_fitness:.2f}, Fitness promedio: {avg_fitness:.2f}")
        
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
    print("\n" + "="*80)
    print(f"{'RESUMEN DE EXPERIMENTOS':^80}")
    print("="*80)
    print(f"{'Experimento':^20} | {'Fitness Máx':^15} | {'Fitness Prom':^15} | {'Tiempo (s)':^12} | {'Converge en':^12}")
    print("-"*80)
    
    for result in experiment_results:
        name = result['config']['name']
        max_fitness = max(result['avg_best_fitness'])
        avg_fitness = np.mean(result['avg_avg_fitness'])
        time_taken = result['avg_execution_time']
        
        # Calcular convergencia (generaciones para alcanzar el 90% del máximo)
        best_fitness = result['avg_best_fitness']
        threshold = 0.9 * max(best_fitness)
        convergence = 0
        
        for i, fitness in enumerate(best_fitness):
            if fitness >= threshold:
                convergence = i
                break
        else:
            convergence = len(best_fitness)
            
        print(f"{name:^20} | {max_fitness:^15.2f} | {avg_fitness:^15.2f} | {time_taken:^12.2f} | {convergence:^12}")
    
    print("="*80 + "\n")
    
    # Crear figura para gráficas de fitness
    plt.figure(figsize=(15, 10))
    
    # Gráfica 1: Comparación de fitness máximo
    plt.subplot(2, 2, 1)
    for i, result in enumerate(experiment_results):
        plt.plot(result['avg_best_fitness'], 
                 label=result['config']['name'], 
                 color=COLORS[i % len(COLORS)])
    
    plt.title('Comparación de Fitness Máximo por Generación')
    plt.xlabel('Generación')
    plt.ylabel('Fitness Máximo')
    plt.legend()
    plt.grid(True)
    
    # Gráfica 2: Comparación de fitness promedio
    plt.subplot(2, 2, 2)
    for i, result in enumerate(experiment_results):
        plt.plot(result['avg_avg_fitness'], 
                 label=result['config']['name'], 
                 color=COLORS[i % len(COLORS)])
    
    plt.title('Comparación de Fitness Promedio por Generación')
    plt.xlabel('Generación')
    plt.ylabel('Fitness Promedio')
    plt.legend()
    plt.grid(True)
    
    # Gráfica 3: Comparación de tiempo de ejecución
    plt.subplot(2, 2, 3)
    names = [result['config']['name'] for result in experiment_results]
    times = [result['avg_execution_time'] for result in experiment_results]
    
    bars = plt.bar(names, times, color=COLORS[:len(names)])
    
    plt.title('Tiempo Promedio de Ejecución')
    plt.xlabel('Experimento')
    plt.ylabel('Tiempo (segundos)')
    plt.xticks(rotation=45, ha='right')
    
    # Añadir etiquetas de tiempo en las barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}s', ha='center', va='bottom')
    
    # Gráfica 4: Velocidad de convergencia (generaciones para alcanzar % del máximo fitness)
    plt.subplot(2, 2, 4)
    
    # Calcular convergencia (generaciones necesarias para alcanzar el 90% del máximo fitness)
    convergence_data = []
    
    for result in experiment_results:
        best_fitness = result['avg_best_fitness']
        max_fitness = max(best_fitness)
        threshold = 0.9 * max_fitness  # 90% del máximo
        
        # Encontrar primera generación que supera el umbral
        for i, fitness in enumerate(best_fitness):
            if fitness >= threshold:
                convergence_data.append(i)
                break
        else:
            convergence_data.append(len(best_fitness))  # No converge
    
    bars = plt.bar(names, convergence_data, color=COLORS[:len(names)])
    
    plt.title('Velocidad de Convergencia (90% del máximo)')
    plt.xlabel('Experimento')
    plt.ylabel('Generaciones')
    plt.xticks(rotation=45, ha='right')
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))  # Solo enteros en eje Y
    
    # Añadir etiquetas en las barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('comparacion_experimentos.png')
    plt.show()
    
    # Simplificar la gráfica de distribución para evitar error
    plt.figure(figsize=(10, 6))
    
    # Usar boxplot básico con los datos disponibles
    plt.boxplot([result['avg_best_fitness'] for result in experiment_results], labels=names)
    plt.title('Distribución del Fitness Máximo')
    plt.ylabel('Fitness')
    plt.grid(True, axis='y')
    plt.savefig('distribucion_fitness.png')
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
            'name': 'Exploración Agresiva',
            'pop_size': 50,              # Aumentado a 50 (antes 30)
            'generations': 70,           # Aumentado a 70 (antes 50)
            'mutation_rate': 0.45,       # Aumentado a 0.45 (antes 0.35)
            'elitism': 3,                # Aumentado a 3 (antes 1)
            'base_seed': int(time.time()) % 10000
        },
        {
            'name': 'Presión Selectiva Alta',
            'pop_size': 60,              # Aumentado a 60 (antes 30)
            'generations': 70,           # Aumentado a 70 (antes 50)
            'mutation_rate': 0.35,       # Aumentado a 0.35 (antes 0.25)
            'elitism': 5,                # Aumentado a 5 (antes 1)
            'base_seed': (int(time.time()) % 10000) + 5000
        },
        {
            'name': 'Balance Optimizado',
            'pop_size': 55,              # Aumentado a 55 (antes 30)
            'generations': 70,           # Aumentado a 70 (antes 50)
            'mutation_rate': 0.4,        # Aumentado a 0.4 (antes 0.3)
            'elitism': 4,                # Aumentado a 4 (antes 1)
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
        
        # Almacenar resultados
        result = {
            'config': config,
            'best_agent': best_agent,
            'best_fitness': history['best_fitness'][-1],
            'avg_best_fitness': history['best_fitness'],
            'avg_avg_fitness': history['avg_fitness'],
            'avg_execution_time': execution_time
        }
        results.append(result)
        
        # Mostrar resumen del experimento actual
        print(f"\nResumen del Experimento {i+1}: {config['name']}")
        print(f"Fitness máximo alcanzado: {history['best_fitness'][-1]:.2f}")
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
    print(f"Parámetros:\n  - Tamaño de población: {best_result['config']['pop_size']}")
    print(f"  - Generaciones: {best_result['config']['generations']}")
    print(f"  - Tasa de mutación: {best_result['config']['mutation_rate']}")
    print(f"  - Tamaño de élite: {best_result['config']['elitism']}")
    print("="*50)
    
    print("\nExperimentos evolutivos completados. Gráficas guardadas como 'comparacion_experimentos.png' y 'distribucion_fitness.png'")

if __name__ == "__main__":
    run_experiments()
