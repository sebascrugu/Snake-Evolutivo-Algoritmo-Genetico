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
from matplotlib.ticker import MaxNLocator
from snakepy import GeneticAlgorithm, DecisionTable, Action, Direction, Point, SnakeGame

# Colores para las gráficas
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

def run_experiment(experiment_config, num_runs=3):
    """
    Ejecuta un experimento con la configuración dada y múltiples repeticiones.
    """
    print(f"\nEjecutando experimento: {experiment_config['name']}")
    print(f"Configuración: {experiment_config}")
    
    all_best_fitness = []
    all_avg_fitness = []
    all_best_agents = []
    all_execution_times = []
    
    for run in range(num_runs):
        print(f"\nEjecución {run+1}/{num_runs}")
        
        ga = GeneticAlgorithm(
            population_size=experiment_config.get('population_size', 50),
            num_generations=experiment_config.get('num_generations', 30),
            mutation_rate=experiment_config.get('mutation_rate', 0.2),
            crossover_rate=experiment_config.get('crossover_rate', 0.8),
            elite_size=experiment_config.get('elite_size', 3)
        )
        
        if 'crossover_type' in experiment_config:
            ga.crossover_type = experiment_config['crossover_type']
        
        start_time = time.time()
        best_agent = ga.evolve(show_progress=True)
        execution_time = time.time() - start_time
        
        all_best_fitness.append(ga.best_fitness_history)
        all_avg_fitness.append(ga.avg_fitness_history)
        all_best_agents.append(best_agent)
        all_execution_times.append(execution_time)
        
        print(f"Ejecución completada en {execution_time:.2f} segundos")
    
    avg_best_fitness = np.mean(all_best_fitness, axis=0)
    avg_avg_fitness = np.mean(all_avg_fitness, axis=0)
    avg_execution_time = np.mean(all_execution_times)
    
    best_run_index = np.argmax([max(fitness) for fitness in all_best_fitness])
    best_agent = all_best_agents[best_run_index]
    
    return {
        'config': experiment_config,
        'avg_best_fitness': avg_best_fitness,
        'avg_avg_fitness': avg_avg_fitness,
        'best_agent': best_agent,
        'avg_execution_time': avg_execution_time,
        'all_best_fitness': all_best_fitness,
        'all_avg_fitness': all_avg_fitness,
        'all_execution_times': all_execution_times
    }

def compare_experiments(experiment_results):
    """
    Compara los resultados de diferentes experimentos generando gráficas.
    """
    plt.figure(figsize=(15, 10))
    
    # Fitness máximo
    plt.subplot(2, 2, 1)
    for i, result in enumerate(experiment_results):
        plt.plot(result['avg_best_fitness'], label=result['config']['name'], color=COLORS[i % len(COLORS)])
    plt.title('Comparación de Fitness Máximo por Generación')
    plt.xlabel('Generación')
    plt.ylabel('Fitness Máximo')
    plt.legend()
    plt.grid(True)
    
    # Fitness promedio
    plt.subplot(2, 2, 2)
    for i, result in enumerate(experiment_results):
        plt.plot(result['avg_avg_fitness'], label=result['config']['name'], color=COLORS[i % len(COLORS)])
    plt.title('Comparación de Fitness Promedio por Generación')
    plt.xlabel('Generación')
    plt.ylabel('Fitness Promedio')
    plt.legend()
    plt.grid(True)
    
    # Tiempos de ejecución
    plt.subplot(2, 2, 3)
    names = [result['config']['name'] for result in experiment_results]
    times = [result['avg_execution_time'] for result in experiment_results]
    bars = plt.bar(names, times, color=COLORS[:len(names)])
    plt.title('Tiempo Promedio de Ejecución')
    plt.xlabel('Experimento')
    plt.ylabel('Tiempo (segundos)')
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height:.1f}s', ha='center', va='bottom')
    
    # Velocidad de convergencia
    plt.subplot(2, 2, 4)
    convergence_data = []
    for result in experiment_results:
        best_fitness = result['avg_best_fitness']
        max_fitness = max(best_fitness)
        threshold = 0.9 * max_fitness
        generations = np.where(best_fitness >= threshold)[0]
        convergence = generations[0] if len(generations) > 0 else len(best_fitness)
        convergence_data.append(convergence)
    bars = plt.bar(names, convergence_data, color=COLORS[:len(names)])
    plt.title('Velocidad de Convergencia (90% del máximo)')
    plt.xlabel('Experimento')
    plt.ylabel('Generaciones')
    plt.xticks(rotation=45, ha='right')
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('comparacion_experimentos.png')
    plt.show()
    
    # Boxplot final
    plt.figure(figsize=(10, 6))
    final_fitness_data = [[fitness[-1] for fitness in result['all_best_fitness']] for result in experiment_results]
    plt.boxplot(final_fitness_data, labels=names)
    plt.title('Distribución del Fitness Máximo Final')
    plt.ylabel('Fitness')
    plt.grid(True, axis='y')
    plt.savefig('distribucion_fitness.png')
    plt.show()

def run_experiments():
    """
    Ejecuta 3 experimentos completos con diferentes parámetros y grafica los resultados.
    """
    print("\n" + "="*50)
    print("EJECUTANDO EXPERIMENTOS CON SNAKE Y ALGORITMO GENÉTICO")
    print("="*50)

    experiments = [
        {
            'name': 'Exp1_Pop30_Mut02',
            'population_size': 30,
            'num_generations': 50,
            'mutation_rate': 0.2,
            'crossover_rate': 0.8,
            'elite_size': 3
        },
        {
            'name': 'Exp2_Pop30_Mut05',
            'population_size': 30,
            'num_generations': 50,
            'mutation_rate': 0.5,
            'crossover_rate': 0.8,
            'elite_size': 3
        },
        {
            'name': 'Exp3_Pop30_CrossLow',
            'population_size': 30,
            'num_generations': 50,
            'mutation_rate': 0.2,
            'crossover_rate': 0.3,
            'elite_size': 3
        }
    ]

    experiment_results = [run_experiment(cfg, num_runs=1) for cfg in experiments]
    compare_experiments(experiment_results)

if __name__ == "__main__":
    run_experiments()
