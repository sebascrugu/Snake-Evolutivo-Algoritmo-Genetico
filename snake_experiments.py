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
    
    Args:
        experiment_config: Diccionario con parámetros de configuración
        num_runs: Número de repeticiones para obtener resultados más robustos
    
    Returns:
        Diccionario con resultados del experimento
    """
    print(f"\nEjecutando experimento: {experiment_config['name']}")
    print(f"Configuración: {experiment_config}")
    
    # Almacenar resultados
    all_best_fitness = []
    all_avg_fitness = []
    all_best_agents = []
    all_execution_times = []
    
    for run in range(num_runs):
        print(f"\nEjecución {run+1}/{num_runs}")
        
        # Crear algoritmo genético con configuración específica
        ga = GeneticAlgorithm(
            population_size=experiment_config.get('population_size', 50),
            num_generations=experiment_config.get('num_generations', 30),
            mutation_rate=experiment_config.get('mutation_rate', 0.2),
            crossover_rate=experiment_config.get('crossover_rate', 0.8),
            elite_size=experiment_config.get('elite_size', 3)
        )
        
        # Asegurarnos de que los juegos en training_mode=True para permitir reinicio durante entrenamiento
        # El método fitness internamente crea los juegos con training_mode=True
        
        # Configurar tipo de cruce si se especifica
        if 'crossover_type' in experiment_config:
            ga.crossover_type = experiment_config['crossover_type']
        
        # Medir tiempo de ejecución
        start_time = time.time()
        
        # Ejecutar evolución
        best_agent = ga.evolve(show_progress=True)
        
        # Calcular tiempo de ejecución
        execution_time = time.time() - start_time
        
        # Almacenar resultados
        all_best_fitness.append(ga.best_fitness_history)
        all_avg_fitness.append(ga.avg_fitness_history)
        all_best_agents.append(best_agent)
        all_execution_times.append(execution_time)
        
        print(f"Ejecución completada en {execution_time:.2f} segundos")
    
    # Calcular promedio de los resultados entre todas las ejecuciones
    avg_best_fitness = np.mean(all_best_fitness, axis=0)
    avg_avg_fitness = np.mean(all_avg_fitness, axis=0)
    avg_execution_time = np.mean(all_execution_times)
    
    # Encontrar mejor agente entre todas las ejecuciones
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
    
    Args:
        experiment_results: Lista de resultados de experimentos
    """
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
        generations = np.where(best_fitness >= threshold)[0]
        if len(generations) > 0:
            convergence = generations[0]
        else:
            convergence = len(best_fitness)  # No converge
            
        convergence_data.append(convergence)
    
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
    
    # Gráfica adicional: Boxplot de distribución de fitness máximo final
    plt.figure(figsize=(10, 6))
    final_fitness_data = []
    
    for result in experiment_results:
        # Obtener fitness final de cada ejecución
        final_fitness = [fitness[-1] for fitness in result['all_best_fitness']]
        final_fitness_data.append(final_fitness)
    
    plt.boxplot(final_fitness_data, labels=names)
    plt.title('Distribución del Fitness Máximo Final')
    plt.ylabel('Fitness')
    plt.grid(True, axis='y')
    plt.savefig('distribucion_fitness.png')
    plt.show()

def run_experiments():
    """
    Ejecuta una única evaluación con exactamente 3 juegos y termina.
    Sin entrenamiento, generaciones ni evolución.
    """
    print("\n" + "="*50)
    print("EXPERIMENTO SIMPLE CON SNAKE")
    print("="*50)
    
    print("\nEjecutando una evaluación con exactamente 3 juegos...")
    
    # Crear un único agente con una tabla de decisión predefinida (valores por defecto)
    # No realizar entrenamiento ni evolución, solo evaluar este agente
    agente = DecisionTable()
    
    print("\nEvaluando agente en 3 juegos consecutivos...")
    
    # Crear instancia del algoritmo genético solo para usar el método fitness
    ga = GeneticAlgorithm()
    
    # Iniciar cronometraje
    start_time = time.time()
    
    # Realizar una única evaluación con 3 juegos
    # El método fitness ya está configurado para jugar exactamente 3 juegos
    fitness_valor = ga.fitness(agente)
    
    # Calcular tiempo de ejecución
    execution_time = time.time() - start_time
    
    # Mostrar resumen de resultados
    print("\n" + "="*50)
    print("RESUMEN DEL EXPERIMENTO:")
    print(f"Fitness obtenido: {fitness_valor:.2f}")
    print(f"Tiempo de ejecución: {execution_time:.2f} segundos")
    print("="*50)
    
    print("\nExperimento completado. No se ejecutarán más evaluaciones.")

if __name__ == "__main__":
    run_experiments()
