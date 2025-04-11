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
        
        # Modificar la función evolve para mostrar progreso detallado
        # Guardar la función original
        original_evolve = ga.evolve
        
        # Tiempo de inicio para calcular duración
        start_time = time.time()
        
        # Crear una función wrapper que muestre progreso detallado
        def evolve_with_progress(show_progress=True):
            # Inicializar población si no está inicializada
            if not ga.population:
                ga.initialize_population()
            
            # Mostrar información de inicio
            total_generations = ga.num_generations
            print(f"\nIniciando evolución con {ga.population_size} agentes durante {total_generations} generaciones")
            print(f"Configuración: Mutación={ga.mutation_rate}, Cruce={ga.crossover_rate}, Élite={ga.elite_size}, Tipo de cruce={ga.crossover_type}")
            print("\nProgreso de la evolución:")
            print("-" * 70)
            
            # Para cada generación
            for generation in range(total_generations):
                # Evaluar fitness
                fitnesses = [ga.fitness(agent) for agent in ga.population]
                
                # Encontrar mejor agente
                max_fitness_idx = np.argmax(fitnesses)
                best_fitness = fitnesses[max_fitness_idx]
                best_agent = ga.population[max_fitness_idx]
                
                # Guardar estadísticas
                ga.best_fitness_history.append(best_fitness)
                ga.avg_fitness_history.append(sum(fitnesses) / len(fitnesses))
                
                # Mostrar progreso detallado
                if show_progress:
                    progress_percent = (generation + 1) / total_generations * 100
                    if generation % 5 == 0 or generation == total_generations - 1:
                        print(f"Generación {generation+1}/{total_generations} ({progress_percent:.1f}%) - Mejor fitness: {best_fitness:.2f}, Promedio: {ga.avg_fitness_history[-1]:.2f}")
                        # Mostrar una barra de progreso visual
                        bar_length = 40
                        filled_length = int(bar_length * (generation + 1) / total_generations)
                        bar = '█' * filled_length + '░' * (bar_length - filled_length)
                        print(f"[{bar}] {progress_percent:.1f}%")
                
                # Crear nueva población
                new_population = []
                
                # Elitismo: pasar los mejores agentes directamente
                sorted_indices = np.argsort(fitnesses)[::-1]
                for i in range(ga.elite_size):
                    elite_idx = sorted_indices[i]
                    new_population.append(DecisionTable(np.copy(ga.population[elite_idx].weights)))
                
                # Generar el resto de la población mediante selección, cruce y mutación
                while len(new_population) < ga.population_size:
                    # Selección
                    parent_indices = ga.selection(fitnesses)
                    parent1 = ga.population[parent_indices[0]]
                    parent2 = ga.population[parent_indices[1]]
                    
                    # Cruce
                    child = ga.crossover(parent1, parent2)
                    
                    # Mutación
                    child = ga.mutate(child)
                    
                    # Agregar a nueva población
                    new_population.append(child)
                
                # Reemplazar población
                ga.population = new_population
            
            # Evaluar fitness final
            final_fitnesses = [ga.fitness(agent) for agent in ga.population]
            best_idx = np.argmax(final_fitnesses)
            
            print(f"\nEvolución completada. Mejor fitness final: {final_fitnesses[best_idx]:.2f}")
            print("-" * 70)
            print(f"Tiempo transcurrido: {time.time() - start_time:.2f} segundos")
            
            return ga.population[best_idx]
        
        # Reemplazar temporalmente la función evolve
        ga.evolve = evolve_with_progress
        
        # Ejecutar evolución con progreso detallado
        best_agent = ga.evolve(show_progress=True)
        
        # Restaurar la función original
        ga.evolve = original_evolve
        
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
    Ejecuta tres experimentos con diferentes configuraciones de algoritmos genéticos
    para el juego Snake y compara los resultados utilizando el proceso evolutivo completo.
    """
    print("\n" + "="*50)
    print("EXPERIMENTOS EVOLUTIVOS COMPARATIVOS PARA SNAKE")
    print("="*50)
    print("\nSe ejecutarán 3 experimentos con diferentes configuraciones.")
    print("Cada experimento ejecutará el proceso evolutivo completo.")
    print("Al finalizar, se compararán los resultados de los tres experimentos.\n")
    
    # Definir las tres configuraciones experimentales
    # Reemplaza temporalmente tus configuraciones actuales con estas
    experiment_configs = [
    {
        'name': 'Mini Base',
        'population_size': 10,       # Reducido de 30
        'num_generations': 5,        # Reducido de 50
        'mutation_rate': 0.1,
        'crossover_rate': 0.8,
        'elite_size': 1,             # Reducido de 2
        'crossover_type': 'one_point'
    },
    {
        'name': 'Mini Exploración',
        'population_size': 15,       # Reducido de 80
        'num_generations': 5,        # Reducido de 50
        'mutation_rate': 0.25,
        'crossover_rate': 0.7,
        'elite_size': 1,             # Reducido de 3
        'crossover_type': 'uniform'
    },
    {
        'name': 'Mini Explotación',
        'population_size': 10,       # Reducido de 50
        'num_generations': 8,        # Reducido de 75
        'mutation_rate': 0.05,
        'crossover_rate': 0.9,
        'elite_size': 2,             # Reducido de 8
        'crossover_type': 'two_point'
    }
    ]
    
    # Almacenar los resultados de cada experimento
    results = []
    total_experiments = len(experiment_configs)
    
    # Ejecutar cada experimento utilizando la función run_experiment() existente
    for i, config in enumerate(experiment_configs):
        print(f"\n\nEJECUTANDO EXPERIMENTO {i+1}/{total_experiments}: {config['name']}")
        print("="*50)
        print(f"\nConfiguración: {config}")
        print(f"Progreso general: {i+1}/{total_experiments} experimentos ({(i+1)/total_experiments*100:.1f}%)")
        
        # Ejecutar el experimento con la configuración actual
        # Usar num_runs=1 para que cada configuración se ejecute una sola vez
        result = run_experiment(config, num_runs=1)
        results.append(result)
        
        # Mostrar resumen del experimento actual
        best_fitness = max(result['avg_best_fitness'])
        print(f"\nResumen del Experimento {i+1}: {config['name']}")
        print(f"Fitness máximo alcanzado: {best_fitness:.2f}")
        print(f"\nExperimento {i+1}/{total_experiments} completado.")
        if i < total_experiments - 1:
            print("Preparando siguiente experimento...\n")
        print(f"Tiempo de ejecución: {result['avg_execution_time']:.2f} segundos")
        print("="*50)
    
    print("\nTodos los experimentos evolutivos han sido completados.")
    
    # Comparar los resultados utilizando la función compare_experiments() existente
    print("\nGenerando gráficas comparativas de los tres experimentos...")
    compare_experiments(results)
    
    # Encontrar la mejor configuración
    best_experiment_idx = np.argmax([max(result['avg_best_fitness']) for result in results])
    best_result = results[best_experiment_idx]
    
    print("\n" + "="*50)
    print("MEJOR CONFIGURACIÓN ENCONTRADA:")
    print(f"Experimento: {best_result['config']['name']}")
    print(f"Fitness máximo: {max(best_result['avg_best_fitness']):.2f}")
    print(f"Parámetros:\n  - Tamaño de población: {best_result['config']['population_size']}")
    print(f"  - Generaciones: {best_result['config']['num_generations']}")
    print(f"  - Tasa de mutación: {best_result['config']['mutation_rate']}")
    print(f"  - Tasa de cruce: {best_result['config']['crossover_rate']}")
    print(f"  - Tamaño de élite: {best_result['config']['elite_size']}")
    print(f"  - Tipo de cruce: {best_result['config']['crossover_type']}")
    print("="*50)
    
    print("\nExperimentos evolutivos completados. Gráficas guardadas como 'comparacion_experimentos.png' y 'distribucion_fitness.png'")

if __name__ == "__main__":
    run_experiments()
