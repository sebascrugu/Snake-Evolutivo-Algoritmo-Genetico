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

# Clase que representa el agente con una tabla de decisiones
class DecisionTableAgent:
    def __init__(self, table=None):
        # Inicializar tabla de decisiones (para representar el "genoma" del agente)
        self.table = DecisionTable() if table is None else table
    
    def get_action(self, snake_head, food_pos, snake_body, width, height):
        """
        Determina la acción basada en el estado actual del juego.
        
        Args:
            snake_head: Posición de la cabeza de la serpiente
            food_pos: Posición de la comida
            snake_body: Lista de posiciones del cuerpo de la serpiente
            width: Ancho del tablero
            height: Alto del tablero
            
        Returns:
            Acción a realizar (0=recto, 1=derecha, 2=izquierda)
        """
        # Crear vector de estado
        state = self._get_state(snake_head, food_pos, snake_body, width, height)
        
        # Usar tabla de decisión para determinar acción
        return self.table.get_action(state)
    
    def _get_state(self, snake_head, food_pos, snake_body, width, height):
        """
        Convierte el estado del juego en un vector de características para la tabla de decisión.
        
        Crea un vector con la siguiente información:
        - Peligro en cada dirección (adelante, derecha, izquierda)
        - Dirección actual de movimiento (izquierda, derecha, arriba, abajo)
        - Dirección relativa de la comida (izquierda, derecha, arriba, abajo)
        - Va hacia la comida
        - Distancia a la comida (normalizada)
        """
        # Vector de estado de 18 dimensiones
        state = np.zeros(18)
        
        # Determinar dirección actual
        directions = [
            Direction.LEFT,
            Direction.RIGHT,
            Direction.UP,
            Direction.DOWN
        ]
        
        # Obtener dirección actual (asumimos que se mueve hacia la derecha por defecto)
        current_dir = Direction.RIGHT
        
        # Si hay al menos dos segmentos, inferimos la dirección
        if len(snake_body) > 0:
            # Comparar cabeza con el siguiente segmento
            if snake_head.x < snake_body[0].x:
                current_dir = Direction.LEFT
            elif snake_head.x > snake_body[0].x:
                current_dir = Direction.RIGHT
            elif snake_head.y < snake_body[0].y:
                current_dir = Direction.UP
            elif snake_head.y > snake_body[0].y:
                current_dir = Direction.DOWN
        
        # Codificar la dirección actual (one-hot encoding)
        dir_idx = directions.index(current_dir)
        state[3 + dir_idx] = 1.0
        
        # Detectar peligro en cada dirección relativa
        # 1. Peligro adelante
        point_ahead = self._get_point_in_direction(snake_head, current_dir)
        state[0] = self._is_danger(point_ahead, snake_body, width, height)
        
        # 2. Peligro a la derecha
        right_dir = directions[(dir_idx + 1) % 4]
        point_right = self._get_point_in_direction(snake_head, right_dir)
        state[1] = self._is_danger(point_right, snake_body, width, height)
        
        # 3. Peligro a la izquierda
        left_dir = directions[(dir_idx - 1) % 4]
        point_left = self._get_point_in_direction(snake_head, left_dir)
        state[2] = self._is_danger(point_left, snake_body, width, height)
        
        # Dirección relativa de la comida (en relación a la cabeza)
        # 4-7: Comida está a la izquierda, derecha, arriba, abajo
        state[7] = 1.0 if food_pos.x < snake_head.x else 0.0  # Comida a la izquierda
        state[8] = 1.0 if food_pos.x > snake_head.x else 0.0  # Comida a la derecha
        state[9] = 1.0 if food_pos.y < snake_head.y else 0.0  # Comida arriba
        state[10] = 1.0 if food_pos.y > snake_head.y else 0.0  # Comida abajo
        
        # Movimiento actual en relación a la comida (4 combinaciones posibles)
        # 11-14: Moviéndose hacia comida desde izq/der/arriba/abajo
        state[11] = 1.0 if current_dir == Direction.RIGHT and food_pos.x > snake_head.x else 0.0
        state[12] = 1.0 if current_dir == Direction.LEFT and food_pos.x < snake_head.x else 0.0
        state[13] = 1.0 if current_dir == Direction.DOWN and food_pos.y > snake_head.y else 0.0
        state[14] = 1.0 if current_dir == Direction.UP and food_pos.y < snake_head.y else 0.0
        
        # 15: Indicador si se está moviendo hacia la comida
        # Si cualquiera de las combinaciones anteriores es verdadera
        state[15] = 1.0 if np.any(state[11:15]) else 0.0
        
        # 16-17: Distancia a la comida (normalizada)
        distance_x = abs(snake_head.x - food_pos.x) / width
        distance_y = abs(snake_head.y - food_pos.y) / height
        state[16] = 1.0 - distance_x  # Más cercano = mayor valor
        state[17] = 1.0 - distance_y  # Más cercano = mayor valor
        
        return state
    
    def _get_point_in_direction(self, point, direction):
        """Obtiene el punto en la dirección dada"""
        if direction == Direction.RIGHT:
            return Point(point.x + 20, point.y)
        elif direction == Direction.LEFT:
            return Point(point.x - 20, point.y)
        elif direction == Direction.DOWN:
            return Point(point.x, point.y + 20)
        elif direction == Direction.UP:
            return Point(point.x, point.y - 20)
    
    def _is_danger(self, point, snake_body, width, height):
        """Verifica si hay peligro en el punto (colisión con borde o cuerpo)"""
        # Colisión con borde
        if point.x < 0 or point.x >= width*BLOCK_SIZE or point.y < 0 or point.y >= height*BLOCK_SIZE:
            return 1.0
        
        # Colisión con cuerpo
        for segment in snake_body:
            if point.x == segment.x and point.y == segment.y:
                return 1.0
        
        return 0.0
    
    def crossover(self, other):
        """
        Realiza cruce entre este agente y otro.
        
        Args:
            other: Otro agente DecisionTableAgent
            
        Returns:
            Un nuevo agente hijo
        """
        # Usar la tabla de decisión de este agente para hacer cruce
        child_table = self.table.crossover(other.table)
        
        # Crear nuevo agente con la tabla resultante
        return DecisionTableAgent(table=child_table)
    
    def mutate(self, mutation_rate):
        """
        Aplica mutación a este agente con la tasa especificada.
        
        Args:
            mutation_rate: Probabilidad de mutación para cada gen
        """
        # Aplicar mutación a la tabla de decisión
        self.table = self.table.mutate(mutation_rate)

# Función auxiliar para evaluación paralela de fitness
def evaluate_agent_fitness(agent, agent_idx, seed):
    """
    Evalúa el fitness de un agente utilizando la semilla proporcionada.
    
    Args:
        agent: Agente a evaluar (tabla de decisiones)
        agent_idx: Índice del agente en la población 
        seed: Semilla para reproducibilidad
    
    Returns:
        Valor de fitness del agente
    """
    # Crear semilla única para cada agente
    agent_seed = seed + agent_idx
    random.seed(agent_seed)
    np.random.seed(agent_seed)
    
    # Crear un juego de Snake en modo silencioso (sin interfaz gráfica)
    game = SnakeGame(
        width=15,
        height=15,
        headless=True,
        ai_control=True
    )
    
    # Inicializar variables para jugar
    total_score = 0
    num_games = 3  # Jugar varias partidas para evaluación más robusta
    
    for game_idx in range(num_games):
        # Reiniciar juego
        game.reset()
        game_seed = agent_seed + game_idx * 1000
        random.seed(game_seed)
        np.random.seed(game_seed)
        
        # Jugar hasta que termine
        game_over = False
        while not game_over:
            # Estado actual del juego
            snake_head = game.snake[0]
            food_pos = game.food
            snake_body = game.snake[1:] if len(game.snake) > 1 else []
            
            # Determinar acción usando la tabla de decisión
            action = agent.get_action(snake_head, food_pos, snake_body, game.grid_width, game.grid_height)
            
            # Aplicar acción
            game_over, _, _ = game.play_step(action)
        
        # Sumar puntuación
        total_score += game.score
    
    # Fitness promedio de las partidas
    fitness = total_score / num_games
    
    return fitness

def run_experiment(config, shared_results, experiment_index):
    """
    Ejecuta un experimento de evolución de agentes Snake utilizando algoritmos genéticos.
    
    Args:
        config: Configuración del experimento 
        shared_results: Diccionario compartido para almacenar resultados
        experiment_index: Índice del experimento actual
    """
    # Extraer parámetros de configuración
    pop_size = config.get('pop_size', 50)
    generations = config.get('generations', 30)
    mutation_rate = config.get('mutation_rate', 0.1)
    elitism = config.get('elitism', 5)
    base_seed = config.get('base_seed', int(time.time()))
    
    print(f"Iniciando experimento {experiment_index} con semilla base: {base_seed}")
    
    # Inicializar población aleatoria
    population = [DecisionTableAgent() for _ in range(pop_size)]
    
    # Historia para seguimiento de progreso
    history = {
        'best_fitness': [],
        'avg_fitness': [],
        'best_score': [],
    }
    
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
        sorted_population = [population[idx] for idx in sorted(
            fitness_dict.keys(), 
            key=lambda idx: fitness_dict[idx], 
            reverse=True
        )]
        
        # Actualizar población ordenada
        population = sorted_population
        
        # Calcular estadísticas
        best_fitness = fitness_dict[sorted(fitness_dict.keys(), key=lambda x: fitness_dict[x], reverse=True)[0]]
        avg_fitness = sum(fitness_dict.values()) / len(fitness_dict)
        
        # Guardar estadísticas
        history['best_fitness'].append(best_fitness)
        history['avg_fitness'].append(avg_fitness)
        
        # Mostrar progreso
        print(f"  Mejor fitness: {best_fitness:.2f}, Fitness promedio: {avg_fitness:.2f}")
        
        # Verificar si estamos en la última generación
        if generation == generations - 1:
            break
        
        # Crear nueva población con elitismo
        new_population = population[:elitism]  # Mantener mejores individuos
        
        # Completar población mediante reproducción
        while len(new_population) < pop_size:
            # Seleccionar padres (usar selección proporcional al fitness)
            parent1 = population[random.randint(0, pop_size//4)]  # De los mejores 25%
            parent2 = population[random.randint(0, pop_size//2)]  # De los mejores 50%
            
            # Crear hijo mediante cruce
            child = parent1.crossover(parent2)
            
            # Aplicar mutación
            child.mutate(mutation_rate)
            
            # Añadir a nueva población
            new_population.append(child)
        
        # Actualizar población
        population = new_population
    
    # Guardar resultado en estructura compartida
    shared_results[experiment_index] = {
        'best_agent': population[0],
        'best_fitness': history['best_fitness'][-1],
        'history': history
    }
    
    print(f"Experimento {experiment_index} completado. Mejor fitness: {history['best_fitness'][-1]:.2f}")
    return population[0], history

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
    
    # Definir las tres configuraciones experimentales con los nuevos nombres de parámetros
    experiment_configs = [
        {
            'name': 'Exploración Agresiva',
            'pop_size': 12,
            'generations': 6,
            'mutation_rate': 0.35,     # Tasa de mutación más alta para explorar más
            'elitism': 1,              # Elitismo mínimo para evitar convergencia prematura
            'base_seed': int(time.time()) % 10000   # Semilla basada en el tiempo actual
        },
        {
            'name': 'Presión Selectiva Alta',
            'pop_size': 15,
            'generations': 5,
            'mutation_rate': 0.25,
            'elitism': 1,
            'base_seed': (int(time.time()) % 10000) + 5000  # Otra semilla única
        },
        {
            'name': 'Balance Optimizado',
            'pop_size': 10,
            'generations': 7,
            'mutation_rate': 0.3,      # Buena tasa de mutación para balance exploración-explotación
            'elitism': 1,
            'base_seed': (int(time.time()) % 10000) + 10000  # Tercera semilla única
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
