# Código modificado para implementar algoritmo genético
# Basado en https://github.com/patrickloeber/snake-ai-pytorch/blob/main/snake_game_human.py

import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from enum import Enum
import time

import pygame

pygame.init()
font = pygame.font.SysFont("arial", 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
# Definimos acciones posibles
class Action(Enum):
    STRAIGHT = 0
    RIGHT_TURN = 1
    LEFT_TURN = 2


Point = namedtuple("Point", "x, y")

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

# Parámetros configurables
BLOCK_SIZE = 20  # Reducido para tener más celdas en el tablero
SPEED = 10  # Velocidad reducida para mejor control y visualización
BOARD_SIZE = (20, 20)  # Tamaño del tablero en celdas (ancho, alto)


class SnakeGame:

    def __init__(self, width=None, height=None, ai_control=False):
        # Configuración del tamaño del tablero
        if width is None or height is None:
            # Usar tamaño predeterminado si no se especifica
            self.grid_width, self.grid_height = BOARD_SIZE
            self.w = self.grid_width * BLOCK_SIZE
            self.h = self.grid_height * BLOCK_SIZE
        else:
            self.grid_width, self.grid_height = width, height
            self.w = self.grid_width * BLOCK_SIZE
            self.h = self.grid_height * BLOCK_SIZE
            
        # Control por IA
        self.ai_control = ai_control
        
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake - AI Genetic")
        self.clock = pygame.time.Clock()

        # init game state
        self.reset()
        
    def reset(self):
        """Reinicia el juego a estado inicial"""
        self.direction = Direction.RIGHT
        
        # Posicionamos la serpiente en el centro, con espacio suficiente
        # Aseguramos que la serpiente no empiece muy cerca de los bordes
        x_mid = (self.grid_width // 2) * BLOCK_SIZE
        y_mid = (self.grid_height // 2) * BLOCK_SIZE
        
        # Verificar que la posición sea válida
        if x_mid >= self.w - 3*BLOCK_SIZE:
            x_mid = self.w // 2
        if y_mid >= self.h - BLOCK_SIZE:
            y_mid = self.h // 2
            
        self.head = Point(x_mid, y_mid)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y),
        ]

        self.score = 0
        self.food = None
        self._place_food()
        
        # Contadores para fitness
        self.steps = 0
        self.steps_without_food = 0
        
        return self.get_state()

    def _place_food(self):
        """Coloca comida en una posición aleatoria del tablero"""
        # Aseguramos que la comida no se coloque fuera de los límites
        max_x = (self.w // BLOCK_SIZE) - 1
        max_y = (self.h // BLOCK_SIZE) - 1
        
        x = random.randint(0, max_x) * BLOCK_SIZE
        y = random.randint(0, max_y) * BLOCK_SIZE
        self.food = Point(x, y)
        
        # Si la comida cae sobre la serpiente, intentar de nuevo
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action=None):
        # 1. collect user input if not AI controlled
        if not self.ai_control:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.direction = Direction.LEFT
                    elif event.key == pygame.K_RIGHT:
                        self.direction = Direction.RIGHT
                    elif event.key == pygame.K_UP:
                        self.direction = Direction.UP
                    elif event.key == pygame.K_DOWN:
                        self.direction = Direction.DOWN
        else:
            # Procesar eventos para poder cerrar la ventana
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
            
            # Utilizar la acción de la IA
            if action is not None:
                # Cambiar dirección según la acción
                # 0=Seguir recto, 1=Girar derecha, 2=Girar izquierda
                self._change_direction(action)

        # 2. move
        self._move(self.direction)  # update the head
        self.snake.insert(0, self.head)
        
        # Incrementar contador de pasos para fitness
        self.steps += 1
        self.steps_without_food += 1

        # 3. check if game over
        reward = 0
        game_over = False
        # Reducir el límite de pasos sin comida para evitar que la serpiente dé vueltas infinitas
        # antes era 100 * len(self.snake), lo reducimos a 50
        if self._is_collision() or self.steps_without_food > 50 * len(self.snake):
            game_over = True
            reward = -10
            return game_over, self.score, reward

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            self.steps_without_food = 0
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        
        # 6. return game over and score
        return game_over, self.score, reward
        
    def _change_direction(self, action):
        """Cambia la dirección de la serpiente según la acción (0=recto, 1=derecha, 2=izquierda)"""
        # Lista de direcciones en orden horario: RIGHT, DOWN, LEFT, UP
        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clockwise.index(self.direction)
        
        if action == Action.STRAIGHT.value:
            # Mantener dirección actual
            new_dir = clockwise[idx]
        elif action == Action.RIGHT_TURN.value:
            # Girar 90 grados a la derecha
            next_idx = (idx + 1) % 4
            new_dir = clockwise[next_idx]
        elif action == Action.LEFT_TURN.value:
            # Girar 90 grados a la izquierda
            next_idx = (idx - 1) % 4
            new_dir = clockwise[next_idx]
            
        self.direction = new_dir

    def _is_collision(self, pt=None):
        """Verifica si hay colisión (con los bordes o con el cuerpo)"""
        if pt is None:
            pt = self.head
            
        # hits boundary - verificación más precisa de los límites
        if (
            pt.x >= self.w
            or pt.x < 0
            or pt.y >= self.h
            or pt.y < 0
        ):
            return True
            
        # hits itself - usar comparación exacta de coordenadas en lugar de in
        for segment in self.snake[1:]:
            if pt.x == segment.x and pt.y == segment.y:
                return True

        return False
        
    def is_collision_at(self, x, y):
        """Verifica si hay colisión en la posición (x,y)"""
        pt = Point(x, y)
        return self._is_collision(pt=pt)

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(
            self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE)
        )

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, direction):
        """Mueve la cabeza de la serpiente según la dirección"""
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
        
    def get_state(self):
        """Obtiene el estado actual como entrada para la IA (sensores de entorno)"""
        head = self.head
        
        # Puntos alrededor de la cabeza en 4 direcciones (derecha, izq, arriba, abajo)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        # Dirección actual
        dir_r = self.direction == Direction.RIGHT
        dir_l = self.direction == Direction.LEFT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN
        
        # Calcular distancia a la comida (Manhattan)
        food_dist_x = abs(self.food.x - head.x) / self.w  # Normalizada
        food_dist_y = abs(self.food.y - head.y) / self.h  # Normalizada
        
        # Calcular si el movimiento nos acerca a la comida
        move_right_good = self.food.x > head.x and dir_r
        move_left_good = self.food.x < head.x and dir_l
        move_up_good = self.food.y < head.y and dir_u
        move_down_good = self.food.y > head.y and dir_d
        moving_toward_food = move_right_good or move_left_good or move_up_good or move_down_good
        
        # Estado del juego (18 sensores en total, añadimos más para mejorar percepción de la comida)
        state = [
            # Peligro adelante [0]
            (dir_r and self.is_collision_at(head.x + BLOCK_SIZE, head.y)) or
            (dir_l and self.is_collision_at(head.x - BLOCK_SIZE, head.y)) or
            (dir_u and self.is_collision_at(head.x, head.y - BLOCK_SIZE)) or
            (dir_d and self.is_collision_at(head.x, head.y + BLOCK_SIZE)),
            
            # Peligro a la derecha [1]
            (dir_r and self.is_collision_at(head.x, head.y + BLOCK_SIZE)) or
            (dir_l and self.is_collision_at(head.x, head.y - BLOCK_SIZE)) or
            (dir_u and self.is_collision_at(head.x + BLOCK_SIZE, head.y)) or
            (dir_d and self.is_collision_at(head.x - BLOCK_SIZE, head.y)),
            
            # Peligro a la izquierda [2]
            (dir_r and self.is_collision_at(head.x, head.y - BLOCK_SIZE)) or
            (dir_l and self.is_collision_at(head.x, head.y + BLOCK_SIZE)) or
            (dir_u and self.is_collision_at(head.x - BLOCK_SIZE, head.y)) or
            (dir_d and self.is_collision_at(head.x + BLOCK_SIZE, head.y)),
            
            # Dirección actual [3-6]
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Ubicación de la comida [7-10]
            self.food.x < head.x,  # comida a la izquierda
            self.food.x > head.x,  # comida a la derecha
            self.food.y < head.y,  # comida arriba
            self.food.y > head.y,  # comida abajo
            
            # Distancia a los bordes (normalizadas) [11-14]
            head.x / self.w,          # distancia relativa al borde izquierdo
            (self.w - head.x) / self.w, # distancia relativa al borde derecho
            head.y / self.h,          # distancia relativa al borde superior
            (self.h - head.y) / self.h, # distancia relativa al borde inferior
            
            # Información adicional sobre la comida [15-17]
            moving_toward_food,  # Si nos estamos moviendo hacia la comida
            1.0 - food_dist_x,   # Cercanía a la comida en X (1=cerca, 0=lejos)
            1.0 - food_dist_y,   # Cercanía a la comida en Y (1=cerca, 0=lejos)
        ]
        
        return np.array(state, dtype=float)


# Tabla de Decisión para el agente (cromosoma)
class DecisionTable:
    def __init__(self, weights=None):
        # Tabla de decisión con pesos para cada sensor hacia cada acción
        # 18 sensores x 3 acciones posibles (actualizado con los nuevos sensores)
        if weights is None:
            # Inicialización aleatoria si no se proporcionan pesos
            # Inicializamos con pesos que favorecen comportamientos deseables
            self.weights = np.random.randn(18, 3)
            
            # BIAS INICIAL: Dar mayor peso negativo a detectar peligros (evitar colisiones)
            self.weights[0, :] = -5.0  # Peligro adelante - evitar esta dirección
            self.weights[1, 1] = -5.0  # Peligro a la derecha - no girar a la derecha
            self.weights[2, 2] = -5.0  # Peligro a la izquierda - no girar a la izquierda
            
            # BIAS INICIAL: Dar peso positivo a ir en dirección de la comida
            self.weights[7:11, :] = np.random.uniform(0.5, 1.5, size=(4, 3))  # Dirección de la comida
            
            # BIAS INICIAL: Premiar el moverse hacia la comida
            self.weights[15, 0] = 3.0  # Si va hacia la comida, seguir recto
            
            # BIAS INICIAL: Premiar estar cerca de la comida
            self.weights[16:18, :] = np.random.uniform(1.0, 2.0, size=(2, 3))
        else:
            self.weights = weights
        
        # Historial de acciones recientes (para detectar ciclos)
        self.recent_actions = []
        self.max_history = 8  # Guardar últimas 8 acciones
    
    def get_action(self, state):
        """Determina la acción basada en el estado actual"""
        # Producto punto de estado y pesos para cada acción
        q_values = np.dot(state, self.weights)
        
        # Verificar patrones cíclicos (repetitivos)
        has_cycle = self._check_for_cycles()
        
        # AJUSTE CRÍTICO 1: Verificar si hay peligro adelante y comida en la misma dirección
        danger_ahead = state[0]          # Peligro adelante
        food_direction = state[7:11]      # Dirección de la comida
        moving_direction = state[3:7]     # Dirección actual
        
        # Si hay peligro adelante, descartar la opción de seguir recto
        if danger_ahead > 0.5:
            q_values[0] = -100  # Penalizar fuertemente ir hacia adelante
            
        # AJUSTE CRÍTICO 2: Evaluar si es razonable moverse hacia la comida
        # Obtener la dirección que nos acercaría a la comida
        best_dir_to_food = -1
        if moving_direction[0] and food_direction[0]:  # Moviendo izq y comida izq
            best_dir_to_food = 0  # Seguir recto
        elif moving_direction[1] and food_direction[1]:  # Moviendo der y comida der
            best_dir_to_food = 0  # Seguir recto
        elif moving_direction[2] and food_direction[2]:  # Moviendo arriba y comida arriba
            best_dir_to_food = 0  # Seguir recto
        elif moving_direction[3] and food_direction[3]:  # Moviendo abajo y comida abajo
            best_dir_to_food = 0  # Seguir recto
        
        # Si hay comida a la derecha pero vamos vertical
        elif food_direction[1] and (moving_direction[2] or moving_direction[3]):
            best_dir_to_food = 1  # Girar derecha
        # Si hay comida a la izquierda pero vamos vertical
        elif food_direction[0] and (moving_direction[2] or moving_direction[3]):
            best_dir_to_food = 2  # Girar izquierda
        # Si hay comida arriba pero vamos horizontal
        elif food_direction[2] and (moving_direction[0] or moving_direction[1]):
            # Determinar qué giro nos lleva arriba dependiendo de la dirección actual
            best_dir_to_food = 1 if moving_direction[0] else 2  # Derecha si vamos izq, izq si vamos der
        # Si hay comida abajo pero vamos horizontal
        elif food_direction[3] and (moving_direction[0] or moving_direction[1]):
            # Determinar qué giro nos lleva abajo dependiendo de la dirección actual
            best_dir_to_food = 2 if moving_direction[0] else 1  # Izq si vamos izq, der si vamos der
        
        # Si identificamos claramente la mejor dirección para ir a la comida, darle bonus
        if best_dir_to_food >= 0 and random.random() < 0.8:  # 80% de ir hacia la comida
            # Verificar que esa dirección no sea peligrosa
            if (best_dir_to_food == 0 and not danger_ahead) or \
               (best_dir_to_food == 1 and not state[1]) or \
               (best_dir_to_food == 2 and not state[2]):
                q_values[best_dir_to_food] += 5.0  # Dar un bonus significativo
        
        # AJUSTE CRÍTICO 3: Romper patrones cíclicos con mayor probabilidad y fuerza
        if has_cycle:
            if random.random() < 0.6:  # 60% de probabilidad de romper ciclos
                # Ruido significativo para romper patrones
                q_values += np.random.normal(0, 2.0, size=q_values.shape)
                # Intentar moverse hacia una dirección completamente diferente
                if random.random() < 0.3:  # 30% chance de cambiar completamente
                    current_action = np.argmax(q_values)
                    q_values[current_action] = -10  # Penalizar acción actual
        
        # Obtener acción final y guardarla en historial
        action = np.argmax(q_values)
        self._update_action_history(action)
        
        return action
    
    def _update_action_history(self, action):
        """Actualiza el historial de acciones recientes"""
        self.recent_actions.append(action)
        if len(self.recent_actions) > self.max_history:
            self.recent_actions.pop(0)  # Eliminar la acción más antigua
    
    def _check_for_cycles(self):
        """Detecta patrones cíclicos en las acciones recientes"""
        if len(self.recent_actions) < 6:  # Necesitamos suficientes acciones para detectar ciclos
            return False
        
        # Verificar ciclos de 2 acciones (ej: 0,1,0,1,0,1)
        if len(self.recent_actions) >= 6:
            pattern = self.recent_actions[-2:]
            previous = self.recent_actions[-4:-2]
            older = self.recent_actions[-6:-4]
            if pattern == previous and pattern == older:
                return True
        
        # Verificar ciclos de 3 acciones (ej: 0,1,2,0,1,2)
        if len(self.recent_actions) >= 6:
            pattern = self.recent_actions[-3:]
            previous = self.recent_actions[-6:-3]
            if pattern == previous:
                return True
                
        return False

# Algoritmo Genético
class GeneticAlgorithm:
    def __init__(self, population_size=30, num_generations=50, mutation_rate=0.1, crossover_rate=0.8, elite_size=2):
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.population = []
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
    def initialize_population(self):
        """Inicializa la población con tablas de decisión aleatorias"""
        self.population = [DecisionTable() for _ in range(self.population_size)]
    
    def fitness(self, agent, num_games=5, show_game=False):
        """Evalúa el fitness de un agente sobre múltiples juegos"""
        total_score = 0
        total_steps = 0
        movement_efficiency = 0  # Eficiencia de movimiento
        unique_positions = 0     # Posiciones únicas visitadas
        foods_reached = 0        # Comidas alcanzadas
        avg_steps_per_food = []  # Pasos promedio para alcanzar comida
        
        # Jugar varios juegos para evaluar mejor al agente
        for game_num in range(num_games):
            # Usar diferentes semillas para evaluar con más robustez
            random.seed(game_num + int(time.time()) % 1000)
            
            game = SnakeGame(ai_control=True)
            done = False
            score = 0
            steps_since_last_food = 0
            
            # Máximo de pasos para evitar juegos infinitos durante evaluación
            max_steps = 400  # Reducido para evaluación más rápida
            steps_played = 0
            prev_distance = None
            positions_set = set()  # Conjunto para rastrear posiciones únicas
            direct_path_bonus = 0  # Bonus por tomar camino directo a la comida
            
            # Jugar hasta que termine
            while not done and steps_played < max_steps:
                state = game.get_state()
                action = agent.get_action(state)
                prev_score = score
                done, score, _ = game.play_step(action)
                steps_played += 1
                steps_since_last_food += 1
                
                # Rastrear posiciones únicas visitadas
                pos = (game.head.x, game.head.y)
                positions_set.add(pos)
                
                # Calcular distancia a la comida (Manhattan)
                curr_distance = abs(game.head.x - game.food.x) + abs(game.head.y - game.food.y)
                
                # Inicializar distancia previa si es la primera iteración
                if prev_distance is None:
                    prev_distance = curr_distance
                
                # Recompensar por acercarse a la comida
                if curr_distance < prev_distance:
                    movement_efficiency += 1
                    # Bonus adicional por moverse directamente hacia la comida
                    direct_path_bonus += 0.1
                else:
                    # Pequeña penalización por alejarse de la comida
                    direct_path_bonus -= 0.05
                
                prev_distance = curr_distance
                
                # Verificar si encontró comida
                if score > prev_score:
                    foods_reached += 1
                    # Registrar cuántos pasos tomó llegar a esta comida
                    avg_steps_per_food.append(steps_since_last_food)
                    steps_since_last_food = 0
                    # Bonus adicional por encontrar comida
                    direct_path_bonus += 5.0
                
                # No mostrar todos los juegos (solo para visualizar el mejor)
                if not show_game:
                    pygame.display.update()
            
            # Terminar si se alcanzó el límite de pasos
            if steps_played >= max_steps and not done:
                done = True
                
            total_score += score
            total_steps += game.steps
            unique_positions += len(positions_set)
        
        # Calcular eficiencia de rutas (menos pasos por comida = mejor)
        route_efficiency = 0
        if avg_steps_per_food:
            # Invertimos la relación: menos pasos = mayor eficiencia
            route_efficiency = 100 / (sum(avg_steps_per_food) / len(avg_steps_per_food) + 1)
        
        # Determinar si hubo muerte temprana
        muerte_temprana = 1 if total_score == 0 else 0
        
        # Fórmula de fitness mejorada para priorizar encuentro de comida
        fitness = (total_score * 30) + \
                 (foods_reached * 20) + \
                 (movement_efficiency * 1.0) + \
                 (route_efficiency * 2.0) + \
                 (unique_positions * 0.2) + \
                 (direct_path_bonus * 2.0) - \
                 (muerte_temprana * 100)  # Penalización más severa por muerte sin comer
                 
        avg_fitness = fitness / num_games
        
        return avg_fitness
    
    def selection(self, fitnesses):
        """Selecciona padres usando método de ruleta"""
        # Normalizar fitness (todos positivos sumando una constante si hay negativos)
        min_fitness = min(fitnesses)
        if min_fitness < 0:
            adjusted_fitnesses = [f - min_fitness + 1 for f in fitnesses]
        else:
            adjusted_fitnesses = fitnesses
            
        # Calcular probabilidades
        total_fitness = sum(adjusted_fitnesses)
        if total_fitness == 0:
            # Si todos tienen 0, selección aleatoria uniforme
            probabilities = [1/len(adjusted_fitnesses) for _ in adjusted_fitnesses]
        else:
            probabilities = [f/total_fitness for f in adjusted_fitnesses]
        
        # Seleccionar dos padres
        selected_indices = np.random.choice(len(self.population), size=2, p=probabilities)
        return selected_indices
    
    def crossover(self, parent1, parent2):
        """Realiza cruce entre dos padres"""
        if random.random() < self.crossover_rate:
            # Cruce de un punto
            child_weights = np.copy(parent1.weights)
            rows, cols = parent1.weights.shape
            
            # Seleccionar punto de cruce
            crossover_point = random.randint(0, rows * cols - 1)
            row_idx = crossover_point // cols
            col_idx = crossover_point % cols
            
            # Realizar cruce
            for i in range(rows):
                for j in range(cols):
                    if i > row_idx or (i == row_idx and j >= col_idx):
                        child_weights[i][j] = parent2.weights[i][j]
            
            return DecisionTable(child_weights)
        else:
            # Si no hay cruce, devolver copia del primer padre
            return DecisionTable(np.copy(parent1.weights))
    
    def mutate(self, agent):
        """Aplica mutación al agente con mayor variabilidad para evitar mínimos locales"""
        child_weights = np.copy(agent.weights)
        rows, cols = child_weights.shape
        
        for i in range(rows):
            for j in range(cols):
                if random.random() < self.mutation_rate:
                    # Añadir ruido gaussiano con mayor desviación (0.8 en vez de 0.5)
                    # para promover más exploración y evitar comportamientos repetitivos
                    child_weights[i][j] += np.random.normal(0, 0.8)
                    
                    # 10% de probabilidad de mutación más drástica para salir de mínimos locales
                    if random.random() < 0.1:
                        # Mutación más agresiva (cambio de signo o reemplazo total)
                        if random.random() < 0.5:
                            # Cambiar signo
                            child_weights[i][j] *= -1
                        else:
                            # Valor completamente nuevo
                            child_weights[i][j] = np.random.normal(0, 1.5)
        
        return DecisionTable(child_weights)
    
    def evolve(self, show_progress=True):
        """Ejecuta el algoritmo genético para evolucionar la población"""
        # Inicializar población si no está inicializada
        if not self.population:
            self.initialize_population()
        
        # Para cada generación
        for generation in range(self.num_generations):
            # Evaluar fitness
            fitnesses = [self.fitness(agent) for agent in self.population]
            
            # Encontrar mejor agente
            max_fitness_idx = np.argmax(fitnesses)
            best_fitness = fitnesses[max_fitness_idx]
            best_agent = self.population[max_fitness_idx]
            
            # Guardar estadísticas
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(sum(fitnesses) / len(fitnesses))
            
            if show_progress and generation % 5 == 0:
                print(f"Generación {generation}: Mejor fitness = {best_fitness}, Promedio = {self.avg_fitness_history[-1]:.2f}")
            
            # Crear nueva población
            new_population = []
            
            # Elitismo: pasar los mejores agentes directamente
            sorted_indices = np.argsort(fitnesses)[::-1]
            for i in range(self.elite_size):
                elite_idx = sorted_indices[i]
                new_population.append(DecisionTable(np.copy(self.population[elite_idx].weights)))
            
            # Generar el resto de la población mediante selección, cruce y mutación
            while len(new_population) < self.population_size:
                # Selección
                parent_indices = self.selection(fitnesses)
                parent1 = self.population[parent_indices[0]]
                parent2 = self.population[parent_indices[1]]
                
                # Cruce
                child = self.crossover(parent1, parent2)
                
                # Mutación
                child = self.mutate(child)
                
                # Agregar a nueva población
                new_population.append(child)
            
            # Reemplazar población
            self.population = new_population
        
        # Evaluar fitness final
        final_fitnesses = [self.fitness(agent) for agent in self.population]
        best_idx = np.argmax(final_fitnesses)
        
        return self.population[best_idx]
    
    def plot_fitness_history(self):
        """Grafica la evolución del fitness a lo largo de las generaciones"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.best_fitness_history, label='Mejor Fitness')
        plt.plot(self.avg_fitness_history, label='Fitness Promedio')
        plt.xlabel('Generación')
        plt.ylabel('Fitness')
        plt.legend()
        plt.title('Evolución del Fitness')
        plt.grid(True)
        plt.savefig('fitness_history.png')
        plt.show()
    
    def demo_best_agent(self, agent):
        """Demuestra el mejor agente en una partida con información de diagnóstico"""
        # Crear un juego con velocidad reducida para la demostración
        game = SnakeGame(ai_control=True)
        done = False
        score = 0
        steps = 0
        max_steps = 1000  # Límite de pasos para la demostración
        comida_encontrada = 0
        
        # Usar una velocidad de actualización más lenta para mejor visualización
        pause_time = 0.2  # Pausa más larga para mejor visualización
        
        # Reiniciar historial de acciones para la demostración
        agent.recent_actions = []
        
        print("\nIniciando demostración del mejor agente con diagnóstico detallado...")
        print("(Observar comportamiento - el juego termina al chocar, NO se reinicia automáticamente)")
        
        # Para visualizar el tablero
        def print_board_state():
            # Creamos una representación del tablero
            grid_width = game.grid_width
            grid_height = game.grid_height
            board = [[' ' for _ in range(grid_width)] for _ in range(grid_height)]
            
            # Marcar comida
            food_x = int(game.food.x // BLOCK_SIZE)
            food_y = int(game.food.y // BLOCK_SIZE)
            if 0 <= food_x < grid_width and 0 <= food_y < grid_height:
                board[food_y][food_x] = 'F'
                
            # Marcar serpiente
            for segment in game.snake:
                seg_x = int(segment.x // BLOCK_SIZE)
                seg_y = int(segment.y // BLOCK_SIZE)
                if 0 <= seg_x < grid_width and 0 <= seg_y < grid_height:
                    board[seg_y][seg_x] = 'S'
            
            # Marcar cabeza
            head_x = int(game.head.x // BLOCK_SIZE)
            head_y = int(game.head.y // BLOCK_SIZE)
            if 0 <= head_x < grid_width and 0 <= head_y < grid_height:
                board[head_y][head_x] = 'H'
                
            # Imprimir tablero
            print("\nEstado del tablero:")
            print(" " + "-" * (grid_width * 2 + 1))
            for row in board:
                print("|", end="")
                for cell in row:
                    print(f" {cell}", end="")
                print(" |")
            print(" " + "-" * (grid_width * 2 + 1))
            
            # Mostrar dirección actual
            direction_str = {
                Direction.RIGHT: "→",
                Direction.LEFT: "←",
                Direction.UP: "↑",
                Direction.DOWN: "↓"
            }.get(game.direction, "?")
            
            print(f"Dirección: {direction_str}")
            
            # Distancia a la comida
            dist_x = abs(game.head.x - game.food.x) // BLOCK_SIZE
            dist_y = abs(game.head.y - game.food.y) // BLOCK_SIZE
            manhattan_dist = dist_x + dist_y
            print(f"Distancia a la comida: {manhattan_dist} casillas (x:{dist_x}, y:{dist_y})")
            
            # Sensores de peligro
            state = game.get_state()
            print(f"Sensores de peligro: Adelante={state[0]}, Derecha={state[1]}, Izquierda={state[2]}")
            
            # Dirección relativa de la comida
            food_dirs = state[7:11]
            food_dir_str = ""
            if food_dirs[0]: food_dir_str += "izquierda "
            if food_dirs[1]: food_dir_str += "derecha "
            if food_dirs[2]: food_dir_str += "arriba "
            if food_dirs[3]: food_dir_str += "abajo "
            print(f"Comida: {food_dir_str}")
        
        # Mostrar tablero inicial
        print_board_state()
        
        # Historial de posiciones para detectar estancamiento
        position_history = []
        stuck_threshold = 10  # Número de posiciones repetidas para considerarse estancado
        
        while not done and steps < max_steps:
            # Obtener estado y acción
            state = game.get_state()
            
            # Calcular valores Q para cada acción
            q_values = np.dot(state, agent.weights)
            
            # Obtener acción
            action = agent.get_action(state)
            
            # Guardar posición actual para detectar estancamiento
            current_pos = (game.head.x, game.head.y)
            position_history.append(current_pos)
            if len(position_history) > stuck_threshold:
                position_history.pop(0)
            
            # Ejecutar acción
            prev_score = score
            done, score, reward = game.play_step(action)
            
            # Verificar si encontró comida
            if score > prev_score:
                comida_encontrada += 1
                print(f"\n¡COMIDA ENCONTRADA! Total: {comida_encontrada}")
                # Mostrar tablero cuando encuentra comida
                print_board_state()
            
            # Pausa para visualización
            time.sleep(pause_time)
            steps += 1
            
            # Mostrar información detallada cada 20 pasos
            if steps % 20 == 0:
                print(f"\nPaso: {steps}, Puntuación: {score}")
                action_names = ["Recto", "Derecha", "Izquierda"]
                print(f"Acción: {action_names[action]} (Q-values: {q_values.round(2)})")
                print_board_state()
                
            # Verificar si hay un patrón cíclico y mostrarlo
            if len(agent.recent_actions) >= 6 and agent._check_for_cycles():
                print(f"\nPatrón cíclico detectado: {agent.recent_actions[-6:]}")
                # Mostrar estado del tablero cuando se detecta un ciclo
                print_board_state()
                
            # Verificar estancamiento (misma posición repetida)
            if len(position_history) >= stuck_threshold:
                if position_history.count(current_pos) >= stuck_threshold // 2:
                    print(f"\n¡ALERTA! Serpiente estancada en la misma posición (x:{current_pos[0]//BLOCK_SIZE}, y:{current_pos[1]//BLOCK_SIZE})")
                    print_board_state()
        
        # Mostrar resumen al finalizar
        print("\n" + "=" * 50)
        print("RESUMEN DE LA DEMOSTRACIÓN:")
        print(f"Pasos totales: {steps}")
        print(f"Puntuación final: {score}")
        print(f"Comida encontrada: {comida_encontrada}")
        
        if steps >= max_steps:
            print("Demostración finalizada por límite de pasos.")
        elif done:
            if game._is_collision():
                # Mostrar dónde chocó
                collision_type = ""
                if game.head.x < 0 or game.head.x >= game.w:
                    collision_type = "pared lateral"
                elif game.head.y < 0 or game.head.y >= game.h:
                    collision_type = "pared superior/inferior"
                elif game.head in game.snake[1:]:
                    collision_type = "su propio cuerpo"
                print(f"La serpiente chocó con: {collision_type}")
            else:
                print("Demostración finalizada por otro motivo.")
                
        print("=" * 50)
            
        return score

def play_human():
    """Función para jugar manualmente"""
    game = SnakeGame(ai_control=False)
    
    while True:
        game_over, score, _ = game.play_step()
        
        if game_over:
            break
    
    print(f"Final Score: {score}")
    pygame.quit()

def train_and_play_ai():
    """Entrena un agente con algoritmo genético y lo demuestra"""
    # Parámetros del algoritmo genético optimizados para aprendizaje eficiente
    population_size = 70  # Aumentar para mayor diversidad
    num_generations = 30  # Reducir para entrenar más rápido
    mutation_rate = 0.25  # Mayor mutación para evitar mínimos locales
    crossover_rate = 0.85 # Mayor crossover para combinar buenas características
    elite_size = 5        # Preservar a los 5 mejores
    
    print(f"Iniciando entrenamiento con {population_size} agentes durante {num_generations} generaciones...")
    print("(Esto puede tomar varios minutos, por favor espera)")
    print("Se ha mejorado el algoritmo para evitar comportamientos cíclicos")
    
    # Crear y ejecutar algoritmo genético
    ga = GeneticAlgorithm(
        population_size=population_size,
        num_generations=num_generations,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        elite_size=elite_size
    )
    
    # Evolucionar población
    best_agent = ga.evolve(show_progress=True)
    
    # Mostrar gráfica de evolución
    ga.plot_fitness_history()
    
    # Demostrar mejor agente
    print("\nDemostrando mejor agente...")
    print("(La serpiente se moverá más lento para mejor visualización)")
    best_score = ga.demo_best_agent(best_agent)
    
    # Preguntar si quiere jugar de nuevo con el mismo agente
    retry = input("\n¿Quieres ver otra demostración con el mejor agente? (s/n): ")
    if retry.lower() == 's':
        print("\nMostrando otra demostración...")
        ga.demo_best_agent(best_agent)
    
    return best_agent, best_score

if __name__ == "__main__":
    # Modo de juego (Human o AI)
    mode = input("Selecciona modo de juego (1: Humano, 2: IA): ")
    
    if mode == "1":
        play_human()
    else:
        train_and_play_ai()
    
    pygame.quit()
