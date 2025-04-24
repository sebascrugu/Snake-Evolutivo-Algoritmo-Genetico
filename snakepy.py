# Código modificado para implementar algoritmo genético
# Basado en https://github.com/patrickloeber/snake-ai-pytorch/blob/main/snake_game_human.py

import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from enum import Enum
import time
import builtins
import os

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

    def __init__(self, width=None, height=None, ai_control=False, training_mode=False, headless=False):
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
        # Modo entrenamiento (si es True, permite reiniciar el juego para entrenar)
        self.training_mode = training_mode
        self.headless = headless
        
        # init display
        if not headless:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption("Snake - AI Genetic")
        else:
            # En modo headless, crear una superficie invisible para simulaciones
            self.display = pygame.Surface((self.w, self.h))
            
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
            # En modo headless no es necesario procesar eventos
            if not self.headless:
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
            
            # Siempre terminar el juego cuando hay colisión
            if self._is_collision():
                # Actualizar la pantalla una última vez para mostrar la colisión
                self._update_ui()
                if not self.headless:
                    pygame.display.flip()
                    time.sleep(0.2)  # Pausa breve para ver la colisión
            
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
        if not self.headless:
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
        
        # Solo actualizar la visualización si no estamos en modo headless
        if not self.headless:
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

    def crossover(self, other):
        """
        Realiza cruce entre esta tabla de decisión y otra.
        
        Args:
            other: Otra tabla de decisión
            
        Returns:
            Una nueva tabla de decisión resultante del cruce
        """
        # Crear una copia de los pesos del primer padre
        child_weights = np.copy(self.weights)
        
        # Cruce uniforme (50% de probabilidad de heredar de cada padre)
        rows, cols = self.weights.shape
        for i in range(rows):
            for j in range(cols):
                # 50% de probabilidad de heredar de cada padre
                if random.random() < 0.5:
                    child_weights[i][j] = other.weights[i][j]
        
        # Crear nueva tabla de decisión con los pesos resultantes
        return DecisionTable(child_weights)
    
    def mutate(self, mutation_rate):
        """
        Aplica mutación simple a esta tabla con la tasa especificada.
        Este método es más simple que el de GeneticAlgorithm y se usa
        para mutaciones directas del agente.
        
        Args:
            mutation_rate: Probabilidad de mutación para cada gen
            
        Returns:
            Esta tabla mutada (para permitir encadenamiento)
        """
        # Aplicar mutación con cierta probabilidad a cada peso
        rows, cols = self.weights.shape
        
        for i in range(rows):
            for j in range(cols):
                if random.random() < mutation_rate:
                    # Añadir ruido gaussiano con desviación moderada
                    self.weights[i][j] += np.random.normal(0, 0.5)
                    
                    # 10% de probabilidad de mutación más drástica
                    if random.random() < 0.1:
                        # Mutación más agresiva (cambio de signo o reemplazo total)
                        if random.random() < 0.5:
                            # Cambiar signo
                            self.weights[i][j] *= -1
                        else:
                            # Valor completamente nuevo
                            self.weights[i][j] = np.random.normal(0, 1.5)
        
        return self

# Algoritmo Genético
class GeneticAlgorithm:
    def __init__(self, population_size=30, num_generations=50, mutation_rate=0.1, crossover_rate=0.8, elite_size=2):
        # Atributo para configurar tipo de cruce (para experimentos)
        self.crossover_type = "one_point"  # Opciones: "one_point", "two_point", "uniform"
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.population = []
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
        # Variables para mecanismos adaptativos
        self.stagnation_counter = 0        # Contador de generaciones sin mejora significativa
        self.improvement_rate_history = [] # Historial de tasas de mejora
        self.best_fitness_ever = 0         # Mejor fitness encontrado hasta ahora
        self.tournament_size = 5           # Tamaño inicial del torneo
        self.offspring_count = {}          # Contador de descendientes por individuo
        self.diversity_threshold = 0.1     # Umbral de diversidad para estrategias correctivas
        
    def initialize_population(self):
        """Inicializa la población con tablas de decisión aleatorias"""
        self.population = [DecisionTable() for _ in range(self.population_size)]
    
    def fitness(self, agent, num_games=7, show_game=False, silent=False):
        """Evalúa el fitness de un agente sobre el número de juegos especificado con semillas consistentes"""
        total_score = 0
        total_steps = 0
        movement_efficiency = 0  # Eficiencia de movimiento
        unique_positions = 0     # Posiciones únicas visitadas
        foods_reached = 0        # Comidas alcanzadas (total acumulado)
        avg_steps_per_food = []  # Pasos promedio para alcanzar comida
        
        # Nuevas variables para el reequilibrio de fitness
        foods_per_game = []      # Comida obtenida en cada juego
        consecutive_avoidance = 0 # Contador de evitaciones consecutivas
        max_snake_length = 3     # Longitud máxima alcanzada por la serpiente
        
        # Semillas fijas para evaluación consistente
        base_seed = 42  # Semilla base arbitraria pero fija
        seeds = [base_seed + i * 1000 for i in range(num_games)]  # Generar semillas espaciadas
        
        if not silent:
            print(f"\nIniciando evaluación de agente en {num_games} juegos...")
        
        # Control de visualización para procesos en paralelo
        # Deshabilitar temporalmente la visualización de Pygame si estamos en modo silencioso
        os_environ_copy = None
        if silent and 'SDL_VIDEODRIVER' not in os.environ:
            os_environ_copy = os.environ.copy()
            os.environ['SDL_VIDEODRIVER'] = 'dummy'  # Usar driver nulo para Pygame
            
        # Jugar el número especificado de juegos para evaluar al agente
        for game_num in range(num_games):
            # Usar semilla fija para este juego
            random.seed(seeds[game_num])
            np.random.seed(seeds[game_num])
            
            # Crear un nuevo juego para cada evaluación
            # Desactivamos los mensajes de depuración durante la evaluación
            # para no saturar la consola
            original_print = print
            if game_num > 0 or silent:  # Silenciar todos los mensajes en modo silencioso
                def silent_print(*args, **kwargs):
                    pass
                builtins.print = silent_print
            
            # Crear juego con visualización desactivada si estamos en modo silencioso
            game = SnakeGame(ai_control=True, training_mode=False, headless=silent)
            done = False
            score = 0
            steps_since_last_food = 0
            
            # Máximo de pasos para evitar juegos infinitos durante evaluación
            max_steps = 400  # Reducido para evaluación más rápida
            steps_played = 0
            prev_distance = None
            positions_set = set()  # Conjunto para rastrear posiciones únicas
            direct_path_bonus = 0  # Bonus por tomar camino directo a la comida
            
            # Nuevas variables para métricas avanzadas
            attempts_toward_food = 0       # Intentos de moverse hacia la comida
            successful_food_approaches = 0  # Acercamientos exitosos a la comida
            wall_avoidance_count = 0       # Veces que evitó una pared
            near_death_avoidance = 0       # Veces que evitó una muerte cercana
            consecutive_approach_food = 0  # Contador de aproximación consistente a la comida
            food_eaten_in_game = 0         # Comida comida en este juego específico
            
            # Rastreo de patrones de movimiento
            last_positions = []            # Últimas posiciones para detectar ciclos
            repeated_cycles = 0            # Contador de ciclos repetidos
            max_cycle_history = 20         # Tamaño máximo del historial de posiciones
            
            # Distancia inicial a la comida
            initial_food_distance = abs(game.head.x - game.food.x) + abs(game.head.y - game.food.y)
            
            print(f"Jugando partida {game_num + 1} de {num_games}...")
            
            # Jugar hasta que termine
            while not done and steps_played < max_steps:
                state = game.get_state()
                action = agent.get_action(state)
                prev_score = score
                
                # Guardar posición antes de moverse
                prev_pos = (game.head.x, game.head.y)
                prev_snake_length = len(game.snake)
                
                # Verificar si va a evitar una pared o colisión inminente
                danger_ahead = state[0]
                danger_right = state[1]
                danger_left = state[2]
                
                # Detectar si se va a evitar una pared
                if danger_ahead and action != 0:  # No va hacia adelante cuando hay peligro
                    wall_avoidance_count += 1
                    near_death_avoidance += 1
                    consecutive_avoidance += 1
                elif danger_right and action != 1:  # No va a la derecha cuando hay peligro
                    wall_avoidance_count += 1
                    consecutive_avoidance += 1
                elif danger_left and action != 2:  # No va a la izquierda cuando hay peligro
                    wall_avoidance_count += 1
                    consecutive_avoidance += 1
                else:
                    # Reiniciar contador si no evitó ningún peligro
                    consecutive_avoidance = 0
                
                done, score, _ = game.play_step(action)
                steps_played += 1
                steps_since_last_food += 1
                
                # Rastrear posiciones únicas visitadas
                pos = (game.head.x, game.head.y)
                positions_set.add(pos)
                
                # Actualizar historial de posiciones para detección de ciclos
                last_positions.append(pos)
                if len(last_positions) > max_cycle_history:
                    last_positions.pop(0)
                
                # Detectar ciclos repetitivos (patrones de 2, 3 o 4 movimientos)
                if len(last_positions) >= 8:
                    for cycle_len in [2, 3, 4]:
                        if len(last_positions) >= cycle_len * 2:
                            recent = last_positions[-cycle_len:]
                            previous = last_positions[-2*cycle_len:-cycle_len]
                            if recent == previous:
                                repeated_cycles += 1
                                break
                
                # Calcular distancia a la comida (Manhattan)
                curr_distance = abs(game.head.x - game.food.x) + abs(game.head.y - game.food.y)
                
                # Inicializar distancia previa si es la primera iteración
                if prev_distance is None:
                    prev_distance = curr_distance
                
                # Analizar si se acerca a la comida
                if curr_distance < prev_distance:
                    movement_efficiency += 1
                    direct_path_bonus += 0.3  # Aumentado para dar más valor a moverse hacia la comida
                    successful_food_approaches += 1
                    consecutive_approach_food += 1
                else:
                    # Penalización por alejarse de la comida
                    direct_path_bonus -= 0.1
                    consecutive_approach_food = 0
                
                # Bonus exponencial por aproximación consistente
                if consecutive_approach_food >= 3:
                    direct_path_bonus += consecutive_approach_food * 0.5
                
                prev_distance = curr_distance
                
                # Verificar si encontró comida
                if score > prev_score:
                    foods_reached += 1
                    food_eaten_in_game += 1
                    # Registrar cuántos pasos tomó llegar a esta comida
                    avg_steps_per_food.append(steps_since_last_food)
                    steps_since_last_food = 0
                    
                    # Bonus por comer comida (mayor bonus mientras más crece)
                    food_bonus_multiplier = 1 + (food_eaten_in_game * 0.5)  # Bonificación creciente
                    direct_path_bonus += 10.0 * food_bonus_multiplier
                    
                    # Reiniciar distancia para próxima comida
                    prev_distance = None
                
                # No mostrar todos los juegos (solo para visualizar el mejor)
                if show_game:
                    pygame.display.update()
            
            # Guardar distancia final a la comida al terminar
            final_food_distance = curr_distance if 'curr_distance' in locals() else initial_food_distance
            
            # Terminar si se alcanzó el límite de pasos
            if steps_played >= max_steps and not done:
                done = True
                
            total_score += score
            total_steps += game.steps
            unique_positions += len(positions_set)
            
            # Registrar comida comida en este juego para análisis de consistencia
            foods_per_game.append(food_eaten_in_game)
            
            # Actualizar la longitud máxima alcanzada
            max_snake_length = max(max_snake_length, 3 + food_eaten_in_game)
            
            # Restaurar la función print original al final de cada juego
            if game_num > 0 or silent:
                builtins.print = original_print
                
            # Mostrar resultados de este juego específico
            if not silent:
                print(f"\nJuego {game_num + 1} de {num_games} completado")
                print(f"  - Puntuación: {score}")
                print(f"  - Pasos totales: {steps_played}")
                print(f"  - Comida encontrada en este juego: {food_eaten_in_game}")
                print(f"  - Posiciones únicas visitadas: {len(positions_set)}")
                print(f"  - Eficiencia de ruta: {(food_eaten_in_game / steps_played * 100):.2f}% (comidas/pasos)")
        
        # COMPONENTE 1: OBJETIVOS PRIMARIOS
        # Puntos por comida (objetivo principal) - REEQUILIBRADO
        base_food_value = 45  # Reducido de 70 a 45 para balancear mejor con otros componentes
        
        # Valor incremental por cada comida adicional (comida más valiosa progresivamente)
        incremental_food_points = 0
        for i in range(total_score):
            # Cada comida vale un 10% más que la anterior
            food_value = base_food_value * (1 + (i * 0.1))
            incremental_food_points += food_value
            
        food_points = incremental_food_points if total_score > 0 else 0
        
        # Puntos por supervivencia (escalados según la longitud) - REEQUILIBRADO
        # Ahora es más significativo para compensar la reducción en valor de comida
        survival_points = total_steps * 0.5
        
        # COMPONENTE 2: EFICIENCIA Y COMPORTAMIENTO INTELIGENTE
        # Eficiencia en conseguir comida - REEQUILIBRADO
        route_efficiency = 0
        food_points_per_step = 0
        if total_steps > 0:
            # Nueva métrica de eficiencia: puntos por paso
            food_points_per_step = (total_score / max(1, total_steps)) * 500
            
            # Mantener la métrica anterior pero con menos peso
            if avg_steps_per_food:
                route_efficiency = 100 / (sum(avg_steps_per_food) / len(avg_steps_per_food) + 1)
        
        # Bonus por aproximación a la comida
        food_approach_bonus = successful_food_approaches * 3
        
        # COMPONENTE 3: CRECIMIENTO Y EXPLORACIÓN
        # Bonus por exploración (evitar quedarse en áreas pequeñas)
        exploration_bonus = unique_positions * 0.5
        
        # NUEVO: Bonificación por consistencia entre juegos
        # Calculamos la desviación estándar de comida entre juegos
        # Una baja desviación indica mayor consistencia
        food_consistency_bonus = 0
        if len(foods_per_game) > 1:
            # Si hay más de un juego, calcular desviación estándar
            std_dev = np.std(foods_per_game)
            # Menor desviación = mayor bonificación
            food_consistency_bonus = 20 * (1 / (1 + std_dev))
        
        # Bonus por evitar paredes y colisiones - AUMENTADO SIGNIFICATIVAMENTE
        # Aumentamos dramáticamente el valor de evitación de colisiones
        basic_avoidance = wall_avoidance_count * 4.0  # Aumentado de 0.5 a 4.0
        critical_avoidance = near_death_avoidance * 10.0  # Aumentado de 2.0 a 10.0
        
        # NUEVO: Bonificación por secuencias exitosas de evitación
        consecutive_avoidance_bonus = 0
        if consecutive_avoidance > 0:
            # Recompensa cuadrática por evitaciones consecutivas
            consecutive_avoidance_bonus = consecutive_avoidance * consecutive_avoidance * 0.5
        
        # NUEVO: Bonificación por supervivencia en espacios reducidos
        confined_space_bonus = 0
        # Si la serpiente es larga y ha visitado pocas posiciones únicas en proporción
        if max_snake_length > 5 and unique_positions < total_steps * 0.5:
            confined_space_bonus = 30  # Bonificación por maniobrar en espacio reducido
        
        # Combinar todas las bonificaciones de evitación
        avoidance_bonus = basic_avoidance + critical_avoidance + consecutive_avoidance_bonus + confined_space_bonus
        
        # COMPONENTE 4: PENALIZACIONES
        # Penalización por muerte temprana - REEQUILIBRADO
        # Reemplazamos la penalización fija con una gradual
        early_death_penalty = 0
        if total_score == 0:  # No comió nada
            # Penalización proporcional a cuánto tiempo le faltó para llegar a 100 pasos
            early_death_penalty = 200 * (1 - min(1.0, total_steps/100))
        elif total_steps < 50:  # Muerte temprana pero comió algo
            # Penalización menor si al menos comió algo
            early_death_penalty = 100 * (1 - min(1.0, total_steps/50))
        
        # Penalización por movimientos repetitivos - REEQUILIBRADO
        # Penalización exponencial para ciclos prolongados
        repetition_penalty = 0
        if repeated_cycles > 0:
            # Penalización crece exponencialmente con el número de ciclos
            repetition_penalty = 5 * (repeated_cycles ** 1.5)
        
        # CÁLCULO FINAL CON PONDERACIONES OPTIMIZADAS
        fitness = (
            food_points +                   # Valor de comida (ahora más equilibrado)
            survival_points +               # Valor de supervivencia (incrementado)
            route_efficiency * 2.0 +        # Eficiencia de ruta (reducido de 4.0 a 2.0)
            food_points_per_step * 2.0 +    # Nueva métrica de eficiencia
            food_approach_bonus +           # Aproximación a comida
            food_consistency_bonus +        # NUEVO: Consistencia entre juegos
            direct_path_bonus * 2.0 +       # Dirección hacia comida (ligeramente reducido)
            exploration_bonus +             # Exploración
            avoidance_bonus +               # Evitación de obstáculos (significativamente aumentado)
            (movement_efficiency * 1.0) -   # Eficiencia de movimiento
            early_death_penalty -           # Penalización gradual por muerte temprana
            repetition_penalty              # Penalización por ciclos (ahora exponencial)
        )
        
        # Variables adicionales que se deben inicializar antes del bucle principal
        consecutive_avoidance = 0           # Para rastrear evitaciones consecutivas
        foods_per_game = []                 # Para rastrear comida por juego
        
        # Resumen de la evaluación
        if not silent:
            print("\n" + "="*60)
            print(f"RESUMEN DE EVALUACIÓN DEL AGENTE")
            print(f"  • Total de comida encontrada: {total_score}")
            print(f"  • Total de pasos realizados: {total_steps}")
            print(f"  • Eficiencia de movimiento: {movement_efficiency}")
            if avg_steps_per_food:
                print(f"  • Promedio de pasos por comida: {sum(avg_steps_per_food) / len(avg_steps_per_food):.2f}")
            print(f"  • Posiciones únicas visitadas: {unique_positions}")
            
            # Mostrar desgloses de la nueva función de fitness
            print(f"  • Puntos por comida: {food_points:.2f}")
            print(f"  • Puntos por supervivencia: {survival_points:.2f}")
            print(f"  • Bonificación por evitación de obstáculos: {avoidance_bonus:.2f}")
            print(f"  • Penalización por muerte temprana: {early_death_penalty:.2f}")
            print(f"  • Penalización por ciclos repetitivos: {repetition_penalty:.2f}")
            if food_consistency_bonus > 0:
                print(f"  • Bonificación por consistencia: {food_consistency_bonus:.2f}")
            
            print(f"  • Fitness calculado: {fitness:.2f}")
            print("="*60)
        
        # Restaurar variables de entorno si las modificamos
        if os_environ_copy is not None:
            os.environ.clear()
            os.environ.update(os_environ_copy)
            
        # Garantizar un valor mínimo positivo para mantener diversidad genética
        fitness = max(1.0, fitness / num_games)
        
        return fitness
    
    def selection(self, fitnesses):
        """
        Selecciona padres utilizando selección por torneo con características
        adaptativas para mantener diversidad genética.
        """
        # Ajustar dinámicamente el tamaño del torneo basado en diversidad y estancamiento
        if self.stagnation_counter > 5:
            # Reducir tamaño de torneo para disminuir presión selectiva y preservar diversidad
            self.tournament_size = max(3, self.tournament_size - 1)
        elif len(self.improvement_rate_history) > 2 and sum(self.improvement_rate_history[-2:]) > 0.05:
            # Aumentar tamaño de torneo cuando hay mejoras consistentes
            self.tournament_size = min(8, self.tournament_size + 1)
        
        selected_indices = []
        
        # Selección ocasional puramente aleatoria para mantener diversidad (10% de probabilidad)
        if random.random() < 0.1:
            # Selección totalmente aleatoria
            selected_indices = random.sample(range(len(self.population)), 2)
        else:
            # Primer torneo - enfocado en calidad
            for _ in range(2):  # Seleccionar dos padres
                # Seleccionar individuos aleatorios para el torneo
                tournament = random.sample(range(len(self.population)), self.tournament_size)
                
                # Comprobar si necesitamos aplicar compartición de fitness
                need_sharing = self.stagnation_counter > 3 and random.random() < 0.7
                
                if need_sharing:
                    # Aplicar compartición de fitness para penalizar individuos similares o sobrerepresentados
                    adjusted_fitnesses = []
                    for idx in tournament:
                        # Penalizar basado en número de descendientes que ya ha tenido
                        offspring_penalty = self.offspring_count.get(idx, 0) * 0.1
                        
                        # Penalizar similitud con otros individuos seleccionados
                        similarity_penalty = 0
                        if len(selected_indices) > 0:
                            # Si ya seleccionamos un padre, añadir penalización por similitud
                            first_parent = self.population[selected_indices[0]]
                            current = self.population[idx]
                            # Calcular similitud basada en pesos (distancia euclidiana normalizada)
                            diff = np.sum((first_parent.weights - current.weights)**2)
                            similarity = 1.0 / (1.0 + diff)  # Mayor similitud = mayor penalización
                            similarity_penalty = similarity * 0.3 * fitnesses[idx]
                        
                        # Fitness ajustado final
                        adjusted_fitness = fitnesses[idx] - offspring_penalty - similarity_penalty
                        adjusted_fitnesses.append(adjusted_fitness)
                    
                    # Seleccionar el mejor individuo del torneo con fitness ajustado
                    best_idx = tournament[np.argmax(adjusted_fitnesses)]
                else:
                    # Selección normal por torneo - elegir el mejor fitness
                    best_idx = tournament[np.argmax([fitnesses[i] for i in tournament])]
                
                selected_indices.append(best_idx)
                
                # Actualizar contador de descendientes
                self.offspring_count[best_idx] = self.offspring_count.get(best_idx, 0) + 1
        
        # Asegurar que no se seleccione dos veces el mismo individuo
        if selected_indices[0] == selected_indices[1]:
            # Reemplazar el segundo índice con otro individuo aleatorio distinto
            candidates = [i for i in range(len(self.population)) if i != selected_indices[0]]
            if candidates:
                selected_indices[1] = random.choice(candidates)
        
        return selected_indices
    
    def crossover(self, parent1, parent2):
        """Realiza cruce entre dos padres"""
        if random.random() < self.crossover_rate:
            # Tipo de cruce configurable (por defecto: cruce de un punto)
            crossover_type = getattr(self, 'crossover_type', 'one_point')
            
            if crossover_type == "uniform":
                # Cruce uniforme (50% de prob. de heredar de cada padre)
                child_weights = np.zeros_like(parent1.weights)
                rows, cols = parent1.weights.shape
                for i in range(rows):
                    for j in range(cols):
                        # 50% de probabilidad de heredar de cada padre
                        if random.random() < 0.5:
                            child_weights[i][j] = parent1.weights[i][j]
                        else:
                            child_weights[i][j] = parent2.weights[i][j]
                
                return DecisionTable(child_weights)
                
            elif crossover_type == "two_point":
                # Cruce de dos puntos
                child_weights = np.copy(parent1.weights)
                rows, cols = parent1.weights.shape
                
                # Seleccionar dos puntos de cruce
                total_genes = rows * cols
                point1 = random.randint(0, total_genes - 2)
                point2 = random.randint(point1 + 1, total_genes - 1)
                
                # Convertir a índices 2D
                for idx in range(total_genes):
                    i = idx // cols
                    j = idx % cols
                    # Asignar genes del segundo padre solo entre los dos puntos
                    if point1 < idx <= point2:
                        child_weights[i][j] = parent2.weights[i][j]
                
                return DecisionTable(child_weights)
                
            else:  # one_point (predeterminado)
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
        """
        Aplica mutación adaptativa al agente con tasas variables según la parte
        de la tabla de decisión, estancamiento y progreso evolutivo.
        
        Args:
            agent: Agente a mutar
            
        Returns:
            DecisionTable: Agente mutado
        """
        child_weights = np.copy(agent.weights)
        rows, cols = child_weights.shape
        
        # 1. Tasa base que disminuye gradualmente con las generaciones
        # Calculamos el progreso de evolución normalizado (0 al inicio, 1 al final)
        if hasattr(self, 'evolve_progress') and self.num_generations > 0:
            evolution_progress = self.evolve_progress / self.num_generations
        else:
            evolution_progress = 0.5  # Valor intermedio por defecto
            
        # La tasa base comienza alta y disminuye gradualmente
        base_rate = self.mutation_rate * (1.0 - evolution_progress * 0.7)
        
        # 2. Aumentar significativamente la tasa cuando hay estancamiento
        stagnation_factor = min(0.5, self.stagnation_counter * 0.1)  # Tope de 50% de aumento
        
        # 3. Aplicar factores dinámicos basados en las mejoras recientes
        recent_improvement = 0
        if len(self.improvement_rate_history) > 0:
            # Promedio de mejoras recientes (negativo si empeoró)
            recent_improvement = sum(self.improvement_rate_history[-3:]) / min(3, len(self.improvement_rate_history))
        
        # Si hay mejoras recientes, reducir mutación. Si hay deterioro, aumentarla.
        improvement_factor = max(-0.2, min(0.2, -recent_improvement))
        
        # Combinar factores - el estancamiento siempre aumenta la mutación
        # mientras que las mejoras pueden reducirla
        adjusted_rate = base_rate * (1.0 + stagnation_factor + improvement_factor)
        
        # Asegurar que la tasa esté dentro de límites razonables
        adjusted_rate = max(0.01, min(0.8, adjusted_rate))
        
        # 4. Diferentes tasas para distintas partes de la tabla de decisión
        
        # Dividir la tabla en secciones funcionales
        # Filas 0-2: Detección de peligros - requiere alta precisión, baja mutación
        # Filas 3-6: Dirección actual - requiere estabilidad, baja mutación
        # Filas 7-10: Ubicación de comida - más flexible, mutación moderada
        # Filas 11-14: Distancias a bordes - más flexible, mutación moderada
        # Filas 15-17: Info avanzada comida - crítica para optimización, mutación adaptable
        section_rates = {
            'danger': adjusted_rate * 0.5,       # Menor tasa para pesos de detección peligro
            'direction': adjusted_rate * 0.6,     # Menor tasa para pesos de dirección
            'food_location': adjusted_rate * 1.2, # Mayor tasa para pesos de ubicación comida
            'borders': adjusted_rate * 1.0,       # Tasa normal para distancias a bordes
            'food_advanced': adjusted_rate * 1.3  # Mayor tasa para info avanzada de comida
        }
        
        # Aplicar mutación con tasas diferenciadas por sección
        for i in range(rows):
            # Determinar la sección a la que pertenece esta fila
            if i <= 2:
                section_rate = section_rates['danger']
            elif i <= 6:
                section_rate = section_rates['direction']
            elif i <= 10:
                section_rate = section_rates['food_location']
            elif i <= 14:
                section_rate = section_rates['borders']
            else:
                section_rate = section_rates['food_advanced']
                
            for j in range(cols):
                if random.random() < section_rate:
                    # Magnitud de mutación también adaptativa 
                    # - Mayor para secciones con alta tasa
                    # - Mayor cuando hay estancamiento
                    mutation_magnitude = 0.6 + stagnation_factor * 0.4
                    
                    # Añadir ruido gaussiano con magnitud adaptativa
                    child_weights[i][j] += np.random.normal(0, mutation_magnitude)
                    
                    # Probabilidad de mutación más drástica basada en estancamiento
                    drastic_prob = 0.1 + stagnation_factor * 0.2
                    if random.random() < drastic_prob:
                        # Mutación más agresiva (cambio de signo o reemplazo total)
                        if random.random() < 0.5:
                            # Cambiar signo
                            child_weights[i][j] *= -1
                        else:
                            # Valor completamente nuevo
                            child_weights[i][j] = np.random.normal(0, 1.5)
        
        # Retornar agente mutado
        return DecisionTable(child_weights)
    
    def evolve(self, show_progress=True):
        """Ejecuta el algoritmo genético para evolucionar la población"""
        # Inicializar población si no está inicializada
        if not self.population:
            self.initialize_population()
        
        # Reiniciar variables para seguimiento
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.stagnation_counter = 0
        self.improvement_rate_history = []
        self.best_fitness_ever = 0
        self.offspring_count = {}
        
        # Para cada generación
        for generation in range(self.num_generations):
            # Guardar progreso actual para cálculos de mutación adaptativa
            self.evolve_progress = generation
            
            # Evaluar fitness
            fitnesses = [self.fitness(agent) for agent in self.population]
            
            # Encontrar mejor agente
            max_fitness_idx = np.argmax(fitnesses)
            best_fitness = fitnesses[max_fitness_idx]
            best_agent = self.population[max_fitness_idx]
            
            # Calcular tasa de mejora respecto a generación anterior
            if len(self.best_fitness_history) > 0:
                prev_best = self.best_fitness_history[-1]
                improvement_rate = (best_fitness - prev_best) / (prev_best + 0.01)  # Evitar división por cero
                self.improvement_rate_history.append(improvement_rate)
                
                # Actualizar contador de estancamiento
                if best_fitness > self.best_fitness_ever * 1.01:  # Mejora de al menos 1%
                    self.best_fitness_ever = best_fitness
                    self.stagnation_counter = 0
                else:
                    self.stagnation_counter += 1
            else:
                # Inicializar en la primera generación
                self.best_fitness_ever = best_fitness
                self.improvement_rate_history.append(0.0)
            
            # Guardar estadísticas
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(sum(fitnesses) / len(fitnesses))
            
            if show_progress and generation % 5 == 0:
                print(f"Generación {generation}: Mejor fitness = {best_fitness}, Promedio = {self.avg_fitness_history[-1]:.2f}")
                if self.stagnation_counter > 0:
                    print(f"  Estancamiento: {self.stagnation_counter} generaciones sin mejora significativa")
            
            # Si es la última generación, terminar
            if generation == self.num_generations - 1:
                break
                
            # Crear nueva población
            new_population = []
            
            # Reiniciar contador de descendientes para esta generación
            self.offspring_count = {}
            
            # Elitismo: pasar los mejores agentes directamente
            sorted_indices = np.argsort(fitnesses)[::-1]
            
            # Elitismo adaptativo: más élites durante estancamiento
            effective_elite_size = self.elite_size
            if self.stagnation_counter > 5:
                # Aumentar elitismo para preservar buenas soluciones
                effective_elite_size = min(self.population_size // 4, self.elite_size * 2)
            
            for i in range(effective_elite_size):
                elite_idx = sorted_indices[i]
                new_population.append(DecisionTable(np.copy(self.population[elite_idx].weights)))
            
            # Generar el resto de la población mediante selección, cruce y mutación
            while len(new_population) < self.population_size:
                # Selección con método de torneo adaptativo
                parent_indices = self.selection(fitnesses)
                parent1 = self.population[parent_indices[0]]
                parent2 = self.population[parent_indices[1]]
                
                # Cruce
                child = self.crossover(parent1, parent2)
                
                # Mutación adaptativa con tasas variables
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

def play_human():
    """Función para jugar manualmente"""
    # En modo humano, no queremos reinicio automático cuando hay colisión
    game = SnakeGame(ai_control=False, training_mode=False)
    
    while True:
        game_over, score, _ = game.play_step()
        
        if game_over:
            break
    
    print(f"Final Score: {score}")
    pygame.quit()

if __name__ == "__main__":
    # Menú principal con opciones más claras
    print("\n" + "="*50)
    print("SNAKE CON ALGORITMO GENÉTICO")
    print("="*50)
    print("\nOPCIONES DISPONIBLES:")
    print("1. Juego manual (control con flechas)")
    print("2. Experimentos comparativos (3 configuraciones)")
    
    try:
        mode = int(input("\nSelecciona una opción (1-2): "))
    except ValueError:
        mode = 0
    
    if mode == 1:
        print("\nIniciando juego manual. Usa las flechas del teclado para controlar la serpiente.")
        play_human()
    elif mode == 2:
        # Importar y ejecutar el experimento de 3 juegos
        try:
            # Verificar si el archivo existe primero
            import os
            if not os.path.exists("snake_experiments.py"):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                print(f"\nBuscando snake_experiments.py en: {script_dir}")
                if os.path.exists(os.path.join(script_dir, "snake_experiments.py")):
                    print(f"Archivo encontrado en el directorio del script")
                else:
                    raise ImportError(f"No se encuentra el archivo snake_experiments.py en {script_dir}")
            
            # Intentar importar usando importlib para mayor robustez
            import importlib.util
            import sys
            
            # Determinar la ruta completa al archivo
            script_dir = os.path.dirname(os.path.abspath(__file__))
            module_path = os.path.join(script_dir, "snake_experiments.py")
            
            # Importar el módulo dinámicamente
            spec = importlib.util.spec_from_file_location("snake_experiments", module_path)
            snake_experiments = importlib.util.module_from_spec(spec)
            sys.modules["snake_experiments"] = snake_experiments
            spec.loader.exec_module(snake_experiments)
            
            print("\nIniciando experimentos comparativos con 3 configuraciones diferentes...")
            snake_experiments.run_experiments()
        except ImportError as e:
            print(f"\nError: {e}")
            print("Verifica que el archivo 'snake_experiments.py' esté en el mismo directorio.")
    else:
        print("\nOpción no válida. Selecciona 1 o 2.")
    
    # Asegurar que pygame se cierre correctamente al finalizar
    pygame.quit()
