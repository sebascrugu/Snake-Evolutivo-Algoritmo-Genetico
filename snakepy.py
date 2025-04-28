# Base Snake

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
    
class Action(Enum):
    STRAIGHT = 0
    RIGHT_TURN = 1
    LEFT_TURN = 2


Point = namedtuple("Point", "x, y")

# Colores
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

# Parámetros
BLOCK_SIZE = 20
SPEED = 10
BOARD_SIZE = (20, 20)


class SnakeGame:

    def __init__(self, width=None, height=None, ai_control=False, training_mode=False, headless=False):
        if width is None or height is None:
            self.grid_width, self.grid_height = BOARD_SIZE
            self.w = self.grid_width * BLOCK_SIZE
            self.h = self.grid_height * BLOCK_SIZE
        else:
            self.grid_width, self.grid_height = width, height
            self.w = self.grid_width * BLOCK_SIZE
            self.h = self.grid_height * BLOCK_SIZE
            
        self.ai_control = ai_control
        self.training_mode = training_mode
        self.headless = headless
        
        if not headless:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption("Snake - AI Genetic")
        else:
            self.display = pygame.Surface((self.w, self.h))
            
        self.clock = pygame.time.Clock()
        self.reset()
        
    def reset(self):
        # Reset
        self.direction = Direction.RIGHT
        
        x_mid = (self.grid_width // 2) * BLOCK_SIZE
        y_mid = (self.grid_height // 2) * BLOCK_SIZE
        
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
        
        self.steps = 0
        self.steps_without_food = 0
        
        return self.get_state()

    def _place_food(self):
        # Comida
        max_x = (self.w // BLOCK_SIZE) - 1
        max_y = (self.h // BLOCK_SIZE) - 1
        
        x = random.randint(0, max_x) * BLOCK_SIZE
        y = random.randint(0, max_y) * BLOCK_SIZE
        self.food = Point(x, y)
        
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action=None):
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
            if not self.headless:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()
            
            if action is not None:
                self._change_direction(action)

        self._move(self.direction)
        self.snake.insert(0, self.head)
        
        self.steps += 1
        self.steps_without_food += 1

        reward = 0
        game_over = False
        
        if self._is_collision() or self.steps_without_food > 50 * len(self.snake):
            game_over = True
            reward = -10
            
            if self._is_collision():
                self._update_ui()
                if not self.headless:
                    pygame.display.flip()
                    time.sleep(0.2)
            
            return game_over, self.score, reward

        if self.head == self.food:
            self.score += 1
            self.steps_without_food = 0
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        self._update_ui()
        if not self.headless:
            self.clock.tick(SPEED)
        
        return game_over, self.score, reward
        
    def _change_direction(self, action):
        # Dirección
        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clockwise.index(self.direction)
        
        if action == Action.STRAIGHT.value:
            new_dir = clockwise[idx]
        elif action == Action.RIGHT_TURN.value:
            next_idx = (idx + 1) % 4
            new_dir = clockwise[next_idx]
        elif action == Action.LEFT_TURN.value:
            next_idx = (idx - 1) % 4
            new_dir = clockwise[next_idx]
            
        self.direction = new_dir

    def _is_collision(self, pt=None):
        # Colisión
        if pt is None:
            pt = self.head
            
        if (
            pt.x >= self.w
            or pt.x < 0
            or pt.y >= self.h
            or pt.y < 0
        ):
            return True
            
        for segment in self.snake[1:]:
            if pt.x == segment.x and pt.y == segment.y:
                return True

        return False
        
    def is_collision_at(self, x, y):
        # Colisión en (x,y)
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
        
        if not self.headless:
            pygame.display.flip()

    def _move(self, direction):
        # Movimiento
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
        # Sensores
        head = self.head
        
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        dir_r = self.direction == Direction.RIGHT
        dir_l = self.direction == Direction.LEFT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN
        
        food_dist_x = abs(self.food.x - head.x) / self.w
        food_dist_y = abs(self.food.y - head.y) / self.h
        
        move_right_good = self.food.x > head.x and dir_r
        move_left_good = self.food.x < head.x and dir_l
        move_up_good = self.food.y < head.y and dir_u
        move_down_good = self.food.y > head.y and dir_d
        moving_toward_food = move_right_good or move_left_good or move_up_good or move_down_good
        
        state = [
            # Peligro adelante
            (dir_r and self.is_collision_at(head.x + BLOCK_SIZE, head.y)) or
            (dir_l and self.is_collision_at(head.x - BLOCK_SIZE, head.y)) or
            (dir_u and self.is_collision_at(head.x, head.y - BLOCK_SIZE)) or
            (dir_d and self.is_collision_at(head.x, head.y + BLOCK_SIZE)),
            
            # Peligro derecha
            (dir_r and self.is_collision_at(head.x, head.y + BLOCK_SIZE)) or
            (dir_l and self.is_collision_at(head.x, head.y - BLOCK_SIZE)) or
            (dir_u and self.is_collision_at(head.x + BLOCK_SIZE, head.y)) or
            (dir_d and self.is_collision_at(head.x - BLOCK_SIZE, head.y)),
            
            # Peligro izquierda
            (dir_r and self.is_collision_at(head.x, head.y - BLOCK_SIZE)) or
            (dir_l and self.is_collision_at(head.x, head.y + BLOCK_SIZE)) or
            (dir_u and self.is_collision_at(head.x - BLOCK_SIZE, head.y)) or
            (dir_d and self.is_collision_at(head.x + BLOCK_SIZE, head.y)),
            
            # Dirección actual
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Ubicación comida
            self.food.x < head.x,
            self.food.x > head.x,
            self.food.y < head.y,
            self.food.y > head.y,
            
            # Distancia bordes
            head.x / self.w,
            (self.w - head.x) / self.w,
            head.y / self.h,
            (self.h - head.y) / self.h,
            
            # Info comida
            moving_toward_food,
            1.0 - food_dist_x,
            1.0 - food_dist_y,
        ]
        
        return np.array(state, dtype=float)

class DecisionTable:
    def __init__(self, weights=None):
        # Tabla decisión
        if weights is None:
            self.weights = np.random.randn(18, 3)
            
            # Pesos iniciales
            self.weights[0, :] = -5.0  # Evitar peligro adelante
            self.weights[1, 1] = -5.0  # Evitar peligro derecha
            self.weights[2, 2] = -5.0  # Evitar peligro izquierda
            
            self.weights[7:11, :] = np.random.uniform(0.5, 1.5, size=(4, 3))  # Dirección comida
            
            self.weights[15, 0] = 3.0  # Moverse hacia comida
            
            self.weights[16:18, :] = np.random.uniform(1.0, 2.0, size=(2, 3))
        else:
            self.weights = weights
        
        # Historial ciclos
        self.recent_actions = []
        self.max_history = 8
    
    def get_action(self, state):
        # Determinar acción
        q_values = np.dot(state, self.weights)
        
        has_cycle = self._check_for_cycles()
        
        danger_ahead = state[0]
        food_direction = state[7:11]
        moving_direction = state[3:7]
        
        # Evitar peligro
        if danger_ahead > 0.5:
            q_values[0] = -100
            
        # Dirección comida
        best_dir_to_food = -1
        if moving_direction[0] and food_direction[0]:
            best_dir_to_food = 0
        elif moving_direction[1] and food_direction[1]:
            best_dir_to_food = 0
        elif moving_direction[2] and food_direction[2]:
            best_dir_to_food = 0
        elif moving_direction[3] and food_direction[3]:
            best_dir_to_food = 0
        
        elif food_direction[1] and (moving_direction[2] or moving_direction[3]):
            best_dir_to_food = 1
        elif food_direction[0] and (moving_direction[2] or moving_direction[3]):
            best_dir_to_food = 2
        elif food_direction[2] and (moving_direction[0] or moving_direction[1]):
            best_dir_to_food = 1 if moving_direction[0] else 2
        elif food_direction[3] and (moving_direction[0] or moving_direction[1]):
            best_dir_to_food = 2 if moving_direction[0] else 1
        
        # Bonus comida
        if best_dir_to_food >= 0 and random.random() < 0.8:
            if (best_dir_to_food == 0 and not danger_ahead) or \
               (best_dir_to_food == 1 and not state[1]) or \
               (best_dir_to_food == 2 and not state[2]):
                q_values[best_dir_to_food] += 5.0
        
        # Romper ciclos
        if has_cycle:
            if random.random() < 0.6:
                q_values += np.random.normal(0, 2.0, size=q_values.shape)
                if random.random() < 0.3:
                    current_action = np.argmax(q_values)
                    q_values[current_action] = -10
        
        action = np.argmax(q_values)
        self._update_action_history(action)
        
        return action
    
    def _update_action_history(self, action):
        # Historial acciones
        self.recent_actions.append(action)
        if len(self.recent_actions) > self.max_history:
            self.recent_actions.pop(0)
    
    def _check_for_cycles(self):
        # Detectar ciclos
        if len(self.recent_actions) < 6:
            return False
        
        # Ciclo 2 acciones
        if len(self.recent_actions) >= 6:
            pattern = self.recent_actions[-2:]
            previous = self.recent_actions[-4:-2]
            older = self.recent_actions[-6:-4]
            if pattern == previous and pattern == older:
                return True
        
        # Ciclo 3 acciones
        if len(self.recent_actions) >= 6:
            pattern = self.recent_actions[-3:]
            previous = self.recent_actions[-6:-3]
            if pattern == previous:
                return True
                
        return False

    def crossover(self, other):
        # Cruce
        child_weights = np.copy(self.weights)
        
        rows, cols = self.weights.shape
        for i in range(rows):
            for j in range(cols):
                if random.random() < 0.5:
                    child_weights[i][j] = other.weights[i][j]
        
        return DecisionTable(child_weights)
    
    def mutate(self, mutation_rate):
        # Mutación
        rows, cols = self.weights.shape
        
        for i in range(rows):
            for j in range(cols):
                if random.random() < mutation_rate:
                    self.weights[i][j] += np.random.normal(0, 0.5)
                    
                    if random.random() < 0.1:
                        if random.random() < 0.5:
                            self.weights[i][j] *= -1
                        else:
                            self.weights[i][j] = np.random.normal(0, 1.5)
        
        return self

class GeneticAlgorithm:
    def __init__(self, population_size=30, num_generations=50, mutation_rate=0.1, crossover_rate=0.8, elite_size=2):
        self.crossover_type = "one_point"
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.population = []
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
        # Variables adaptativas
        self.stagnation_counter = 0
        self.improvement_rate_history = []
        self.best_fitness_ever = 0
        self.tournament_size = 5
        self.offspring_count = {}
        self.diversity_threshold = 0.1
        
    def initialize_population(self):
        # Inicializar población
        self.population = [DecisionTable() for _ in range(self.population_size)]
    
    def fitness(self, agent, num_games=7, show_game=False, silent=False):
        # Evaluar fitness
        total_score = 0
        total_steps = 0
        movement_efficiency = 0
        unique_positions = 0
        foods_reached = 0
        avg_steps_per_food = []
        
        foods_per_game = []
        consecutive_avoidance = 0
        max_snake_length = 3
        
        # Semillas 
        base_seed = 42
        seeds = [base_seed + i * 1000 for i in range(num_games)]
        
        if not silent:
            print(f"\nIniciando evaluación de agente en {num_games} juegos...")
        
        # Control visualización
        os_environ_copy = None
        if silent and 'SDL_VIDEODRIVER' not in os.environ:
            os_environ_copy = os.environ.copy()
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
            
        for game_num in range(num_games):
            random.seed(seeds[game_num])
            np.random.seed(seeds[game_num])
            
            original_print = print
            if game_num > 0 or silent:
                def silent_print(*args, **kwargs):
                    pass
                builtins.print = silent_print
            
            game = SnakeGame(ai_control=True, training_mode=False, headless=silent)
            done = False
            score = 0
            steps_since_last_food = 0
            
            max_steps = 400
            steps_played = 0
            prev_distance = None
            positions_set = set()
            direct_path_bonus = 0
            
            # Métricas adicionales
            attempts_toward_food = 0
            successful_food_approaches = 0
            wall_avoidance_count = 0
            near_death_avoidance = 0
            consecutive_approach_food = 0
            food_eaten_in_game = 0
            
            # Patrones de movimiento
            last_positions = []
            repeated_cycles = 0
            max_cycle_history = 20
            
            initial_food_distance = abs(game.head.x - game.food.x) + abs(game.head.y - game.food.y)
            
            print(f"Jugando partida {game_num + 1} de {num_games}...")
            
            while not done and steps_played < max_steps:
                state = game.get_state()
                action = agent.get_action(state)
                prev_score = score
                
                prev_pos = (game.head.x, game.head.y)
                prev_snake_length = len(game.snake)
                
                # Detección de peligros
                danger_ahead = state[0]
                danger_right = state[1]
                danger_left = state[2]
                
                # Evitar peligros
                if danger_ahead and action != 0:
                    wall_avoidance_count += 1
                    near_death_avoidance += 1
                    consecutive_avoidance += 1
                elif danger_right and action != 1:
                    wall_avoidance_count += 1
                    consecutive_avoidance += 1
                elif danger_left and action != 2:
                    wall_avoidance_count += 1
                    consecutive_avoidance += 1
                else:
                    consecutive_avoidance = 0
                
                done, score, _ = game.play_step(action)
                steps_played += 1
                steps_since_last_food += 1
                
                # Posiciones únicas
                pos = (game.head.x, game.head.y)
                positions_set.add(pos)
                
                # Detección ciclos
                last_positions.append(pos)
                if len(last_positions) > max_cycle_history:
                    last_positions.pop(0)
                
                # Patrones repetitivos
                if len(last_positions) >= 8:
                    for cycle_len in [2, 3, 4]:
                        if len(last_positions) >= cycle_len * 2:
                            recent = last_positions[-cycle_len:]
                            previous = last_positions[-2*cycle_len:-cycle_len]
                            if recent == previous:
                                repeated_cycles += 1
                                break
                
                # Distancia a comida
                curr_distance = abs(game.head.x - game.food.x) + abs(game.head.y - game.food.y)
                
                if prev_distance is None:
                    prev_distance = curr_distance
                
                # Acercamiento a comida
                if curr_distance < prev_distance:
                    movement_efficiency += 1
                    direct_path_bonus += 0.3
                    successful_food_approaches += 1
                    consecutive_approach_food += 1
                else:
                    direct_path_bonus -= 0.1
                    consecutive_approach_food = 0
                
                # Bonus aproximación
                if consecutive_approach_food >= 3:
                    direct_path_bonus += consecutive_approach_food * 0.5
                
                prev_distance = curr_distance
                
                # Recompensa por comida
                if score > prev_score:
                    foods_reached += 1
                    food_eaten_in_game += 1
                    avg_steps_per_food.append(steps_since_last_food)
                    steps_since_last_food = 0
                    
                    food_bonus_multiplier = 1 + (food_eaten_in_game * 0.5)
                    direct_path_bonus += 10.0 * food_bonus_multiplier
                    
                    prev_distance = None
                
                if show_game:
                    pygame.display.update()
            
            final_food_distance = curr_distance if 'curr_distance' in locals() else initial_food_distance
            
            if steps_played >= max_steps and not done:
                done = True
                
            total_score += score
            total_steps += game.steps
            unique_positions += len(positions_set)
            
            foods_per_game.append(food_eaten_in_game)
            
            max_snake_length = max(max_snake_length, 3 + food_eaten_in_game)
            
            if game_num > 0 or silent:
                builtins.print = original_print
                
            if not silent:
                print(f"\nJuego {game_num + 1} de {num_games} completado")
                print(f"  - Puntuación: {score}")
                print(f"  - Pasos totales: {steps_played}")
                print(f"  - Comida encontrada en este juego: {food_eaten_in_game}")
                print(f"  - Posiciones únicas visitadas: {len(positions_set)}")
                print(f"  - Eficiencia de ruta: {(food_eaten_in_game / steps_played * 100):.2f}% (comidas/pasos)")
        
        # Cálculo fitness
        # 1. Objetivos primarios
        base_food_value = 45
        
        # Valor incremental por comida
        incremental_food_points = 0
        for i in range(total_score):
            food_value = base_food_value * (1 + (i * 0.1))
            incremental_food_points += food_value
            
        food_points = incremental_food_points if total_score > 0 else 0
        
        # Supervivencia
        survival_points = total_steps * 0.5
        
        # 2. Eficiencia
        route_efficiency = 0
        food_points_per_step = 0
        if total_steps > 0:
            food_points_per_step = (total_score / max(1, total_steps)) * 500
            
            if avg_steps_per_food:
                route_efficiency = 100 / (sum(avg_steps_per_food) / len(avg_steps_per_food) + 1)
        
        # Aproximación comida
        food_approach_bonus = successful_food_approaches * 3
        
        # 3. Exploración
        exploration_bonus = unique_positions * 0.5
        
        # Consistencia entre juegos
        food_consistency_bonus = 0
        if len(foods_per_game) > 1:
            std_dev = np.std(foods_per_game)
            food_consistency_bonus = 20 * (1 / (1 + std_dev))
        
        # Evitar peligros
        basic_avoidance = wall_avoidance_count * 4.0
        critical_avoidance = near_death_avoidance * 10.0
        
        consecutive_avoidance_bonus = 0
        if consecutive_avoidance > 0:
            consecutive_avoidance_bonus = consecutive_avoidance * consecutive_avoidance * 0.5
        
        confined_space_bonus = 0
        if max_snake_length > 5 and unique_positions < total_steps * 0.5:
            confined_space_bonus = 30
        
        avoidance_bonus = basic_avoidance + critical_avoidance + consecutive_avoidance_bonus + confined_space_bonus
        
        # 4. Penalizaciones
        early_death_penalty = 0
        if total_score == 0:
            early_death_penalty = 200 * (1 - min(1.0, total_steps/100))
        elif total_steps < 50:
            early_death_penalty = 100 * (1 - min(1.0, total_steps/50))
        
        repetition_penalty = 0
        if repeated_cycles > 0:
            repetition_penalty = 5 * (repeated_cycles ** 1.5)
        
        # Cálculo final
        fitness = (
            food_points +
            survival_points +
            route_efficiency * 2.0 +
            food_points_per_step * 2.0 +
            food_approach_bonus +
            food_consistency_bonus +
            direct_path_bonus * 2.0 +
            exploration_bonus +
            avoidance_bonus +
            (movement_efficiency * 1.0) -
            early_death_penalty -
            repetition_penalty
        )
        
        consecutive_avoidance = 0
        foods_per_game = []
        
        if not silent:
            print("\n" + "="*60)
            print(f"RESUMEN DE EVALUACIÓN DEL AGENTE")
            print(f"  • Total de comida encontrada: {total_score}")
            print(f"  • Total de pasos realizados: {total_steps}")
            print(f"  • Eficiencia de movimiento: {movement_efficiency}")
            if avg_steps_per_food:
                print(f"  • Promedio de pasos por comida: {sum(avg_steps_per_food) / len(avg_steps_per_food):.2f}")
            print(f"  • Posiciones únicas visitadas: {unique_positions}")
            
            print(f"  • Puntos por comida: {food_points:.2f}")
            print(f"  • Puntos por supervivencia: {survival_points:.2f}")
            print(f"  • Bonificación por evitación de obstáculos: {avoidance_bonus:.2f}")
            print(f"  • Penalización por muerte temprana: {early_death_penalty:.2f}")
            print(f"  • Penalización por ciclos repetitivos: {repetition_penalty:.2f}")
            if food_consistency_bonus > 0:
                print(f"  • Bonificación por consistencia: {food_consistency_bonus:.2f}")
            
            print(f"  • Fitness calculado: {fitness:.2f}")
            print("="*60)
        
        if os_environ_copy is not None:
            os.environ.clear()
            os.environ.update(os_environ_copy)
            
        fitness = max(1.0, fitness / num_games)
        
        return fitness
    
    def selection(self, fitnesses):
        # Selección padres
        if self.stagnation_counter > 5:
            self.tournament_size = max(3, self.tournament_size - 1)
        elif len(self.improvement_rate_history) > 2 and sum(self.improvement_rate_history[-2:]) > 0.05:
            self.tournament_size = min(8, self.tournament_size + 1)
        
        selected_indices = []
        
        if random.random() < 0.1:
            selected_indices = random.sample(range(len(self.population)), 2)
        else:
            for _ in range(2):
                tournament = random.sample(range(len(self.population)), self.tournament_size)
                
                need_sharing = self.stagnation_counter > 3 and random.random() < 0.7
                
                if need_sharing:
                    adjusted_fitnesses = []
                    for idx in tournament:
                        offspring_penalty = self.offspring_count.get(idx, 0) * 0.1
                        
                        similarity_penalty = 0
                        if len(selected_indices) > 0:
                            first_parent = self.population[selected_indices[0]]
                            current = self.population[idx]
                            diff = np.sum((first_parent.weights - current.weights)**2)
                            similarity = 1.0 / (1.0 + diff)
                            similarity_penalty = similarity * 0.3 * fitnesses[idx]
                        
                        adjusted_fitness = fitnesses[idx] - offspring_penalty - similarity_penalty
                        adjusted_fitnesses.append(adjusted_fitness)
                    
                    best_idx = tournament[np.argmax(adjusted_fitnesses)]
                else:
                    best_idx = tournament[np.argmax([fitnesses[i] for i in tournament])]
                
                selected_indices.append(best_idx)
                
                self.offspring_count[best_idx] = self.offspring_count.get(best_idx, 0) + 1
        
        if selected_indices[0] == selected_indices[1]:
            candidates = [i for i in range(len(self.population)) if i != selected_indices[0]]
            if candidates:
                selected_indices[1] = random.choice(candidates)
        
        return selected_indices
    
    def crossover(self, parent1, parent2):
        # Cruce padres
        if random.random() < self.crossover_rate:
            crossover_type = getattr(self, 'crossover_type', 'one_point')
            
            if crossover_type == "uniform":
                child_weights = np.zeros_like(parent1.weights)
                rows, cols = parent1.weights.shape
                for i in range(rows):
                    for j in range(cols):
                        if random.random() < 0.5:
                            child_weights[i][j] = parent1.weights[i][j]
                        else:
                            child_weights[i][j] = parent2.weights[i][j]
                
                return DecisionTable(child_weights)
                
            elif crossover_type == "two_point":
                child_weights = np.copy(parent1.weights)
                rows, cols = parent1.weights.shape
                
                total_genes = rows * cols
                point1 = random.randint(0, total_genes - 2)
                point2 = random.randint(point1 + 1, total_genes - 1)
                
                for idx in range(total_genes):
                    i = idx // cols
                    j = idx % cols
                    if point1 < idx <= point2:
                        child_weights[i][j] = parent2.weights[i][j]
                
                return DecisionTable(child_weights)
                
            else:  # one_point
                child_weights = np.copy(parent1.weights)
                rows, cols = parent1.weights.shape
                
                crossover_point = random.randint(0, rows * cols - 1)
                row_idx = crossover_point // cols
                col_idx = crossover_point % cols
                
                for i in range(rows):
                    for j in range(cols):
                        if i > row_idx or (i == row_idx and j >= col_idx):
                            child_weights[i][j] = parent2.weights[i][j]
                
                return DecisionTable(child_weights)
        else:
            return DecisionTable(np.copy(parent1.weights))
    
    def mutate(self, agent):
        # Mutación adaptativa
        child_weights = np.copy(agent.weights)
        rows, cols = child_weights.shape
        
        if hasattr(self, 'evolve_progress') and self.num_generations > 0:
            evolution_progress = self.evolve_progress / self.num_generations
        else:
            evolution_progress = 0.5
            
        base_rate = self.mutation_rate * (1.0 - evolution_progress * 0.7)
        
        stagnation_factor = min(0.5, self.stagnation_counter * 0.1)
        
        recent_improvement = 0
        if len(self.improvement_rate_history) > 0:
            recent_improvement = sum(self.improvement_rate_history[-3:]) / min(3, len(self.improvement_rate_history))
        
        improvement_factor = max(-0.2, min(0.2, -recent_improvement))
        
        adjusted_rate = base_rate * (1.0 + stagnation_factor + improvement_factor)
        
        adjusted_rate = max(0.01, min(0.8, adjusted_rate))
        
        # Tasas por sección
        section_rates = {
            'danger': adjusted_rate * 0.5,
            'direction': adjusted_rate * 0.6,
            'food_location': adjusted_rate * 1.2,
            'borders': adjusted_rate * 1.0,
            'food_advanced': adjusted_rate * 1.3
        }
        
        for i in range(rows):
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
                    mutation_magnitude = 0.6 + stagnation_factor * 0.4
                    
                    child_weights[i][j] += np.random.normal(0, mutation_magnitude)
                    
                    drastic_prob = 0.1 + stagnation_factor * 0.2
                    if random.random() < drastic_prob:
                        if random.random() < 0.5:
                            child_weights[i][j] *= -1
                        else:
                            child_weights[i][j] = np.random.normal(0, 1.5)
        
        return DecisionTable(child_weights)
    
    def evolve(self, show_progress=True):
        # Evolución
        if not self.population:
            self.initialize_population()
        
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.stagnation_counter = 0
        self.improvement_rate_history = []
        self.best_fitness_ever = 0
        self.offspring_count = {}
        
        for generation in range(self.num_generations):
            self.evolve_progress = generation
            
            fitnesses = [self.fitness(agent) for agent in self.population]
            
            max_fitness_idx = np.argmax(fitnesses)
            best_fitness = fitnesses[max_fitness_idx]
            best_agent = self.population[max_fitness_idx]
            
            if len(self.best_fitness_history) > 0:
                prev_best = self.best_fitness_history[-1]
                improvement_rate = (best_fitness - prev_best) / (prev_best + 0.01)
                self.improvement_rate_history.append(improvement_rate)
                
                if best_fitness > self.best_fitness_ever * 1.01:
                    self.best_fitness_ever = best_fitness
                    self.stagnation_counter = 0
                else:
                    self.stagnation_counter += 1
            else:
                self.best_fitness_ever = best_fitness
                self.improvement_rate_history.append(0.0)
            
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(sum(fitnesses) / len(fitnesses))
            
            if show_progress and generation % 5 == 0:
                print(f"Generación {generation}: Mejor fitness = {best_fitness}, Promedio = {self.avg_fitness_history[-1]:.2f}")
                if self.stagnation_counter > 0:
                    print(f"  Estancamiento: {self.stagnation_counter} generaciones sin mejora significativa")
            
            if generation == self.num_generations - 1:
                break
                
            new_population = []
            
            self.offspring_count = {}
            
            sorted_indices = np.argsort(fitnesses)[::-1]
            
            effective_elite_size = self.elite_size
            if self.stagnation_counter > 5:
                effective_elite_size = min(self.population_size // 4, self.elite_size * 2)
            
            for i in range(effective_elite_size):
                elite_idx = sorted_indices[i]
                new_population.append(DecisionTable(np.copy(self.population[elite_idx].weights)))
            
            while len(new_population) < self.population_size:
                parent_indices = self.selection(fitnesses)
                parent1 = self.population[parent_indices[0]]
                parent2 = self.population[parent_indices[1]]
                
                child = self.crossover(parent1, parent2)
                
                child = self.mutate(child)
                
                new_population.append(child)
            
            self.population = new_population
        
        final_fitnesses = [self.fitness(agent) for agent in self.population]
        best_idx = np.argmax(final_fitnesses)
        
        return self.population[best_idx]
    
    def plot_fitness_history(self):
        # Gráfica fitness
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
    # Juego manual
    game = SnakeGame(ai_control=False, training_mode=False)
    
    while True:
        game_over, score, _ = game.play_step()
        
        if game_over:
            break
    
    print(f"Final Score: {score}")
    pygame.quit()

if __name__ == "__main__":
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
        try:
            import os
            if not os.path.exists("snake_experiments.py"):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                print(f"\nBuscando snake_experiments.py en: {script_dir}")
                if os.path.exists(os.path.join(script_dir, "snake_experiments.py")):
                    print(f"Archivo encontrado en el directorio del script")
                else:
                    raise ImportError(f"No se encuentra el archivo snake_experiments.py en {script_dir}")
            
            import importlib.util
            import sys
            
            script_dir = os.path.dirname(os.path.abspath(__file__))
            module_path = os.path.join(script_dir, "snake_experiments.py")
            
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
    
    pygame.quit()
