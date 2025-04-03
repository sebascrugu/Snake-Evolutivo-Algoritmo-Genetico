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
        
        # Estado del juego (11 sensores en total)
        state = [
            # Peligro adelante
            (dir_r and self.is_collision_at(head.x + BLOCK_SIZE, head.y)) or
            (dir_l and self.is_collision_at(head.x - BLOCK_SIZE, head.y)) or
            (dir_u and self.is_collision_at(head.x, head.y - BLOCK_SIZE)) or
            (dir_d and self.is_collision_at(head.x, head.y + BLOCK_SIZE)),
            
            # Peligro a la derecha
            (dir_r and self.is_collision_at(head.x, head.y + BLOCK_SIZE)) or
            (dir_l and self.is_collision_at(head.x, head.y - BLOCK_SIZE)) or
            (dir_u and self.is_collision_at(head.x + BLOCK_SIZE, head.y)) or
            (dir_d and self.is_collision_at(head.x - BLOCK_SIZE, head.y)),
            
            # Peligro a la izquierda
            (dir_r and self.is_collision_at(head.x, head.y - BLOCK_SIZE)) or
            (dir_l and self.is_collision_at(head.x, head.y + BLOCK_SIZE)) or
            (dir_u and self.is_collision_at(head.x - BLOCK_SIZE, head.y)) or
            (dir_d and self.is_collision_at(head.x + BLOCK_SIZE, head.y)),
            
            # Dirección actual
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Ubicación de la comida
            self.food.x < head.x,  # comida a la izquierda
            self.food.x > head.x,  # comida a la derecha
            self.food.y < head.y,  # comida arriba
            self.food.y > head.y   # comida abajo
        ]
        
        return np.array(state, dtype=int)


# Tabla de Decisión para el agente (cromosoma)
class DecisionTable:
    def __init__(self, weights=None):
        # Tabla de decisión con pesos para cada sensor hacia cada acción
        # 11 sensores x 3 acciones posibles
        if weights is None:
            # Inicialización aleatoria si no se proporcionan pesos
            self.weights = np.random.randn(11, 3)
        else:
            self.weights = weights
    
    def get_action(self, state):
        """Determina la acción basada en el estado actual"""
        # Producto punto de estado y pesos para cada acción
        output = np.dot(state, self.weights)
        # Devuelve la acción con mayor valor
        return np.argmax(output)

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
    
    def fitness(self, agent, num_games=3, show_game=False):
        """Evalúa el fitness de un agente sobre múltiples juegos"""
        total_score = 0
        total_steps = 0
        
        for _ in range(num_games):
            game = SnakeGame(ai_control=True)
            done = False
            score = 0
            
            # Máximo de pasos para evitar juegos infinitos durante evaluación
            max_steps = 2000  # Límite razonable para evaluación
            steps_played = 0
            
            # Jugar hasta que termine
            while not done and steps_played < max_steps:
                state = game.get_state()
                action = agent.get_action(state)
                done, score, _ = game.play_step(action)
                steps_played += 1
                
                # No mostrar todos los juegos (solo para visualizar el mejor)
                if not show_game:
                    pygame.display.update()
            
            # Si se alcanzó el máximo de pasos sin terminar el juego
            if steps_played >= max_steps and not done:
                done = True
                
            total_score += score
            total_steps += game.steps
        
        # Calcular fitness: (comida × 10) + pasos − (muertetemprana × 50)
        # Consideramos muerte temprana si no come nada
        muerte_temprana = 1 if total_score == 0 else 0
        # Ajuste de la fórmula de fitness para priorizar más la comida
        fitness = (total_score * 15) + total_steps - (muerte_temprana * 50)
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
        """Aplica mutación al agente"""
        child_weights = np.copy(agent.weights)
        rows, cols = child_weights.shape
        
        for i in range(rows):
            for j in range(cols):
                if random.random() < self.mutation_rate:
                    # Añadir ruido gaussiano
                    child_weights[i][j] += np.random.normal(0, 0.5)
        
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
        """Demuestra el mejor agente en una partida"""
        # Crear un juego con velocidad reducida para la demostración
        game = SnakeGame(ai_control=True)
        done = False
        score = 0
        steps = 0
        max_steps = 1000  # Límite de pasos para la demostración
        
        # Usar una velocidad de actualización más lenta para mejor visualización
        pause_time = 0.1  # Pausa más larga para mejor visualización
        
        while not done and steps < max_steps:
            # Obtener estado y acción
            state = game.get_state()
            action = agent.get_action(state)
            
            # Ejecutar acción
            done, score, _ = game.play_step(action)
            
            # Pausa para visualización
            time.sleep(pause_time)
            steps += 1
            
            # Mostrar información en tiempo real
            if steps % 10 == 0:
                print(f"Paso: {steps}, Puntuación: {score}")
        
        if steps >= max_steps:
            print(f"Demostración finalizada por límite de pasos. Puntuación: {score}")
        else:
            print(f"Demostración finalizada. Puntuación final: {score}")
            
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
    # Parámetros del algoritmo genético
    population_size = 50  # Mantener 50 agentes
    num_generations = 30  # Reducir a 30 generaciones para pruebas más rápidas
    mutation_rate = 0.15  # Reducir ligeramente para más estabilidad
    crossover_rate = 0.8  # Aumentar para más diversidad
    elite_size = 3        # Mantener los 3 mejores
    
    print(f"Iniciando entrenamiento con {population_size} agentes durante {num_generations} generaciones...")
    print("(Esto puede tomar varios minutos, por favor espera)")
    
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
    
    return best_agent, best_score

if __name__ == "__main__":
    # Modo de juego (Human o AI)
    mode = input("Selecciona modo de juego (1: Humano, 2: IA): ")
    
    if mode == "1":
        play_human()
    else:
        train_and_play_ai()
    
    pygame.quit()
