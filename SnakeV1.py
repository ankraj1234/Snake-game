import os
import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple
import neat
import math
import time
import pickle
import matplotlib.pyplot as plt

pygame.init()
font = pygame.font.SysFont('arial', 25)

# Global counters for reporting
gen_counter = 0
overall_best_fitness = float('-inf')
overall_best_apples = 0
# Metrics lists
best_fitness_history = []
avg_fitness_history = []
best_apples_history = []

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED   = (200, 0,   0)
BLUE1 = (0,   0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,   0,   0)

BLOCK_SIZE = 20
SPEED = 60  # used only for demo when render=True

direction_order = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]

def turn(current, action):
    idx = direction_order.index(current)
    if action == 0:  # straight
        return current
    elif action == 1:  # right turn
        return direction_order[(idx + 1) % 4]
    elif action == 2:  # left turn
        return direction_order[(idx - 1) % 4]

class SnakeGame:
    def __init__(self, w=640, h=480, render=True):
        self.w = w
        self.h = h
        self.render = render
        if self.render:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake')
            self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - 2 * BLOCK_SIZE, self.head.y)
        ]
        self.score = 0
        self.hunger = 100
        self.frame_iteration = 0
        self.food = None
        self.place_food()

    def place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self.place_food()

    def play(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.move()
        self.snake.insert(0, self.head)
        self.frame_iteration += 1

        # Base: 100 frames + 200 per body segment 
        max_idle_frames = 100 + (len(self.snake) * 200)
        if self.frame_iteration > max_idle_frames:
            return True, self.score
        if self.is_collision():
            return True, self.score

        if self.head == self.food:
            self.score += 1
            self.hunger = 100
            self.frame_iteration = 0  
            self.place_food()
        else:
            self.snake.pop()
            self.hunger -= 1
            if self.hunger <= 0:
                return True, self.score

        if self.render:
            self.update_screen()
            self.clock.tick(SPEED)
        return False, self.score

    def update_screen(self):
        self.display.fill(BLACK)
        pygame.draw.rect(self.display, WHITE, pygame.Rect(self.head.x, self.head.y, BLOCK_SIZE, BLOCK_SIZE))
        for pt in self.snake[1:]:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        text = font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def move(self):
        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        x %= self.w
        y %= self.h
        self.head = Point(x, y)

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        return pt in self.snake[1:]

def get_relative_state(game):
    head = game.head
    point_l = Point(head.x - BLOCK_SIZE, head.y)
    point_r = Point(head.x + BLOCK_SIZE, head.y)
    point_u = Point(head.x, head.y - BLOCK_SIZE)
    point_d = Point(head.x, head.y + BLOCK_SIZE)

    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    # Danger straight, right, left
    danger_straight = int((dir_r and game.is_collision(point_r)) or
                          (dir_l and game.is_collision(point_l)) or
                          (dir_u and game.is_collision(point_u)) or
                          (dir_d and game.is_collision(point_d)))

    right_dir = direction_order[(direction_order.index(game.direction) + 1) % 4]
    left_dir = direction_order[(direction_order.index(game.direction) - 1) % 4]

    point_right = Point(head.x + BLOCK_SIZE if right_dir == Direction.RIGHT else head.x - BLOCK_SIZE if right_dir == Direction.LEFT else head.x,
                        head.y + BLOCK_SIZE if right_dir == Direction.DOWN else head.y - BLOCK_SIZE if right_dir == Direction.UP else head.y)

    point_left = Point(head.x + BLOCK_SIZE if left_dir == Direction.RIGHT else head.x - BLOCK_SIZE if left_dir == Direction.LEFT else head.x,
                       head.y + BLOCK_SIZE if left_dir == Direction.DOWN else head.y - BLOCK_SIZE if left_dir == Direction.UP else head.y)

    danger_right = int(game.is_collision(point_right))
    danger_left = int(game.is_collision(point_left))

    food_ahead = int((dir_r and game.food.x > head.x) or
                     (dir_l and game.food.x < head.x) or
                     (dir_u and game.food.y < head.y) or
                     (dir_d and game.food.y > head.y))

    food_right = int((right_dir == Direction.RIGHT and game.food.x > head.x) or
                     (right_dir == Direction.LEFT and game.food.x < head.x) or
                     (right_dir == Direction.UP and game.food.y < head.y) or
                     (right_dir == Direction.DOWN and game.food.y > head.y))

    food_left = int((left_dir == Direction.RIGHT and game.food.x > head.x) or
                    (left_dir == Direction.LEFT and game.food.x < head.x) or
                    (left_dir == Direction.UP and game.food.y < head.y) or
                    (left_dir == Direction.DOWN and game.food.y > head.y))

    return np.array([danger_straight, danger_right, danger_left,
                     food_ahead, food_right, food_left], dtype=int)

def eval_genomes(genomes, config):
    global gen_counter, overall_best_fitness, overall_best_apples
    gen_counter += 1
    best_fitness = float('-inf')
    sum_fitness = 0.0
    best_apples = 0
    best_genome = None

    for _, genome in genomes:
        game = SnakeGame(render=False)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0.0
        steps = 0
        previous_score = 0
        
        while True:
            prev_dist = math.dist([game.head.x, game.head.y], [game.food.x, game.food.y])

            state = get_relative_state(game)
            output = net.activate(state)
            action = np.argmax(output)
            game.direction = turn(game.direction, action)
            done, score = game.play()
            steps += 1
            
            curr_dist = math.dist([game.head.x, game.head.y], [game.food.x, game.food.y])

            delta = prev_dist - curr_dist
            genome.fitness += delta * 0.5

            if score > previous_score:
                genome.fitness += 10
                previous_score = score

            genome.fitness += 0.1
            
            if done:           
                final_apples = game.score
                genome.fitness -= 0
                break
        
        final_apples = game.score
        sum_fitness += genome.fitness
        
        if final_apples > best_apples:
            best_apples = final_apples
            best_genome = genome
        
        best_fitness = max(best_fitness, genome.fitness)

    avg_fitness = sum_fitness / len(genomes)
    
    # Update global tracking
    if best_fitness > overall_best_fitness:
        overall_best_fitness = best_fitness
    overall_best_apples = max(overall_best_apples, best_apples)
    
    # Store metrics
    best_fitness_history.append(best_fitness)
    avg_fitness_history.append(avg_fitness)
    best_apples_history.append(best_apples)

    print(f"****** Generation {gen_counter} ******")
    print(f"  Best fitness this gen:     {best_fitness:.3f}")
    print(f"  Avg fitness this gen:      {avg_fitness:.3f}")
    print(f"  Best apples eaten this gen:{best_apples}")
    print(f"  Overall best apples eaten: {overall_best_apples}")

    if best_genome:
        best_genome_fitness = best_genome.fitness if hasattr(best_genome, 'fitness') else 0
        
        genome_data = {
            'genome': best_genome,
            'score': best_apples, 
            'fitness': best_genome_fitness,
            'generation': gen_counter
        }
        
        should_save = False
        if os.path.exists("best_genome.pkl"):
            try:
                with open("best_genome.pkl", "rb") as f:
                    old_data = pickle.load(f)
                old_best_score = old_data.get('score', 0) if isinstance(old_data, dict) else 0
                if best_apples > old_best_score:
                    should_save = True
            except:
                should_save = True  # Save if we can't read the old file
        else:
            should_save = True  # Save if no file exists
        
        if should_save:
            try:
                with open("best_genome.pkl", "wb") as f:
                    pickle.dump(genome_data, f)
                print(f"NEW HIGH SCORE! Saved genome")
            except Exception as e:
                print(f"  Error saving genome: {e}")

def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)
    p = neat.Population(config)
    # p.add_reporter(neat.StdOutReporter(True))
    # p.add_reporter(neat.StatisticsReporter())
    
    try:
        winner = p.run(eval_genomes, 500)
        print(f"\nTraining completed! Winner fitness: {winner.fitness}")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    # Plot results
    if best_fitness_history:
        gens = list(range(1, len(best_fitness_history) + 1))
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(gens, best_fitness_history, label='Best Fitness', color='blue')
        plt.plot(gens, avg_fitness_history, label='Avg Fitness', color='orange')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness over Generations')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(gens, best_apples_history, label='Best Apples Eaten', color='green')
        plt.xlabel('Generation')
        plt.ylabel('Apples')
        plt.title('Best Apples Eaten over Generations')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.scatter(best_fitness_history, best_apples_history, alpha=0.6, color='red')
        plt.xlabel('Best Fitness')
        plt.ylabel('Best Apples')
        plt.title('Fitness vs Apples Correlation')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

def demo_from_pickle(pkl_path, config_path):

    if not os.path.exists(pkl_path):
        print(f"Error: Pickle file '{pkl_path}' not found!")
        return
    
    if not os.path.exists(config_path):
        print(f"Error: Config file '{config_path}' not found!")
        return
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            genome = data['genome']
            print(f"Loading genome from generation {data.get('generation', 'unknown')}")
            print(f"Best score in training: {data.get('score', data.get('apples', 'unknown'))}")
            print(f"Fitness: {data.get('fitness', 'unknown'):.3f}" if isinstance(data.get('fitness'), (int, float)) else f"Fitness: {data.get('fitness', 'unknown')}")
        else:
            genome = data
            print("Loading genome (legacy format)")
            
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return

    try:
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                    config_path)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
    except Exception as e:
        print(f"Error creating network: {e}")
        return
    
    print("Starting demo... Press ESC or close window to exit")
    game = SnakeGame(render=True)
    
    try:
        while True:
            state = get_relative_state(game)
            output = net.activate(state)
            action = np.argmax(output)
            game.direction = turn(game.direction, action)
            done, score = game.play()
            
            if done:
                print(f"Game Over! Final Score: {score}")
                print("Press any key to restart or close window to exit...")
                
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return
                        elif event.type == pygame.KEYDOWN:
                            waiting = False
                            break
                
                # Restart the game
                game.reset()
                
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    finally:
        pygame.quit()

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    
    print("Snake Game with NEAT AI")
    print("=" * 30)
    
    mode = input("Enter 'train' to train or 'demo' to run saved model: ").strip().lower()
    if mode == 'train':
        if not os.path.exists(config_path):
            print(f"Error: Config file '{config_path}' not found!")
        else:
            run(config_path)
    elif mode == 'demo':
        pkl_path = os.path.join(local_dir, 'best_genome.pkl')
        demo_from_pickle(pkl_path, config_path)
    else:
        print("Invalid option. Choose 'train' or 'demo'.")