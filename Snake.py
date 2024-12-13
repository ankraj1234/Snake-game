import os
import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple
import neat
import math
import time
import multiprocessing

pygame.init()
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')
l =[]
# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 20
MIN_TIME_TO_EAT_FOOD = 1000
 

class SnakeGame:
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):    
        # init game state
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [self.head, 
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self.place_food()
        self.frame_iteration = 0
        self.hunger = 100

    def place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self.place_food()

    def play(self):
        # collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # move
        self.move(self.direction)  # updates the head
        self.snake.insert(0, self.head)
        
        # check if game over
        self.game_over = False
        if self.is_collision():
            self.game_over = True
            return self.game_over, self.score
            
        # place new food or just move
        if self.head == self.food:
            self.score += 1
            self.place_food()
        else:
            self.snake.pop()
        
        # update ui and clock
        self.update_screen()
        self.clock.tick(SPEED)
        # return game over and score
        return self.game_over, self.score   

    def update_screen(self):
        self.display.fill(BLACK)

        # Draw the snake's head with a different color
        pygame.draw.rect(self.display, WHITE, pygame.Rect(self.head.x, self.head.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # Draw the rest of the snake's body
        for pt in self.snake[1:]:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        # Draw the food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # Draw the score
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def move(self, direction):
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

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Check if it hits the boundaries
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Check if it hits itself
        if pt in self.snake[1:]:
            return True
        return False

def eval_genomes(genomes, config):
    gen_best_score = 0
    for genome_id, genome in genomes:
        game = SnakeGame()
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0

        # Initialize distances and hunger lists
        dis_list1 = [float('inf')]
        dis_list2 = [float('inf')]
        dis_list3 = [game.w - game.head.x]
        dis_list4 = [game.h - game.head.y]
        hunger_list = [game.hunger]

        
        
        while True:
            # Get the current state of the game
            state = get_game_state(game)
            action = net.activate(state)
            #print(f"Action: {action}")  # Debugging statement

            # Check the length of the action array
            if len(action) < 3:
                print(f"Error: action array too short: {action}")
                break

            # Choose the direction based on the network's output
            direction = get_direction_from_action(action, game.direction)
            game.direction = direction
            
            game_over, score = game.play()

            if gen_best_score < score:
                gen_best_score = score

            # Evaluate the genome based on the game state
            if game_over:
                genome.fitness -= 10
                break

            # Increase the genome's fitness based on the snake's performance
            if game.head == game.food:
                genome.fitness += 100
                game.hunger = 100

            genome.fitness += 0.2

            # Calculate distances
            snakefoodDistEuclidean = math.sqrt((game.head.x - game.food.x) ** 2 + (game.head.y - game.food.y) ** 2)
            snakefoodDisManhattan = abs(game.head.x - game.food.x) + abs(game.head.y - game.food.y)
            snakeheadBottomDis = game.h - game.head.y
            snakeheadRightDis = game.w - game.head.x

            dis_list1.append(snakefoodDistEuclidean)
            dis_list2.append(snakefoodDisManhattan)
            dis_list3.append(snakeheadRightDis)
            dis_list4.append(snakeheadBottomDis)
            hunger_list.append(game.hunger)
            
            # Adjust fitness based on distances
            if dis_list1[-1] > dis_list1[-2]:
                genome.fitness -= 5

            if dis_list1[-1] < dis_list1[-2]:
                genome.fitness += 2.5

            if dis_list2[-1] > dis_list2[-2]:
                genome.fitness -= 5

            if dis_list2[-1] < dis_list2[-2]:
                genome.fitness += 2.5

            if hunger_list[-1] < hunger_list[-2]:
                genome.fitness -= 1.5

            # Check the output and move the snake
            if action[0] >= 0 and action[1] < 0 and action[2] < 0 and action[3] < 0:
                game.move(Direction.RIGHT)

            if action[1] >= 0 and action[0] < 0 and action[2] < 0 and action[3] < 0:
                game.move(Direction.LEFT)

            if action[2] >= 0 and action[1] < 0 and action[0] < 0 and action[3] < 0:
                game.move(Direction.DOWN)

            if action[3] >= 0 and action[1] < 0 and action[2] < 0 and action[0] < 0:
                game.move(Direction.UP)
                
            '''if action[0] == max(action):
                game.move(Direction.RIGHT)
            elif action[1] == max(action):
                game.move(Direction.LEFT)
            else:
                game.move(Direction.UP)'''

            # Add more fitness if axis aligns
            if game.head.x == game.food.x:
                genome.fitness += 10
            if game.head.y == game.food.y:
                genome.fitness += 10

            # Check hunger
            game.hunger -= 1
            if game.hunger == 0:
                genome.fitness -= 15
                break

            # Snake popping on other side of screen if screen limit reached
            if game.head.x >= game.w:
                game.head = Point(0, game.head.y)
            if game.head.x < 0:
                game.head = Point(game.w - BLOCK_SIZE, game.head.y)
            if game.head.y >= game.h:
                game.head = Point(game.head.x, 0)
            if game.head.y < 0:
                game.head = Point(game.head.x, game.h - BLOCK_SIZE)

            # Check if snake runs into itself
            if game.head in game.snake[1:]:
                genome.fitness -= 30
                break

            # Check if snake collides with food
            if game.head == game.food:
                genome.fitness += 100
                game.hunger = 100
                game.snake.insert(0, game.head)
                game.place_food()

    print(f"Best Score {gen_best_score}")
    l.append(gen_best_score)


def get_game_state(game):
    head = game.head
    point_l = Point(head.x - BLOCK_SIZE, head.y)
    point_r = Point(head.x + BLOCK_SIZE, head.y)
    point_u = Point(head.x, head.y - BLOCK_SIZE)
    point_d = Point(head.x, head.y + BLOCK_SIZE)
    
    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN
    
    state = [
        # Danger straight
        int((dir_r and game.is_collision(point_r)) or 
        (dir_l and game.is_collision(point_l)) or 
        (dir_u and game.is_collision(point_u)) or 
        (dir_d and game.is_collision(point_d))),
        
        # Danger right
        int((dir_u and game.is_collision(point_r)) or 
        (dir_d and game.is_collision(point_l)) or 
        (dir_l and game.is_collision(point_u)) or 
        (dir_r and game.is_collision(point_d))),
        
        # Danger left
        int((dir_d and game.is_collision(point_r)) or 
        (dir_u and game.is_collision(point_l)) or 
        (dir_r and game.is_collision(point_u)) or 
        (dir_l and game.is_collision(point_d))),
        
        # Move direction
        int(dir_l),
        int(dir_r),
        int(dir_u),
        int(dir_d),
        
        # Food location 
        int(game.food.x < game.head.x),  # food left
        int(game.food.x > game.head.x),  # food right
        int(game.food.y < game.head.y),  # food up
        int(game.food.y > game.head.y)   # food down
    ]

    return np.array(state)

def get_direction_from_action(action, current_direction):
    directions = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP] 
    max_index = np.argmax(action)
    return directions[max_index]

def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    #p.add_reporter(neat.StdOutReporter(True))
    #stats = neat.StatisticsReporter()
    #p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to 500 generations.
    winner = p.run(eval_genomes, 500)

    print(f'\nOverall Best Score: {max(l)}')
    #print('\nBest genome:\n{!s}'.format(winner))

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)