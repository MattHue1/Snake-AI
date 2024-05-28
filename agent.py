import torch
import random
import numpy as np
from collections import deque
from snake import SnakeAI, Direction, Point

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.002
BLOCK_SIZE = 20

class Agent:
    
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.85
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        

    def get_state(self, game):
        head = game.snake[0]
        lPoint = Point(head.x - BLOCK_SIZE, head.y)
        rPoint = Point(head.x + BLOCK_SIZE, head.y)
        uPoint = Point(head.x, head.y - BLOCK_SIZE)
        dPoint = Point(head.x, head.y + BLOCK_SIZE)

        ldir = game.direction == Direction.LEFT
        rdir = game.direction == Direction.RIGHT
        udir = game.direction == Direction.UP
        ddir = game.direction == Direction.DOWN

        # [danger straight, danger right, danger left, l, r, u, d, food left, food right, food up, food down]
        state = [
            # Danger straight
            (dir_r and game.is_collision(rPoint)) or
            (dir_l and game.is_collision(lPoint)) or
            (dir_u and game.is_collision(uPoint)) or
            (dir_d and game.is_collision(dPoint)),

            # Danger right
            (dir_u and game.is_collision(rPoint)) or
            (dir_d and game.is_collision(lPoint)) or
            (dir_l and game.is_collision(uPoint)) or
            (dir_r and game.is_collision(dPoint)),

            # Danger left
            (dir_d and game.is_collision(rPoint)) or
            (dir_u and game.is_collision(lPoint)) or
            (dir_r and game.is_collision(uPoint)) or
            (dir_l and game.is_collision(dPoint)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=int)
    
    def remember(self, state, direction, reward, next_state, over):
        self.memory.append((state, direction, reward, next_state, over)) #pop first when max memory reached
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, over = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, over)
    
    def train_short_memory(self, state, direction, reward, next_state, over):
        self.trainer.train_step(state, direction, reward, next_state, over)
    
    def get_action(self, state):
        # random
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move
    
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeAI()
    while True:
        # get old state
        old = agent.get_state(game)

        # get dir
        final_move = agent.get_action(old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        new_state = agent.get_state(game)

        # train short memory
        agent.train_short_memory(old, final_move, reward, new_state, done)

        agent.remember(old, final_move, reward, new_state, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
            
            print("Game", agent.n_games, "Score", score, "Record", record)

            # plot


if __name__ == '__main__':
    train()
