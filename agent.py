# Inspired / refactored from https://github.com/patrickloeber/snake-ai-pytorch/tree/main

import random
import numpy as np
from collections import deque
from klask_simulator import KlaskSimulator
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(6, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        # TODO refactor how we define state 
        state = [
            game.bodies["puck1"].position[0],
            game.bodies["puck1"].position[1],
            game.bodies["puck2"].position[0],
            game.bodies["puck2"].position[1],
            game.bodies["ball"].position[0],
            game.bodies["ball"].position[1]
        ]

        return np.array(state, dtype=int)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Extract positions from the state
        puck1_x, puck1_y = state[0], state[1]
        ball_x, ball_y = state[4], state[5]

        # Define a simple strategy to always push towards the ball
        # Calculate direction vector from puck to ball
        direction_x = ball_x - puck1_x
        direction_y = ball_y - puck1_y

        # Normalize the direction vector to apply a consistent force regardless of distance
        norm = np.sqrt(direction_x**2 + direction_y**2)
        if norm == 0:
            force_x, force_y = 0, 0
        else:
            force_x, force_y = direction_x / norm, direction_y / norm

        # Apply some force magnitude
        force_magnitude = 1.0  # This is arbitrary and should be chosen according to the game's requirements
        action = {'force': (force_x * force_magnitude, force_y * force_magnitude)}

        return action

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    agent = Agent()
    game = KlaskSimulator()
    game.reset()

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)
        final_move_list = list(final_move['force'])

        # perform move and get new state
        reward, done, score = game.step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move_list, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move_list, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()
