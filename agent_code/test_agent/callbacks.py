import os
import pickle
import random
from matplotlib import pyplot as plt
import torch
import numpy as np
from .data import create_input
from .train import EPS_END,EPS_DECAY,BATCH_SIZE,GAMMA,EPS_START
from .dqn import DQN
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
direction_to_action = {
    (0, 1): 'RIGHT',
    (0, -1): 'LEFT',
    (1, 0): 'DOWN',
    (-1, 0): 'UP'
}
#setting up a rulebased agent's setup clone to use instead of random
from random import shuffle
from collections import deque
def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.
        Same as the rule_based_agent's rules
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()
    
    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.bomb_dropped = False
    self.move = 'WAIT'
    self.logger.debug('Successfully entered setup code')
    np.random.seed()
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    self.current_round = 0
    self.batch_size = BATCH_SIZE  # manually delete model to restart training
    self.gamma = GAMMA
    self.epsilon = EPS_START
    self.epsilon_decay = EPS_DECAY
    self.epsilon_min = EPS_END
    if self.train and not os.path.isfile("my-saved-model.pt"): # and not to continue training,
        self.logger.info("Setting up model from scratch.")
        self.model = DQN(n_actions=6)  
    else:
        self.logger.info("Loading model from saved state.")
        self.model = DQN(n_actions=6)  
        self.model.load_state_dict(torch.load("my-saved-model.pt"))

def reset_self(self):
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0



def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    #random_prob = .1
    
    #print(self.train)
    
    if self.train and random.random() < self.epsilon:
        
        #self.logger.debug("Choosing action purely at random (exploration).")
        #print("i am here going random")
        #action = np.random.choice(ACTIONS, p=[.21, .21, .21, .21, .05, .11])
        #action = np.random.choice(ACTIONS, p=weights)
        #print(action)
        #print(game_state['self'][3])
        #following the rule based agent list of valid actions
        
        #print(action)
        #create_input(game_state)
        self.logger.info('------Picking action according to rule set--------------')
        # Check if we are in a different round
        if game_state["round"] != self.current_round:
            reset_self(self)
            self.current_round = game_state["round"]
        # Gather information about the game state
        arena = game_state['field'] 
        _, score, bombs_left, (x, y) = game_state['self']
        bombs = game_state['bombs']
        
        bomb_xys = [xy for (xy, t) in bombs]
        others = [xy for (n, s, b, xy) in game_state['others']]
        coins = game_state['coins']
        bomb_map = np.ones(arena.shape) * 5
        for (xb, yb), t in bombs:
            for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
                if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                    bomb_map[i, j] = min(bomb_map[i, j], t)

        # If agent has been in the same location three times recently, it's a loop
        if self.coordinate_history.count((x, y)) > 2:
            self.ignore_others_timer = 5
        else:
            self.ignore_others_timer -= 1
        self.coordinate_history.append((x, y))

        # Check which moves make sense at all
        directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        valid_tiles, valid_actions = [], []
        for d in directions:
            if ((arena[d] == 0) and
                    (game_state['explosion_map'][d] < 1) and
                    (bomb_map[d] > 0) and
                    (not d in others) and
                    (not d in bomb_xys)):
                valid_tiles.append(d)
        if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
        if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
        if (x, y - 1) in valid_tiles: valid_actions.append('UP')
        if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
        if (x, y) in valid_tiles: valid_actions.append('WAIT')
        # Disallow the BOMB action if agent dropped a bomb in the same spot recently
        if (bombs_left > 0) and (x, y) not in self.bomb_history: valid_actions.append('BOMB')
        self.logger.debug(f'Valid actions: {valid_actions}')

        # Collect basic action proposals in a queue
        # Later on, the last added action that is also valid will be chosen
        action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        shuffle(action_ideas)

        # Compile a list of 'targets' the agent should head towards
        cols = range(1, arena.shape[0] - 1)
        rows = range(1, arena.shape[0] - 1)
        dead_ends = [(x, y) for x in cols for y in rows if (arena[x, y] == 0)
                    and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
        crates = [(x, y) for x in cols for y in rows if (arena[x, y] == 1)]
        targets = coins + dead_ends + crates
        # Add other agents as targets if in hunting mode or no crates/coins left
        if self.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
            targets.extend(others)

        # Exclude targets that are currently occupied by a bomb
        targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

        # Take a step towards the most immediately interesting target
        free_space = arena == 0
        if self.ignore_others_timer > 0:
            for o in others:
                free_space[o] = False
        d = look_for_targets(free_space, (x, y), targets, self.logger)
        if d == (x, y - 1): action_ideas.append('UP')
        if d == (x, y + 1): action_ideas.append('DOWN')
        if d == (x - 1, y): action_ideas.append('LEFT')
        if d == (x + 1, y): action_ideas.append('RIGHT')
        if d is None:
            self.logger.debug('All targets gone, nothing to do anymore')
            action_ideas.append('WAIT')

        # Add proposal to drop a bomb if at dead end
        if (x, y) in dead_ends:
            action_ideas.append('BOMB')
        # Add proposal to drop a bomb if touching an opponent
        if len(others) > 0:
            if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
                action_ideas.append('BOMB')
        # Add proposal to drop a bomb if arrived at target and touching crate
        if d == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
            action_ideas.append('BOMB')

        # Add proposal to run away from any nearby bomb about to blow
        for (xb, yb), t in bombs:
            if (xb == x) and (abs(yb - y) < 4):
                # Run away
                if (yb > y): action_ideas.append('UP')
                if (yb < y): action_ideas.append('DOWN')
                # If possible, turn a corner
                action_ideas.append('LEFT')
                action_ideas.append('RIGHT')
            if (yb == y) and (abs(xb - x) < 4):
                # Run away
                if (xb > x): action_ideas.append('LEFT')
                if (xb < x): action_ideas.append('RIGHT')
                # If possible, turn a corner
                action_ideas.append('UP')
                action_ideas.append('DOWN')
        # Try random direction if directly on top of a bomb
        for (xb, yb), t in bombs:
            if xb == x and yb == y:
                action_ideas.extend(action_ideas[:4])

        # Pick last action added to the proposals list that is also valid
        while len(action_ideas) > 0:
            a = action_ideas.pop()
            if a in valid_actions:
                # Keep track of chosen action for cycle detection
                if a == 'BOMB':
                    self.bomb_history.append((x, y))

                return a
        if valid_actions:
            return random.choice(valid_actions)
        else:
            return 'WAIT'
    else:
        #print("Using the model")
        self.logger.debug("---------------Querying model for action (exploitation).-------------")
        state_tensor = create_input(game_state).unsqueeze(0)  # Add batch dimension
        #print(state_tensor)
        
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            q_values = self.model(state_tensor)
            self.logger.debug(q_values)
        action_idx = q_values.argmax().item()
        action = ACTIONS[action_idx]
        
        #print(action)
    self.logger.debug("-----end of turn------")

    if not self.train:
        self.logger.debug(action)

    if self.train:   
        if self.epsilon > EPS_END:
            self.epsilon *= EPS_DECAY
        self.logger.debug(self.epsilon )
    return action

    