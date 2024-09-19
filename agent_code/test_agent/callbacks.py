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
from ..rule_based_agent.callbacks import act as rule_act
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
    '''
    self.logger.debug("Choosing action purely at random (exploration).")
    print("i am here going random")
    action = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    print(action)
    '''
    '''
    num_actions = len(ACTIONS)
    weights = np.ones(num_actions) / num_actions  # Start with uniform distribution

    # Extract agent's position
    agent_pos = game_state['self'][3]

    # Initialize the explosion map
    explosion_map = game_state['explosion_map']
    
    # Define possible move directions: up, down, left, right
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    # Initialize matrix for logging
    logging_matrix = np.copy(explosion_map)
    logging_matrix[agent_pos[0], agent_pos[1]] = 0.5  # Mark the agent's current position

    # Iterate over each direction to check possible bomb placements
    for direction in directions:
        bomb_pos = (agent_pos[0] + direction[0], agent_pos[1] + direction[1])

        # Check if bomb position is within bounds
        if 0 <= bomb_pos[0] < game_state['field'].shape[0] and 0 <= bomb_pos[1] < game_state['field'].shape[1]:
            # Assume action index 5 is 'place bomb'
            bomb_action_index = 5

            # Check if the agent can move to a safe position immediately after placing the bomb
            safe_move_found = False
            for move_direction in directions:
                new_x, new_y = agent_pos[0] + move_direction[0], agent_pos[1] + move_direction[1]

                # Ensure the new position is within the bounds of the field
                if 0 <= new_x < game_state['field'].shape[0] and 0 <= new_y < game_state['field'].shape[1]:
                    # Check if the new position is safe from explosions
                    if explosion_map[new_x, new_y] == 0:
                        safe_move_found = True
                        move=direction_to_action.get(move_direction,'WAIT')
                        break
            
            if safe_move_found:
                self.logger.debug("Safe move found.")
                self.logger.debug(move)
                weights[bomb_action_index] = 0.5  # Favor placing a bomb if safe
            else:
                weights[bomb_action_index] = 0

    weights = np.clip(weights, 0, None)  
    total_weight = weights.sum()
    if total_weight > 0:
        weights /= total_weight  
    else:
        weights = np.ones(num_actions) / num_actions 
        '''
    #print(self.train)
    
    if self.train and random.random() < self.epsilon:
        
        self.logger.debug("Choosing action purely at random (exploration).")
        #print("i am here going random")
        action = np.random.choice(ACTIONS, p=[.21, .21, .21, .21, .05, .11])
        #action = np.random.choice(ACTIONS, p=weights)
        #print(action)
        #print(game_state['self'][3])
        '''
        # Adjust weights based on game state 
        num_actions = len(ACTIONS)
        weights = np.ones(num_actions) / num_actions
        agent_pos = game_state['self'][3]

        # Check nearby bombs
        for bomb, _ in game_state['bombs']:
            distance = abs(agent_pos[0] - bomb[0]) + abs(agent_pos[1] - bomb[1])
        
            # Adjust weights if bomb is within radius of 2
            if distance <= 4:
                weights[0:4] = 0
            
        explosion_map = game_state['explosion_map']
        if explosion_map[agent_pos[0], agent_pos[1]] > 0:
            weights[0:4] = 0
            
        weights /= weights.sum()  
        action = np.random.choice(ACTIONS, p=weights)
        '''
        #following the rule based agent list of valid actions
        
        print(action)
        
    else:
        #print("Using the model")
        self.logger.debug("Querying model for action (exploitation).")
        state_tensor = create_input(game_state).unsqueeze(0)  # Add batch dimension
        #print(state_tensor)
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            q_values = self.model(state_tensor)
        action_idx = q_values.argmax().item()
        action = ACTIONS[action_idx]
        #print(action)

    if not self.train:
        print(action)

    if self.train:   
        if self.epsilon > EPS_END:
            self.epsilon *= EPS_DECAY
        #print(self.epsilon)
       
    return action

'''
def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)
'''