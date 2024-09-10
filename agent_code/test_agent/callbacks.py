import os
import pickle
import random
import torch
import numpy as np
from .data import create_input
from .train import EPS_END,EPS_DECAY,BATCH_SIZE,GAMMA,EPS_START
from .dqn import DQN
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


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
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.epsilon = EPS_START
        self.epsilon_decay = EPS_DECAY
        self.epsilon_min = EPS_END
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
    if self.train and random.random() < self.epsilon:
        self.logger.debug("Choosing action purely at random (exploration).")
        print("i am here going random")
        action = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        #print(action)
    else:
        print("i am here using model")
        self.logger.debug("Querying model for action (exploitation).")
        state_tensor = create_input(game_state).unsqueeze(0)  # Add batch dimension
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            q_values = self.model(state_tensor)
        action_idx = q_values.argmax().item()
        action = ACTIONS[action_idx]
        print(action)
    if self.epsilon > EPS_END:
        self.epsilon *= EPS_DECAY
        print(self.epsilon)
       
    return action
'''
def select_action(self, state):
    if random.random() < self.epsilon:
        return random.choice(ACTIONS)  # Choose a random action from the list
    else:
        with torch.no_grad():
            action_index = self.model(state).argmax().item()  # Get the index of the best action
            return ACTIONS[action_index]  # Return the corresponding action
'''

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