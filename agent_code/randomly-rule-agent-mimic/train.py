from collections import namedtuple, deque
import torch.optim as optim
import pickle
from typing import List
import torch
import torch.nn.functional as F
import random

import events
import events as e
from .dqn import DQN, n_actions
from .data import create_input
# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 5  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.7
EPS_END = 0.1
EPS_DECAY = 0.999
TAU = 0.01
LR = 1e-4



def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.model = DQN(n_actions)
    self.target_model = DQN(n_actions)
    self.target_model.load_state_dict(self.model.state_dict())
    self.optimizer = optim.Adam(self.model.parameters())
    self.replay_buffer = deque(maxlen=10000)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    # Idea: Add your own events to hand out rewards
    '''if ...:
        events.append(PLACEHOLDER_EVENT)
    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
    '''
    if old_game_state is None:
        return
    else:
        # Compute the reward based on the occurred events
        reward = reward_from_events(self, events)

        # Convert game states to the feature representation used by the model
        old_state_features = create_input(old_game_state)  
        new_state_features = create_input(new_game_state)

        # Create a transition and store it
        self.transitions.append(Transition(old_state_features, self_action, new_state_features, reward))

       
def train(self):
    if len(self.replay_buffer) < BATCH_SIZE:
        return

        # Sample a batch of transitions
    batch = random.sample(self.replay_buffer, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)
    """
    if len(self.replay_buffer) < self.batch_size:
        return

        # Sample a batch of transitions
    batch = random.sample(self.replay_buffer, self.batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    """
    # Convert to tensors
    states = torch.stack(states)
    actions = torch.tensor(actions).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.stack(next_states)
    dones = torch.tensor(dones, dtype=torch.float32)

    # Compute Q values
    q_values = self.model(states).gather(1, actions).squeeze(1)  # Q(s, a)

    # Compute target Q values
    with torch.no_grad():
        next_q_values = self.target_model(next_states).max(1)[0]  # max_a' Q(s', a')
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute the loss
    loss = F.mse_loss(q_values, target_q_values)

    # Perform backpropagation
    self.optimizer.zero_grad()
    loss.backward()

    # Update model parameters
    self.optimizer.step()

    # Epsilon decay
    if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(create_input(last_game_state), last_action, None, reward_from_events(self, events)))
    train(self)
    torch.save(self.model.state_dict(), "my-saved-model.pt")  # Save model parameters



def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: +100,
        e.CRATE_DESTROYED: +50,
        e.COIN_FOUND: +100,
        e.KILLED_OPPONENT: +100,
        e.KILLED_SELF: -1000,
        e.SURVIVED_ROUND: +10,
        e.INVALID_ACTION: -10,
        e.BOMB_DROPPED: +0,
        e.WAITED: -1
    }
    reward_sum = 0
    self.logger.info("------------------------")
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    self.logger.info("------------------------")
    return reward_sum


def store_experience(self, state, action, reward, next_state, done):
    self.replay_buffer.append((state, action, reward, next_state, done))


def update_target_model(self):
    # Perform soft update of target network
    for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
        target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
