

# bomberman_rl
Setup for a project/competition amongst students to train a winning Reinforcement Learning agent for the classic game Bomberman.

To run the agent use command
python3 main.py play --agents <agent_name> <agent_name> <agent_name> <agent_name> --n-rounds <no-of-rounds>

Please read the main.py for further instructions

# Notable Agents
## Agent Descriptions

### Only-random
This agent determines its actions purely based on randomness. Specifically, it decides its next move using the following line of code:
action = np.random.choice(ACTIONS, p=[.21, .21, .21, .21, .05, .11])

This means that each action is chosen with a probability of 21%, except for one action with a 5% probability and another with an 11% probability.

### Bomb-avoid
The agent employs a strategy to avoid danger from bombs. Before placing a bomb, it checks whether its next move will lead it into a section where the bomb's explosion could harm it. Only if the agent determines that it will be safe from the bomb's blast radius will it proceed to place the bomb.

### Rule-based-agent-mimic
This agent mimics the logic of a rule-based agent. It follows the same decision-making process as the rule-based agent when it is not relying on a model to decide its actions. Essentially, it uses predefined rules to determine its behavior in various situations.

### Randomly-rule-based-agent-mimic
This agent also mimics the rule-based agent, but with an added layer of randomness. It randomly decides whether to follow the rule-based logic or to choose an action at random. There is a 1 in 5 chance (20% probability) that it will select an action randomly instead of using the rule-based logic.



