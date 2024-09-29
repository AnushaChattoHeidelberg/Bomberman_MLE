import numpy as np
import torch
import settings as s
import matplotlib.pyplot as plt
def create_input(game_state: dict) -> torch.tensor:
    # Initialize the image
    image = np.zeros((s.COLS, s.ROWS), dtype=np.float32)

    try:
        # Stones
        image[game_state['field'] == -1] += 0.1  # Assign value 0.1 for stones
        
        # Crates
        image[game_state['field'] == 1] += 0.25  # Assign value 0.25 for crates
        
        # Coins
        for coin in game_state['coins']:
            image[coin] += 0.3  # Assign value 0.3 for coins
            
        # Own agent
        agent_pos = game_state['self'][3]
        image[agent_pos] += 0.2  # Assign value 0.2 for own agent
        
        # Other agents
        for opponent in game_state['others']:
            opponent_pos = opponent[3]
            image[opponent_pos] += 0.4  # Assign value 0.3 for other agents
        
        # Bombs
        for bomb, _ in game_state['bombs']:
            image[bomb] += 0.5  # Assign value 0.5 for bombs
            
        # Active explosions
        image[game_state['explosion_map'] != 0] += 0.5  # Assign value 0.5 for active explosions

    except KeyError as e:
        print(f"Key error: {e}. Please check the game_state dictionary.")

    # Normalize to the range [0, 1]
    max_value = image.max()
    if max_value > 0:
        image /= max_value  # Normalize to [0, 1]

    # Convert to torch tensor and add a channel dimension
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # (1, x, y)

    #visualize image
    '''
    plt.imshow(image, cmap='hot', interpolation='nearest') 
    plt.colorbar()  
    plt.title("Game State Visualization")
    plt.axis('off')  
    plt.savefig("game_state.png", bbox_inches='tight', pad_inches=0)  # Save without borders
    plt.close()
    '''
    return image_tensor

