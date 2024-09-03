import numpy as np
import torch
import settings as s
import torch.nn.functional as F
import matplotlib.pyplot as plt

def create_input(game_state: dict) -> torch.tensor:
    
    colors = {
        'danger': np.array([1, 0, 0], dtype=np.float32),       # bombs and explosions
        'm_agent': np.array([0, 1, 0], dtype=np.float32),     # own agents
        'o_agent': np.array([0, 0, 1], dtype=np.float32),      # other agents
        'stones': np.array([0.5, 0.5, 0.5], dtype=np.float32),     # stones
        'coins': np.array([1, 1, 0], dtype=np.float32),    # coins
        'crates': np.array([0.5, 0.5, 0], dtype=np.float32)   # crates
    }

    
    image = np.zeros((s.COLS, s.ROWS, 3), dtype=np.float32)

    #Mark all stones
    mask = game_state['field'] == -1
    image[mask] = colors['stones'] * 0.1

    # Mark all crates
    is_crate = game_state['field'] == 1
    image[is_crate] = colors['crates']

    # Mark all coins
    for coin in game_state['coins']:
        image[coin] = colors['coins']

    # Mark own agent
    agent = game_state['self'][3]
    image[agent] = colors['m_agent']

    # Mark other agents
    for opponent in game_state['others']:
        agent = opponent[3]
        image[agent] = colors['o_agent']

    # Mark all bombs and explosions
    for bomb, _ in game_state['bombs']:
        image[bomb] = colors['danger']
    active_bombs = game_state['explosion_map'] != 0
    image[active_bombs] = colors['danger']

    # Convert to torch tensor and permute dimensions
    image = torch.tensor(image).permute(2, 0, 1)  # (channels, x, y)

    # Convert to grayscale
    image = image.mean(dim=0, keepdim=True)

    # Convert the tensor to a NumPy array
    image_np = image.squeeze().cpu().numpy()

    # Plot and save the image using Matplotlib
    plt.imshow(image_np, cmap='gray')
    plt.axis('off')  # Turn off axis labels
    plt.savefig('mean_image.png', bbox_inches='tight', pad_inches=0)
    
    return image