"""
Neural Atari Game Player - Supports both Breakout and Skiing

Controls:

BREAKOUT:
SPACE: Fire
LEFT ARROW: Move Left
RIGHT ARROW: Move Right
. (PERIOD): No Operation (NOOP) -> DEFAULT

SKIING:
LEFT ARROW: Left
RIGHT ARROW: Right
. (PERIOD): No Operation (NOOP) -> DEFAULT

ESC or Q: Quit

Optional Parameters:

--game: Choose between 'breakout' and 'skiing'
--temperature: Control the randomness of predictions (lower = more deterministic)
--vqvae_type: Choose between 'standard' and 'ema' codebook types
--model: Choose between 'action_to_latent' and 'action_state_to_latent'

Examples:
python play_neural_atari.py --game breakout --temperature 0.05 --vqvae_type ema
python play_neural_atari.py --game skiing --temperature 0.01 --vqvae_type standard --model action_state_to_latent

The game runs at a lower FPS (15) to make changes more visible. You can adjust this by changing the FPS variable if you want faster gameplay.
"""

import os
import sys
import torch
import numpy as np
import pygame
from PIL import Image
import time
import argparse
from collections import OrderedDict
from models.latent_action_model import load_latent_action_model, ActionStateToLatentMLP, ActionToLatentMLP
import cv2
from typing import Dict, Any, Tuple

import torch._dynamo
torch._dynamo.config.suppress_errors = True

# --- Configuration ---
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 840
FPS = 15  # Slower FPS for visible changes

class GameConfig:
    """Configuration class for different Atari games"""
    
    def __init__(self, game_name: str, world_model_path: str = None):
        self.game_name = game_name.lower()
        
        if self.game_name == 'breakout':
            self.action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
            self.key_to_action = {
                pygame.K_PERIOD: 0,  # NOOP
                pygame.K_SPACE: 1,   # FIRE
                pygame.K_RIGHT: 2,   # RIGHT
                pygame.K_LEFT: 3     # LEFT
            }
            self.continuous_keys = {
                pygame.K_SPACE: (1, "FIRE"),
                pygame.K_RIGHT: (2, "RIGHT"),
                pygame.K_LEFT: (3, "LEFT"),
                pygame.K_PERIOD: (0, "NOOP")
            }
            self.initial_frame_path = 'data/breakout/0.png'
            if world_model_path:
                self.world_model_path = world_model_path
                self.action_state_to_latent_path = os.path.join(os.path.dirname(world_model_path), 'action_state_to_latent_best.pt')
                self.action_to_latent_path = os.path.join(os.path.dirname(world_model_path), 'action_to_latent_best.pt')
            else:
                self.world_model_path = 'checkpoints/breakout/latent_action/best.pt'
                self.action_to_latent_path = 'checkpoints/breakout/latent_action/action_to_latent_best.pt'
                self.action_state_to_latent_path = 'checkpoints/breakout/latent_action/action_state_to_latent_best.pt'
            
        elif self.game_name == 'skiing':
            self.action_names = ['NOOP', 'RIGHT', 'LEFT']
            self.key_to_action = {
                pygame.K_PERIOD: 0,  # NOOP
                pygame.K_RIGHT: 1,   # RIGHT
                pygame.K_LEFT: 2     # LEFT
            }
            self.continuous_keys = {
                pygame.K_RIGHT: (1, "RIGHT"),
                pygame.K_LEFT: (2, "LEFT"),
                pygame.K_PERIOD: (0, "NOOP")
            }
            self.initial_frame_path = 'data/skiing/0.png'
            if world_model_path:
                self.world_model_path = world_model_path
                self.action_state_to_latent_path = os.path.join(os.path.dirname(world_model_path), 'action_state_to_latent_best.pt')
                self.action_to_latent_path = os.path.join(os.path.dirname(world_model_path), 'action_to_latent_best.pt')
            else:
                self.world_model_path = 'checkpoints/skiing/latent_action/best.pt'
                self.action_to_latent_path = 'checkpoints/skiing/latent_action/action_to_latent_best.pt'
                self.action_state_to_latent_path = 'checkpoints/skiing/latent_action/action_state_to_latent_best.pt'
            
        else:
            raise ValueError(f"Unsupported game: {game_name}. Supported games: 'breakout', 'skiing'")
    
    def get_control_instructions(self) -> str:
        if self.game_name == 'breakout':
            return """
BREAKOUT Controls:
-----------------
SPACE - Fire
LEFT ARROW - Move Left
RIGHT ARROW - Move Right
. (PERIOD) - No Operation
ESC or Q - Quit
"""
        elif self.game_name == 'skiing':
            return """
SKIING Controls:
---------------
LEFT ARROW - Move Left
RIGHT ARROW - Move Right
. (PERIOD) - No Operation
ESC or Q - Quit
"""

# --- Device selection ---
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def action_to_onehot(action_idx, num_actions, device):
    onehot = torch.zeros(1, num_actions, device=device)
    onehot[0, action_idx] = 1.0
    return onehot

def get_quantized_from_indices(indices, world_model, vqvae_type):
    """
    Get quantized embeddings from indices, handling both standard and EMA VQ-VAE types.
    """
    indices = indices.view(1, 5, 7)
    indices = indices.to(world_model.vq.codebook.device if vqvae_type == 'ema' else world_model.vq.embeddings.weight.device)
    
    if vqvae_type == 'ema':
        # For EMA VQ-VAE, codebook is a buffer
        quantized = world_model.vq.codebook[indices.flatten()].view(indices.shape + (-1,))
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
    else:
        # For standard VQ-VAE, embeddings is a nn.Embedding layer
        embeddings = world_model.vq.embeddings
        quantized = embeddings(indices)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
    
    return quantized

def load_action_model(game_config, model_type, device):
    """Load the appropriate action model for the game"""
    if model_type == 'action_state_to_latent':
        # Create model with appropriate action space size
        num_actions = len(game_config.action_names)
        action_model = ActionStateToLatentMLP(action_dim=num_actions).to(device)
        checkpoint_path = game_config.action_state_to_latent_path
    else:
        num_actions = len(game_config.action_names)
        action_model = ActionToLatentMLP(input_dim=num_actions).to(device)
        checkpoint_path = game_config.action_to_latent_path
    
    return action_model, checkpoint_path

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, choices=['breakout', 'skiing'], default='breakout',
                        help='Game to play: breakout or skiing')
    parser.add_argument('--temperature', type=float, default=0.01, help='Sampling temperature for latent prediction')
    parser.add_argument('--model', type=str, choices=['action_to_latent', 'action_state_to_latent'], default='action_to_latent',
                        help='Model type to use for action to latent mapping')
    parser.add_argument('--vqvae_type', type=str, choices=['standard', 'cvivit', 'ema'], default='standard',
                        help='Type of VQ-VAE codebook: standard (learnable embeddings) or ema (exponential moving average)')
    parser.add_argument('--world_model_path', type=str, default=None,
                        help='Path to the world model checkpoint (overrides default)')
    args = parser.parse_args()
    
    # Initialize game configuration
    game_config = GameConfig(args.game, world_model_path=args.world_model_path)
    
    # Override world model path if provided
    if args.world_model_path:
        game_config.world_model_path = args.world_model_path
    
    # Initialize pygame
    pygame.init()
    window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption(f"Neural {args.game.title()}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 18)
    
    # Set up device and models
    device = get_device()
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Game: {args.game.title()}")
    print(f"[INFO] VQ-VAE type: {args.vqvae_type}")
    
    # Load world model
    print("[INFO] Loading world model...")
    if args.vqvae_type == 'ema':
        model_type = "LatentActionVQVAE_EMA"
    elif args.vqvae_type == 'cvivit':
        model_type = "CViViTEncoderCrossAttention"
    else:
        model_type = "LatentActionVQVAE"
    
    try:
        world_model, _ = load_latent_action_model(game_config.world_model_path, model_type, device, codebook_size=256)
        world_model.to(device)
        world_model.eval()
        if device.type == 'cuda':
            world_model = torch.compile(world_model)
        print(f"[INFO] Successfully loaded world model from {game_config.world_model_path}")
    except FileNotFoundError:
        print(f"[ERROR] World model not found: {game_config.world_model_path}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to load world model: {e}")
        sys.exit(1)
    
    # Load action-to-latent model
    print("[INFO] Loading action-to-latent model...")
    action_model, checkpoint_path = load_action_model(game_config, args.model, device)
    
    # Load checkpoint with error handling
    try:
        ckpt = torch.load(checkpoint_path, map_location=device)
        original_state_dict = ckpt['model_state_dict']
        
        # Adjust state dict keys for compatibility (handle torch.compile artifacts)
        new_state_dict = OrderedDict()
        for key, value in original_state_dict.items():
            if key.startswith('_orig_mod'):
                new_key = key.replace('_orig_mod.', '')
            else:
                new_key = key
            new_state_dict[new_key] = value
        
        action_model.load_state_dict(new_state_dict)
        print(f"[INFO] Successfully loaded action model from {checkpoint_path}")
    except FileNotFoundError:
        print(f"[ERROR] Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to load action model: {e}")
        sys.exit(1)
    
    action_model.eval()
    if device.type == 'cuda':
        action_model = torch.compile(action_model)
    
    # Load initial frame
    print("[INFO] Loading initial frame...")
    try:
        init_img = Image.open(game_config.initial_frame_path).convert('RGB')
        init_frame_np = np.array(init_img, dtype=np.float32) / 255.0
        current_frame = torch.from_numpy(init_frame_np).permute(2, 0, 1).unsqueeze(0).to(device)
        print(f"[INFO] Initial frame shape: {current_frame.shape}")
    except FileNotFoundError:
        print(f"[ERROR] Initial frame '{game_config.initial_frame_path}' not found!")
        sys.exit(1)
    
    # Set up frame history
    last2_frames = [current_frame.clone(), current_frame.clone()]
    
    # Game state
    action_idx = 0  # Default to NOOP
    last_displayed_action = ""  # For display purposes
    score = 0
    step = 0
    num_actions = len(game_config.action_names)
    
    # Display the configuration and controls
    print(f"\nNeural {args.game.title()} Configuration:")
    print("=" * 40)
    print(f"Game: {args.game.title()}")
    print(f"VQ-VAE Type: {args.vqvae_type}")
    print(f"Action Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Device: {device}")
    print(f"Action Space Size: {num_actions}")
    print(game_config.get_control_instructions())
    
    # Main game loop
    running = True
    while running:
        # Default to NOOP (0) each frame
        action_idx = 0
        
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # Handle specific quit keys
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False
                # Set the action based on key press
                elif event.key in game_config.key_to_action:
                    action_idx = game_config.key_to_action[event.key]
        
        # Check currently pressed keys (for continuous input)
        keys = pygame.key.get_pressed()
        for key, (action, name) in game_config.continuous_keys.items():
            if keys[key]:
                action_idx = action
                last_displayed_action = name
                break
        
        # Only print non-NOOP actions to console
        if action_idx != 0:
            print(f"Action: {game_config.action_names[action_idx]}")
        
        # Generate next frame
        with torch.no_grad():
            try:
                # Stack last 2 frames
                stacked_frames = torch.cat([last2_frames[0], last2_frames[1]], dim=1)
                
                # Get action prediction
                onehot = action_to_onehot(action_idx, num_actions, device)
                if args.model == 'action_state_to_latent':
                    logits = action_model(onehot, stacked_frames)
                else:
                    logits = action_model(onehot)
                indices = action_model.sample_latents(logits, temperature=args.temperature)
                
                # Get quantized embeddings (handling both VQ-VAE types)
                quantized = get_quantized_from_indices(indices, world_model, args.vqvae_type)
                
                # Generate next frame
                frame_in = current_frame.permute(0, 1, 3, 2)
                quantized = quantized.to(device)
                frame_in = frame_in.to(device)
                next_frame = world_model.decoder(quantized, frame_in)
                next_frame = next_frame.permute(0, 1, 3, 2)
                
                # Update frame history
                last2_frames[0] = last2_frames[1]
                last2_frames[1] = next_frame.clone()
                current_frame = next_frame.clone()
                
            except Exception as e:
                print(f"[ERROR] Failed to generate next frame: {e}")
                # Keep the current frame if generation fails
                pass
        
        # Convert the frame to a pygame surface for display
        frame_np = current_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
        frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)
        
        # Create surface from numpy array
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame_np, (1, 0, 2)))
        
        # Scale the surface to fit the window
        scaled_surface = pygame.transform.scale(frame_surface, (WINDOW_WIDTH, WINDOW_HEIGHT - 80))
        
        # Render to the window
        window.fill((0, 0, 0))
        window.blit(scaled_surface, (0, 0))
        
        # Display game info - always show the last action
        step += 1
        info_text = f"Step: {step}"
        if last_displayed_action:
            info_text += f" | Last Action: {last_displayed_action}"
        info_text += f" | Temp: {args.temperature}"
        
        # Display game and model info
        game_info = f"Game: {args.game.upper()} | Actions: {num_actions}"
        model_info = f"VQ-VAE: {args.vqvae_type.upper()} | Model: {args.model}"
        
        text_surface = font.render(info_text, True, (255, 255, 255))
        game_surface = font.render(game_info, True, (200, 200, 200))
        model_surface = font.render(model_info, True, (200, 200, 200))
        
        window.blit(text_surface, (10, WINDOW_HEIGHT - 60))
        window.blit(game_surface, (10, WINDOW_HEIGHT - 40))
        window.blit(model_surface, (10, WINDOW_HEIGHT - 20))
        
        # Update the display
        pygame.display.flip()
        
        # Limit framerate
        clock.tick(FPS)
    
    # Clean up
    pygame.quit()
    print("Game closed.")

if __name__ == "__main__":
    main()