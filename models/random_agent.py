import numpy as np
from typing import List, Dict
import json
import os
import cv2

class RandomAgent:
    """Agent that selects random actions."""
    
    def __init__(self, n_actions: int):
        """Initialize the random agent.
        
        Args:
            n_actions: Number of possible actions
        """
        self.n_actions = n_actions
        self.transitions = []  # Store transitions for the current episode
    
    def select_action(self, temperature: float = None) -> int:
        """Select a random action uniformly, ignoring temperature."""
        return np.random.randint(0, self.n_actions)
    
    def record_transition(self, frame: np.ndarray, action: int, reward: float, 
                         episode: int, step: int) -> None:
        """Record a transition for data collection."""
        self.transitions.append({
            'frame': frame,
            'action': action,
            'reward': reward,
            'episode': episode,
            'step': step
        })
    
    def save_episode_data(self, episode_dir: str, actions_file: str) -> None:
        """Save episode data to disk."""
        # Save frames as PNG files
        for transition in self.transitions:
            frame_path = os.path.join(
                episode_dir, 
                f"frame_{transition['step']:05d}.png"
            )
            cv2.imwrite(frame_path, transition['frame'])
        
        # Save actions and rewards to JSON
        actions_data = {
            'episode': self.transitions[0]['episode'],
            'actions': [
                {
                    'step': t['step'],
                    'action': int(t['action']),
                    'reward': float(t['reward'])
                }
                for t in self.transitions
            ]
        }
        
        with open(actions_file, 'w') as f:
            json.dump(actions_data, f, indent=2)
        
        # Clear transitions after saving
        self.transitions = [] 