import gymnasium as gym
import numpy as np
import cv2
from typing import Tuple, Dict, Any

class AtariBreakoutEnv:
    """
    Wrapper for Atari Breakout environment with preprocessing.
    If return_rgb=True, returns original RGB frames (210, 160, 3) instead of preprocessed grayscale (84, 84).
    """
    
    def __init__(self, return_rgb: bool = False):
        """Initialize the environment with specific settings.
        Args:
            return_rgb: If True, return original RGB frames in reset/step.
        """
        # Create the base Atari environment with specific settings
        self.env = gym.make(
            "ALE/Breakout-v5",
            render_mode='rgb_array',  # Enable rgb_array rendering
            frameskip=4,  # Skip 4 frames between actions
            repeat_action_probability=0.0,  # Disable sticky actions
            full_action_space=False  # Use minimal action space
        )
        
        # Store action space size for the agent
        self.action_space = self.env.action_space
        print(f"Action space: {self.action_space}")
        print(f"Action meanings: {self.env.unwrapped.get_action_meanings()}")
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84), dtype=np.uint8
        )
        
        # Track lives for monitoring
        self.lives = 5  # Breakout starts with 5 lives
        self.was_real_done = True  # Track if the episode was actually done
        
        # Reward shaping configuration
        self.living_penalty = -0.005  # Small negative reward per time step
        self.life_loss_penalty = 0.0  # Substantial penalty for losing a life
        
        self.return_rgb = return_rgb
    
    def preprocess_observation(self, obs: np.ndarray) -> np.ndarray:
        """Convert RGB observation to 84x84 grayscale."""
        # Convert to grayscale
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        
        # Resize to 84x84
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        
        return resized
    
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and return initial observation.
        Returns original RGB if self.return_rgb else preprocessed grayscale.
        """
        obs, info = self.env.reset()
        self.lives = info.get('lives', 5)  # Get initial lives
        
        if self.return_rgb:
            return obs, info
        processed_obs = self.preprocess_observation(obs)
        return processed_obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.
        Returns original RGB if self.return_rgb else preprocessed grayscale.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Track lives to detect life loss
        lives = info.get('lives', 0)
        life_loss_reward = 0.0
        if lives < self.lives:
            # Lost a life - apply penalty
            life_loss_reward = self.life_loss_penalty
            self.lives = lives
        
        # Apply living penalty and life loss penalty
        shaped_reward = reward + self.living_penalty + life_loss_reward
        #shaped_reward = reward
        
        if self.return_rgb:
            return obs, shaped_reward, terminated, truncated, info
        processed_obs = self.preprocess_observation(obs)
        return processed_obs, shaped_reward, terminated, truncated, info
    
    def close(self):
        """Close the environment."""
        self.env.close() 