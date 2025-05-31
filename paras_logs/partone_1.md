# Implementation Plan for DQN Breakout Improvements

## 1. Visualize Current Agent Gameplay

Create a utility function to record gameplay videos from the current agent:

- Generate videos of 5 random-policy episodes and 5 episodes from the latest trained model
- Save these as MP4 files at 30fps 
- Include frame number, action taken, and cumulative reward as overlay text
- Output video resolution should be 160x210 (original Atari) for clarity

## 2. Upgrade Environment to Full RGB and Higher Resolution

Modify environment wrapper to:
- Use RGB instead of grayscale (3 channels)
- Use native Atari resolution (160x210) or higher resolution like 160x160
- Reduce frame skip from 4 to 2 frames for better ball tracking
- Keep 4-frame stacking for temporal information

Corresponding network architecture changes:
- Modify input channels from 4 to 12 (3 RGB channels × 4 frames)
- Increase network capacity to handle RGB data:
  - First layer: Conv2d(12, 48, kernel_size=8, stride=4)
  - Second layer: Conv2d(48, 96, kernel_size=4, stride=2)
  - Third layer: Conv2d(96, 96, kernel_size=3, stride=1)
- Increase FC layer sizes proportionally

## 3. Implement Prioritized Experience Replay

Add prioritized replay buffer that:
- Stores transitions with priority values based on TD errors
- Uses sum-tree data structure for efficient sampling
- Includes hyperparameters:
  - α = 0.6 (prioritization exponent)
  - β = 0.4 → 1.0 (importance sampling exponent, annealed)
- Implements importance sampling weights for unbiased updates
- Updates sample priorities after each learning step

## 4. Implement Reward Shaping

Enhance reward function as follows (as implemented in `AtariBreakoutEnv`):
- Original reward: +1 for breaking a brick (unchanged)
- Living penalty: -0.001 per time step (applied every step)
- Life loss penalty: -1.0 for losing a life (applied when number of lives decreases)
- Implementation details:
  - Track lives between steps to detect life loss (compare `info['lives']` to previous value)
  - In the `step` method, shaped reward is computed as: `shaped_reward = reward + living_penalty + life_loss_penalty (if life lost)`
  - The shaped reward is returned by the environment for each step
  - Make the shaping configurable (easy to turn on/off) if needed for ablation
  - Keep track of original vs shaped rewards for evaluation if required

## Training Protocol

1. First implement environment visualization to verify current behavior
2. Implement and test RGB + higher resolution separately 
3. Add prioritized replay and verify implementation
4. Finally add reward shaping
5. Conduct ablation studies with different combinations:
   - Baseline (original)
   - RGB + higher resolution only
   - Prioritized replay only
   - Reward shaping only
   - All improvements combined

Evaluate each condition with 100 evaluation episodes and report average scores, max scores, and learning curves.