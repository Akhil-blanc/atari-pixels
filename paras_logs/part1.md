# Part 1: DQN Training with Exploration Modes

## Overview

This project supports two exploration strategies for DQN training in Atari Breakout:

### 1. Temperature-based Exploration (Default)
- **Action selection:** Softmax over Q-values with a temperature parameter (Boltzmann exploration).
- **Temperature annealing:**
  - Initial temperature: 1.0
  - Minimum temperature: 0.05
  - Exponential decay per episode: 0.995 (slowed by a factor of 3 for smoother annealing)
- **Prioritized Experience Replay (PER):**
  - Enabled by default in this mode.
  - Uses a sum-tree for efficient sampling and priority updates.
  - Alpha (priority exponent) and beta (importance sampling correction) are annealed during training.
  - Importance sampling weights are applied to the loss for bias correction.
- **Reward:** Only extrinsic (environment) reward is used. No intrinsic or combined reward is tracked in this mode.
- **Usage:**
  ```bash
  python train_dqn.py
  ```
- **Logging:** Progress bar shows extrinsic reward as `reward`, the current temperature as `temp`, and loss. PER diagnostics are periodically printed.

### 2. RND-based Exploration (Optional)
- **Action selection:** Dual-agent setup with Random Network Distillation (RND) for intrinsic motivation.
- **Combined reward:** Weighted sum of extrinsic and intrinsic rewards, controlled by `alpha`.
- **Usage:**
  ```bash
  python train_dqn.py --exploration_mode rnd
  ```
- **Logging:** Progress bar shows both combined and extrinsic rewards, as well as `alpha`.

## Command-Line Arguments
- `--exploration_mode`: Choose between `temperature` (default) and `rnd`.
- Other arguments for controlling episodes, buffer size, and saving remain unchanged.

## Refactoring Notes
- The training loop, buffer prefill, evaluation, and checkpointing are all mode-aware.
- The code is modular and clear, with minimal risk of introducing bugs.
- Only the relevant rewards are tracked and logged for each mode.

## Example Usage
- **Default (temperature):**
  ```bash
  python train_dqn.py
  ```
- **RND mode:**
  ```bash
  python train_dqn.py --exploration_mode rnd
  ```

---

This update improves clarity, maintainability, and flexibility for experimentation with different exploration strategies in DQN training.

## Related Components
- [part2.md](part2.md): Uses these videos to learn latent actions
- [part4.md](part4.md): Uses action labels recorded here to map to latent actions

## Implementation Details

### Environment Setup
1. Set up the Atari Breakout environment using OpenAI Gym/Gymnasium
2. Configure the environment for:
   - Lower resolution (84x84 pixels) for efficiency
   - Grayscale frames to reduce dimensionality
   - Frame skipping (4 frames) to speed up training
   - End-of-episode detection based on lives lost
   - 8-frame stacking for better temporal information

### Agent Training
1. Implement a DQN (Deep Q-Network) agent using PyTorch
   - Use a simple CNN architecture (3 convolutional layers + 2 fully connected)
   - 8-frame stacking for better temporal information
   - **Prioritized Experience Replay (PER) with sum-tree for efficient sampling and priority updates**
   - Experience replay buffer with capacity of 1,000,000 transitions
   - Target network update frequency of 500 steps
   - Epsilon-greedy and softmax exploration (temperature-based)
   - Learning rate of 0.0001–0.00025 with Adam optimizer
   - Batch size of 128 for training
   - Double DQN implementation to reduce Q-value overestimation
   - HuberLoss for robust learning
   - Proper gradient clipping (1.0) for stable training
   - Modular, test-driven design
   - Proper device selection (CUDA/MPS/CPU) and torch.compile for speed

2. **Random Network Distillation (RND) for Intrinsic Rewards**
   - Implement RND using a fixed random target network and a trainable predictor network
   - Compute intrinsic reward as the prediction error between the two networks
   - Normalize and scale intrinsic rewards using running statistics
   - Use orthogonal initialization for RND networks
   - Intrinsic rewards encourage exploration of novel states

3. **Dual-Agent Training: Exploration and Exploitation**
   - Use two DQN agents sharing a replay buffer:
     - **Exploration Agent**: Trained with a combined reward (extrinsic + intrinsic from RND)
     - **Exploitation Agent**: Trained with extrinsic (environment) reward only
   - Combine rewards for exploration agent as: `combined_reward = (1 - alpha) * extrinsic + alpha * intrinsic`, with `alpha` decaying over time
   - Both agents are optimized in parallel during training
   - Periodically evaluate the exploitation agent for skill assessment and checkpointing

### Data Collection
- For each training run, record:
  - Complete gameplay episodes
  - Store sequential frames as PNG images at 84x84 resolution
  - Save corresponding actions (LEFT, RIGHT, NOTHING) in a JSON file
  - Include reward and episode information for reference

- Organize data structure as:
data/
  raw_gameplay/
    episodes/
      episode_001/
        frame_00000.png
        frame_00001.png
        ...
      episode_002/
        ...
  actions/
    actions.json

- Create validation splits (10% of episodes) for later evaluation

### Tools and Libraries
- PyTorch for agent implementation
- Gymnasium (OpenAI Gym successor) for Atari environment
- OpenCV for image processing and saving
- **Random Network Distillation (RND) for novelty-based exploration**
- Weights & Biases (optional) for experiment tracking

### Success Criteria
- Agent achieves a score up to 20 points
- Complete and accurate action logs corresponding to all frames
- **Demonstrated use of RND and dual-agent setup for improved exploration and skill acquisition**
- **Demonstrated use of Prioritized Experience Replay (PER) and temperature-based softmax exploration for improved sample efficiency and learning**

## Progress Overview (Implementation Status)

### Implemented so far
- **AtariBreakoutEnv**: Custom wrapper for Gymnasium's Breakout environment with:
  - 84x84 grayscale preprocessing
  - Frame skipping (4 frames)
  - Minimal action space
- **RandomAgent**: Agent that selects random actions, records frames, actions, and rewards, and saves them to disk as PNGs and JSON.
- **DQNAgent**: Deep Q-Network agent implemented in PyTorch, including:
  - 3-layer CNN + 2 fully connected layers for Q-value prediction
  - Input of 8 stacked frames (instead of traditional 4) for better temporal information
  - Experience replay buffer (capacity 1,000,000)
  - Double DQN implementation to reduce Q-value overestimation
  - HuberLoss for more robust learning
  - Target network and synchronization logic (every 500 steps)
  - Epsilon-greedy and softmax action selection
  - Optimized hyperparameters: learning rate 2.5e-4, batch size 128
  - Proper gradient clipping (1.0) for stable training
  - Modular, test-driven design
  - DQN training step (optimize_model) and full training loop
  - Proper device selection (CUDA/MPS/CPU) and torch.compile for speed
- **Random Network Distillation (RND)**: Implemented in `rnd.py` and integrated into training:
  - Fixed random target network and trainable predictor network
  - Intrinsic reward is prediction error, normalized and scaled
  - Used for novelty-based exploration
- **Dual-Agent Training**: Implemented in `train_dqn.py`:
  - **Exploration agent**: Trained with combined (extrinsic + intrinsic) reward
  - **Exploitation agent**: Trained with extrinsic reward only
  - Shared replay buffer
  - Alpha parameter controls reward mixing and decays over time
  - Both agents optimized in parallel
- **Training Script**: `train_dqn.py` implements the full DQN training loop, including:
  - Pre-filling of replay buffer with random experiences
  - Multiple optimization steps per environment step (10x)
  - Epsilon decay with faster schedule (200k steps)
  - Target network updates, and checkpointing
  - Running average reward tracking
  - Command-line overrides for max_episodes, min_buffer, save_freq, and no_save (for flexible runs)
  - Loss logging: average loss per episode, running stats for last 10 episodes, and reward tracking
  - Option to skip saving frames/actions for dry runs (`--no_save`)
  - Saving model checkpoints
  - Recording gameplay data (frames, actions, rewards) for each episode as PNGs and JSON
  - Progress bar and improved logging with formatted metrics
- **Test Suite**: Comprehensive pytest-based tests for environment, agents (random and DQN), replay buffer, target network sync, epsilon-greedy policy, and DQN training/optimization.
- **ROM Installation**: Requirements and instructions updated to ensure ROMs are installed automatically.
- **DQN Training Optimizations**: Implemented various techniques to accelerate learning:
  - Double Q-learning to reduce overestimation bias
  - Increased learning rate and batch size
  - More frequent optimization steps
  - Pre-filling of replay buffer
  - Better exploration strategy
- **Gameplay Video Recording**: Implementation of `record_videos.py` to:
  - Generate videos of gameplay episodes for random and trained agents
  - Save videos as MP4 files at 30fps
  - Include frame number, action taken, and cumulative reward as overlay text
  - Support for different checkpoints
- **Reward Shaping**: Implementation in `AtariBreakoutEnv` including:
  - Living penalty (-0.001 per time step)
  - Life loss penalty (-1.0 for losing a life)
  - Tracking lives between steps to detect life loss

### Remaining tasks
- Train DQN agent for longer durations
- Increase max steps per episode from 1000 to 10000
- Integrate experiment tracking (e.g., with WandB)

## Next Steps
- **Train for longer durations:** Extend training to allow the agent to learn more complex strategies and improve its score.
- **Increase max steps per episode:** Change the environment or training configuration to allow up to 10,000 steps per episode (currently 1,000), enabling the agent to play longer and potentially achieve higher scores.
- **Target high scores:** Aim for the agent to reach and surpass a score of 200 points as training progresses.
- **Continue experiment tracking and checkpointing:** Use tools like WandB for logging and analysis.

## Planned Improvements Status

1. ✅ **Visualize Current Agent Gameplay**
   - Videos of random and trained agents are implemented in `record_videos.py`
   - Proper overlay with frame number, action, and reward
   - MP4 output at 30fps with correct resolution (160x210 plus text area)

2. ❌ **Upgrade Environment to Full RGB and Higher Resolution**
   - **Status: Not implemented**
   - Need to modify environment wrapper to use RGB instead of grayscale
   - Need to use native Atari resolution (160x210) or higher
   - Need to reduce frame skip from 4 to 2
   - Need to update network architecture to handle larger input size
   - Keep 8-frame stacking for temporal information (currently implemented)

3. ✅ **Implement Prioritized Experience Replay**
   - **Status: Implemented**
   - Prioritized Experience Replay (PER) is now implemented using a sum-tree data structure for efficient sampling and priority updates (see `dqn_agent.py`).
   - Supports alpha (priority exponent), beta (importance sampling correction), and dynamic priority updates after each optimization step.
   - Importance sampling weights are used to correct for bias during training.
   - PER is enabled by default in temperature-based exploration mode.

4. ✅ **Implement Reward Shaping**
   - Living penalty of -0.001 per time step implemented
   - Life loss penalty of -1.0 implemented
   - Proper tracking of lives between steps implemented

## Environment Setup & Troubleshooting

- After installing dependencies from `requirements.txt`, run:
  
  ```bash
  python -m AutoROM --accept-license
  ```
  This will download and install all required Atari ROMs. If you see errors about missing ROMs, re-run the above command.
- For more details, see the comments in `requirements.txt`.

## Saving Videos and Frames

You can now generate a large number of gameplay videos and save all frames for each video using the bulk video generation feature. This will save both the video and a folder containing all frames (as PNGs) for each episode in the `bulk_videos` directory.

**Example command:**
```bash
python record_videos.py --bulk --output_dir bulk_videos --total_videos 20 --percent_random 30
```
- This will generate 20 videos in `bulk_videos/`, with 30% from the random agent and the rest from the trained agent.
- For each video (e.g., `bulk_random_agent_1.mp4`), a folder with the same name (e.g., `bulk_random_agent_1/`) will be created containing all frames as `0.png`, `1.png`, etc.

## Remaining Tasks and Next Steps

- **Train DQN agent for longer duration:**
  - Increase the maximum number of training steps to 10,000 for improved performance.
  - (Experiment tracking is already implemented.)

---