# Random Network Distillation (RND) Implementation Guide

## Overview

This guide outlines how to implement Random Network Distillation (RND) to enhance exploration in your DQN agent for Atari Breakout. RND will help the agent discover novel states and avoid local minima where the agent just moves left.

## Core Components

1. **Target Network**: A randomly initialized neural network with fixed weights
2. **Predictor Network**: A neural network trained to predict the output of the target network
3. **Intrinsic Reward**: Generated based on the prediction error
4. **Reward Combination**: Blend intrinsic and extrinsic rewards using an alpha parameter

## Implementation Steps

### 1. Create the RND Networks

#### RND Network Architecture
- Create a base network class for both target and predictor networks
- Architecture should be similar to the feature extractor of your DQN (convolutional layers)
- Output should be a feature vector (e.g., 512 dimensions) rather than Q-values
- Use orthogonal initialization for better feature representation:
  - Implement using PyTorch's nn.init.orthogonal_() function
  - Set gain parameter to sqrt(2) for ReLU activations
  - Orthogonal initialization creates weight matrices with orthogonal columns/rows
  - This helps preserve gradient magnitudes and prevent vanishing/exploding gradients
  - It also improves feature diversity and speeds up convergence

#### Target and Predictor Networks
- Initialize target network with random weights and freeze them (no gradient updates)
- Initialize predictor network with random weights (will be trained)
- Both networks should have identical architecture

### 2. Implement Running Statistics Tracking

#### RunningMeanStd Class
- Create a class to track running mean and variance of values
- Include methods to update statistics with new batches of data
- Use these statistics to normalize intrinsic rewards and observations

### 3. Main RND Implementation

#### RandomNetworkDistillation Class
- Initialize with state shape, device, output dimension, learning rate, etc.
- Implement method to calculate prediction error and intrinsic reward
- Include reward normalization using running statistics
- Implement update method for training the predictor network
- Add parameters to control update frequency (update_proportion)
- Include methods for tracking and reporting statistics

### 4. Create Two DQN Agents

#### Setup Two Separate Agents
- Create two identical DQN agents with different purposes:
  1. **Exploration Agent**:
     - Responsible for interacting with the environment
     - Policy updated using combined intrinsic and extrinsic rewards
     - Focused on discovering novel states and avoiding local minima
  2. **Exploitation Agent**:
     - Never directly interacts with the environment during training
     - Policy updated using only extrinsic rewards
     - Focused purely on maximizing game score
     - Used for evaluation and final deployment

#### Shared Components
- Both agents should share the same replay buffer
- RND module is only needed for the exploration agent
- Both agents can share the same state preprocessing code

### 5. Implement Training Loop with Dual Agents

#### Reward Combination
- Implement the formula: `total_reward = (1 - alpha) * extrinsic_reward + alpha * intrinsic_reward`
- Start with alpha = 0.5 and adjust based on performance
- Consider implementing an alpha decay schedule

#### Environmental Interactions
- Calculate intrinsic reward for each state encountered
- Store the combined reward in the replay buffer
- Track both intrinsic and extrinsic rewards separately for monitoring

#### Monitoring and Logging
- Log intrinsic reward statistics (mean, max) during training
- Track the policy's performance with pure extrinsic rewards for evaluation
- Monitor exploration metrics (state visitation counts if possible)

## Key Hyperparameters

- **output_dim**: Dimensionality of the RND network output (e.g., 512)
- **reward_scale**: Scaling factor for intrinsic rewards (e.g., 0.1)
- **update_proportion**: Percentage of transitions used for RND updates (e.g., 0.25)
- **alpha**: Weight for intrinsic vs extrinsic rewards (e.g., 0.5 initially, decay over time)
- **learning_rate**: Separate learning rate for the RND predictor network (e.g., 1e-4)
- **evaluation_frequency**: How often to evaluate the exploitation agent (e.g., every 100 episodes)
- **evaluation_episodes**: Number of episodes to run during each evaluation (e.g., 10)

## Best Practices

### Normalizing Intrinsic Rewards
- Track running mean and std of intrinsic rewards
- Normalize by dividing by running std: `normalized_reward = reward / (sqrt(var) + epsilon)`
- Apply clipping to prevent extreme values: `clipped_reward = clip(normalized_reward, -5, 5)`

### Training Frequency
- Train RND predictor less frequently than policy network
- Use update_proportion parameter to control frequency
- Start with update_proportion = 0.25 (update on 25% of transitions)

### Alpha Parameter Management
- Start with alpha = 0.5 (equal weight to intrinsic and extrinsic rewards)
- Implement alpha decay to gradually shift focus to extrinsic rewards
- Consider different alpha values for different stages of training

### Observation Normalization
- Normalize observations before feeding to RND networks
- Use running statistics for normalization
- Apply clipping to normalized observations

## Integration with Existing Code

### Files to Modify

1. **dqn_agent.py**:
   - Add RND class implementation with orthogonal initialization
   - Add RunningMeanStd class for normalizing rewards
   - Modify DQNAgent to incorporate RND
   - Add intrinsic reward calculation

2. **train_dqn.py**:
   - Create two DQN agents (exploration and exploitation)
   - Update training loop to implement dual-agent architecture
   - Use exploration agent for collecting experiences
   - Update exploitation agent with extrinsic rewards only
   - Add periodic evaluation of exploitation agent
   - Add alpha parameter and decay schedule
   - Include RND statistics in logs

### Data Structures to Update

1. **Replay Buffer**:
   - Store transitions with both intrinsic and extrinsic rewards
   - Implement methods to retrieve transitions with either combined or extrinsic-only rewards

2. **Training Statistics**:
   - Track intrinsic rewards over time
   - Track exploration agent performance (combined rewards)
   - Track exploitation agent performance (extrinsic rewards only)
   - Monitor prediction errors
   - Track evaluation metrics for exploitation agent
   - Compare visitation statistics between baseline and RND agents

## Debugging and Monitoring

- Print RND statistics periodically during training:
  ```
  if episode % log_frequency == 0:
      print(f"RND Stats: {rnd.get_stats()}")
  ```
- Track both exploration and exploitation agent performances separately:
  ```
  exploration_rewards = []  # Combined rewards
  exploitation_rewards = []  # Extrinsic rewards only
  evaluation_scores = []  # Periodic evaluation results
  ```
- Monitor the ratio of intrinsic to extrinsic rewards:
  ```
  intrinsic_ratio = mean(intrinsic_rewards) / (mean(abs(extrinsic_rewards)) + 1e-8)
  print(f"Intrinsic/Extrinsic ratio: {intrinsic_ratio:.2f}")
  ```
- Check that intrinsic rewards gradually decrease for frequently visited states
- Compare evaluation performance between exploitation agent and a baseline without RND
- Create visualizations showing:
  1. Evaluation scores over time for exploitation agent
  2. Training scores for both agents
  3. Intrinsic reward magnitude over time
  4. Alpha parameter value over time

## Expected Outcomes

- Increased exploration in the early stages of training
- Discovery of optimal strategies more consistently
- Avoidance of local optima (like staying at the left side)
- Potentially longer training time but better final performance
- More stable learning across different random seeds

## Potential Issues and Solutions

### Issue: Intrinsic rewards dominate training
- Solution: Decrease alpha or reward_scale
- Solution: Implement more aggressive normalization

### Issue: No improvement over baseline
- Solution: Increase alpha or reward_scale
- Solution: Check for bugs in intrinsic reward calculation
- Solution: Ensure RND predictor is actually training

### Issue: Unstable training
- Solution: Add gradient clipping to RND updates
- Solution: Decrease learning rate for RND predictor
- Solution: Implement more aggressive reward normalization