import pytest
import numpy as np
import os
from atari_env.atari_env import AtariskiingEnv 
from models.random_agent import RandomAgent

def test_env_initialization():
    env = AtariskiingEnv()
    assert hasattr(env, 'reset'), "Environment should have reset method"
    assert hasattr(env, 'step'), "Environment should have step method"
    assert hasattr(env, 'close'), "Environment should have close method"

def test_env_reset():
    env = AtariskiingEnv()
    obs, info = env.reset()
    
    # Test observation shape (84x84 grayscale)
    assert isinstance(obs, np.ndarray), "Observation should be numpy array"
    assert obs.shape == (84, 84), "Observation should be 84x84"
    assert obs.dtype == np.uint8, "Observation should be uint8"
    
    # Test info dict
    assert isinstance(info, dict), "Info should be a dictionary"
    env.close()

def test_env_step():
    env = AtariskiingEnv()
    env.reset()
    
    # Test step with action
    obs, reward, terminated, truncated, info = env.step(0)  # NOOP action
    
    assert isinstance(obs, np.ndarray), "Observation should be numpy array"
    assert obs.shape == (84, 84), "Observation should be 84x84"
    assert isinstance(reward, float), "Reward should be a float"
    assert isinstance(terminated, bool), "Terminated should be a boolean"
    assert isinstance(truncated, bool), "Truncated should be a boolean"
    assert isinstance(info, dict), "Info should be a dictionary"
    env.close()

def test_random_agent_initialization():
    agent = RandomAgent(n_actions=4)
    assert hasattr(agent, 'select_action'), "Agent should have select_action method"
    assert hasattr(agent, 'record_transition'), "Agent should have record_transition method"
    assert hasattr(agent, 'save_episode_data'), "Agent should have save_episode_data method"

def test_random_agent_action_selection():
    agent = RandomAgent(n_actions=4)
    action = agent.select_action()
    assert isinstance(action, int), "Action should be an integer"
    assert 0 <= action < 4, "Action should be in range [0, 3]"

def test_data_recording():
    # Create temporary directories for testing
    os.makedirs("test_data/episode_001", exist_ok=True)
    
    agent = RandomAgent(n_actions=4)
    dummy_frame = np.zeros((84, 84), dtype=np.uint8)
    
    # Record some transitions
    agent.record_transition(dummy_frame, action=1, reward=1.0, episode=1, step=0)
    agent.record_transition(dummy_frame, action=2, reward=0.0, episode=1, step=1)
    
    # Save episode data
    agent.save_episode_data(
        episode_dir="test_data/episode_001",
        actions_file="test_data/actions.json"
    )
    
    # Check if files were created
    assert os.path.exists("test_data/episode_001/frame_00000.png"), "Frame 0 should be saved"
    assert os.path.exists("test_data/episode_001/frame_00001.png"), "Frame 1 should be saved"
    assert os.path.exists("test_data/actions.json"), "Actions file should be saved"
    
    # Clean up
    import shutil
    shutil.rmtree("test_data")

def test_full_episode():
    env = AtariskiingEnv()
    agent = RandomAgent(n_actions=4)
    
    obs, info = env.reset()
    done = False
    step = 0
    
    while not done and step < 10:  # Run for max 10 steps for testing
        action = agent.select_action()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.record_transition(obs, action, reward, episode=1, step=step)
        step += 1
    
    assert step > 0, "Should complete at least one step"
    env.close()

# def test_dqn_agent_initialization():
#     from dqn_agent import DQNAgent
#     agent = DQNAgent(n_actions=4, state_shape=(8, 84, 84))
#     assert hasattr(agent, 'select_action'), "DQNAgent should have select_action method"
#     assert hasattr(agent, 'optimize_model'), "DQNAgent should have optimize_model method"
#     assert hasattr(agent, 'update_target_network'), "DQNAgent should have update_target_network method"
#     assert hasattr(agent, 'replay_buffer'), "DQNAgent should have a replay_buffer attribute"

# def test_dqn_agent_action_selection():
#     from dqn_agent import DQNAgent
#     agent = DQNAgent(n_actions=4, state_shape=(8, 84, 84))
#     dummy_state = np.zeros((8, 84, 84), dtype=np.uint8)
#     action = agent.select_action(dummy_state)
#     assert isinstance(action, int), "Action should be an integer"
#     assert 0 <= action < 4, "Action should be in range [0, 3]"

# def test_dqn_replay_buffer():
#     from dqn_agent import ReplayBuffer
#     buffer = ReplayBuffer(capacity=100)
#     dummy_transition = (np.zeros((8, 84, 84), dtype=np.uint8), 1, 1.0, 0.0, np.zeros((8, 84, 84), dtype=np.uint8), False)
#     buffer.push(*dummy_transition)
#     assert len(buffer) == 1, "ReplayBuffer should store transitions"
#     sample = buffer.sample(1)
#     states, actions, rewards, next_states, dones = sample
#     assert len(states) == 1, "Sampled batch should have correct size"

# def test_dqn_target_network_sync():
#     from dqn_agent import DQNAgent
#     agent = DQNAgent(n_actions=4, state_shape=(8, 84, 84))
#     # Simulate parameter change
#     for param in agent.policy_net.parameters():
#         param.data += 1.0
#     agent.update_target_network()
#     for p, t in zip(agent.policy_net.parameters(), agent.target_net.parameters()):
#         assert np.allclose(p.detach().cpu().numpy(), t.detach().cpu().numpy()), "Target network should sync with policy network"

# # TDD: Test for DQNAgent.optimize_model (should fail until implemented)
# def test_dqn_optimize_model():
#     import torch
#     from dqn_agent import DQNAgent
#     agent = DQNAgent(n_actions=4, state_shape=(8, 84, 84))
#     # Fill replay buffer with dummy transitions
#     for _ in range(128):  # Match the agent's batch_size
#         state = np.zeros((8, 84, 84), dtype=np.uint8)
#         action = np.random.randint(0, 4)
#         extrinsic_reward = np.random.randn()
#         intrinsic_reward = 0.0
#         next_state = np.zeros((8, 84, 84), dtype=np.uint8)
#         done = np.random.choice([True, False])
#         agent.replay_buffer.push(state, action, extrinsic_reward, intrinsic_reward, next_state, done)
#     # Should raise NotImplementedError or fail until implemented
#     try:
#         loss = agent.optimize_model()
#         assert loss is not None, "optimize_model should return a loss value after update"
#     except NotImplementedError:
#         pass  # Acceptable for TDD

# # TDD: Test for a minimal DQN training loop (should fail until implemented)
# def test_dqn_training_loop():
#     from dqn_agent import DQNAgent
#     from atari_env.atari_env import AtariskiingEnv
#     agent = DQNAgent(n_actions=4, state_shape=(8, 84, 84))
#     env = AtariskiingEnv()
#     obs, info = env.reset()
#     state_stack = np.stack([obs]*8, axis=0)  # (8, 84, 84)
#     for step in range(10):
#         action = agent.select_action(state_stack)
#         next_obs, reward, terminated, truncated, info = env.step(action)
#         next_state_stack = np.roll(state_stack, shift=-1, axis=0)
#         next_state_stack[-1] = next_obs
#         extrinsic_reward = reward
#         intrinsic_reward = 0.0
#         agent.replay_buffer.push(state_stack, action, extrinsic_reward, intrinsic_reward, next_state_stack, terminated or truncated)
#         agent.optimize_model()  # Should not crash
#         state_stack = next_state_stack
#         if terminated or truncated:
#             obs, info = env.reset()
#             state_stack = np.stack([obs]*8, axis=0)
#     # Try updating target network
#     agent.update_target_network()
#     env.close()

# # TDD: Prioritized Replay Buffer tests (should fail until implemented)
# def test_prioritized_replay_buffer_init():
#     from dqn_agent import PrioritizedReplayBuffer
#     buffer = PrioritizedReplayBuffer(capacity=100)
#     assert hasattr(buffer, 'push'), "PrioritizedReplayBuffer should have push method"
#     assert hasattr(buffer, 'sample'), "PrioritizedReplayBuffer should have sample method"
#     assert hasattr(buffer, 'update_priorities'), "PrioritizedReplayBuffer should have update_priorities method"
#     assert len(buffer) == 0, "Buffer should be empty after initialization"

# def test_prioritized_replay_buffer_push_and_capacity():
#     from dqn_agent import PrioritizedReplayBuffer
#     buffer = PrioritizedReplayBuffer(capacity=10)
#     dummy = (np.zeros((8, 84, 84), dtype=np.uint8), 1, 1.0, 0.0, np.zeros((8, 84, 84), dtype=np.uint8), False)
#     for i in range(15):
#         buffer.push(*dummy, priority=abs(i+1))
#     assert len(buffer) == 10, "Buffer should not exceed capacity"

# def test_prioritized_replay_buffer_sample_returns_is_weights_and_indices():
#     from dqn_agent import PrioritizedReplayBuffer
#     buffer = PrioritizedReplayBuffer(capacity=20, alpha=0.6, beta=0.4)
#     dummy = (np.zeros((8, 84, 84), dtype=np.uint8), 1, 1.0, 0.0, np.zeros((8, 84, 84), dtype=np.uint8), False)
#     for i in range(20):
#         buffer.push(*dummy, priority=abs(i+1))
#     batch = buffer.sample(batch_size=5)
#     states, actions, rewards, next_states, dones, weights, indices = batch
#     assert len(states) == 5, "Sampled batch should have correct size"
#     assert weights.shape == (5,), "Importance sampling weights should be returned with correct shape"
#     assert len(indices) == 5, "Indices should be returned for updating priorities"

# def test_prioritized_replay_buffer_update_priorities():
#     from dqn_agent import PrioritizedReplayBuffer
#     buffer = PrioritizedReplayBuffer(capacity=10)
#     dummy = (np.zeros((8, 84, 84), dtype=np.uint8), 1, 1.0, 0.0, np.zeros((8, 84, 84), dtype=np.uint8), False)
#     for i in range(10):
#         buffer.push(*dummy, priority=1.0)
#     batch = buffer.sample(batch_size=4)
#     _, _, _, _, _, _, indices = batch
#     new_priorities = np.random.rand(4) + 1.0
#     buffer.update_priorities(indices, new_priorities)
#     # No assertion: just check that it does not crash

# def test_prioritized_replay_buffer_beta_annealing():
#     from dqn_agent import PrioritizedReplayBuffer
#     buffer = PrioritizedReplayBuffer(capacity=10, beta=0.4)
#     assert buffer.beta == 0.4, "Initial beta should be set"
#     buffer.anneal_beta(1.0)
#     assert buffer.beta == 1.0, "Beta should be updated after annealing" 