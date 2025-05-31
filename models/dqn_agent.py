"""Deep Q-Network (DQN) implementation with several improvements.

Key Features:
- Double DQN to prevent Q-value overestimation
- Huber Loss for robustness to outliers
- Q-value normalization
- Gradient clipping
- Experience replay for sample efficiency
- Target network for training stability
- Mixed precision training on CUDA
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, OrderedDict
import random



class ReplayBuffer:
    """Experience replay buffer for DQN with support for intrinsic and extrinsic rewards."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity) # FIFO buffer

    def push(self, state, action, extrinsic_reward, intrinsic_reward, next_state, done):
        """Store a transition in the buffer with both extrinsic and intrinsic rewards."""
        self.buffer.append((state, action, extrinsic_reward, intrinsic_reward, next_state, done))

    def sample(self, batch_size: int, mode: str = 'exploration', alpha: float = 0.5):
        """
        Sample a batch of transitions.
        mode: 'exploration' returns combined reward, 'exploitation' returns extrinsic only.
        alpha: weight for intrinsic vs extrinsic reward (used in exploration mode)
        """
        indices = random.sample(range(len(self.buffer)), batch_size)
        batch = [self.buffer[idx] for idx in indices]
        states, actions, extrinsic_rewards, intrinsic_rewards, next_states, dones = zip(*batch)
        if mode == 'exploration':
            # Combined reward
            rewards = [(1 - alpha) * er + alpha * ir for er, ir in zip(extrinsic_rewards, intrinsic_rewards)]
        else:
            # Exploitation: extrinsic only
            rewards = extrinsic_rewards
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class SumTree:
    """SumTree data structure for efficient sampling and priority updates."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = np.full(capacity, None, dtype=object)  # Use None for uninitialized
        self.size = 0
        self.write = 0

    def add(self, priority, data):
        assert isinstance(data, tuple) and len(data) == 6, f"SumTree.add: data must be a tuple of length 6, got {type(data)} with len {len(data) if isinstance(data, tuple) else 'N/A'}: {data}"
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, s):
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - (self.capacity - 1)
        # Check if index is valid (within current size)
        if data_idx >= self.size or self.data[data_idx] is None:
            # Sample from a valid range instead
            #print(f"[DEBUG] Sampling from valid range instead of {data_idx}")
            valid_data_idx = random.randint(0, self.size - 1)
            data_idx = valid_data_idx
            idx = data_idx + self.capacity - 1

        data = self.data[data_idx]
        if not (isinstance(data, tuple) and len(data) == 6):
            raise ValueError(f"SumTree.get: Invalid data at index {data_idx}: {data} (type={type(data)})")
        return idx, self.tree[idx], data

    def total(self):
        return self.tree[0]

    def __len__(self):
        return self.size

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer using a sum-tree."""
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_initial = beta
        self.epsilon = 1e-6
        self.tree = SumTree(capacity)
        self.max_priority = 1.0

    def push(self, state, action, extrinsic_reward, intrinsic_reward, next_state, done, priority=None):
        transition = (state, action, extrinsic_reward, intrinsic_reward, next_state, done)
        assert isinstance(transition, tuple) and len(transition) == 6, f"PrioritizedReplayBuffer.push: transition must be tuple of length 6, got {type(transition)} with len {len(transition)}: {transition}"
        if priority is None:
            priority = self.max_priority
        p = (abs(priority) + self.epsilon) ** self.alpha
        self.tree.add(p, transition)
        self.max_priority = max(self.max_priority, p)

    def sample(self, batch_size, mode='exploration', alpha=0.5):
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total() / batch_size
        for i in range(batch_size):
            while True:
                a = segment * i
                b = segment * (i + 1)
                s = np.random.uniform(a, b)
                try:
                    idx, p, data = self.tree.get(s)
                    batch.append(data)
                    idxs.append(idx)
                    priorities.append(p)
                    break  # Only break if valid
                except Exception as e:
                    print(f"[DEBUG] Resampling due to invalid data: {e}")
                    continue
        try:
            states, actions, extrinsic_rewards, intrinsic_rewards, next_states, dones = zip(*batch)
        except Exception as e:
            print("[DEBUG] Error in zip(*batch):", e)
            print("[DEBUG] Batch contents:")
            for i, item in enumerate(batch):
                print(f"  batch[{i}]: type={type(item)}, value={item}")
            raise
        if mode == 'exploration':
            rewards = [(1 - alpha) * er + alpha * ir for er, ir in zip(extrinsic_rewards, intrinsic_rewards)]
        else:
            rewards = extrinsic_rewards
        priorities = np.array(priorities, dtype=np.float32)
        probs = priorities / (self.tree.total() + 1e-8)
        weights = (len(self.tree) * probs) ** (-self.beta)
        weights /= weights.max() + 1e-8
        weights = weights.astype(np.float32)
        return states, actions, rewards, next_states, dones, weights, idxs

    def update_priorities_nodebug(self, idxs, priorities):
        for idx, priority in zip(idxs, priorities):
            p = (abs(priority) + self.epsilon) ** self.alpha
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, p)
    
    def update_priorities(self, idxs, priorities):
        #print(f"[DEBUG] Raw priorities - Max: {np.max(priorities):.6f}, Mean: {np.mean(priorities):.6f}")
        
        transformed_priorities = []
        for idx, priority in zip(idxs, priorities):
            p = (abs(priority) + self.epsilon) ** self.alpha
            transformed_priorities.append(p)
            self.tree.update(idx, p)
        
        new_max = max(transformed_priorities) if transformed_priorities else 0
        self.max_priority = max(self.max_priority, new_max)
        #print(f"[DEBUG] Transformed priorities - Max: {new_max:.6f}, Alpha: {self.alpha}")
        #print(f"[DEBUG] Updated max_priority: {self.max_priority:.6f}")

    def anneal_beta(self, new_beta):
        self.beta = new_beta

    def __len__(self):
        return len(self.tree)

class DQNCNN(nn.Module):
    """CNN architecture for DQN, based on the original DQN paper.
    
    Architecture:
    1. Input: (batch_size, 8, 84, 84) - 8 stacked frames
    2. Conv layers process spatial features
    3. FC layers compute Q-values for each action
    """
    def __init__(self, input_shape, n_actions):
        super().__init__()
        c, h, w = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # Compute conv output size
        def conv2d_size_out(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        self.fc = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights using Kaiming initialization."""
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass normalizes inputs and computes Q-values."""
        x = x.float() / 255.0  # Normalize pixel values
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class DQNAgent:
    """Deep Q-Network agent implementing several DQN improvements.
    
    Key Features:
    1. Double DQN: Uses two networks to prevent Q-value overestimation
       - Policy network: Selects actions
       - Target network: Evaluates actions
       This prevents the positive bias in Q-value estimation that occurs
       when the same network both selects and evaluates actions.
       
    2. Experience Replay: Stores and randomly samples transitions
       - Breaks correlation between consecutive samples
       - Allows multiple updates from each experience
       - Improves sample efficiency
       
    3. Q-value normalization: Prevents any action from dominating
       - Subtracts mean and divides by std
       - Helps maintain reasonable Q-value ranges
       
    4. Huber Loss: More robust to outliers than MSE
       - Combines L2 loss for small errors
       - L1 loss for large errors
       - Helps prevent unstable updates
       
    5. Gradient clipping: Prevents explosive gradients
       - Clips gradient norm to 1.0
       - Maintains stable updates
    """
    def __init__(self, n_actions: int, state_shape, replay_buffer=None, prioritized=False, per_alpha=0.6, per_beta=0.4):
        """Initialize DQN agent with networks and replay buffer."""
        self.n_actions = n_actions
        self.state_shape = state_shape
        self.policy_net = DQNCNN(state_shape, n_actions)
        self.target_net = DQNCNN(state_shape, n_actions)
        self.prioritized = prioritized
        if replay_buffer is not None:
            self.replay_buffer = replay_buffer
        else:
            if prioritized:
                self.replay_buffer = PrioritizedReplayBuffer(capacity=1000000, alpha=per_alpha, beta=per_beta)
            else:
                self.replay_buffer = ReplayBuffer(capacity=1000000)
        self.gamma = 0.99  # Discount factor
        self.batch_size = 32
        self.learning_rate = 2.5e-4
        # Device selection
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        # Compile for speed if available and on CUDA only
        if hasattr(torch, 'compile') and self.device.type == 'cuda':
            self.policy_net = torch.compile(self.policy_net)

    def select_action(self, state, mode: str = 'greedy', temperature: float = 1.0, epsilon: float = 0.0) -> int:
        """
        Select action using specified exploration strategy.
        
        Args:
            state: Current state
            mode: 'greedy' or 'softmax'
            temperature: Temperature for softmax exploration
            epsilon: Probability of random action (for training only)
        """
        
        # Normal policy-based selection (without normalization)
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            if mode == 'softmax':
                # Softmax action selection (Boltzmann exploration)
                logits = q_values / max(temperature, 1e-6)
                probs = torch.softmax(logits, dim=1)
                action = int(torch.multinomial(probs, num_samples=1).item())
            elif mode == 'epsilon':
                if np.random.rand() < epsilon:
                    action = np.random.randint(0, self.n_actions)
                else:
                    action = int(torch.argmax(q_values, dim=1).item())  
            elif mode == 'greedy':
                # Greedy with 5% random action
                if np.random.rand() < 0.05:
                    action = np.random.randint(0, self.n_actions)
                else:
                    action = int(torch.argmax(q_values, dim=1).item())
            else:
                action = int(torch.argmax(q_values, dim=1).item())
        return action

    def optimize_model(self, mode: str = 'exploitation', alpha: float = 0.5):
        """Update policy network using double DQN algorithm with NaN protection."""
        if len(self.replay_buffer) < self.batch_size:
            return None
            
        # Sample from replay buffer
        batch = self.replay_buffer.sample(self.batch_size, mode=mode, alpha=alpha)
        
        # Unpack batch (simplifying PER logic for clarity)
        if self.prioritized:
            states, actions, rewards, next_states, dones, weights, idxs = batch
            weights = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)
        else:
            states, actions, rewards, next_states, dones = batch
            weights = None
            idxs = None
        
        # Convert to tensors
        states = torch.from_numpy(np.stack(states)).to(self.device).float()
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.from_numpy(np.stack(next_states)).to(self.device).float()
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        # Switch to training mode
        self.policy_net.train()
        
        # Calculate current Q-values
        q_values = self.policy_net(states).gather(1, actions)
        
        # Calculate target Q-values (Double DQN)
        with torch.no_grad():
            # Select actions using policy net
            next_actions = self.policy_net(next_states).max(1, keepdim=True)[1]
            # Evaluate actions using target net
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            # Calculate targets
            target_q = rewards + self.gamma * next_q_values * (1.0 - dones)
        
        # Calculate TD errors and loss
        td_errors = q_values - target_q
        mean_td = td_errors.abs().mean().item()
        
        # Use Huber loss for stability
        loss = nn.HuberLoss()(q_values, target_q)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        # Check for NaN before updating
        if any(torch.isnan(p.grad).any() for p in self.policy_net.parameters() if p.grad is not None):
            print("WARNING: NaN detected in gradients! Skipping update.")
            return None
            
        # Update weights
        self.optimizer.step()
        
        # Check for NaN after updating
        if any(torch.isnan(p).any() for p in self.policy_net.parameters()):
            print("WARNING: NaN detected in weights! Restoring from target network.")
            self.policy_net.load_state_dict(self.target_net.state_dict())
            return None
        
        return loss.item(), mean_td

    def update_target_network(self):
        """Copy policy network weights to target network.
        
        This periodic update helps maintain stable Q-learning targets.
        Too frequent updates can lead to unstable training,
        while too infrequent updates can lead to stale targets.
        """
        state_dict = self.policy_net.state_dict()
        cleaned_state_dict = OrderedDict()
        for k,v in state_dict.items():
            if k.startswith('_orig_mod.'):
                cleaned_state_dict[k.replace('_orig_mod.', '')]= v
            else:
                cleaned_state_dict[k] = v
        self.target_net.load_state_dict(cleaned_state_dict)

    def anneal_per_beta(self, new_beta):
        if self.prioritized and hasattr(self.replay_buffer, 'anneal_beta'):
            self.replay_buffer.anneal_beta(new_beta) 

    def diagnostic_sampling_comparison(self, batch_size=128, num_batches=10):
        """Compare average TD errors from prioritized vs random sampling"""
        if len(self.replay_buffer) < batch_size:
            return None
        
        # Track average TD errors
        prioritized_td_errors = []
        random_td_errors = []
        
        for _ in range(num_batches):
            # 1. Get a batch using prioritized sampling
            if self.prioritized:
                per_batch = self.replay_buffer.sample(batch_size)
                per_states, per_actions, per_rewards, per_next_states, per_dones, per_weights, per_idxs = per_batch
            
            # 2. Get a batch using uniform random sampling
            random_indices = np.random.randint(0, len(self.replay_buffer), size=batch_size)
            random_batch = []
            for idx in random_indices:
                data_idx = idx % self.replay_buffer.capacity
                if hasattr(self.replay_buffer, 'tree'):
                    data = self.replay_buffer.tree.data[data_idx]
                else:
                    data = list(self.replay_buffer.buffer)[data_idx]
                if data is not None:
                    random_batch.append(data)
            
            if len(random_batch) < batch_size:
                continue
                
            # Unpack random batch
            rand_states, rand_actions, rand_rewards, rand_intrinsic_rewards, rand_next_states, rand_dones = zip(*random_batch)
            
            # Convert to tensors for both batches
            per_states_tensor = torch.from_numpy(np.stack(per_states)).to(self.device).float() / 255.0
            per_actions_tensor = torch.tensor(per_actions, dtype=torch.long, device=self.device).unsqueeze(1)
            # Add this line to convert rewards to tensor
            per_rewards_tensor = torch.tensor(per_rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
            per_next_states_tensor = torch.from_numpy(np.stack(per_next_states)).to(self.device).float() / 255.0
            per_dones_tensor = torch.tensor(per_dones, dtype=torch.float32, device=self.device).unsqueeze(1)
            
            rand_states_tensor = torch.from_numpy(np.stack(rand_states)).to(self.device).float() / 255.0
            rand_actions_tensor = torch.tensor(rand_actions, dtype=torch.long, device=self.device).unsqueeze(1)
            # Add this line to convert rewards to tensor
            rand_rewards_tensor = torch.tensor(rand_rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
            rand_next_states_tensor = torch.from_numpy(np.stack(rand_next_states)).to(self.device).float() / 255.0
            rand_dones_tensor = torch.tensor(rand_dones, dtype=torch.float32, device=self.device).unsqueeze(1)
            
            # Calculate TD errors for both batches
            with torch.no_grad():
                # For prioritized batch
                per_q_values = self.policy_net(per_states_tensor).gather(1, per_actions_tensor)
                per_next_actions = self.policy_net(per_next_states_tensor).max(1, keepdim=True)[1]
                per_next_q_values = self.target_net(per_next_states_tensor).gather(1, per_next_actions)
                # Use per_rewards_tensor instead of per_rewards
                per_target_q = per_rewards_tensor + self.gamma * per_next_q_values * (1.0 - per_dones_tensor)
                per_td_error = (per_q_values - per_target_q).abs().mean().item()
                prioritized_td_errors.append(per_td_error)
                
                # For random batch
                rand_q_values = self.policy_net(rand_states_tensor).gather(1, rand_actions_tensor)
                rand_next_actions = self.policy_net(rand_next_states_tensor).max(1, keepdim=True)[1]
                rand_next_q_values = self.target_net(rand_next_states_tensor).gather(1, rand_next_actions)
                # Use rand_rewards_tensor instead of rand_rewards
                rand_target_q = rand_rewards_tensor + self.gamma * rand_next_q_values * (1.0 - rand_dones_tensor)
                rand_td_error = (rand_q_values - rand_target_q).abs().mean().item()
                random_td_errors.append(rand_td_error)
        
        # Calculate overall stats
        avg_prioritized_td = np.mean(prioritized_td_errors)
        avg_random_td = np.mean(random_td_errors)
        improvement_ratio = avg_prioritized_td / avg_random_td if avg_random_td > 0 else 0
        
        result = {
            'avg_prioritized_td_error': avg_prioritized_td,
            'avg_random_td_error': avg_random_td,
            'improvement_ratio': improvement_ratio,
            'prioritized_td_errors': prioritized_td_errors,
            'random_td_errors': random_td_errors
        }
        
        print(f"PER Diagnostics: Prioritized TD Error = {avg_prioritized_td:.6f}, Random TD Error = {avg_random_td:.6f}")
        print(f"Improvement Ratio: {improvement_ratio:.3f}x (higher is better for PER)")
        
        return result

    def get_action_softmax_probs(self, state, temperature: float = 1.0):
        """
        Return softmax probabilities for all actions given a state and temperature.
        Args:
            state: Current state (np.ndarray)
            temperature: Softmax temperature
        Returns:
            probs: np.ndarray of shape (n_actions,)
        """
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            logits = q_values / max(temperature, 1e-6)
            probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
        return probs

    def get_q_values(self, state):
        """
        Return Q-values for all actions given a state.
        Args:
            state: Current state (np.ndarray)
        Returns:
            q_values: np.ndarray of shape (n_actions,)
        """
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).cpu().numpy().flatten()
        return q_values

    def get_weight_norms(self):
        """Return L2 norm of weights for policy and target networks as a dict."""
        policy_weight_norm = float(torch.norm(torch.stack([p.detach().float().norm(2) for p in self.policy_net.parameters()]), 2).item())
        target_weight_norm = float(torch.norm(torch.stack([p.detach().float().norm(2) for p in self.target_net.parameters()]), 2).item())
        # Add monitoring for extreme values
        max_weight = max([p.detach().abs().max().item() for p in self.policy_net.parameters()])
        min_weight = min([p.detach().abs().min().item() for p in self.policy_net.parameters() if p.numel() > 0])
        
        return {
            'policy_weight_norm': policy_weight_norm,
            'target_weight_norm': target_weight_norm,
            'policy_max_weight': max_weight,
            'policy_min_weight': min_weight
        }

    def get_grad_norms(self):
        """Return L2 norm of gradients for policy and target networks as a dict. If gradients are None, returns 0."""
        def grad_norm(model):
            grads = [p.grad.detach().float().norm(2) for p in model.parameters() if p.grad is not None]
            if len(grads) == 0:
                return 0.0
            return float(torch.norm(torch.stack(grads), 2).item())
        policy_grad_norm = grad_norm(self.policy_net)
        target_grad_norm = grad_norm(self.target_net)
        return {
            'policy_grad_norm': policy_grad_norm,
            'target_grad_norm': target_grad_norm
        }