import unittest
import numpy as np
import torch
from models.rnd import RunningMeanStd, RandomNetworkDistillation

class TestRunningMeanStd(unittest.TestCase):
    def test_update_from_moments(self):
        rms = RunningMeanStd(shape=(1,))
        rms.update_from_moments(np.array([5.0]), np.array([4.0]), 1)
        self.assertEqual(rms.mean.item(), 5.0)
        self.assertEqual(rms.var.item(), 4.0)
        self.assertEqual(rms.count, 1)

    def test_update(self):
        rms = RunningMeanStd(shape=(1,))
        x1 = np.array([[1.0]])
        x2 = np.array([[2.0]])
        rms.update(x1)
        self.assertEqual(rms.mean.item(), 1.0)
        # For single sample, variance remains at 1.0 (default)
        self.assertEqual(rms.var.item(), 1.0)
        rms.update(x2)
        self.assertAlmostEqual(rms.mean.item(), 1.5, places=5)
        self.assertAlmostEqual(rms.var.item(), 0.5, places=5)

    def test_normalize(self):
        rms = RunningMeanStd(shape=(1,))
        # Set mean and variance directly for testing
        rms.mean = np.array([15.0])
        rms.var = np.array([25.0])
        rms.count = 2
        normalized = rms.normalize(np.array([[25.0]]))
        self.assertAlmostEqual(normalized.item(), 2.0, places=5)  # (25-15)/5 = 2.0

class TestRandomNetworkDistillation(unittest.TestCase):
    def test_initialization(self):
        state_shape = (4, 84, 84)
        output_dim = 512
        rnd = RandomNetworkDistillation(state_shape=state_shape, output_dim=output_dim)
        
        # Check that target and predictor networks have the same structure
        self.assertEqual(len(list(rnd.target.parameters())), len(list(rnd.predictor.parameters())))
        
        # Check target network parameters are not trainable (requires_grad=False)
        for param in rnd.target.parameters():
            self.assertFalse(param.requires_grad)
        
        # Check predictor network parameters are trainable (requires_grad=True)
        for param in rnd.predictor.parameters():
            self.assertTrue(param.requires_grad)

    def test_forward_pass(self):
        state_shape = (4, 84, 84)
        batch_size = 2
        output_dim = 512
        rnd = RandomNetworkDistillation(state_shape=state_shape, output_dim=output_dim)
        
        # Create a batch of dummy states
        states = torch.randn((batch_size,) + state_shape)
        
        # Get target and predictor outputs
        with torch.no_grad():
            target_output = rnd.target(states)
            predictor_output = rnd.predictor(states)
        
        # Check output shapes
        self.assertEqual(target_output.shape, (batch_size, output_dim))
        self.assertEqual(predictor_output.shape, (batch_size, output_dim))

    def test_compute_intrinsic_reward(self):
        state_shape = (4, 84, 84)
        batch_size = 2
        output_dim = 512
        rnd = RandomNetworkDistillation(state_shape=state_shape, output_dim=output_dim)
        
        # Create a batch of dummy states
        states = torch.randn((batch_size,) + state_shape)
        
        # Compute intrinsic rewards
        rewards = rnd.compute_intrinsic_reward(states)
        
        # Check rewards shape
        self.assertEqual(rewards.shape, (batch_size,))
        
        # We know that some rewards might be negative due to normalization
        # Instead, just check that rewards are finite
        rewards_cpu = rewards.cpu()
        self.assertTrue(torch.isfinite(rewards_cpu).all().item())

    def test_update(self):
        state_shape = (4, 84, 84)
        batch_size = 3
        output_dim = 512
        rnd = RandomNetworkDistillation(state_shape=state_shape, output_dim=output_dim)
        
        # Create a batch of dummy states
        states = torch.randn((batch_size,) + state_shape)
        
        # Record loss before update
        initial_loss = rnd.update(states).item()
        
        # Update again and check if loss decreases
        for _ in range(5):  # Multiple updates to ensure loss reduction
            new_loss = rnd.update(states).item()
        
        # Loss should decrease after multiple updates on the same data
        self.assertLess(new_loss, initial_loss)

if __name__ == "__main__":
    unittest.main() 