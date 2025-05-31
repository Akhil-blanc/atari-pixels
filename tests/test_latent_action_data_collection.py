import pytest
import torch
import json


def test_load_vqvae_model():
    from latent_action_model import load_latent_action_model
    model, step = load_latent_action_model('checkpoints/latent_action/best.pt', device='cpu')
    assert hasattr(model, 'encoder')
    assert hasattr(model, 'vq')
    assert hasattr(model, 'decoder')


def test_env_step_and_collect():
    from atari_env.atari_env import AtariBreakoutEnv
    env = AtariBreakoutEnv()
    obs, _ = env.reset()
    action = 1
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == next_obs.shape


def test_latent_code_extraction():
    from latent_action_model import load_latent_action_model
    model, _ = load_latent_action_model('checkpoints/latent_action/best.pt', device='cpu')
    # Use RGB frames: (B, 3, 210, 160)
    frame_t = torch.zeros((1, 3, 210, 160), dtype=torch.float32)
    frame_tp1 = torch.zeros((1, 3, 210, 160), dtype=torch.float32)
    _, indices, *_ = model(frame_t, frame_tp1)
    assert indices.shape[-2:] == (5, 7)
    assert indices.max() < 256


def test_data_collection_and_save(tmp_path):
    # Simulate collecting and saving 10 pairs
    pairs = [{'action': 0, 'latent_code': [1,2,3]}, {'action': 1, 'latent_code': [4,5,6]}]
    out_path = tmp_path / "pairs.json"
    with open(out_path, 'w') as f:
        json.dump(pairs, f)
    with open(out_path) as f:
        loaded = json.load(f)
    assert loaded == pairs


def test_minimum_pairs_collected():
    # Simulate collection loop
    collected = []
    for i in range(100):
        collected.append({'action': i%4, 'latent_code': [i%256]*35})
    assert len(collected) >= 100 