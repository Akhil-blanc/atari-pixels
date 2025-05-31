import pytest
import torch
from torch.utils.data import DataLoader
from dataloaders.latent_action_data import AtariFramePairDataset, get_action_latent_dataloaders
from models.latent_action_model import LatentActionVQVAE
import os
import tempfile
import numpy as np
from PIL import Image
import glob

def create_dummy_episode(dir_path, num_frames=5, grayscale=False):
    os.makedirs(dir_path, exist_ok=True)
    for i in range(num_frames):
        arr = np.random.randint(0, 256, (160, 210, 1 if grayscale else 3), dtype=np.uint8)
        img = Image.fromarray(arr.squeeze() if grayscale else arr)
        img.save(os.path.join(dir_path, f"{i}.png"))

# --- Data Preparation Tests ---
def test_dataset_split():
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(10):
            create_dummy_episode(os.path.join(tmpdir, f"ep{i}"), num_frames=3)
        ds_train = AtariFramePairDataset(tmpdir, split='train', grayscale=False, seed=123, split_ratio=(0.6,0.2,0.2))
        ds_val = AtariFramePairDataset(tmpdir, split='val', grayscale=False, seed=123, split_ratio=(0.6,0.2,0.2))
        ds_test = AtariFramePairDataset(tmpdir, split='test', grayscale=False, seed=123, split_ratio=(0.6,0.2,0.2))
        total_eps = 10
        assert len(ds_train.episode_dirs) == 6
        assert len(ds_val.episode_dirs) == 2
        assert len(ds_test.episode_dirs) == 2

def test_frame_pair_extraction():
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            create_dummy_episode(os.path.join(tmpdir, f"ep{i}"), num_frames=4)
        ds = AtariFramePairDataset(tmpdir, split='train', grayscale=False, split_ratio=(1,0,0))
        # Dynamically compute expected pairs
        expected_pairs = sum(max(0, len(glob.glob(os.path.join(ep, '*.png'))) - 1) for ep in ds.episode_dirs)
        assert len(ds) == expected_pairs
        for f0, f1 in ds.pairs:
            assert os.path.dirname(f0) == os.path.dirname(f1)

def test_grayscale_conversion():
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            create_dummy_episode(os.path.join(tmpdir, f"ep{i}"), num_frames=2, grayscale=True)
        ds = AtariFramePairDataset(tmpdir, split='train', grayscale=True, split_ratio=(1,0,0))
        x, y = ds[0]
        assert x.shape[0] == 1 and y.shape[0] == 1
        assert x.max() <= 1.0 and x.min() >= 0.0

def test_normalization():
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            create_dummy_episode(os.path.join(tmpdir, f"ep{i}"), num_frames=2)
        ds = AtariFramePairDataset(tmpdir, split='train', grayscale=False, split_ratio=(1,0,0))
        x, y = ds[0]
        assert x.max() <= 1.0 and x.min() >= 0.0
        assert y.max() <= 1.0 and y.min() >= 0.0

# --- Latent Action Model Tests ---
def test_latent_action_model_forward():
    model = LatentActionVQVAE()
    frame_t = torch.randn(2, 3, 210, 160)
    frame_tp1 = torch.randn(2, 3, 210, 160)
    recon, indices, commitment_loss, codebook_loss = model(frame_t, frame_tp1)
    assert recon.shape == (2, 3, 210, 160)
    assert indices.shape == (2, 5, 7)
    assert commitment_loss.shape == ()
    assert codebook_loss.shape == ()

def test_codebook_usage():
    model = LatentActionVQVAE()
    frame_t = torch.randn(2, 3, 160, 210)
    frame_tp1 = torch.randn(2, 3, 160, 210)
    _, indices, _, _ = model(frame_t, frame_tp1)
    # At least some codes should be used
    unique_codes = torch.unique(indices)
    assert unique_codes.numel() > 1

def test_loss_computation():
    model = LatentActionVQVAE()
    frame_t = torch.randn(2, 3, 210, 160)
    frame_tp1 = torch.randn(2, 3, 210, 160)
    recon, indices, commitment_loss, codebook_loss = model(frame_t, frame_tp1)
    mse = torch.nn.functional.mse_loss(recon, frame_tp1)
    total_loss = mse + commitment_loss + codebook_loss
    assert total_loss.requires_grad

@pytest.mark.parametrize("batch_size", [8, 32])
def test_dataloader_shapes_and_types(batch_size):
    train_loader, val_loader = get_action_latent_dataloaders(batch_size=batch_size)
    batch = next(iter(train_loader))
    actions, latents = batch
    # actions: (batch, 4), latents: (batch, 35)
    assert actions.shape[1] == 4
    assert latents.shape[1] == 35
    assert actions.dtype == torch.float32
    assert latents.dtype in (torch.int64, torch.long)
    assert actions.shape[0] == batch_size
    # Check value ranges
    assert torch.all((latents >= 0) & (latents < 256))

def test_train_val_split():
    train_loader, val_loader = get_action_latent_dataloaders(batch_size=16)
    train_len = len(train_loader.dataset)
    val_len = len(val_loader.dataset)
    total = train_len + val_len
    assert abs(train_len / total - 0.8) < 0.05
    assert abs(val_len / total - 0.2) < 0.05 