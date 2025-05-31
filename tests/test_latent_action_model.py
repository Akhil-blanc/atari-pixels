import pytest
import torch
from models.latent_action_model import ActionToLatentMLP

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def test_model_forward_shape_and_dtype():
    device = get_device()
    model = ActionToLatentMLP().to(device)
    batch_size = 16
    x = torch.eye(4)[torch.randint(0, 4, (batch_size,))].to(device)  # (batch, 4)
    logits = model(x)
    assert logits.shape == (batch_size, 35, 256)
    assert logits.dtype == torch.float32

def test_model_on_all_devices():
    for device_str in ['cpu', 'cuda', 'mps']:
        if device_str == 'cuda' and not torch.cuda.is_available():
            continue
        if device_str == 'mps' and not torch.backends.mps.is_available():
            continue
        device = torch.device(device_str)
        model = ActionToLatentMLP().to(device)
        x = torch.eye(4)[torch.randint(0, 4, (8,))].to(device)
        logits = model(x)
        assert str(logits.device).replace(':0', '') == str(device).replace(':0', '')

def test_temperature_sampling():
    device = get_device()
    model = ActionToLatentMLP().to(device)
    x = torch.eye(4)[torch.randint(0, 4, (4,))].to(device)
    logits = model(x)
    for temp in [0.1, 1.0, 2.0]:
        latents = model.sample_latents(logits, temperature=temp)
        assert latents.shape == (4, 35)
        assert latents.dtype in (torch.int64, torch.long)
        assert torch.all((latents >= 0) & (latents < 256)) 