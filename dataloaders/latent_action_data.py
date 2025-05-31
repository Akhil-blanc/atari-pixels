import os
import glob
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Tuple, Optional
import json

class AtariFramePairDataset(Dataset):
    def __init__(self, root_dir: str, split: str = 'train', grayscale: bool = True, seed: int = 42, split_ratio=(0.95, 0.05, 0)):
        """
        Args:
            root_dir: Directory containing episode folders with PNG frames.
            split: 'train', 'val', or 'test'.
            grayscale: Whether to convert frames to grayscale.
            seed: Random seed for reproducibility.
            split_ratio: Tuple for train/val/test split.
        """
        self.root_dir = root_dir
        self.grayscale = grayscale
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed
        self.episode_dirs = sorted([os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        random.Random(seed).shuffle(self.episode_dirs)
        n = len(self.episode_dirs)
        n_train = max(1, int(n * split_ratio[0]))
        n_val = max(1, int(n * split_ratio[1]))
        n_test = n - n_train - n_val
        if n_test < 1:
            n_test = 1
            if n_val > 1:
                n_val -= 1
            else:
                n_train -= 1
        if split == 'train':
            self.episode_dirs = self.episode_dirs[:n_train]
        elif split == 'val':
            self.episode_dirs = self.episode_dirs[n_train:n_train+n_val]
        elif split == 'test':
            self.episode_dirs = self.episode_dirs[n_train+n_val:]
        else:
            raise ValueError(f"Unknown split: {split}")
        self.pairs = self._collect_pairs()

    def _collect_pairs(self) -> List[Tuple[str, str]]:
        pairs = []
        for ep_dir in self.episode_dirs:
            frame_files = sorted(glob.glob(os.path.join(ep_dir, '*.png')), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            for i in range(len(frame_files) - 1):
                pairs.append((frame_files[i], frame_files[i+1]))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def _load_frame(self, path: str) -> np.ndarray:
        img = Image.open(path)
        if self.grayscale:
            img = img.convert('L')  # 1 channel
        else:
            img = img.convert('RGB')  # 3 channels
        arr = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        if self.grayscale:
            arr = arr[None, ...]  # (1, H, W)
        else:
            arr = arr.transpose(2, 0, 1)  # (3, H, W)
        return arr

    def __getitem__(self, idx):
        frame_t_path, frame_tp1_path = self.pairs[idx]
        frame_t = self._load_frame(frame_t_path)
        frame_tp1 = self._load_frame(frame_tp1_path)
        return torch.from_numpy(frame_t), torch.from_numpy(frame_tp1)

class ActionLatentPairDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.num_classes = 4
        self.latent_dim = 35
        self.codebook_size = 256

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        action = item['action']
        latent_code = item['latent_code']
        # One-hot encode action
        action_onehot = np.zeros(self.num_classes, dtype=np.float32)
        action_onehot[action] = 1.0
        latent_code = np.array(latent_code, dtype=np.int64)
        return torch.from_numpy(action_onehot), torch.from_numpy(latent_code)

class ActionStateLatentTripleDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.num_classes = 4
        self.latent_dim = 35
        self.codebook_size = 256

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        action = item['action']
        frames = np.array(item['frames'], dtype=np.float32)  # (6, 210, 160), already normalized
        latent_code = item['latent_code']
        # One-hot encode action
        action_onehot = np.zeros(self.num_classes, dtype=np.float32)
        action_onehot[action] = 1.0
        latent_code = np.array(latent_code, dtype=np.int64)
        return (
            torch.from_numpy(action_onehot),
            torch.from_numpy(frames),
            torch.from_numpy(latent_code)
        )

class ActionStateLatentTripleNPZDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.actions = data['actions']  # (N,)
        self.frames = data['frames']    # (N, 6, 210, 160)
        self.latents = data['latents']  # (N, 35)
        self.num_classes = 4
        self.latent_dim = 35
        self.codebook_size = 256

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        action = self.actions[idx]
        frames = self.frames[idx]
        latent_code = self.latents[idx]
        # One-hot encode action
        action_onehot = np.zeros(self.num_classes, dtype=np.float32)
        action_onehot[action] = 1.0
        return (
            torch.from_numpy(action_onehot),
            torch.from_numpy(frames.astype(np.float32)),
            torch.from_numpy(latent_code.astype(np.int64))
        )

def get_action_latent_dataloaders(batch_size=128, num_workers=0, pin_memory=True, seed=42):
    json_path = os.path.join('data', 'actions', 'action_latent_pairs.json')
    dataset = ActionLatentPairDataset(json_path)
    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = n - n_train
    torch.manual_seed(seed)
    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader

def get_action_state_latent_dataloaders(batch_size=128, num_workers=0, pin_memory=True, seed=42):
    npz_path = os.path.join('data', 'actions', 'action_state_latent_triples.npz')
    json_path = os.path.join('data', 'actions', 'action_state_latent_triples.json')
    if os.path.exists(npz_path):
        dataset = ActionStateLatentTripleNPZDataset(npz_path)
    elif os.path.exists(json_path):
        dataset = ActionStateLatentTripleDataset(json_path)
    else:
        raise FileNotFoundError(f"Neither {npz_path} nor {json_path} found.")
    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = n - n_train
    torch.manual_seed(seed)
    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader 