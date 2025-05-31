import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import argparse
from models.latent_action_model import ActionToLatentMLP, ActionStateToLatentMLP
from dataloaders.latent_action_data import get_action_latent_dataloaders, get_action_state_latent_dataloaders

# Hyperparameters
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 250
GRAD_CLIP = 1.0
CHECKPOINT_DIR = 'checkpoints/latent_action/'
MODEL_NAME = 'action_to_latent_best.pt'
SEED = 42

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, with_frames=False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch in tqdm(loader, desc='Train', leave=False):
        if with_frames:
            actions, frames, latents = batch
            actions = actions.to(device)
            frames = frames.to(device)
        else:
            actions, latents = batch
            actions = actions.to(device)
            frames = None
        latents = latents.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            if with_frames:
                logits = model(actions, frames)
            else:
                logits = model(actions)
            loss = criterion(logits.view(-1, 256), latents.view(-1))
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        running_loss += loss.item() * actions.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == latents).sum().item()
        total += latents.numel()
    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def eval_one_epoch(model, loader, criterion, device, with_frames=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc='Val', leave=False):
            if with_frames:
                actions, frames, latents = batch
                actions = actions.to(device)
                frames = frames.to(device)
            else:
                actions, latents = batch
                actions = actions.to(device)
                frames = None
            latents = latents.to(device)
            if with_frames:
                logits = model(actions, frames)
            else:
                logits = model(actions)
            loss = criterion(logits.view(-1, 256), latents.view(-1))
            running_loss += loss.item() * actions.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == latents).sum().item()
            total += latents.numel()
    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--with_frames', action='store_true', help='Use action+state (frames) model and data')
    args = parser.parse_args()

    set_seed(SEED)
    device = get_device()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # wandb init
    wandb.init(project='atari-action-to-latent', config={
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'epochs': EPOCHS,
        'grad_clip': GRAD_CLIP,
        'seed': SEED,
        'with_frames': args.with_frames
    })

    # Data
    if args.with_frames:
        train_loader, val_loader = get_action_state_latent_dataloaders(batch_size=BATCH_SIZE)
        model = ActionStateToLatentMLP()
        model_name = 'action_state_to_latent_best.pt'
    else:
        train_loader, val_loader = get_action_latent_dataloaders(batch_size=BATCH_SIZE)
        model = ActionToLatentMLP()
        model_name = 'action_to_latent_best.pt'
    model = model.to(device)
    if device.type == 'cuda':
        model = torch.compile(model)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    best_val_acc = 0.0
    best_ckpt_path = os.path.join(CHECKPOINT_DIR, model_name)

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, with_frames=args.with_frames)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device, with_frames=args.with_frames)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            try:
                torch.save({'model_state_dict': model.state_dict(),
                            'val_acc': val_acc,
                            'epoch': epoch}, best_ckpt_path)
                wandb.run.summary['best_val_acc'] = best_val_acc
                wandb.run.summary['best_ckpt_path'] = best_ckpt_path
                print(f"[Checkpoint] Saved best model at epoch {epoch} with val acc {val_acc:.4f}")
            except Exception as e:
                print(f"[Warning] Failed to save checkpoint: {e}")
    wandb.finish()
    print(f"Training complete. Best val acc: {best_val_acc:.4f}. Model saved to {best_ckpt_path}")

if __name__ == '__main__':
    main() 