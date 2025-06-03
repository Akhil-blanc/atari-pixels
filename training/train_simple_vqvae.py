import glob
import os
import math
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import make_grid
import wandb
import matplotlib.pyplot as plt
from datetime import datetime

from dataloaders.latent_action_data import AtariFramePairDataset
from models.simple_vqvae import SimplifiedVQVAE  # Import your model

# ----------------------
# Utility Functions
# ----------------------


def psnr(pred, target):
    mse = F.mse_loss(pred, target, reduction='mean').item()
    if mse == 0:
        return 100
    return 20 * math.log10(1.0 / math.sqrt(mse))


def ssim(img1, img2):
    # Simple SSIM for 3-channel images, batchwise
    # For full SSIM, use torchmetrics or skimage
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu1 = img1.mean(dim=[2, 3], keepdim=True)
    mu2 = img2.mean(dim=[2, 3], keepdim=True)
    sigma1 = ((img1 - mu1) ** 2).mean(dim=[2, 3], keepdim=True)
    sigma2 = ((img2 - mu2) ** 2).mean(dim=[2, 3], keepdim=True)
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean(dim=[2, 3], keepdim=True)
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))
    return ssim_map.mean().item()


def codebook_entropy(indices, codebook_size):
    # indices: (B, H, W) or (N,)
    hist = torch.bincount(indices.flatten(), minlength=codebook_size).float()
    prob = hist / hist.sum()
    entropy = -(prob[prob > 0] * prob[prob > 0].log()).sum().item()
    return entropy, hist.cpu().numpy()


def save_checkpoint(state, is_best, checkpoint_dir, step):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"ckpt_{step:06d}.pt")
    torch.save(state, path)
    if is_best:
        torch.save(state, os.path.join(checkpoint_dir, "best.pt"))


def load_model(checkpoint_path, device):
    """Load SimplifiedVQVAE model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SimplifiedVQVAE(
        input_channels=3,
        action_dim=4,
        codebook_size=256,
        latent_dim=128
    )
    model.load_state_dict(checkpoint['model'])
    step = checkpoint.get('step', 0)
    return model, step


def check_latent_encoding(model, val_loader, device, step, output_dir=None, use_wandb=True):
    """Check if the latent space encodes the ball position."""
    os.makedirs(output_dir, exist_ok=True) if output_dir else None

    with torch.no_grad():
        # Get a batch containing the ball
        frame_t, frame_tp1 = next(iter(val_loader))
        frame_t, frame_tp1 = frame_t.to(device), frame_tp1.to(device)

        # For SimplifiedVQVAE, we need to create action vectors
        # Using dummy actions for now - in real training you'd have actual actions
        batch_size = frame_t.size(0)
        dummy_action = torch.randn(batch_size, 4).to(device)

        # 1. Identify ball position by frame difference
        frame_diff = torch.abs(frame_tp1 - frame_t).sum(1)
        ball_mask = (frame_diff > 0.05).float()

        # 2. Get latent representation using your model
        z = model.encoder(frame_t, dummy_action)

        # 3. Visualize to compare ball position and latent activations
        for i in range(min(4, frame_t.size(0))):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Original frame with ball
            axes[0].imshow(frame_tp1[i].permute(1, 2, 0).cpu().numpy())
            axes[0].set_title('Frame with Ball')

            # Ball mask (where movement is detected)
            axes[1].imshow(ball_mask[i].cpu().numpy(), cmap='hot')
            axes[1].set_title('Ball Position')

            # Latent space activation (mean across channels)
            latent_vis = z[i].mean(0).cpu().numpy()
            latent_vis = (latent_vis - latent_vis.min()) / \
                (latent_vis.max() - latent_vis.min() + 1e-8)
            axes[2].imshow(latent_vis, cmap='viridis')
            axes[2].set_title('Latent Space Activation')

            plt.tight_layout()

            # Save locally if output_dir provided
            if output_dir:
                filename = os.path.join(
                    output_dir, f'ball_encoding_ex{i}_step{step}.png')
                fig.savefig(filename)
                print(f"Saved {filename}")

            # Log to wandb if requested
            if use_wandb:
                wandb.log({f'ball_encoding_{i}': wandb.Image(fig)}, step=step)

            plt.close(fig)


def test_encoder_decoder(model, val_loader, device, step, output_dir=None, use_wandb=True):
    """Test encoder/decoder by enhancing ball signal in latent space."""
    os.makedirs(output_dir, exist_ok=True) if output_dir else None

    with torch.no_grad():
        # Get frames with ball movement
        frame_t, frame_tp1 = next(iter(val_loader))
        frame_t, frame_tp1 = frame_t.to(device), frame_tp1.to(device)

        # Create dummy actions
        batch_size = frame_t.size(0)
        dummy_action = torch.randn(batch_size, 4).to(device)

        # 1. Normal forward pass
        recon, indices, _, _ = model(frame_t, dummy_action)

        # 2. Get latent representation
        z = model.encoder(frame_t, dummy_action)

        # 3. Find ball position
        frame_diff = torch.abs(frame_tp1 - frame_t).sum(1, keepdim=True)
        ball_mask = (frame_diff > 0.05).float()

        # 4. Upsample ball mask to latent space dimensions
        ball_mask_latent = F.interpolate(
            ball_mask, size=z.shape[2:], mode='nearest')

        # 5. Enhance latent representation at ball position
        z_enhanced = z.clone()
        z_enhanced = z_enhanced + 20.0 * ball_mask_latent.expand_as(z_enhanced)

        # 6. Check code changes
        quantized_original, indices_original, _, _ = model.vq(z)
        quantized_enhanced, indices_enhanced, _, _ = model.vq(z_enhanced)
        
        code_changes = (indices_original != indices_enhanced).float()

        # 7. Decode enhanced latent
        recon_enhanced = model.decoder(quantized_enhanced)

        # 8. Compare results
        for i in range(min(4, frame_t.size(0))):
            fig, axes = plt.subplots(1, 5, figsize=(25, 5))

            def normalize_for_display(img_tensor):
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                img_np = np.clip(img_np, 0, 1)
                return img_np

            # Target frame (using frame_tp1 as target)
            axes[0].imshow(normalize_for_display(frame_tp1[i]))
            axes[0].set_title('Target Frame')

            # Normal reconstruction
            axes[1].imshow(normalize_for_display(recon[i]))
            axes[1].set_title('Normal Reconstruction')

            # Enhanced latent reconstruction
            axes[2].imshow(normalize_for_display(recon_enhanced[i]))
            axes[2].set_title('Enhanced Reconstruction')

            # Difference between normal and enhanced
            diff = torch.abs(recon_enhanced[i] - recon[i]).sum(0).cpu().numpy()
            diff_max = diff.max()
            if diff_max > 1e-6:
                diff = diff / diff_max
            else:
                diff = np.zeros_like(diff)
            axes[3].imshow(diff, cmap='hot')
            axes[3].set_title('Reconstruction Difference')

            # Code changes visualization
            axes[4].imshow(code_changes[i].cpu().numpy(), cmap='hot')
            axes[4].set_title('Codebook Changes')

            plt.tight_layout()

            if output_dir:
                filename = os.path.join(
                    output_dir, f'latent_manipulation_ex{i}_step{step}.png')
                fig.savefig(filename)
                print(f"Saved {filename}")

            if use_wandb:
                wandb.log(
                    {f'latent_manipulation_{i}': wandb.Image(fig)}, step=step)

            plt.close(fig)


def analyze_checkpoint(checkpoint_path, data_dir, output_dir, device=None, grayscale=False):
    """Load a model from checkpoint and run diagnostic visualizations."""
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    model, step = load_model(checkpoint_path, device)
    model = model.to(device)
    model.eval()

    print(f"Loaded checkpoint from step {step}")

    # Setup data
    val_set = AtariFramePairDataset(data_dir, split='val', grayscale=grayscale)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=True, num_workers=2)

    # Run diagnostics
    print("Running latent encoding analysis...")
    check_latent_encoding(model, val_loader, device, step,
                          output_dir=os.path.join(output_dir, 'latent_encoding'),
                          use_wandb=False)

    print("Running encoder-decoder test...")
    test_encoder_decoder(model, val_loader, device, step,
                         output_dir=os.path.join(output_dir, 'encoder_decoder_test'),
                         use_wandb=False)

    print(f"Analysis complete. Results saved to {output_dir}")


# ----------------------
# Training Function
# ----------------------

def train(args):
    # Device setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
        compile_mode = "default"
        use_amp = True
        pin_memory = True
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        compile_mode = "eager"
        use_amp = False
        pin_memory = False
    else:
        device = torch.device('cpu')
        compile_mode = None
        use_amp = False
        pin_memory = False
    print(f"Using device: {device}")

    # WandB setup
    wandb.init(project="latent-action", config=vars(args))

    # Data
    train_set = AtariFramePairDataset(
        args.data_dir, split='train', grayscale=args.grayscale)
    val_set = AtariFramePairDataset(
        args.data_dir, split='val', grayscale=args.grayscale)
    test_set = AtariFramePairDataset(
        args.data_dir, split='test', grayscale=args.grayscale)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)

    # Print data statistics
    def get_stats(dataset, name):
        n_episodes = len(dataset.episode_dirs)
        n_pairs = len(dataset)
        sample = dataset[0]
        shape = sample[0].shape
        print(f"{name}: {n_episodes} episodes, {n_pairs} pairs, frame shape: {shape}")
    print("--- Data Statistics ---")
    get_stats(train_set, "Train")
    get_stats(val_set, "Val")
    get_stats(test_set, "Test")
    print("-----------------------")

    # Model - Using SimplifiedVQVAE
    input_channels = 1 if args.grayscale else 3
    model = SimplifiedVQVAE(
        input_channels=input_channels,
        action_dim=args.action_dim,
        codebook_size=args.codebook_size,
        latent_dim=args.latent_dim
    )
    model = model.to(device)

    # Conditionally compile the model
    if compile_mode is not None:
        try:
            print(f"Compiling model with {compile_mode} backend...")
            model = torch.compile(model, backend=compile_mode)
            print("Model compilation successful")
        except Exception as e:
            print(f"Model compilation failed: {e}")
            print("Continuing without compilation")

    # Optimizer & Scheduler
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=args.max_iters, eta_min=args.lr_min)

    scaler = torch.amp.GradScaler() if use_amp and device.type == 'cuda' else None

    # Training state
    best_val_loss = float('inf')
    global_step = 0
    grad_accum = args.grad_accum
    checkpoint_dir = os.path.join(args.checkpoint_dir, 'latent_action')

    model.train()
    optimizer.zero_grad()
    data_iter = iter(train_loader)
    pbar = tqdm(range(args.max_iters), desc="Training", dynamic_ncols=True)
    
    while global_step < args.max_iters:
        try:
            frame_t, frame_tp1 = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            frame_t, frame_tp1 = next(data_iter)
        frame_t, frame_tp1 = frame_t.to(device), frame_tp1.to(device)
        
        # Create dummy actions - in real training you'd have actual actions
        batch_size = frame_t.size(0)
        dummy_action = torch.randn(batch_size, args.action_dim).to(device)
        
        with torch.amp.autocast(device.type, enabled=(scaler is not None)):
            # Use your SimplifiedVQVAE model
            predicted_frame, indices, commit_loss, codebook_loss = model(frame_t, dummy_action)
            
            # Reconstruction loss - compare predicted next frame with actual next frame
            frame_diff = torch.abs(frame_tp1 - frame_t)
            motion_weight = 1.0 + 10.0 * (frame_diff.sum(dim=1, keepdim=True) > 0.05).float()
            rec_loss = (motion_weight * (predicted_frame - frame_tp1)**2).mean()

            # Entropy regularization
            entropy, hist = codebook_entropy(indices, model.vq.num_embeddings)
            entropy_reg = -args.entropy_weight * entropy
            loss = rec_loss + commit_loss + codebook_loss + entropy_reg
            
        if scaler is not None:
            scaler.scale(loss / grad_accum).backward()
        else:
            (loss / grad_accum).backward()
            
        if (global_step + 1) % grad_accum == 0:
            if args.grad_clip > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
        # Logging
        if (global_step + 1) % args.log_interval == 0:
            with torch.no_grad():
                psnr_val = psnr(predicted_frame, frame_tp1)
                ssim_val = ssim(predicted_frame, frame_tp1)
                l1 = F.l1_loss(predicted_frame, frame_tp1).item()
                l2 = F.mse_loss(predicted_frame, frame_tp1).item()
                wandb.log({
                    'loss/total': loss.item(),
                    'loss/rec': rec_loss.item(),
                    'loss/commit': commit_loss.item(),
                    'loss/codebook': codebook_loss.item(),
                    'loss/entropy_reg': entropy_reg,
                    'metric/psnr': psnr_val,
                    'metric/ssim': ssim_val,
                    'metric/l1': l1,
                    'metric/l2': l2,
                    'codebook/entropy': entropy,
                    'codebook/hist': wandb.Histogram(hist),
                    'lr': scheduler.get_last_lr()[0],
                    'step': global_step + 1
                }, step=global_step + 1)
                
        # Save reconstructions
        if (global_step + 1) % args.recon_interval == 0:
            with torch.no_grad():
                grid = make_grid(
                    torch.cat([frame_t, frame_tp1, predicted_frame], dim=0), nrow=args.batch_size)
                wandb.log({'reconstructions': [wandb.Image(
                    grid, caption=f'Step {global_step+1}')]}, step=global_step+1)
                    
        # Checkpoint
        if (global_step + 1) % args.ckpt_interval == 0:
            save_checkpoint({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'step': global_step + 1
            }, is_best=False, checkpoint_dir=checkpoint_dir, step=global_step + 1)

        # Periodic diagnostics for latent space:
        if (global_step + 1) % args.diagnostic_interval == 0:
            print(f"Running diagnostics at step {global_step+1}")
            check_latent_encoding(
                model, val_loader, device, global_step + 1,
                output_dir=None,
                use_wandb=True
            )
            test_encoder_decoder(
                model, val_loader, device, global_step + 1,
                output_dir=None,
                use_wandb=True
            )
            
        # Validation
        if (global_step + 1) % args.val_interval == 0:
            val_loss = evaluate(model, val_loader, device, scaler, args.action_dim, pbar_desc='Val')
            wandb.log({'val/loss': val_loss}, step=global_step+1)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'step': global_step + 1
                }, is_best=True, checkpoint_dir=checkpoint_dir, step=global_step + 1)
                
        global_step += 1
        pbar.update(1)
        
    pbar.close()
    
    # Final test evaluation
    test_loss = evaluate(model, test_loader, device, scaler, args.action_dim, pbar_desc='Test')
    wandb.log({'test/loss': test_loss}, step=global_step)
    print(f"Test loss: {test_loss}")


def evaluate(model, loader, device, scaler, action_dim, pbar_desc=None):
    model.eval()
    losses = []
    with torch.no_grad():
        if pbar_desc:
            loader_iter = tqdm(loader, desc=pbar_desc, dynamic_ncols=True)
        else:
            loader_iter = loader
        for frame_t, frame_tp1 in loader_iter:
            frame_t, frame_tp1 = frame_t.to(device), frame_tp1.to(device)
            
            # Create dummy actions
            batch_size = frame_t.size(0)
            dummy_action = torch.randn(batch_size, action_dim).to(device)
            
            with torch.amp.autocast(device.type, enabled=(scaler is not None)):
                predicted_frame, indices, commit_loss, codebook_loss = model(frame_t, dummy_action)
                rec_loss = F.mse_loss(predicted_frame, frame_tp1)
                loss = rec_loss + commit_loss + codebook_loss
            losses.append(loss.item())
    model.train()
    return np.mean(losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='bulk_videos',
                        help='Directory with episode folders')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_iters', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lr_min', type=float, default=1e-4)
    parser.add_argument('--grad_accum', type=int, default=1)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--grayscale', action='store_true', default=False)
    parser.add_argument('--entropy_weight', type=float, default=0.1)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--recon_interval', type=int, default=1000)
    parser.add_argument('--ckpt_interval', type=int, default=5000)
    parser.add_argument('--val_interval', type=int, default=1000)
    parser.add_argument('--codebook_size', type=int, default=256,
                        help='Size of the VQ codebook (number of latent codes)')
    parser.add_argument('--action_dim', type=int, default=4,
                        help='Dimension of action vector')
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='Dimension of latent space')

    parser.add_argument('--analyze', action='store_true',
                        help='Analyze latest checkpoint')
    parser.add_argument('--analysis_output_dir', type=str,
                        default='analysis', help='Output directory for analysis')
    parser.add_argument('--diagnostic_interval', type=int,
                        default=50000, help='Steps between diagnostic visualizations')
    args = parser.parse_args()

    if args.analyze:
        # Find the latest checkpoint
        checkpoint_dir = os.path.join(args.checkpoint_dir, 'latent_action')
        checkpoints = sorted(
            glob.glob(os.path.join(checkpoint_dir, 'ckpt_*.pt')))
        if not checkpoints:
            checkpoint_path = os.path.join(checkpoint_dir, 'best.pt')
            if not os.path.exists(checkpoint_path):
                print("No checkpoints found!")
                exit(1)
        else:
            checkpoint_path = checkpoints[-1]
        # Run analysis
        analyze_checkpoint(
            checkpoint_path=checkpoint_path,
            data_dir=args.data_dir,
            output_dir=args.analysis_output_dir,
            device=None
        )
        exit(0)

    train(args)