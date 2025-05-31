import torch
import numpy as np
from PIL import Image
import imageio
import random
import argparse
from models.latent_action_model import load_latent_action_model, ActionToLatentMLP, ActionStateToLatentMLP

# --- Device selection ---
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

device = get_device()
print(f"[DEBUG] Using device: {device}")

# --- Load initial frame ---
init_img = Image.open('data/0.png').convert('RGB')
init_frame = np.array(init_img, dtype=np.float32) / 255.0  # (210, 160, 3), [0,1]
print(f"[DEBUG] Loaded initial frame shape (np): {init_frame.shape}")
init_frame = torch.from_numpy(init_frame).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 3, 210, 160)
print(f"[DEBUG] Initial frame tensor shape: {init_frame.shape}, dtype: {init_frame.dtype}")



# --- Load models ---
world_model, _ = load_latent_action_model('checkpoints/latent_action/best.pt', device)
world_model.to(device)
world_model.eval()
if device.type == 'cuda':
    world_model = torch.compile(world_model)
print(f"[DEBUG] Loaded world model. VQ codebook shape: {world_model.vq.embeddings.weight.shape}, device: {next(world_model.parameters()).device}")






def action_to_onehot(action_idx, device):
    onehot = torch.zeros(1, 4, device=device)
    onehot[0, action_idx] = 1.0
    return onehot

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--with_frames', action='store_true', help='Use action+state (frames) model for action-to-latent')
    parser.add_argument('--steps', type=int, default=100, help='Number of steps to generate')
    parser.add_argument('--temperature', type=float, default=0.001, help='Sampling temperature for latent prediction')
    args = parser.parse_args()

    # --- Load action-to-latent model ---
    if args.with_frames:
        print("[INFO] Using ActionStateToLatentMLP (action + last 2 frames)")
        model = ActionStateToLatentMLP().to(device)
        ckpt = torch.load('checkpoints/latent_action/action_state_to_latent_best.pt', map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        print("[INFO] Using ActionToLatentMLP (action only)")
        model = ActionToLatentMLP().to(device)
        ckpt = torch.load('checkpoints/latent_action/action_to_latent_best.pt', map_location=device)
        # Fix state dict keys by removing the '_orig_mod.' prefix if present
        fixed_state_dict = {}
        for k, v in ckpt['model_state_dict'].items():
            if k.startswith('_orig_mod.'):
                fixed_state_dict[k.replace('_orig_mod.', '')] = v
            else:
                fixed_state_dict[k] = v
        model.load_state_dict(fixed_state_dict)
    model.eval()
    if device.type == 'cuda':
        model = torch.compile(model)

    action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
    frames = []
    current_frame = init_frame.clone()
    current_frame_for_frame = (current_frame.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    frames.append(current_frame_for_frame)

    if args.with_frames:
        # Frame buffer: last 2 frames (each (1, 3, 210, 160))
        last2_frames = [current_frame.clone(), current_frame.clone()]
    all_latent_indices = []
    
    for step in range(args.steps):
        print(f"\n[DEBUG] === Step {step} ===")
        action_idx = random.randint(0, 3)
        print(f"[DEBUG] Random action: {action_idx} ({action_names[action_idx]})")
        with torch.no_grad():
            if args.with_frames:
                # Stack last 2 frames: (1, 6, 210, 160)
                stacked_frames = torch.cat([last2_frames[0], last2_frames[1]], dim=1)  # (1, 6, 210, 160)
                onehot = action_to_onehot(action_idx, device)
                logits = model(onehot, stacked_frames)
                indices = model.sample_latents(logits, temperature=args.temperature)
            else:
                onehot = action_to_onehot(action_idx, device)
                logits = model(onehot)
                indices = model.sample_latents(logits, temperature=args.temperature)
            all_latent_indices.append(indices.cpu().numpy())
            # Additional debugging info
            if step > 0:
                # Count how many latent positions changed from previous step
                changes = (all_latent_indices[-1] != all_latent_indices[-2]).sum()
                print(f"Step {step}: {changes} latent positions changed")
                
            indices = indices.view(1, 5, 7)
            embeddings = world_model.vq.embeddings
            indices = indices.to(embeddings.weight.device)
            quantized = embeddings(indices)  # (1, 5, 7, 128)
            quantized = quantized.permute(0, 3, 1, 2).contiguous()  # (1, 128, 5, 7)
            # Frame for decoder
            frame_in = current_frame.permute(0, 1, 3, 2)  # (1, 3, 210, 160) -> (1, 3, 160, 210)
            quantized = quantized.to(next(world_model.parameters()).device)
            frame_in = frame_in.to(next(world_model.parameters()).device)
            next_frame = world_model.decoder(quantized, frame_in)  # (1, 3, 160, 210)
            next_frame = next_frame.permute(0, 1, 3, 2)  # (1, 3, 210, 160)
            frame_np = (next_frame.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
            frames.append(frame_np)
            current_frame = next_frame.clone()
            if args.with_frames:
                last2_frames[0] = last2_frames[1]
                last2_frames[1] = current_frame.clone()

    print(f"[DEBUG] Saving video with {len(frames)} frames...")
    imageio.mimsave('data/neural_random_game.gif', frames, fps=3)
    print('Saved video to data/neural_random_game.gif')
    visualize_latent_changes(
    world_model=world_model, 
    action_to_latent=model,
    init_frame=init_frame, 
    device=device,
    action_names=['NOOP', 'FIRE', 'RIGHT', 'LEFT'],
    num_steps=40,
    with_frames=args.with_frames  # Pass this parameter
    )
    # visualize_ground_truth_comparison(
    #     world_model=world_model, 
    #     action_to_latent=model,
    #     device=device,
    #     frames_dir='comparison_frames', 
    #     num_steps=20, 
    #     temperature=0.01)

def visualize_ground_truth_comparison(
        world_model, action_to_latent, device,
        frames_dir='comparison_frames', num_steps=20, temperature=0.1,
        gt_frames_path='bulk_videos/bulk_random_agent_1'):
    """
    Visualize a comparison between:
    1. Ground truth frames and their encoded latents
    2. Predicted latents and their decoded frames
    
    This helps separate encoder/decoder errors from latent prediction errors.
    """
    import os
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    import imageio
    
    os.makedirs(frames_dir, exist_ok=True)
    all_viz_frames = []
    
    # Load ground truth frames
    print(f"Loading ground truth frames from {gt_frames_path}")
    gt_frame_paths = sorted([os.path.join(gt_frames_path, f) for f in os.listdir(gt_frames_path) 
                          if f.endswith('.png') and f[0].isdigit()],
                         key=lambda x: int(os.path.basename(x).split('.')[0]))
    
    # Ensure we have enough frames
    if len(gt_frame_paths) < num_steps + 2:
        print(f"Warning: Only {len(gt_frame_paths)} ground truth frames available, need at least {num_steps+2}")
        num_steps = len(gt_frame_paths) - 2
    
    # Load action sequence (assuming these actions generated the ground truth frames)
    # If you don't have this, you can use random actions or infer them from frames
    action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
    actions = [1]  # Start with FIRE, then alternate RIGHT/LEFT
    for i in range(1, num_steps):
        actions.append(2 if i % 2 == 0 else 3)  # Alternate RIGHT/LEFT
    
    # Helper functions
    def load_frame(path):
        img = Image.open(path).convert('RGB')
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        return tensor
    
    def action_to_onehot(action_idx):
        onehot = torch.zeros(1, 4, device=device)
        onehot[0, action_idx] = 1.0
        return onehot
    
    def get_gt_latent(frame_t, frame_tp1):
        """Extract ground truth latent from consecutive frames"""
        # Permute for encoder
        frame_t_perm = frame_t.permute(0, 1, 3, 2)
        frame_tp1_perm = frame_tp1.permute(0, 1, 3, 2)
        
        # Concatenate for encoder
        x = torch.cat([frame_t_perm, frame_tp1_perm], dim=1)
        
        # Get latent
        with torch.no_grad():
            z = world_model.encoder(x)
            _, indices, _, _ = world_model.vq(z)
        
        return indices.view(1, 5, 7), z
    
    # Initial setup
    current_pred_frame = load_frame(gt_frame_paths[0])
    frame_buffer = [load_frame(gt_frame_paths[0]), load_frame(gt_frame_paths[0])]
    
    for step in range(num_steps):
        print(f"\n=== Step {step} ===")
        action_idx = actions[min(step, len(actions)-1)]
        print(f"Action: {action_names[action_idx]}")
        
        # Ground truth frames for this step
        gt_frame_t = load_frame(gt_frame_paths[step])
        gt_frame_tp1 = load_frame(gt_frame_paths[step+1])
        
        # Get ground truth latent from consecutive GT frames
        gt_latent_indices, gt_z = get_gt_latent(gt_frame_t, gt_frame_tp1)
        
        # Get predicted latent from action+state model
        with torch.no_grad():
            # Stack last 2 frames for frame buffer
            stacked_frames = torch.cat(frame_buffer, dim=1)
            onehot = action_to_onehot(action_idx)
            logits = action_to_latent(onehot, stacked_frames)
            pred_latent_indices = action_to_latent.sample_latents(logits, temperature=temperature)
            
        # Decoder for both latents
        pred_latent_indices_reshaped = pred_latent_indices.view(1, 5, 7)
        embeddings = world_model.vq.embeddings
        
        # Ground truth latent -> decoded frame
        gt_latent_indices = gt_latent_indices.to(embeddings.weight.device)
        gt_quantized = embeddings(gt_latent_indices)
        gt_quantized = gt_quantized.permute(0, 3, 1, 2).contiguous()
        
        # Predicted latent -> decoded frame
        pred_latent_indices = pred_latent_indices_reshaped.to(embeddings.weight.device)
        pred_quantized = embeddings(pred_latent_indices)
        pred_quantized = pred_quantized.permute(0, 3, 1, 2).contiguous()
        
        # Decode both
        with torch.no_grad():
            frame_in = gt_frame_t.permute(0, 1, 3, 2)
            
            # Ground truth latent -> decoded frame
            gt_decoded = world_model.decoder(gt_quantized, frame_in)
            gt_decoded = gt_decoded.permute(0, 1, 3, 2)
            
            # Predicted latent -> decoded frame
            pred_decoded = world_model.decoder(pred_quantized, frame_in)
            pred_decoded = pred_decoded.permute(0, 1, 3, 2)
        
        # Update frame buffer for next iteration
        frame_buffer[0] = frame_buffer[1].clone()
        frame_buffer[1] = pred_decoded.clone()
        
        # Convert to numpy for visualization
        gt_frame_t_np = gt_frame_t.squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)
        gt_frame_tp1_np = gt_frame_tp1.squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)
        gt_decoded_np = gt_decoded.squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)
        pred_decoded_np = pred_decoded.squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Row 1: Ground truth
        axes[0, 0].imshow(gt_frame_t_np)
        axes[0, 0].set_title('Ground Truth Frame t')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(gt_frame_tp1_np)
        axes[0, 1].set_title('Ground Truth Frame t+1')
        axes[0, 1].axis('off')
        
        # Show ground truth latent
        gt_latent_grid = gt_latent_indices.cpu().numpy().reshape(5, 7)
        im1 = axes[0, 2].imshow(gt_latent_grid, cmap='viridis')
        axes[0, 2].set_title('Ground Truth Latent')
        plt.colorbar(im1, ax=axes[0, 2])
        
        # Row 2: Predictions
        axes[1, 0].imshow(gt_decoded_np)
        axes[1, 0].set_title('Decoded from GT Latent')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(pred_decoded_np)
        axes[1, 1].set_title('Decoded from Predicted Latent')
        axes[1, 1].axis('off')
        
        # Show predicted latent and differences
        pred_latent_grid = pred_latent_indices.cpu().numpy().reshape(5, 7)
        im2 = axes[1, 2].imshow(pred_latent_grid, cmap='viridis')
        axes[1, 2].set_title('Predicted Latent')
        plt.colorbar(im2, ax=axes[1, 2])
        
        # Add info about differences
        latent_diff = (gt_latent_grid != pred_latent_grid).sum()
        pixel_diff_gt = np.mean(np.abs(gt_frame_tp1_np - gt_decoded_np))
        pixel_diff_pred = np.mean(np.abs(gt_frame_tp1_np - pred_decoded_np))
        
        plt.suptitle(f'Step {step}: Action={action_names[action_idx]}\n'
                    f'Latent differences: {latent_diff}/35 positions\n'
                    f'Pixel error (GT decoder): {pixel_diff_gt:.4f}  |  '
                    f'Pixel error (Pred decoder): {pixel_diff_pred:.4f}', 
                    fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(frames_dir, f'comparison_{step:03d}.png'))
        all_viz_frames.append(imageio.imread(os.path.join(frames_dir, f'comparison_{step:03d}.png')))
        plt.close()
    
    # Create GIF
    print("Creating GIF of comparison...")
    imageio.mimsave(os.path.join(frames_dir, 'comparison.gif'), all_viz_frames, fps=2)
    print(f"GIF saved to {os.path.join(frames_dir, 'comparison.gif')}")

def visualize_latent_changes(
        world_model, action_to_latent, 
        init_frame, device, action_names=['NOOP', 'FIRE', 'RIGHT', 'LEFT'],
        frames_dir='debug_frames', make_gif=True, num_steps=50, temperature=0.001,
        with_frames=False):
    """Create visualizations showing latent changes alongside game frames"""
    
    import random
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import imageio
    import torch
    
    os.makedirs(frames_dir, exist_ok=True)
    
    # Initialize storage for previous values
    prev_latent = None
    prev_frame = None
    all_frames = []  # For GIF creation
    current_frame = init_frame.clone()
    
    # For frame-based model, initialize frame history
    if with_frames:
        last2_frames = [current_frame.clone(), current_frame.clone()]
    
    # Create a colorful discrete colormap for latent codes
    cmap = plt.cm.get_cmap('viridis', 16)
    
    # Helper function
    def action_to_onehot(action_idx):
        onehot = torch.zeros(1, 4, device=device)
        onehot[0, action_idx] = 1.0
        return onehot
    
    # Helper function to map action to latent embedding
    def map_action_to_latent_embedding(action_idx):
        onehot = action_to_onehot(action_idx)
        with torch.no_grad():
            if with_frames:
                # Stack last 2 frames
                stacked_frames = torch.cat([last2_frames[0], last2_frames[1]], dim=1)
                logits = action_to_latent(onehot, stacked_frames)
            else:
                logits = action_to_latent(onehot)
            indices = action_to_latent.sample_latents(logits, temperature=temperature)
        indices = indices.view(1, 5, 7)
        embeddings = world_model.vq.embeddings
        indices = indices.to(embeddings.weight.device)
        quantized = embeddings(indices)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return quantized, indices
    
    for step in range(num_steps):
        print(f"\n[DEBUG] === Step {step} ===")
        action_idx = random.randint(0, 3)
        print(f"[DEBUG] Random action: {action_idx} ({action_names[action_idx]})")
        
        # Generate the next frame using action
        latent, latent_indices = map_action_to_latent_embedding(action_idx)
        
        # Reshape latent to 5x7 grid for visualization
        latent_grid = latent_indices.view(5, 7).cpu().numpy()
        
        with torch.no_grad():
            frame_in = current_frame.permute(0, 1, 3, 2)  # (1, 3, 210, 160) -> (1, 3, 160, 210)
            latent = latent.to(device)
            frame_in = frame_in.to(device)
            next_frame = world_model.decoder(latent, frame_in)
            next_frame = next_frame.permute(0, 1, 3, 2)
        
        # Convert frames to numpy for visualization
        current_np = current_frame.squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)
        next_np = next_frame.squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)
        
        # Create visualization with 2x3 grid
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Top row: Frames
        if with_frames:
            # Frame t-2 (Input to encoder 1)
            frame_t2_np = last2_frames[0].squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)
            axes[0, 0].imshow(frame_t2_np)
            axes[0, 0].set_title(f'Frame t-2 (Encoder Input 1)')
            axes[0, 0].axis('off')
            
            # Frame t-1 (Input to encoder 2)
            frame_t1_np = last2_frames[1].squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)
            axes[0, 1].imshow(frame_t1_np)
            axes[0, 1].set_title(f'Frame t-1 (Encoder Input 2)')
            axes[0, 1].axis('off')
        else:
            # For action-only model, show placeholder
            axes[0, 0].imshow(np.zeros_like(current_np))
            axes[0, 0].set_title(f'Action-only model (no frames)')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(np.zeros_like(current_np))
            axes[0, 1].set_title(f'Action-only model (no frames)')
            axes[0, 1].axis('off')
        
        # Frame t (Input to decoder)
        axes[0, 2].imshow(current_np)
        axes[0, 2].set_title(f'Frame t (Decoder Input)\nAction: {action_names[action_idx]}')
        axes[0, 2].axis('off')
        
        # Bottom row: Generated frame, latent codes, and changes
        # Generated frame t+1
        axes[1, 0].imshow(next_np)
        axes[1, 0].set_title(f'Generated Frame t+1')
        axes[1, 0].axis('off')
        
        # Latent codes
        im = axes[1, 1].imshow(latent_grid, cmap=cmap, vmin=0, vmax=15)
        axes[1, 1].set_title('Latent Codes (5x7)')
        for i in range(5):
            for j in range(7):
                axes[1, 1].text(j, i, f'{latent_grid[i,j]}', ha='center', va='center', 
                             color='white' if latent_grid[i,j] > 7 else 'black', fontsize=8)
        
        # Latent changes
        if prev_latent is not None:
            change_mask = (latent_grid != prev_latent).astype(float)
            axes[1, 2].imshow(change_mask, cmap='Reds')
            axes[1, 2].set_title(f'Changed Positions: {int(change_mask.sum())}/35')
            
            # Add values to show what changed from->to
            for i in range(5):
                for j in range(7):
                    if change_mask[i,j] > 0:
                        axes[1, 2].text(j, i, f'{prev_latent[i,j]}->{latent_grid[i,j]}', 
                                     ha='center', va='center', color='black', fontsize=7)
        else:
            axes[1, 2].imshow(np.ones((5, 7)), cmap='Greys')
            axes[1, 2].set_title('Initial Frame (No Changes)')
        
        # Add colorbar for latent codes
        cbar = plt.colorbar(im, ax=axes[1, 1])
        cbar.set_label('Latent Code Value')
        
        plt.suptitle(f'Step {step}: Action={action_names[action_idx]}', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(frames_dir, f'step_{step:03d}.png'))
        all_frames.append(imageio.imread(os.path.join(frames_dir, f'step_{step:03d}.png')))
        plt.close()
        
        # Update current frame and latent for next iteration
        current_frame = next_frame.clone()
        prev_latent = latent_grid.copy()
        prev_frame = next_np.copy()
        
        # Update frame history for frame-based model
        if with_frames:
            last2_frames[0] = last2_frames[1].clone()
            last2_frames[1] = current_frame.clone()
        
        # Check if we're hitting a static state
        if prev_latent is not None:
            changes = (latent_grid != prev_latent).sum()
            print(f"Step {step}: {changes} latent positions changed")
            if changes == 0 and step > 3:
                print("WARNING: Game has become static!")
    
    # Create GIF
    if make_gif and all_frames:
        print("Creating GIF of visualization...")
        imageio.mimsave(os.path.join(frames_dir, 'latent_changes.gif'), all_frames, fps=2)
        print(f"GIF saved to {os.path.join(frames_dir, 'latent_changes.gif')}")

if __name__ == '__main__':
    main() 
    # Call with your variables from the main script
    