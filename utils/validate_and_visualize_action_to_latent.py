import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from models.latent_action_model import ActionStateToLatentMLP, load_latent_action_model
from dataloaders.latent_action_data import get_action_state_latent_dataloaders

def validate_initial_fire_action():
    """
    Specifically validate cases where:
    1. Input frames are identical or highly similar
    2. Action is FIRE (index 1)
    
    This mimics the first step in neural_random_game.py
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('debug', exist_ok=True)
    
    # Load models
    action_model = ActionStateToLatentMLP().to(device)
    ckpt = torch.load('checkpoints/latent_action/action_state_to_latent_best.pt', map_location=device)
    action_model.load_state_dict(ckpt['model_state_dict'])
    action_model.eval()
    
    world_model, _ = load_latent_action_model('checkpoints/latent_action/best.pt', device)
    world_model.to(device)
    world_model.eval()
    
    # Load validation data
    _, val_loader = get_action_state_latent_dataloaders(batch_size=32)
    
    # For storing results
    fire_errors = []
    fire_examples = []
    
    # Extract cases of interest
    with torch.no_grad():
        for batch_idx, (actions, frames, latents) in enumerate(val_loader):
            actions = actions.to(device)
            frames = frames.to(device)
            latents = latents.to(device)
            
            # Find examples where action is FIRE (index 1)
            fire_indices = (actions.argmax(dim=1) == 1).nonzero(as_tuple=True)[0]
            
            if len(fire_indices) == 0:
                continue
            
            # For FIRE actions, check frame similarity
            for idx in fire_indices:
                # Get the two frames (each 3 channels)
                frame_pair = frames[idx]  # Shape: [6, 210, 160]
                frame1 = frame_pair[:3]   # First frame 
                frame2 = frame_pair[3:]   # Second frame
                
                # Calculate similarity between frames
                similarity = 1.0 - torch.mean(torch.abs(frame1 - frame2)).item()
                
                # If frames are similar (like in game start)
                if similarity > 0.98:  # Highly similar
                    # Get predictions
                    logits = action_model(actions[idx:idx+1], frames[idx:idx+1])
                    pred = logits.argmax(dim=-1)
                    
                    # Calculate error
                    gt = latents[idx]
                    error_count = (pred != gt).sum().item()
                    error_rate = error_count / 35.0
                    fire_errors.append(error_count)
                    
                    # Store for visualization 
                    fire_examples.append({
                        'frames': frames[idx].cpu(),
                        'action': actions[idx].cpu(),
                        'gt_latent': gt.cpu(),
                        'pred_latent': pred.cpu(),
                        'similarity': similarity,
                        'error_count': error_count
                    })
                    
                    # If this closely matches what we're seeing in visual debugging
                    if 16 <= error_count <= 18:
                        print(f"Found match for 17/35 error pattern! Error count: {error_count}")
                        
                        # Visualize this specific case
                        visualize_specific_example(fire_examples[-1], world_model, batch_idx, len(fire_examples)-1)
    
    # Summarize findings
    if fire_errors:
        print(f"Found {len(fire_errors)} examples of FIRE with similar frames")
        print(f"Average error count: {np.mean(fire_errors):.2f}/35 ({(1-np.mean(fire_errors)/35)*100:.2f}% accuracy)")
        print(f"Distribution of errors: {np.bincount(fire_errors)}")
        
        # Plot distribution
        plt.figure(figsize=(10, 6))
        plt.hist(fire_errors, bins=range(36), alpha=0.7)
        plt.xlabel('Number of Errors per Example (out of 35)')
        plt.ylabel('Count')
        plt.title('Error Distribution for FIRE Action with Similar Frames')
        plt.savefig('debug/fire_error_distribution.png')
        plt.close()
        
        # Visualize the first few examples we found
        for i, example in enumerate(fire_examples[:5]):
            visualize_specific_example(example, world_model, 0, i)
    else:
        print("No examples found with FIRE action and similar frames")

def visualize_specific_example(example, world_model, batch_idx, example_idx):
    """Visualize a specific example to match our comparison visualization"""
    device = next(world_model.parameters()).device
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Extract data
    frames = example['frames']
    frame1 = frames[:3].unsqueeze(0).to(device)
    frame2 = frames[3:].unsqueeze(0).to(device)
    gt_latent = example['gt_latent'].reshape(5, 7)
    pred_latent = example['pred_latent'].reshape(5, 7)
    
    # Display frames
    axes[0, 0].imshow(frame1.squeeze(0).permute(1, 2, 0).cpu().numpy())
    axes[0, 0].set_title('Input Frame 1')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(frame2.squeeze(0).permute(1, 2, 0).cpu().numpy())
    axes[0, 1].set_title('Input Frame 2')
    axes[0, 1].axis('off')
    
    # Display latents
    im1 = axes[0, 2].imshow(gt_latent.numpy(), cmap='viridis')
    axes[0, 2].set_title('Ground Truth Latent')
    plt.colorbar(im1, ax=axes[0, 2])
    
    im2 = axes[1, 2].imshow(pred_latent.numpy(), cmap='viridis')
    axes[1, 2].set_title('Predicted Latent')
    plt.colorbar(im2, ax=axes[1, 2])
    
    # Generate decoded frames
    with torch.no_grad():
        # Get embeddings for both latents
        gt_latent_reshaped = gt_latent.reshape(1, 5, 7).to(device)
        pred_latent_reshaped = pred_latent.reshape(1, 5, 7).to(device)
        
        embeddings = world_model.vq.embeddings
        
        # Process ground truth latent
        gt_quantized = embeddings(gt_latent_reshaped)
        gt_quantized = gt_quantized.permute(0, 3, 1, 2).contiguous()
        
        # Process predicted latent
        pred_quantized = embeddings(pred_latent_reshaped)
        pred_quantized = pred_quantized.permute(0, 3, 1, 2).contiguous()
        
        # Use frame1 as input to decoder
        frame_in = frame1.permute(0, 1, 3, 2)  # Permute for decoder
        
        # Decode both
        gt_decoded = world_model.decoder(gt_quantized, frame_in)
        gt_decoded = gt_decoded.permute(0, 1, 3, 2)
        
        pred_decoded = world_model.decoder(pred_quantized, frame_in)
        pred_decoded = pred_decoded.permute(0, 1, 3, 2)
    
    # Display decoded frames
    axes[1, 0].imshow(gt_decoded.squeeze(0).permute(1, 2, 0).cpu().numpy())
    axes[1, 0].set_title('Decoded from GT Latent')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(pred_decoded.squeeze(0).permute(1, 2, 0).cpu().numpy())
    axes[1, 1].set_title('Decoded from Predicted Latent')
    axes[1, 1].axis('off')
    
    # Add error stats 
    error_count = example['error_count']
    error_rate = error_count / 35.0
    frame_similarity = example['similarity']
    
    plt.suptitle(f'FIRE Action with Similar Frames (Similarity: {frame_similarity:.4f})\n'
                f'Latent differences: {error_count}/35 positions ({error_rate*100:.1f}% error rate)',
                fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'debug/fire_example_batch{batch_idx}_ex{example_idx}.png')
    plt.close()

def validate_and_visualize():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = ActionStateToLatentMLP().to(device)
    ckpt = torch.load('checkpoints/latent_action/action_state_to_latent_best.pt', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Load validation data
    _, val_loader = get_action_state_latent_dataloaders(batch_size=16)
    
    all_correct = 0
    all_total = 0
    example_errors = []
    
    # Get predictions and calculate accuracy
    with torch.no_grad():
        for batch_idx, (actions, frames, latents) in enumerate(val_loader):
            actions = actions.to(device)
            frames = frames.to(device)
            latents = latents.to(device)
            
            # Get predictions (argmax, no temperature)
            logits = model(actions, frames)
            preds = logits.argmax(dim=-1)
            
            # Calculate accuracy
            correct = (preds == latents).sum().item()
            total = latents.numel()
            all_correct += correct
            all_total += total
            
            # Track per-example errors
            for i in range(actions.size(0)):
                errors = (preds[i] != latents[i]).sum().item()
                example_errors.append(errors)
            
            # Visualize first few examples
            if batch_idx < 5:
                for i in range(min(2, actions.size(0))):
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Ground truth
                    gt_latent = latents[i].reshape(5, 7).cpu().numpy()
                    axes[0].imshow(gt_latent, cmap='viridis')
                    axes[0].set_title(f'Ground Truth Latent')
                    
                    # Prediction
                    pred_latent = preds[i].reshape(5, 7).cpu().numpy()
                    axes[1].imshow(pred_latent, cmap='viridis') 
                    axes[1].set_title(f'Predicted Latent')
                    
                    # Compute error
                    error_count = (pred_latent != gt_latent).sum()
                    plt.suptitle(f'Action: {actions[i].nonzero().item()}, Error Count: {error_count}/35')
                    plt.tight_layout()
                    plt.savefig(f'debug/batch{batch_idx}_example{i}.png')
                    plt.close()
    
    # Overall accuracy
    accuracy = all_correct / all_total
    print(f"Overall accuracy: {accuracy:.4f}")
    
    # Distribution of errors
    plt.figure(figsize=(10, 6))
    plt.hist(example_errors, bins=range(36), alpha=0.7)
    plt.xlabel('Number of Errors per Example (out of 35)')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Errors')
    plt.savefig('debug/error_distribution.png')
    plt.close()
    
    # Calculate mean errors
    mean_errors = np.mean(example_errors)
    print(f"Mean errors per example: {mean_errors:.2f}/35 ({(35-mean_errors)/35*100:.2f}% accuracy)")

if __name__ == "__main__":
    #validate_and_visualize()
    validate_initial_fire_action()