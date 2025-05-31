# Part 2: Latent Action Prediction Model

## Overview
This part focuses on building a model that can infer latent actions between consecutive frames. Following the LAPA paper, we'll use a VQ-VAE-inspired approach to learn a discrete codebook of latent actions that explain the transitions between frames.

## Related Components
- [part1.md](part1.md): Provides the raw video data for training
- [part3.md](part3.md): Uses the latent actions from this model to predict next frames
- [part4.md](part4.md): Maps these latent actions to actual game controls

## Implementation Details

### Data Preparation
1. Load consecutive frame pairs from gameplay recordings in the `bulk_videos` directory
   - Each episode folder contains sequentially numbered PNG files (0.png, 1.png, etc.)
   - Frame resolution: 160 × 210 pixels, RGB format (3 channels)
2. Create a PyTorch dataset of (frame_t, frame_t+1) pairs
   - Maintain episode boundaries (don't create pairs across episode boundaries)
   - Create train (80%), validation (10%), and test (10%) splits at the episode level
3. Apply preprocessing:
   - Normalize pixel values to [0, 1] (divide by 255)
   - Take an optional parameter (default yes) that converts to grayscale to reduce model complexity

### Latent Action Model Architecture
1. Implement a VQ-VAE based model with:
   - Encoder: Takes both current frame and next frame as input
     - Concatenate frames along channel dimension (6 channels input for RGB)
     - 5 convolutional layers with stride 2 (downsampling)
     - Channel progression: 6→64→128→256→512→512
     - ReLU activations and batch normalization
     - Final spatial dimensions: 5×7×512 (after 5 stride-2 layers from 160×210)
     - Apply a 1×1 convolution to project 512 channels to 128 channels (resulting in 5×7×128)
   
   - Vector Quantization Layer:
     - Codebook size: 256 vectors (8-bit)
     - Embedding dimension: 128 (increased from 64 to handle higher complexity)
     - Implement with straight-through estimator for backpropagation
     - Commitment loss weight: 0.25
   
   - Decoder:
     - Input: Current frame (3 channels) and quantized latent representation (5×7×128)
     - Process current frame through initial convolutional layers
     - Combine with quantized latent information through cross-attention or FiLM conditioning
     - 5 transposed convolutional layers for upsampling
     - Channel progression: 512→512→256→128→64→3
     - Final output: Reconstructed next frame (160×210×3)
     - **No skip connections** to ensure information flows through latent bottleneck

2. Loss Function:
   - Reconstruction loss: MSE between predicted next frame and actual next frame
   - Codebook commitment loss: To ensure encoder outputs stay close to codebook entries
   - Codebook entropy regularization: To encourage usage of full codebook
   - Optional (to be implemented later): Perceptual loss using pretrained VGG features for better visual quality

### Training Process
1. Train for 100,000 iterations with batch size 32 (reduced from 64 due to larger frame size)
2. Use Adam optimizer with learning rate 3e-4, decaying to 1e-4
3. Implement gradient accumulation if batch size needs to be reduced further
4. Implement codebook reset for unused embeddings (every 10,000 iterations)
5. Log using Wandb:
   - Training/validation losses at each step
   - Reconstruction quality metrics (PSNR, SSIM) every 1,000 steps
   - Codebook usage statistics (histograms) every 5,000 steps
   - Example reconstructions every 2,500 steps
   - Model architecture and hyperparameters
   - GPU memory usage and training speed metrics
6. Save model checkpoints:
   - Every 10,000 iterations
   - Best model based on validation reconstruction loss (stored at `checkpoints/latent_action/best.pt`)

**Only latent action extraction remains to be implemented.**

### Latent Action Extraction
1. Create a function to extract latent action indices from any pair of consecutive frames
2. Process the entire dataset to generate a new dataset of:
   - Current frame
   - Corresponding latent action index
   - Next frame
3. Save this processed dataset for use in Part 3
4. Create visualizations of which latent actions correspond to different gameplay events
   - Map common game events (paddle movement, ball bouncing, brick breaking) to latent codes

### Evaluation
1. Measure reconstruction quality:
   - PSNR (Peak Signal-to-Noise Ratio)
   - SSIM (Structural Similarity Index)
   - L1 and L2 pixel errors
2. Analyze codebook usage:
   - Histogram of code utilization
   - Visualization of code embeddings using t-SNE
   - Track perplexity of the discrete distribution
3. Qualitatively evaluate by visualizing:
   - Original frame pairs
   - Reconstructed next frames
   - Differences between actual and predicted (heatmaps)
4. Evaluation by game elements:
   - Separate metrics for ball movement accuracy
   - Paddle movement reconstruction quality
   - Brick state preservation accuracy

### Success Criteria
- Reconstruction PSNR > 28 dB for RGB frames
- At least 50% of codebook being actively used
- Visually plausible next frame reconstructions
- Clear patterns in latent code assignment for similar transitions
- Successful train/validation/test splits with consistent performance

### Optimization Considerations
1. Memory optimization:
   - Mixed precision training (fp16) to reduce VRAM usage
   - Gradient checkpointing if needed
   - Consider downsampling frames if full resolution proves too memory-intensive

2. Training efficiency:
   - Benchmark different batch sizes to find optimal training speed
   - Monitor GPU utilization and adjust model accordingly

### Hardware and Resource Requirements
- GPU with at least 12GB VRAM recommended (16GB+ preferred)
- Estimated training time: 12-24 hours on a single GPU
- Storage requirements: ~15GB for processed data and checkpoints