# Part 3: Next Frame Prediction Model

## Redundant
It seems part 3 is redundant, since decoder trained in part 2 is the world model that we need.

## Overview
In this part, we'll build a model that can predict the next frame given the current frame and a latent action. This model will serve as the core world model for our Breakout game simulation.

## Related Components
- [part2.md](part2.md): Provides the latent action representations and dataset
- [part5.md](part5.md): Uses this model as the world dynamics engine for the playable interface

## Implementation Details

### Data Preparation
1. Load the processed dataset from Part 2, containing:
   - Current frames
   - Latent action indices
   - Next frames
2. Create train/validation splits (90%/10%)
3. Set up data loaders with batch size 32

### Model Architecture
1. Implement a conditional U-Net architecture:
   - Input: Current frame (84x84x1)
   - Condition: Latent action embedding (from codebook)
   
   - Encoder Path:
     - 4 blocks of (Conv2D, BatchNorm, LeakyReLU, MaxPool)
     - Channel progression: 1→64→128→256→512
     - Resolution reduction: 84→42→21→10→5
   
   - Latent Action Integration:
     - Convert latent action index to embedding vector
     - Spatial broadcast to match encoder feature maps
     - Concatenate with encoder features at each level
   
   - Decoder Path:
     - 4 blocks of (TransposedConv2D, BatchNorm, ReLU)
     - Skip connections from encoder (concatenation)
     - Channel progression: 512→256→128→64→1
     - Final Sigmoid activation for pixel values
   
   - Optional: Implement temporal context
     - Allow for conditioning on previous 3-4 frames
     - Use a small convolutional LSTM to encode temporal context

2. Loss Function:
   - Primary: L1 loss between predicted and actual next frame
   - Perceptual loss using pre-trained VGG features (optional)
   - Adversarial loss with small discriminator (optional, if time permits)

### Training Process
1. Train for 50,000 iterations
2. Use Adam optimizer with learning rate 1e-4
3. Implement learning rate scheduling (reduce on plateau)
4. Gradient clipping to stabilize training
5. Save model checkpoints every 5,000 iterations
6. Log training/validation losses and example predictions

### Evaluation
1. Quantitative metrics:
   - L1 error on validation set
   - PSNR and SSIM scores
   - Multi-step prediction error (rollout of 10 steps)
2. Qualitative evaluation:
   - Single-step prediction visualization
   - Multi-step rollout visualization (applying the model iteratively)
   - Comparison against ground truth sequences

### Multi-step Prediction Testing
1. Implement a function for multi-step prediction:
   - Start with a real frame
   - Predict next frame using current model
   - Use predicted frame as input for next prediction
   - Repeat for N steps
2. Analyze how errors accumulate over multiple steps
3. Identify failure modes (e.g., ball disappearing, physics violations)

### Success Criteria
- Single-step PSNR > 35 dB
- Visually convincing predictions for at least 10-step rollouts
- Ball physics and paddle movements maintained correctly
- Brick collision and destruction accurately modeled