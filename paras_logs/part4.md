# Part 4: Action Mapping Model

## Progress & Status
- [x] **Data Collection:** Complete
    - Collected (action, latent_code) pairs using a random agent and trained VQ-VAE
    - **Stored in:** `data/actions/action_latent_pairs.json`
    - **Format:** JSON list of dicts, each with:
      - `'action'`: int (0–3, corresponding to NOOP, FIRE, RIGHT, LEFT)
      - `'latent_code'`: list of 35 ints (each 0–255, codebook indices, flattened from (5, 7) grid)
      - Example: `{ "action": 2, "latent_code": [123, 45, ..., 87] }`
    - Codebook size: 256 (indices 0–255), embedding size: 128
    - Latent code shape: (5, 7) grid, flattened to 35 indices per frame pair
- [x] **Model Design:** Complete
    - Implemented `ActionToLatentMLP` in `latent_action_model.py` (MLP, 4→512→256→35×256)
    - Supports temperature-based sampling for latent code prediction
- [x] **Data Loader:** Complete
    - Implemented `ActionLatentPairDataset` and `get_action_latent_dataloaders` in `latent_action_data.py`
    - Loads, splits, and batches (action, latent_code) pairs for training/validation
- [x] **Training:** Complete
    - Trained for 50 epochs using Adam (lr=1e-3), cross-entropy loss, mixed precision, and gradient clipping
    - **Best validation accuracy:** 82.7%
    - **Best checkpoint saved to:** `checkpoints/latent_action/action_to_latent_best.pt`
    - All metrics and checkpoints logged to wandb (project: `atari-action-to-latent`)
- [ ] **Analysis:** Pending
- [ ] **Testing:** Pending
- [ ] **Refinement:** Pending
- [ ] **Integration:** Pending

## Overview
In this part, we'll create a model that maps the actual game controls (LEFT, RIGHT, NOTHING) to the latent action space we learned in Part 2. This will allow us to control the world model with normal game inputs.

## Related Components
- [part1.md](part1.md): Provides the original action labels
- [part2.md](part2.md): Provides the latent action codes
- [part5.md](part5.md): Uses this mapping to convert user inputs into world model updates

## Implementation Steps

### 1. Data Collection
- **[Done]** Load your trained VQ-VAE from Part 2
- **[Done]** Use random agent to play Breakout and collect data:
  - Record action taken at each step
  - Process frame pairs through VQ-VAE
  - Extract quantized latent code indices (35 per pair, from a codebook of 256)
  - Store (action, latent_code) pairs as JSON in `data/actions/action_latent_pairs.json`
- **[Done]** Aim for 100,000+ pairs across multiple episodes
- **[Done]** Calculate action frequency and latent code distribution statistics

### 2. Model Design
- **[Done]** Created a simple MLP classifier in `latent_action_model.py`:
  - Input: One-hot encoded action (4 dimensions)
  - Hidden layers: 512 → 256 with ReLU and dropout
  - Output: 35 × 256 logits (for each latent code position, 256-way softmax)
  - Includes temperature-based sampling for latent code prediction

### 3. Data Loader
- **[Done]** Implemented in `latent_action_data.py`:
  - Loads `data/actions/action_latent_pairs.json`
  - Splits into 80% training, 20% validation
  - Returns PyTorch DataLoader objects with (one-hot action, latent code indices)

### 4. Training
- **[Done]** Split data: 80% training, 20% validation
- **[Done]** Trained with cross-entropy loss and Adam optimizer (lr=1e-3) for 250 epochs
- **[Done]** Monitored validation accuracy and saved best model checkpoint
- **[Done]** Best model checkpoint: `checkpoints/latent_action/action_to_latent_best.pt` (val acc: 96.3%)

### 5. Analysis
- [Pending] For each action, analyze predicted vs actual latent code distributions
- [Pending] Visualize top-k most likely codes per action
- [Pending] Create histograms showing latent code frequency by action
- [Pending] Measure prediction accuracy per action type

### 6. Testing
- [Pending] Implement test function: action → predicted latent code
- [Pending] Validate in environment by comparing predicted vs actual codes
- [Pending] Calculate overall and per-action accuracy
- [Pending] Identify patterns in successful/failed predictions

### 7. Refinement (if needed)
- [Pending] If accuracy < 60%, consider:
  - Adding game state features as context
  - Using probabilistic output instead of single prediction
  - Creating separate models per action
  - Balancing dataset

### 8. Integration
- [Pending] Create simple inference pipeline for your world model
- [Pending] Document performance characteristics and limitations

## Expected Outcome
- Model with 60-70%+ accuracy mapping actions to latent codes (achieved 82.7%)
- Clear differentiation between action effects
- Documented interface for world model integration