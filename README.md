# Disclaimer

**This codebase has only been tested for Atari Breakout. It is very hacky, trained on a small number of samples, and is not at all optimized. Use at your own risk!**

---

# Atari Pixels Project

This projects aims to create a neural playable version of Atari Breakout by learning purely from videos of the game. It's a small replica of what [Google's Genie project](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/), where they learned an interactive and playable world models purely through videos of game.

**Watch the video**

[![Watch the video](https://img.youtube.com/vi/H8Eh1HlLzZM/0.jpg)](https://www.youtube.com/watch?v=H8Eh1HlLzZM)


## Getting Started

### 1. Install Dependencies and Atari ROMs
```bash
pip install -r requirements.txt
python -m AutoROM --accept-license
```
## Experiments
The following experiments were conducted from my side:

1. Convolution stems with temporal cross attention encoder inspired by C-ViViT
2. Exponential moving average for codebook updates to prevent codebook collapse
3. Decrease number of embeddings in codebook to 32
4. Try the model with different atari game with clock - Skiing-v5

You can run the experiments by running the following command:

Download the checkpoints from [here](https://drive.google.com/drive/folders/1igtad_5vXiiHaEYYfZ8gt0y-yjfg-2P0?usp=sharing) and place them in the `checkpoints` folder.

Then run the following commands to play the game with different models:

```bash

0. Original VQ-VAE model with 256 embeddings and no EMA updates

```bash
python play_neural_breakout.py --model action_state_to_latent --vqvae_type standard --world_model_path .\checkpoints\original\best.pt --game breakout

```

1. Convolution stems with temporal cross attention encoder inspired by C-ViViT
```bash
python play_neural_breakout.py --model action_state_to_latent --vqvae_type cvivit --world_model_path .\checkpoints\cvivit\best.pt --game breakout
```

2. Exponential moving average for codebook updates to prevent codebook collapse
```bash
python play_neural_breakout.py --model action_state_to_latent --vqvae_type ema --world_model_path .\checkpoints\cvivit-ema\best.pt --game breakout
```
3. Decrease number of embeddings in codebook to 32
```bash
python play_neural_breakout.py --model action_state_to_latent --vqvae_type ema --world_model_path .\checkpoints\cvivit-ema-32\best.pt --game breakout
```
4. Try the model with different atari game with clock - Skiing-v5
```bash
python play_neural_breakout.py --model action_state_to_latent --vqvae_type cvivit --world_model_path .\checkpoints\new_game\best.pt --game skiing
```

## Part 1: DQN Training & Exploration
- Implements a Deep Q-Network (DQN) agent for Atari Breakout.
- Supports two exploration strategies:
  - **Temperature-based (Boltzmann) exploration** with Prioritized Experience Replay (PER).
  - **Random Network Distillation (RND)** for intrinsic motivation and improved exploration.
- Modular, efficient, and test-driven codebase.
- Current progress: Agent achieves a score up to 20.
- Next steps: Train for longer, increase max steps per episode to 10,000, and target scores of 200+.

See [part1.md](part1.md) for full details, implementation, and next steps.

---

### 2. Train the DQN Agent
- **Default (temperature-based exploration):**
  ```bash
  python train_dqn.py
  ```
- **RND-based exploration:**
  ```bash
  python train_dqn.py --exploration_mode rnd
  ```
- Additional arguments (see `python train_dqn.py --help`) allow you to control episodes, buffer size, and more.

### 3. Generate Gameplay Videos
- To record videos of a trained (or random) agent:
  ```bash
  python record_videos.py --checkpoint_path <path_to_checkpoint> --output_dir videos/
  e.g. python record_videos.py --bulk --total_videos 100 --percent_random 15 --output_dir bulk_videos
  ```
- Replace `<path_to_checkpoint>` with the path to your saved model checkpoint (see `checkpoints/`).
- Videos will be saved as MP4 files in the specified output directory.

---

## Part 2: Train Latent Action Prediction Model
- **Prepare data:** Ensure gameplay videos are available in `bulk_videos/` (from Part 1).
- **Train VQ-VAE latent action model:**
  ```bash
  python train_latent_action.py
  ```
- **Checkpoints and logs:**
  - Best model: `checkpoints/latent_action/best.pt`
  - Processed data: see `data/` and `vqvae_recons/`
  - Training logs and metrics: Weights & Biases (wandb)
- **Evaluation:**
  - Run `test_latent_action_model.py` for automated tests and metrics
  - Visualize reconstructions and codebook usage as described in part2.md

See [part2.md](part2.md) for full details, implementation, and next steps.

---

## Part 3: Next Frame Prediction Model (World Model)
- **Note:** The decoder from Part 2 serves as the world model. No separate training is required unless you wish to experiment with alternative architectures.
- **Evaluate world model:**
  - Use the decoder to predict next frames given current frame and latent action index
  - For multi-step prediction and rollout analysis, see evaluation code in `test_latent_action_model.py`

---

## Part 4: Train Action-to-Latent Mapping Model
- **Extract (action, latent_code) pairs using trained VQ-VAE:**
  ```bash
  python collect_action_latent_pairs.py
  ```
  - Output: `data/actions/action_latent_pairs.json`
- **Train the action-to-latent MLP:**
  ```bash
  python train_action_to_latent.py
  ```
  - Best checkpoint: `checkpoints/latent_action/action_to_latent_best.pt`
- **Evaluate mapping accuracy:**
  - Run `test_latent_action_data_collection.py` and `test_latent_action_model.py`
  - Analyze accuracy and code distributions as described in part4.md

See [part4.md](part4.md) for full details, implementation, and next steps.

---

## Part 5: Playable Neural Breakout (World Model Demo)
- **Run a random agent in the neural world model:**
  ```bash
  python neural_random_game.py
  ```
  - Output: `data/neural_random_game.gif` (video of neural gameplay)
- **Play Breakout using the neural world model:**
  ```bash
  python play_neural_breakout.py
  ```
  - Controls: SPACE (Fire), LEFT/RIGHT ARROW, PERIOD (NOOP), ESC/Q (Quit)
  - Requires trained models from Parts 2 and 4
- **All inference runs on GPU if available, otherwise MPS/CPU.**
- **For best performance, ensure torch.compile is enabled and models are on CUDA.**

See [part5.md](part5.md) for full details, implementation, and next steps.

---

## Utility & Debug Scripts
- **debug_color_channels.py:** Visualizes and compares color channels between environment frames and PNG files to debug color mismatches.
- **debug_first_step_difference.py:** Compares predicted vs. ground truth latents and reconstructions for the first step of the neural random game, helping debug model discrepancies.
- **neural_random_game.py:** Runs a random agent in the neural world model and saves a GIF of the generated gameplay for qualitative evaluation.

---
## Suggested Improvements
- Train DQN agent for longer (by increasing max steps per episode to 10,000 and target scores of 200+)
- Generate 1000s of videos of agent playing with a higher percentage of random agent actions:
  ```bash
  python record_videos.py --bulk --total_videos 1000 --percent_random 20 --output_dir bulk_videos
  ```
- Train the latent action model for much longer to achieve convergence
- Collect much more data for action → latent code mapping
- Try with different Atari games
