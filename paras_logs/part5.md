# Part 5: Playable World Model Interface

## Overview
This part demonstrates a playable Breakout game running entirely through the learned world model. The game loop, user interface, and neural inference are implemented in `play_neural_breakout.py`. A random agent version for video generation is implemented in `neural_random_game.py`.

## Related Components
- [part2.md](part2.md): Latent action codebook
- [part3.md](part3.md): Next frame prediction model
- [part4.md](part4.md): Action-to-latent mapping

## Implementation Details

### Neural Game with a Random Player (`neural_random_game.py`)
- Loads the initial frame (`data/0.png`).
- Loads the world model (`checkpoints/latent_action/best.pt`) using `load_latent_action_model` from `latent_action_model.py`.
- Loads the action-to-latent model (`checkpoints/latent_action/action_to_latent_best.pt` or `action_to_latent_best.pt` for action-only).
- For each step:
  - Samples a random discrete action (`[NOOP, FIRE, RIGHT, LEFT]`).
  - Maps the action (and optionally last 2 frames) to a latent code using the action-to-latent model.
  - Decodes the next frame using the world model's decoder.
  - Feeds the generated frame back for the next step.
- Repeats for a specified number of steps (default 100).
- Saves all frames as a video (`data/neural_random_game.gif`).

### Playable Neural Game (`play_neural_breakout.py`)
- Loads the initial frame (`data/0.png`).
- Loads the world model and action-to-latent model (always using the action+state variant).
- Uses PyGame for the user interface:
  - Displays the current frame (scaled up for visibility).
  - Shows step count, last action, and temperature at the bottom.
- Controls:
  - `SPACE`: Fire
  - `LEFT ARROW`: Move Left
  - `RIGHT ARROW`: Move Right
  - `.` (PERIOD): No Operation (NOOP)
  - `ESC` or `Q`: Quit
- Game loop:
  - Captures user input each frame (default action is NOOP).
  - Maps the action and last 2 frames to a latent code.
  - Uses the world model to generate the next frame.
  - Updates the display at 15 FPS.
- No explicit score, lives, or advanced UI elements are shown.
- No reality anchoring, frame caching, async prediction, or frame skipping is implemented.
- All neural inference runs on GPU if available, otherwise MPS or CPU.
- Uses `torch.compile` for model speedup on CUDA.

### Notes
- The game is a pure neural simulation: all transitions are predicted by the learned model, not the real environment.
- The only feedback to the player is the visual game state and action display.
- For evaluation, users can play and observe the model's consistency and visual quality.

### Success Criteria
- Game runs at stable 15 FPS.
- Paddle responds to user input.
- Visual transitions are plausible for extended play.
- The game is playable and demonstrates the learned world model in action.