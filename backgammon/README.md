# Backgammon RL — Starter Kit

This repo is the starter code for the Computational Intelligence backgammon project. It includes:

- **Fast game engine** (`backgammon.py`) accelerated with Numba
- **Baselines**
  - `random_player.py` — uniformly random legal move
  - `pubeval_player.py` — Tesauro’s *pubeval* reimplemented in pure Python + Numba
- **Training harness** (`train.py`) — algorithm-agnostic self-play + periodic evaluation + autosave
- **Example agent** (`agent.py`) — compact PyTorch DQN that evaluates *after-states* (you can replace this with policy-gradient, actor-critic, TD(λ), etc.)

The harness auto-saves the **best checkpoint** to `checkpoints/best.pt`, and the example agent auto-loads it when playing in evaluation/competition.

---

## 1) Quickstart

### Set up (new virtual env recommended)
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install numpy numba matplotlib torch
```
> PyTorch via `pip` installs a CPU build by default. Install a CUDA wheel if you have a GPU.

### Train (self-play) & evaluate
```bash
python train.py
```

By default `train.py` trains via self-play and evaluates vs **pubeval** every few thousand games. To evaluate vs the random baseline instead, edit the last line in `train.py`:

```python
if __name__ == "__main__":
    # eval_vs can be "pubeval" or "random"
    train(n_games=250_000, n_epochs=2_500, n_eval=100, eval_vs="pubeval")
```

The harness prints periodic win-rates and saves the best model to `checkpoints/best.pt`.

---

## 2) Repository layout

```
backgammon.py            # Fast game engine (Numba)
agent.py                 # PyTorch example agent (DQN after-state) — replace/extend as you like
train.py                 # Algorithm-agnostic training + periodic evaluation + autosave
pubeval_numba.py         # Tesauro's pubeval (Numba); strong fixed baseline
random_fast.py           # Uniform random legal move baseline
flipped_agent_fast.py    # Fast board/move flip utilities (Numba)
checkpoints/             # Saved models (created automatically)
```

---

## 3) Board & moves (very important)

- The board is a length-29 NumPy array:
  - `1..24` — on-board points
  - `25` — bar for **player +1**
  - `26` — bar for **player −1**
  - `27` — borne off for **player +1** (0 → 15)
  - `28` — borne off for **player −1** (0 → −15)
  - `0` — unused
- The **sign** encodes owner (`>0` for +1, `<0` for −1); the magnitude is the count.

**Move representation**
- A *single* move is `[start, end]` (two integers).
- A *move sequence* for one dice roll is either:
  - shape **`(1, 2)`** — one move, or
  - shape **`(2, 2)`** — two moves (when both dice are used in an order)
- **Doubles:** the engine will call your agent **twice** in the same turn (not four times). Each call returns at most a two-move sequence; the harness requests the second application separately.

---

## 4) Engine API essentials

```python
import backgammon
board = backgammon.init_board()
dice  = backgammon.roll_dice()
moves, boards = backgammon.legal_moves(board, dice, player=+1)  # after-states
next_board = backgammon.update_board(board, moves[0], player=+1)
done = backgammon.game_over(next_board)
```

- `legal_moves` returns **all legal move sequences** for the current dice and the corresponding **after-state** boards.
- The engine is Numba-accelerated; feed/return simple NumPy arrays for speed.

---

## 5) Training harness (`train.py`)

- **Algorithm-agnostic**: works with policy-gradient, TD(λ), DQN, actor-critic, etc.
- **Self-play**: `agent` vs `agent` for learning.
- **Periodic evaluation** vs `pubeval_numba` (default) or `random_fast`.
- **Autosave**: when your evaluation win-rate improves, the harness saves to `checkpoints/best.pt`.

Example usage inside `train.py`:
```python
from train import train
train(n_games=20_000, n_epochs=2_000, n_eval=200, eval_vs="random")
```

---

## 6) Example agent (`agent.py`)

The example is a compact **PyTorch DQN** that scores **after-states**:

- Enumerate legal after-states for the current dice.
- Encode each after-state to a feature vector.
- Batch the features through a small MLP to get Q-values.
- ε-greedy during training; greedy during evaluation.
- Stores transitions in a replay buffer and learns with 1-step bootstrapped targets.

**Features used (simple & fast):**
```
[ points(1..24), bar_self, bar_opp, off_self, off_opp, moves_left ]
```
Scaled to modest ranges (see `_encode_state` in `agent.py`). Feel free to replace with TD-Gammon-style features or raw encodings.

**Save/load behavior**
- The harness calls `agent.save()` automatically on new best eval; it writes `checkpoints/best.pt`.
- In evaluation/competition, `agent.action(...)` auto-loads `checkpoints/best.pt` on first call and plays **greedily** (no exploration).

**Agent API your code must expose**
- `action(board, dice, player, i, train=False, train_config=None) -> [] or (k,2) array`
- Optional hooks used by the harness (implemented in the example):
  - `set_eval_mode(is_eval: bool)`
  - `episode_start()`, `end_episode(outcome, final_board, perspective)`
  - `game_over_update(board, reward)` (legacy compatibility; still called)

> You can completely replace `agent.py` with your own method as long as `action(...)` matches the signature above.

---

## 7) Playing both colors (flipping)

Common and recommended: **always act as player +1** by flipping the board before selecting a move, and flip the chosen move back. Use `flipped_agent_fast.py` (`flip_board`, `flip_move`) or copy the snippet from `agent.py`.

---

## 8) Evaluation & competition

- Manual evaluation example:
  ```python
  from train import evaluate
  import pubeval_numba as pubeval
  import agent
  wr = evaluate(agent, pubeval, n_eval=1000, label="manual check")
  ```
- **Competition**: submit `agent.py` and `checkpoints/best.pt`. The agent auto-loads the checkpoint and plays greedily.

---

## 9) Tips & common gotchas

- **Numba + PyTorch** is fine: Numba speeds up the environment; PyTorch runs your model.
- Don’t decorate `agent.action()` with `@torch.no_grad()` if you train inside it. Use a **local** `with torch.no_grad():` only around the Q-value **selection** forward pass.
- When applying a sequence, call:
  ```python
  board = backgammon.update_board(board, move_seq, player)
  ```
  It accepts both a single `[start, end]` or a sequence shaped `(k, 2)`.
- For deterministic experiments, seed NumPy & Torch (dice randomness and ε-greedy still add variance).

---

## 10) FAQ

**Q: Can we change features / the network / the algorithm?**  
Yes. `train.py` is agnostic. Replace `agent.py` with policy-gradient, actor-critic, TD(λ), etc.

**Q: How often are we evaluated and saved?**  
Every `n_epochs` games, the harness evaluates and, if the win-rate improves, saves `checkpoints/best.pt`.

**Q: Does the engine check for illegal moves?**  
`legal_moves` only returns legal sequences. If you post-process moves, be careful. Tournament scripts will validate legality.
