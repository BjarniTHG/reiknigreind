#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Submission-only Actor–Critic agent (inference only).

- No training; only greedy action selection.
- Loads critic weights from checkpoints/agent_ac.pt (on first action call).
- CPU by default, matching the original script's behavior.
- Public API kept compatible with train.py:
    - action(board, dice, player, i, train=False, train_config=None)
    - episode_start(), end_episode(...), game_over_update(...)
    - set_eval_mode(is_eval)

Notes:
- Uses the critic (w1, w2, b1, b2) for greedy selection, identical to your
  greedy_action logic in the working AC file.
- Uses flipped_agent.flip_board / flip_move for perspective handling.
"""

from pathlib import Path
import numpy as np
import torch
from torch import Tensor

# ---- engine & flip helpers ----
try:
    import backgammon as Backgammon
except Exception:
    import Backgammon  # fallback if module name is capitalized

import flipped_agent  # expects flip_board(board_np) and flip_move(move_seq)

# -------------------- Device --------------------
device = torch.device("cpu")
print(f"[agent_ac_submit] Using device: {device}")

# -------------------- Features --------------------
nx = 24 * 2 * 6 + 4 + 1  # same as your working AC

def one_hot_encoding(board, nSecondRoll: bool) -> np.ndarray:
    oneHot = np.zeros(24 * 2 * 6 + 4 + 1)
    # player +1 bins
    for i in range(0, 5):
        idx = np.where(board[1:25] == i)[0] - 1
        if idx.size > 0:
            oneHot[i * 24 + idx] = 1
    idx = np.where(board[1:25] >= 5)[0] - 1
    if idx.size > 0:
        oneHot[5 * 24 + idx] = 1
    # player -1 bins (negative counts)
    for i in range(0, 5):
        idx = np.where(board[1:25] == -i)[0] - 1
        if idx.size > 0:
            oneHot[6 * 24 + i * 24 + idx] = 1
    idx = np.where(board[1:25] <= -5)[0] - 1
    if idx.size > 0:
        oneHot[6 * 24 + 5 * 24 + idx] = 1
    # bars/offs + second-roll flag
    oneHot[12 * 24 + 0] = board[25]
    oneHot[12 * 24 + 1] = board[26]
    oneHot[12 * 24 + 2] = board[27]
    oneHot[12 * 24 + 3] = board[28]
    oneHot[12 * 24 + 4] = 1.0 if nSecondRoll else 0.0
    return oneHot

# -------------------- Parameters (critic only, inference) --------------------
# Shapes follow your AC:
#   w1: (nx/2, nx), b1: (nx/2, 1), w2: (1, nx/2), b2: (1, 1)
w1: Tensor = torch.zeros((nx // 2, nx),  dtype=torch.float, device=device)
b1: Tensor = torch.zeros((nx // 2, 1),   dtype=torch.float, device=device)
w2: Tensor = torch.zeros((1, nx // 2),   dtype=torch.float, device=device)
b2: Tensor = torch.zeros((1, 1),         dtype=torch.float, device=device)

# lazy loading flag & checkpoint path
_CKPT_PATH = Path("checkpoints/agent_ac.pt")
_loaded_once = False

def _load_weights_if_available():
    """Lazy-load critic weights from checkpoints/agent_ac.pt if present."""
    global _loaded_once, w1, w2, b1, b2
    if _loaded_once:
        return
    if not _CKPT_PATH.exists():
        _loaded_once = True  # avoid repeated checks
        return
    try:
        state = torch.load(_CKPT_PATH, map_location=device)
        # Accept either raw tensors or .data wrappers
        def _get(k): 
            v = state[k]
            return v.data if hasattr(v, "data") else v

        # primary expected keys
        if all(k in state for k in ("w1", "w2", "b1", "b2")):
            w1.copy_(_get("w1"))
            w2.copy_(_get("w2"))
            b1.copy_(_get("b1"))
            b2.copy_(_get("b2"))
        else:
            # soft fallback: handle nested dicts or alt naming if needed
            # (customize here if your checkpoint has a different layout)
            raise KeyError("Missing critic keys w1,w2,b1,b2 in checkpoint.")

        _loaded_once = True
        print("[agent_ac_submit] Loaded critic weights from checkpoints/agent_ac.pt")
    except Exception as e:
        _loaded_once = True
        print(f"[agent_ac_submit] Warning: failed to load checkpoint: {e}")

# -------------------- Eval switch (no-op for inference) --------------------
_eval_mode = True
def set_eval_mode(is_eval: bool):
    global _eval_mode
    _eval_mode = bool(is_eval)

# -------------------- Hooks required by harness (no-ops) --------------------
def episode_start():
    pass

def end_episode(outcome, final_board, perspective):
    pass

def game_over_update(board, reward):
    pass

# -------------------- Greedy policy using critic (matches your AC) --------------------
@torch.no_grad()
def _greedy_action(board_np: np.ndarray, dice, oplayer: int, nSecondRoll: bool):
    flippedplayer = -1
    # view from +1 POV if needed
    if flippedplayer == oplayer:
        board_eff = flipped_agent.flip_board(np.copy(board_np))
        player_eff = -oplayer  # becomes +1
    else:
        board_eff = board_np
        player_eff = oplayer   # already +1

    possible_moves, possible_boards = Backgammon.legal_moves(board_eff, dice, player_eff)
    na = len(possible_boards)
    if na == 0:
        return []

    # Build features for all after-states
    xa = np.zeros((na, nx), dtype=np.float32)
    for i in range(na):
        xa[i, :] = one_hot_encoding(possible_boards[i], nSecondRoll)
    # shape for matmuls: x -> (nx, na)
    x = torch.tensor(xa.T, dtype=torch.float, device=device)

    # Critic forward: va = sigmoid(w2·tanh(w1·x + b1) + b2)
    h      = w1 @ x + b1         # (nx/2, na)
    h_tanh = torch.tanh(h)       # (nx/2, na)
    y      = w2 @ h_tanh + b2    # (1, na)
    va     = torch.sigmoid(y).cpu().numpy().reshape(-1)  # (na,)

    action = possible_moves[int(np.argmax(va))]
    if flippedplayer == oplayer:
        action = flipped_agent.flip_move(action)
    return action

# -------------------- Main API --------------------
def action(board_copy, dice, player, i, train=False, train_config=None):
    """
    Return the greedy move only. No learning occurs.
    - board_copy: numpy array (len 29)
    - dice: (d1, d2)
    - player: +1 or -1
    - i: move index within the roll (0 for the first application)
    """
    _load_weights_if_available()
    nSecondRoll_flag = bool((dice[0] == dice[1]) and (i == 0))
    return _greedy_action(np.copy(board_copy), dice, player, nSecondRoll_flag)

