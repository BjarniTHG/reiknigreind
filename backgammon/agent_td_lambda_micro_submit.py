#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tournament agent (submit) — faithful to the shallow TD(λ) learner

- Same parameterization and shapes: w1, b1, w2, b2
- Same feature encoder (one_hot_encoding) and POV flipping
- Greedy selection over after-states via critic value (no epsilon, no learning)
- Loads weights from checkpoints/td_lambda.pt (override via load(path=...))

Public API kept for harness compatibility:
  - action(board, dice, player, i, train=False, train_config=None)
  - set_eval_mode(is_eval)
  - episode_start(), end_episode(...), game_over_update(...)  (no-ops)
  - load(path), save(path)  (save is optional)
"""

from typing import Optional
from pathlib import Path
import numpy as np
import torch

# ---- engine imports (lower-case backgammon as requested) ----
try:
    import backgammon as Backgammon
except Exception:
    import Backgammon  # fallback if your file happens to be upper-case

import flipped_agent  # uses your existing flip helpers

# -------------------- Device --------------------
# The original learner defaulted to CPU. We keep that for fidelity, but allow CUDA.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[agent-submit] Using device: {device}")
if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
device = torch.device("cpu")

# -------------------- Features --------------------
nx = 24 * 2 * 6 + 4 + 1  # unchanged from original
H  = nx // 2

def one_hot_encoding(board, nSecondRoll):
    # IDENTICAL layout to the learner
    oneHot = np.zeros(24 * 2 * 6 + 4 + 1, dtype=np.float32)
    # +1 bins
    for i in range(0, 5):
        idx = np.where(board[1:25] == i)[0] - 1
        if idx.size > 0:
            oneHot[i*24 + idx] = 1
    idx = np.where(board[1:25] >= 5)[0] - 1
    if idx.size > 0:
        oneHot[5*24 + idx] = 1
    # -1 bins
    for i in range(0, 5):
        idx = np.where(board[1:25] == -i)[0] - 1
        if idx.size > 0:
            oneHot[6*24 + i*24 + idx] = 1
    idx = np.where(board[1:25] <= -5)[0] - 1
    if idx.size > 0:
        oneHot[6*24 + 5*24 + idx] = 1
    # bars / off / second-roll flag (note: learner stores the flag as-is)
    oneHot[12 * 24 + 0] = board[25]
    oneHot[12 * 24 + 1] = board[26]
    oneHot[12 * 24 + 2] = board[27]
    oneHot[12 * 24 + 3] = board[28]
    oneHot[12 * 24 + 4] = 1.0 if nSecondRoll else 0.0
    return oneHot

# -------------------- Parameters (faithful shapes) --------------------
# Leaf tensors with no grad; we only infer.
w1 = torch.empty(H,  nx, device=device, dtype=torch.float32)
b1 = torch.empty(H,  1,  device=device, dtype=torch.float32)
w2 = torch.empty(1,  H,  device=device, dtype=torch.float32)
b2 = torch.empty(1,  1,  device=device, dtype=torch.float32)

@torch.no_grad()
def critic_forward(x: torch.Tensor) -> torch.Tensor:
    # x: (nx, N)
    h = torch.tanh(torch.mm(w1, x) + b1)  # (H, N)
    y = torch.mm(w2, h) + b2              # (1, N)
    return torch.sigmoid(y)               # (1, N)

# -------------------- Checkpoint IO --------------------
CKPT_DEFAULT = Path("checkpoints/agent_td_lambda.pt")
_loaded_once = False
_eval_mode   = True  # submit agent is eval-only

def set_eval_mode(is_eval: bool):
    # Kept for API parity (tournament may call it)
    global _eval_mode
    _eval_mode = bool(is_eval)

def save(path: Optional[str] = None):
    # Rarely needed in tournament, but provided for symmetry
    p = Path(path) if path else CKPT_DEFAULT
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"w1": w1, "w2": w2, "b1": b1, "b2": b2}, p)

def load(path: Optional[str] = None, map_location: Optional[torch.device] = None):
    p  = Path(path) if path else CKPT_DEFAULT
    ml = map_location or device
    state = torch.load(p, map_location=ml)
    with torch.no_grad():
        # Accept either raw tensors or Variables
        w1.copy_(state["w1"].data if hasattr(state["w1"], "data") else state["w1"])
        w2.copy_(state["w2"].data if hasattr(state["w2"], "data") else state["w2"])
        b1.copy_(state["b1"].data if hasattr(state["b1"], "data") else state["b1"])
        b2.copy_(state["b2"].data if hasattr(state["b2"], "data") else state["b2"])

def _lazy_load():
    global _loaded_once
    if _loaded_once:
        return
    if CKPT_DEFAULT.exists():
        try:
            load(str(CKPT_DEFAULT), map_location=device)
        except Exception as e:
            print(f"[agent-submit] Warning: failed to load '{CKPT_DEFAULT}': {e}")
    else:
        print(f"[agent-submit] Warning: checkpoint not found: {CKPT_DEFAULT}")
    _loaded_once = True

# -------------------- Greedy policy (faithful) --------------------
@torch.no_grad()
def greedy_action(board, dice, oplayer, nSecondRoll):
    flippedplayer = -1
    if (flippedplayer == oplayer):  # view from +1 POV
        board_eff = flipped_agent.flip_board(np.copy(board))
        player_eff = -oplayer
    else:
        board_eff = board
        player_eff = oplayer

    possible_moves, possible_boards = Backgammon.legal_moves(board_eff, dice, player_eff)
    na = len(possible_boards)
    if na == 0:
        return []

    xa = np.zeros((na, nx), dtype=np.float32)
    for i in range(na):
        xa[i, :] = one_hot_encoding(possible_boards[i], nSecondRoll)

    x = torch.from_numpy(xa.T).to(device)  # (nx, na)
    va = critic_forward(x).squeeze(0)      # (na,)
    m  = int(torch.argmax(va).item())
    action = possible_moves[m]

    if (flippedplayer == oplayer):
        action = flipped_agent.flip_move(action)
    return action

@torch.no_grad()
def greedy_policy(board, dice, oplayer, nRoll):
    """
    Faithful shape & return to the learner's greedy_policy
    but used only internally; tournament calls action().
    """
    flippedplayer = -1
    nSecondRoll = ((dice[0] == dice[1]) & (nRoll == 0))
    flipped_flag = (flippedplayer == oplayer)

    if flipped_flag:
        board_eff = flipped_agent.flip_board(np.copy(board))
        player_eff = -oplayer
    else:
        board_eff = board
        player_eff = oplayer

    possible_moves, possible_boards = Backgammon.legal_moves(board_eff, dice, player_eff)
    na = len(possible_moves)
    if na == 0:
        return [], None, None, None, flipped_flag  # mirror learner arity

    xa = np.zeros((na, nx), dtype=np.float32)
    for i in range(na):
        xa[i, :] = one_hot_encoding(possible_boards[i], nSecondRoll)
    x = torch.from_numpy(xa.T).to(device)          # (nx, na)

    va = critic_forward(x).squeeze(0)              # (na,)
    m  = int(torch.argmax(va).item())
    target = va[m].detach()

    action = possible_moves[m]
    if flipped_flag:
        action = flipped_agent.flip_move(action)

    x_selected = torch.from_numpy(xa[m]).to(device).view(nx, 1)
    chosen_after_eff = possible_boards[m].reshape(-1)  # numpy, +1 POV

    return action, x_selected, target, chosen_after_eff, flipped_flag

# -------------------- Harness hooks --------------------
def episode_start():
    pass

def end_episode(outcome, final_board, perspective):
    pass

def game_over_update(board, reward):
    pass

def _expand_dice_micro(dice: np.ndarray) -> list[int]:
    d1, d2 = int(dice[0]), int(dice[1])
    return [d1, d1] if d1 == d2 else [d1, d2]

@torch.no_grad()
def greedy_action_micro(board, dice, oplayer):
    """
    Build the turn via two micro-steps max, scoring single-die moves with the critic.
    Returns [] (no-ops) or an np.ndarray of shape (k,2), k in {1,2}.
    """
    # POV flip (evaluate from +1 perspective if current player is -1)
    flipped_flag = (oplayer == -1)
    if flipped_flag:
        board_eff  = flipped_agent.flip_board(np.copy(board))
        player_eff = +1    # we flipped -1 board to +1 POV
    else:
        board_eff  = board
        player_eff = oplayer

    dice_steps = _expand_dice_micro(dice)
    chosen_moves_eff = []  # in effective POV

    # Greedy per-die loop (at most two micro-moves with this backend)
    for step_idx, die in enumerate(dice_steps):
        # Enumerate single-die legal moves
        legals = Backgammon.legal_move(board_eff, die, player_eff)  # list of [start,end]
        if len(legals) == 0:
            # cannot move on this die ⇒ pass this micro-step
            continue

        # nSecondRoll flag: True only on first micro-step of a double
        nSecondRoll = bool((dice_steps[0] == dice_steps[-1]) and (len(dice_steps) == 2) and (step_idx == 0))

        # Evaluate each single-die candidate by after-state value
        xa_list = []
        boards_after = []
        for mv in legals:
            bb = Backgammon.update_board(board_eff, np.asarray(mv, dtype=np.int32), player_eff)
            boards_after.append(bb)
            xa_list.append(one_hot_encoding(bb, nSecondRoll))

        x = torch.from_numpy(np.stack(xa_list, axis=1)).to(device)  # (nx, na)
        va = critic_forward(x).squeeze(0)                           # (na,)
        m  = int(torch.argmax(va).item())

        # Commit best micro-move in effective POV, update effective board
        best_move = legals[m]
        chosen_moves_eff.append(np.array(best_move, dtype=np.int32))
        board_eff = boards_after[m]  # advance after-state

        # Nothing chosen this turn
    if len(chosen_moves_eff) == 0:
        return []

    # Pack moves to shape (k,2) first, then flip once if needed
    moves_arr = np.stack(chosen_moves_eff, axis=0).astype(np.int32)  # (k,2)

    if flipped_flag:
        moves_arr = flipped_agent.flip_move(moves_arr)  # expects (k,2)

    return moves_arr

def action(board_copy, dice, player, i, train: bool = False, train_config=None):
    """
    Tournament entry point (eval-only). Build a turn by greedy micro-actions.
    Returns [] if no legal micro-moves exist for either die.
    """
    _lazy_load()
    return greedy_action_micro(np.copy(board_copy), dice, player)
