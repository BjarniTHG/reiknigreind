#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backgammon TD(λ) — Critic-only with eligibility traces (no actor), DEEP critic.

Network (critic):
    x (nx) -> H -> 256 -> H -> 1
    with tanh at each hidden, sigmoid on output (value in [0,1])

Learning:
    - Greedy selection via the critic (ε=0).
    - TD(λ) with eligibility traces for critic params only.
    - Two branches maintained (normal and 'flipped' POV) exactly like your original:
        * accumulate traces on the previous x for that branch,
        * delta = reward{branch} + γ * tgt - V(prev),
        * update all critic layers: w1,b1,w3,b3,w4,b4 with α1, and w2,b2 with α2.

Public API:
  - action(board, dice, player, i, train=False, train_config=None)
  - episode_start(), end_episode(...), game_over_update(...)
  - set_eval_mode(is_eval)
  - save(path), load(path)

Assumptions:
  - `backgammon` module provides legal_moves/update mechanics.
  - `flipped_agent` provides flip_board(board_np), flip_move(moves_array).
"""

from typing import Optional
import numpy as np
import torch

# ---- engine imports ----
try:
    import backgammon as Backgammon
except Exception:
    import Backgammon

import flipped_agent  # flip helpers

# -------------------- Device --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[agent] Using device: {device}")

if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass

# -------------------- Hyperparameters --------------------
alpha1 = 0.001  # lr for hidden critic layers
alpha2 = 0.001  # lr for output critic layer
lam    = 0.7    # TD(λ)
gamma  = 1.0    # episodic

# -------------------- Features --------------------
nx = 24 * 2 * 6 + 4 + 1

def one_hot_encoding(board, nSecondRoll):
    oneHot = np.zeros(24 * 2 * 6 + 4 + 1, dtype=np.float32)
    # +1 side bins
    for i in range(0,5):
        idx = np.where(board[1:25] == i)[0] - 1
        if idx.size > 0:
            oneHot[i*24 + idx] = 1
    idx = np.where(board[1:25] >= 5)[0] - 1
    if idx.size > 0:
        oneHot[5*24 + idx] = 1
    # -1 side bins
    for i in range(0,5):
        idx = np.where(board[1:25] == -i)[0] - 1
        if idx.size > 0:
            oneHot[6*24 + i*24 + idx] = 1
    idx = np.where(board[1:25] <= -5)[0] - 1
    if idx.size > 0:
        oneHot[6*24 + 5*24 + idx] = 1
    # bars/offs + second-roll flag
    oneHot[12 * 24 + 0] = board[25]
    oneHot[12 * 24 + 1] = board[26]
    oneHot[12 * 24 + 2] = board[27]
    oneHot[12 * 24 + 3] = board[28]
    oneHot[12 * 24 + 4] = 1.0 if nSecondRoll else 0.0
    return oneHot

# -------------------- Critic parameters (DEEP) --------------------
H = int(nx/2)
H_EXPAND = 256

# Critic: nx -> H -> 256 -> H -> 1
# Make parameters LEAF tensors; scale weights in-place under no_grad.
w1 = torch.randn(H,        nx,       device=device, dtype=torch.float32, requires_grad=True)
b1 = torch.zeros(H, 1,     device=device, dtype=torch.float32, requires_grad=True)
w3 = torch.randn(H_EXPAND, H,        device=device, dtype=torch.float32, requires_grad=True)
b3 = torch.zeros(H_EXPAND, 1,        device=device, dtype=torch.float32, requires_grad=True)
w4 = torch.randn(H,        H_EXPAND, device=device, dtype=torch.float32, requires_grad=True)
b4 = torch.zeros(H, 1,     device=device, dtype=torch.float32, requires_grad=True)
w2 = torch.randn(1,        H,        device=device, dtype=torch.float32, requires_grad=True)
b2 = torch.zeros(1, 1,     device=device, dtype=torch.float32, requires_grad=True)
with torch.no_grad():
    w1.mul_(0.1); w3.mul_(0.1); w4.mul_(0.1); w2.mul_(0.1)

def critic_forward(x):  # x: (nx, N) column-batched features
    h1 = torch.mm(w1, x) + b1;  h1 = torch.tanh(h1)      # (H, N)
    h2 = torch.mm(w3, h1) + b3; h2 = torch.tanh(h2)      # (256, N)
    h3 = torch.mm(w4, h2) + b4; h3 = torch.tanh(h3)      # (H, N)
    y  = torch.mm(w2, h3) + b2                           # (1, N)
    return torch.sigmoid(y)                               # (1, N)

# -------------------- Per-episode state --------------------
# Critic traces (+1 POV & flipped)
Z_w1 = Z_b1 = Z_w2 = Z_b2 = None
Z_w3 = Z_b3 = Z_w4 = Z_b4 = None
Zf_w1 = Zf_b1 = Zf_w2 = Zf_b2 = None
Zf_w3 = Zf_b3 = Zf_w4 = Zf_b4 = None

# Previous after-state feature caches
xold = xold_flipped = None

# Move counter / eval
moveNumber = 0
_eval_mode = False

# -------------------- Save / Load --------------------
from pathlib import Path
CKPT_DEFAULT = Path("checkpoints/td_lambda_critic_only_deep.pt")

def save(path: Optional[str] = None):
    p = Path(path) if path else CKPT_DEFAULT
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "w1": w1, "b1": b1, "w2": w2, "b2": b2,
        "w3": w3, "b3": b3, "w4": w4, "b4": b4,
    }, p)

def load(path: Optional[str] = None, map_location: Optional[torch.device] = None):
    p = Path(path) if path else CKPT_DEFAULT
    ml = map_location or device
    state = torch.load(p, map_location=ml)
    with torch.no_grad():
        for name, param in [("w1", w1), ("b1", b1), ("w2", w2), ("b2", b2),
                            ("w3", w3), ("b3", b3), ("w4", w4), ("b4", b4)]:
            src = state[name]
            src = src.data if hasattr(src, "data") else src
            param.copy_(src)
    set_eval_mode(True)

def set_eval_mode(is_eval: bool):
    global _eval_mode
    _eval_mode = bool(is_eval)

# -------------------- Greedy policy via critic --------------------
def greedy_policy(board, dice, oplayer, nRoll):
    flippedplayer = -1
    nSecondRoll = ((dice[0] == dice[1]) & (nRoll == 0))
    flipped_flag = (flippedplayer == oplayer)

    if flipped_flag:
        board_eff = flipped_agent.flip_board(np.copy(board))
        player_eff = -oplayer  # +1 POV
    else:
        board_eff = board
        player_eff = oplayer   # +1 POV

    possible_moves, possible_boards = Backgammon.legal_moves(board_eff, dice, player_eff)
    na = len(possible_moves)
    if na == 0:
        return [], None, None, flipped_flag  # keep arity 4 for caller

    # batch score after-states
    xa = np.zeros((na, nx), dtype=np.float32)
    for i in range(na):
        xa[i, :] = one_hot_encoding(possible_boards[i], nSecondRoll)

    x = torch.from_numpy(xa.T).to(device)              # (nx, na)
    va = critic_forward(x).squeeze(0)                  # (na,)
    m = int(torch.argmax(va).item())                   # arg on GPU, scalar to host

    target = va[m].detach()                            # scalar tensor on device
    x_selected = torch.from_numpy(xa[m]).to(device).view(nx, 1)  # (nx,1)

    action = possible_moves[m]
    if flipped_flag:
        action = flipped_agent.flip_move(action)

    chosen_after_eff = possible_boards[m].reshape(-1)  # +1 POV numpy board
    return action, x_selected, target, chosen_after_eff

def greedy_action(board, dice, oplayer, nSecondRoll):
    # same as greedy_policy but only returns the move (used in eval)
    flippedplayer = -1
    if (flippedplayer == oplayer):
        board_eff = flipped_agent.flip_board(np.copy(board))
        player_eff = -oplayer
    else:
        board_eff = board
        player_eff = oplayer

    possible_moves, possible_boards = Backgammon.legal_moves(board_eff, dice, player_eff)
    na = len(possible_boards)
    if (na == 0):
        return []

    xa = np.zeros((na, nx), dtype=np.float32)
    for i in range(na):
        xa[i, :] = one_hot_encoding(possible_boards[i], nSecondRoll)

    x = torch.from_numpy(xa.T).to(device)              # (nx, na)
    va = critic_forward(x).squeeze(0)                  # (na,)
    m = int(torch.argmax(va).item())

    action = possible_moves[m]
    if (flippedplayer == oplayer):
        action = flipped_agent.flip_move(action)
    return action

# -------------------- Episode hooks --------------------
def episode_start():
    global Z_w1, Z_b1, Z_w2, Z_b2, Z_w3, Z_b3, Z_w4, Z_b4
    global Zf_w1, Zf_b1, Zf_w2, Zf_b2, Zf_w3, Zf_b3, Zf_w4, Zf_b4
    global xold, xold_flipped, moveNumber

    # critic traces (normal)
    Z_w1 = torch.zeros_like(w1)
    Z_b1 = torch.zeros_like(b1)
    Z_w3 = torch.zeros_like(w3)
    Z_b3 = torch.zeros_like(b3)
    Z_w4 = torch.zeros_like(w4)
    Z_b4 = torch.zeros_like(b4)
    Z_w2 = torch.zeros_like(w2)
    Z_b2 = torch.zeros_like(b2)

    # critic traces (flipped)
    Zf_w1 = torch.zeros_like(w1)
    Zf_b1 = torch.zeros_like(b1)
    Zf_w3 = torch.zeros_like(w3)
    Zf_b3 = torch.zeros_like(b3)
    Zf_w4 = torch.zeros_like(w4)
    Zf_b4 = torch.zeros_like(b4)
    Zf_w2 = torch.zeros_like(w2)
    Zf_b2 = torch.zeros_like(b2)

    xold = None
    xold_flipped = None
    moveNumber = 0

def end_episode(outcome, final_board, perspective):
    pass

def game_over_update(board, reward):
    pass

# -------------------- Utilities --------------------
def set_eval_mode(is_eval: bool):
    global _eval_mode
    _eval_mode = bool(is_eval)

def _zero_param_grads():
    for p in (w1, b1, w3, b3, w4, b4, w2, b2):
        if p.grad is not None:
            p.grad.zero_()

def _g(p: torch.Tensor) -> torch.Tensor:
    # Return grad or zeros_like if grad hasn't populated (safety).
    return p.grad if p.grad is not None else torch.zeros_like(p)

# -------------------- Main action --------------------
def action(board_copy, dice, player, i, train=False, train_config=None):
    global Z_w1, Z_b1, Z_w2, Z_b2, Z_w3, Z_b3, Z_w4, Z_b4
    global Zf_w1, Zf_b1, Zf_w2, Zf_b2, Zf_w3, Zf_b3, Zf_w4, Zf_b4
    global xold, xold_flipped, moveNumber

    nSecondRoll_flag = bool((dice[0] == dice[1]) and (i == 0))

    # Greedy during eval
    if not train or _eval_mode:
        return greedy_action(np.copy(board_copy), dice, player, nSecondRoll_flag)

    # Greedy critic selection; also get features and +1 POV after-state
    act, x, target_val, chosen_after_eff = greedy_policy(
        np.copy(board_copy), dice, player, nRoll=i
    )
    if isinstance(act, list) and len(act) == 0:
        return []

    # terminal check in +1 POV
    is_terminal = (chosen_after_eff[27] == 15)
    flippedplayer = -1

    # Rewards and tgt (critic-only)
    if is_terminal:
        reward  = 1.0 if (player != flippedplayer) else 0.0
        rewardf = 1.0 - reward
        tgt = torch.tensor(0.0, device=device)
    else:
        reward  = 0.0
        rewardf = 0.0
        tgt = target_val  # tensor on device

    # Start updates after at least one full turn has passed and a move happened
    if (moveNumber > 1) and (len(act) > 0):
        # ---- flipped branch OR terminal ----
        if (flippedplayer == player) or is_terminal:
            if xold_flipped is not None:
                y_sigmoid = critic_forward(xold_flipped)  # (1,1)
                y_sigmoid.backward()

                # accumulate traces (flipped)
                Zf_w1 = gamma * lam * Zf_w1 + _g(w1)
                Zf_b1 = gamma * lam * Zf_b1 + _g(b1)
                Zf_w3 = gamma * lam * Zf_w3 + _g(w3)
                Zf_b3 = gamma * lam * Zf_b3 + _g(b3)
                Zf_w4 = gamma * lam * Zf_w4 + _g(w4)
                Zf_b4 = gamma * lam * Zf_b4 + _g(b4)
                Zf_w2 = gamma * lam * Zf_w2 + _g(w2)
                Zf_b2 = gamma * lam * Zf_b2 + _g(b2)

                _zero_param_grads()

                # TD error (flipped)
                delta = rewardf + gamma * tgt - y_sigmoid  # (1,1)
                with torch.no_grad():
                    w1.add_(alpha1 * delta * Zf_w1)
                    b1.add_(alpha1 * delta * Zf_b1)
                    w3.add_(alpha1 * delta * Zf_w3)
                    b3.add_(alpha1 * delta * Zf_b3)
                    w4.add_(alpha1 * delta * Zf_w4)
                    b4.add_(alpha1 * delta * Zf_b4)
                    w2.add_(alpha2 * delta * Zf_w2)
                    b2.add_(alpha2 * delta * Zf_b2)

        # ---- non-flipped branch OR terminal ----
        if (flippedplayer != player) or is_terminal:
            if xold is not None:
                y_sigmoid = critic_forward(xold)  # (1,1)
                y_sigmoid.backward()

                Z_w1 = gamma * lam * Z_w1 + _g(w1)
                Z_b1 = gamma * lam * Z_b1 + _g(b1)
                Z_w3 = gamma * lam * Z_w3 + _g(w3)
                Z_b3 = gamma * lam * Z_b3 + _g(b3)
                Z_w4 = gamma * lam * Z_w4 + _g(w4)
                Z_b4 = gamma * lam * Z_b4 + _g(b4)
                Z_w2 = gamma * lam * Z_w2 + _g(w2)
                Z_b2 = gamma * lam * Z_b2 + _g(b2)

                _zero_param_grads()

                delta = reward + gamma * tgt - y_sigmoid  # (1,1)
                with torch.no_grad():
                    w1.add_(alpha1 * delta * Z_w1)
                    b1.add_(alpha1 * delta * Z_b1)
                    w3.add_(alpha1 * delta * Z_w3)
                    b3.add_(alpha1 * delta * Z_b3)
                    w4.add_(alpha1 * delta * Z_w4)
                    b4.add_(alpha1 * delta * Z_b4)
                    w2.add_(alpha2 * delta * Z_w2)
                    b2.add_(alpha2 * delta * Z_b2)

    # Cache current side’s x as previous (for next turn’s update)
    if len(act) > 0:
        if player == -1:
            xold_flipped = x
        else:
            xold = x

    if not nSecondRoll_flag:
        moveNumber += 1

    return act
