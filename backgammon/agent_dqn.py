#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Greedy-only PyTorch DQN agent for Backgammon (after-state evaluation).

API:
  - action(board, dice, player, i, train=False, train_config=None)
  - game_over_update(board, reward)
  - set_eval_mode(is_eval)
  - save(path="checkpoints/best.pt")
  - load(path="checkpoints/best.pt")

Key points:
- SAME network architecture as your attached agent.py:
    QNet: Linear(state_dim, 256) + ReLU -> Linear(256, 256) + ReLU -> Linear(256, 1)
- Selection is ALWAYS greedy (no epsilon).
- We evaluate legal after-states for the current roll (single call to update_board in env).
- Replay stores (s = chosen after-state, r, s_next = next decision point after opponent's turn, done).
- Targets: r (terminal) else r + γ * V_targ(s_next).
"""

from collections import deque
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import backgammon  # engine

# ---------------- Config ----------------
class Config:
    state_dim = 24 + 4 + 1      # 24 points + (bar_self, bar_opp, off_self, off_opp) + moves_left
    gamma = 0.99
    lr = 1e-3
    batch_size = 256
    buffer_size = 100_000
    start_learning_after = 2_000
    target_update_every = 2_000
    train_every = 1
    hidden = 256                # <- same width as your agent.py
    device = "cuda" if torch.cuda.is_available() else "cpu"

CFG = Config()

# ------------- Flip helpers (29-length boards) -------------
_FLIP_IDX = np.array(
    [0, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13,
     12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 26, 25, 28, 27],
    dtype=np.int32
)

def _flip_board(board):
    out = np.empty(29, dtype=board.dtype)
    out[:] = -board[_FLIP_IDX]
    return out

def _flip_move(move):
    if len(move) == 0:
        return move
    mv = np.asarray(move, dtype=np.int32).copy()
    for r in range(mv.shape[0]):
        mv[r, 0] = _FLIP_IDX[mv[r, 0]]
        mv[r, 1] = _FLIP_IDX[mv[r, 1]]
    return mv

# ------------- Replay Buffer -------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, s, r, s_next, done):
        # Normalize types to avoid np.bool_ surprises downstream
        self.buf.append((s, float(r), s_next, bool(done)))

    def __len__(self):
        return len(self.buf)

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, r, s2, d = zip(*batch)

        # Explicit, dense arrays (types are important here)
        S  = np.stack(s).astype(np.float32, copy=False)
        S2 = np.stack(s2).astype(np.float32, copy=False)
        D  = np.asarray(d, dtype=np.bool_)   # <— key line

        return (
            torch.as_tensor(S,  dtype=torch.float32, device=CFG.device),
            torch.as_tensor(r,  dtype=torch.float32, device=CFG.device).unsqueeze(1),
            torch.as_tensor(S2, dtype=torch.float32, device=CFG.device),
            torch.from_numpy(D).to(device=CFG.device).unsqueeze(1),  # <— avoids the error
        )

# ------------- Model (IDENTICAL architecture to your agent.py) -------------
class QNet(nn.Module):
    def __init__(self, in_dim=CFG.state_dim, hid=CFG.hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid),    nn.ReLU(),
            nn.Linear(hid, 1),
        )
    def forward(self, x):
        return self.net(x)

_qnet = QNet().to(CFG.device)
_tnet = QNet().to(CFG.device)
_tnet.load_state_dict(_qnet.state_dict())
_opt = torch.optim.Adam(_qnet.parameters(), lr=CFG.lr)
_buf = ReplayBuffer(CFG.buffer_size)

_steps = 0
_eval_mode = False

# ------------- Save / Load -------------
CHECKPOINT_PATH = Path("checkpoints/best.pt")
_loaded_from_disk = False

def save(path: str = str(CHECKPOINT_PATH)):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"qnet": _qnet.state_dict()}, p)

def load(path: str = str(CHECKPOINT_PATH), map_location: str | torch.device = "cpu"):
    global _loaded_from_disk
    state = torch.load(path, map_location=map_location)
    _qnet.load_state_dict(state["qnet"])
    _tnet.load_state_dict(_qnet.state_dict())
    set_eval_mode(True)
    _loaded_from_disk = True

def _lazy_load_if_available():
    global _loaded_from_disk
    if _loaded_from_disk:
        return
    if CHECKPOINT_PATH.exists():
        try:
            load(str(CHECKPOINT_PATH), map_location="cpu")
        except Exception:
            pass
        _loaded_from_disk = True

# ------------- Features -------------
def _moves_left(dice, i):
    # doubles -> two applications; otherwise one
    return 1 + int(dice[0] == dice[1]) - i

def _encode_state(board_plus_one, moves_left):
    """
    +1 POV features: 24 points (scaled), bar_self, bar_opp, off_self, off_opp, moves_left
    """
    x = np.zeros(CFG.state_dim, dtype=np.float32)
    x[:24] = board_plus_one[1:25] * 0.2
    x[24]  = board_plus_one[25] * 0.2
    x[25]  = board_plus_one[26] * 0.2
    x[26]  = board_plus_one[27] / 15.0
    x[27]  = board_plus_one[28] / 15.0
    x[28]  = float(moves_left)
    return x

def _is_terminal_plus_one(board_plus_one):
    return board_plus_one[27] == 15

# ------------- Learning (after-state → after-state) -------------
def _maybe_learn():
    global _steps
    if _eval_mode: return
    if len(_buf) < CFG.start_learning_after: return
    if _steps % CFG.train_every != 0: return

    S, R, S2, D = _buf.sample(CFG.batch_size)
    with torch.no_grad():
        Q2 = _tnet(S2)
        mask = (~D).float()
        target = R + mask * (CFG.gamma * Q2)

    Q = _qnet(S)
    loss = F.mse_loss(Q, target)

    _opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(_qnet.parameters(), 1.0)
    _opt.step()

    if _steps % CFG.target_update_every == 0:
        _tnet.load_state_dict(_qnet.state_dict())

# ------------- Hooks -------------
def set_eval_mode(is_eval: bool):
    global _eval_mode
    _eval_mode = bool(is_eval)
    if _eval_mode: _qnet.eval()
    else:          _qnet.train()

def episode_start():
    pass

def end_episode(outcome, final_board, perspective):
    pass

def game_over_update(board, reward):
    """
    Trainer may call this at episode end (twice: normal & flipped).
    Encode as +1 POV with moves_left=0 and push a terminal sample.
    """
    s = _encode_state(board, moves_left=0)
    _buf.push(s, float(reward), s, True)

# ------------- Opponent turn → next after-state features -------------
def _s_next_after_opponent(chosen_board_plus_one: np.ndarray) -> np.ndarray:
    """
    From our chosen after-state (+1 POV), roll opponent dice, enumerate their
    legal after-states (player=-1), flip to +1 POV, encode with moves_left=1
    (our next first move), pick best-for-+1 using the TARGET net.
    """
    opp_dice = backgammon.roll_dice()
    opp_moves, opp_boards = backgammon.legal_moves(chosen_board_plus_one, opp_dice, player=-1)

    if len(opp_boards) == 0:
        return _encode_state(chosen_board_plus_one, moves_left=1)

    feats = np.stack([_encode_state(_flip_board(b), moves_left=1) for b in opp_boards], axis=0)
    feats_t = torch.as_tensor(feats, dtype=torch.float32, device=CFG.device)
    with torch.no_grad():
        vals = _tnet(feats_t).squeeze(1)
        idx = int(torch.argmax(vals).item())
    return feats[idx]

# ------------- Policy (ALWAYS greedy) -------------
def action(board_copy, dice, player, i, train=False, train_config=None):
    """
    Returns [] if no legal moves, else an array of shape (k,2) of [start,end] moves.
    Selection is ALWAYS greedy (no epsilon). If train=True, learn with after-state tuples.
    """
    global _steps
    if not train:
        _lazy_load_if_available()

    # Work in +1 POV
    board_pov = _flip_board(board_copy) if player == -1 else board_copy

    # Enumerate legal after-states from current dice
    possible_moves, possible_boards = backgammon.legal_moves(board_pov, dice, player=1)
    nA = len(possible_moves)
    if nA == 0:
        return []

    moves_left_now = _moves_left(dice, i)
    moves_left_after = max(0, moves_left_now - 1)
    Sp = np.stack([_encode_state(b, moves_left_after) for b in possible_boards], axis=0)
    Sp_t = torch.as_tensor(Sp, dtype=torch.float32, device=CFG.device)

    # Greedy selection (no exploration)
    with torch.no_grad():
        q_after = _qnet(Sp_t).squeeze(1)  # [nA]
        a_idx = int(torch.argmax(q_after).item())

    chosen_move = possible_moves[a_idx]
    chosen_board_plus_one = possible_boards[a_idx]
    s_after = Sp[a_idx]

    # Terminal & reward from +1 POV
    done = _is_terminal_plus_one(chosen_board_plus_one)
    r = 1.0 if done else 0.0

    if train and (not _eval_mode):
        if not done:
            s_next = _s_next_after_opponent(chosen_board_plus_one)
        else:
            s_next = s_after  # masked by 'done'
        _buf.push(s_after, float(r), s_next, bool(done))
        _steps += 1
        _maybe_learn()

    # Return move in ORIGINAL POV
    if player == -1:
        chosen_move = _flip_move(chosen_move)
    return chosen_move
