#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal greedy agent for competition.
Loads checkpoints/best.pt (produced by train.py) and selects the move
that maximizes Q(after-state) with no exploration.

Exposes:
  action(board_copy, dice, player, i=0, **_) -> [] or (k,2) array
"""

import numpy as np
import torch
import torch.nn as nn
import Backgammon

# -------- Constants (must match training) --------
STATE_DIM = 24 + 4 + 1
HID = 256
WEIGHTS_PATH = "checkpoints/best.pt"

# -------- Flip helpers --------
_FLIP_IDX = np.array([0,24,23,22,21,20,19,18,17,16,15,14,13,
                      12,11,10,9,8,7,6,5,4,3,2,1,26,25,28,27], dtype=np.int32)

def _flip_board(b):
    return -b[_FLIP_IDX]

def _flip_move(m):
    if len(m) == 0:
        return m
    m = np.asarray(m, dtype=np.int32).copy()
    for r in range(m.shape[0]):
        m[r,0] = _FLIP_IDX[m[r,0]]
        m[r,1] = _FLIP_IDX[m[r,1]]
    return m

def _moves_left(dice, i):
    return 1 + int(dice[0] == dice[1]) - i

def _encode_state(board_flipped, moves_left):
    x = np.zeros(STATE_DIM, dtype=np.float32)
    x[:24] = board_flipped[1:25] * 0.2         # points
    x[24]  = board_flipped[25] * 0.2           # self bar
    x[25]  = board_flipped[26] * 0.2           # opp bar
    x[26]  = board_flipped[27] / 15.0          # self off
    x[27]  = board_flipped[28] / 15.0          # opp off
    x[28]  = float(moves_left)                 # moves left this turn
    return x

# -------- Model (must match training) --------
class QNet(nn.Module):
    def __init__(self, in_dim=STATE_DIM, hid=HID):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, 1),
        )
    def forward(self, x):
        return self.net(x)

_q = QNet()
_q.eval()

# Load weights if present
try:
    state = torch.load(WEIGHTS_PATH, map_location="cpu")
    _q.load_state_dict(state["qnet"])
except Exception:
    # If weights are missing, the agent will run with random init (weak)
    pass

@torch.no_grad()
def action(board_copy, dice, player, i=0, **_):
    # Always evaluate from +1 perspective
    board_pov = _flip_board(board_copy) if player == -1 else board_copy

    # Enumerate legal move sequences & after-states
    possible_moves, possible_boards = Backgammon.legal_moves(board_pov, dice, player=1)
    if not possible_moves:
        return []

    moves_left = _moves_left(dice, i)
    S_primes = np.stack([_encode_state(b, moves_left - 1) for b in possible_boards], axis=0)
    Sp_t = torch.as_tensor(S_primes, dtype=torch.float32)  # CPU

    # Greedy argmax
    qvals = _q(Sp_t).squeeze(1)
    a_idx = int(torch.argmax(qvals).item())

    move = possible_moves[a_idx]
    return _flip_move(move) if player == -1 else move
