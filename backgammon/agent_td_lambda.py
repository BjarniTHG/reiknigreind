#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Actor–Critic with TD(λ) eligibility traces — faithful to the original script,
adapted to the train.py harness (online updates inside action()).

- Keep the original parameterization (w1,w2,b1,b2) for the critic and
  (theta_w1, theta_w2, theta_b1, theta_b2) for the actor.
- Preserve one_hot_encoding, softmax_policy (sampling) and greedy_action (eval).
- Maintain separate eligibility traces for +1 POV and 'flipped' (-1) updates.
- Move-level updates replicate the original: use stored theta grads from the
  *previous* step per side, multiply traces by γλ, and apply TD error δ.

Note: The environment file is lower-case 'backgammon.py'. We import with a
fallback in case your local module is named 'Backgammon.py'.
"""

import numpy as np
import torch
from torch.autograd import Variable

# ---- engine imports (lower-case backgammon as requested) ----
try:
    import backgammon as Backgammon
except Exception:
    import Backgammon  # fallback if your file happens to be upper-case

import flipped_agent as flipped_agent  # uses your existing flip helpers

# -------------------- Device --------------------
# Original code ran on CPU by default; keep that behavior.
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print(f"[agent] Using device: {device}")

# (optional, faster matmul on Ada/Lovelace)
if device.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True

# -------------------- Hyperparameters (faithful) --------------------
alpha  = 0.05   # (kept from original; not used directly in updates below)
alpha1 = 0.001  # step size layer 1 (critic & actor heads respectively)
alpha2 = 0.001  # step size layer 2
lam    = 0.7    # TD(λ)
gamma  = 1.0    # episodic, terminal reward

# -------------------- Features --------------------
nx = 24 * 2 * 6 + 4 + 1  # unchanged from original

def one_hot_encoding(board, nSecondRoll):
    oneHot = np.zeros(24 * 2 * 6 + 4 + 1)
    # where are the zeros, single, double, ... discs (player +1)
    for i in range(0,5):
        idx = np.where(board[1:25] == i)[0] - 1
        if idx.size > 0:
            oneHot[i*24 + idx] = 1
    # anything >= 5
    idx = np.where(board[1:25] >= 5)[0] - 1
    if idx.size > 0:
        oneHot[5*24 + idx] = 1
    # now repeat for player -1 (negative counts)
    for i in range(0,5):
        idx = np.where(board[1:25] == -i)[0] - 1
        if idx.size > 0:
            oneHot[6*24 + i*24 + idx] = 1
    idx = np.where(board[1:25] <= -5)[0] - 1
    if idx.size > 0:
        oneHot[6*24 + 5*24 + idx] = 1
    # jail and home indicators + second-roll flag
    oneHot[12 * 24 + 0] = board[25]
    oneHot[12 * 24 + 1] = board[26]
    oneHot[12 * 24 + 2] = board[27]
    oneHot[12 * 24 + 3] = board[28]
    oneHot[12 * 24 + 4] = nSecondRoll
    return oneHot

# -------------------- Parameters (faithful init) --------------------
# Critic
w1 = Variable(0.1*torch.randn(int(nx/2), nx,                device=device, dtype=torch.float), requires_grad=True)
b1 = Variable(torch.zeros((int(nx/2),1),                     device=device, dtype=torch.float), requires_grad=True)
w2 = Variable(0.1*torch.randn(1,           int(nx/2),        device=device, dtype=torch.float), requires_grad=True)
b2 = Variable(torch.zeros((1,1),                              device=device, dtype=torch.float), requires_grad=True)

# -------------------- Per-episode state (faithful) --------------------
# Eligibility traces (critic & actor) for +1 and flipped (-1) branches
Z_w1 = Z_b1 = Z_w2 = Z_b2 = None
Zf_w1 = Zf_b1 = Zf_w2 = Zf_b2 = None

# Previous-step caches (board encodings and actor grads) per side
xold = xold_flipped = None

# Discount factors used in traces for actor (faithful names)
I  = 1.0
If = 1.0

# Move counter (faithful logic uses moveNumber > 1 gate)
moveNumber = 0

# Eval/train switch
_eval_mode = False

# -------------------- Save / Load (for train.py checkpoints) --------------------
from pathlib import Path
CKPT_DEFAULT = Path("checkpoints/agent_td_lambda.pt")

def save(path: str | None = None):
    p = Path(path) if path else CKPT_DEFAULT
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "w1": w1, "w2": w2, "b1": b1, "b2": b2,
    }, p)

def load(self, path=None, map_location=None):
    p = Path(path) if path else CKPT_DEFAULT
    ml = map_location or device
    state = torch.load(p, map_location=ml)
    # assign (Variables keep requires_grad=True)
    w1.data.copy_(state["w1"].data);  w2.data.copy_(state["w2"].data)
    b1.data.copy_(state["b1"].data);  b2.data.copy_(state["b2"].data)
    set_eval_mode(True)  # play greedily after load

def set_eval_mode(is_eval: bool):
    global _eval_mode
    _eval_mode = bool(is_eval)

# -------------------- Policy helpers (faithful) --------------------
def random_action(board_copy, dice, player, nSecondRoll):
    possible_moves, _ = Backgammon.legal_moves(board_copy, dice, player)
    if len(possible_moves) == 0:
        return []
    move = possible_moves[np.random.randint(len(possible_moves))]
    return move

def greedy_policy(board, dice, oplayer, nRoll):
    flippedplayer = -1
    nSecondRoll = ((dice[0] == dice[1]) & (nRoll == 0))
    flipped_flag = (flippedplayer == oplayer)

    if flipped_flag:  # view it from player 1 perspective
        board_eff = flipped_agent.flip_board(np.copy(board))
        player_eff = -oplayer  # +1
    else:
        board_eff = board
        player_eff = oplayer   # +1

    possible_moves, possible_boards = Backgammon.legal_moves(board_eff, dice, player_eff)
    na = len(possible_moves)
    if na == 0:
        return [], [], [], None, flipped_flag

    xa = np.zeros((na, nx))
    for i in range(na):
        xa[i, :] = one_hot_encoding(possible_boards[i], nSecondRoll)
    x = Variable(torch.tensor(xa.transpose(), dtype=torch.float, device=device))

    # ---- CRITIC on after-states ----
    h = torch.mm(w1, x) + b1
    h_tanh = h.tanh()
    y = torch.mm(w2, h_tanh) + b2
    va = y.sigmoid()
    # select move index m that maximizes va
    m = int(torch.argmax(va).item())
    target = va.data[0, m]

    # action in caller's orientation
    action = possible_moves[m]
    if flipped_flag:
        action = flipped_agent.flip_move(action)

    # features of chosen after-state (not strictly needed here, kept from original)
    x_selected = Variable(
        torch.tensor(one_hot_encoding(possible_boards[m], nSecondRoll),
                     dtype=torch.float, device=device)
    ).view(nx, 1)

    # return also the chosen after-state board in +1 POV (unflipped)
    chosen_after_eff = possible_boards[m].reshape(-1)  # numpy 1D

    return action, x_selected, target, chosen_after_eff, flipped_flag


def greedy_action(board, dice, oplayer, nSecondRoll):
    flippedplayer = -1
    if (flippedplayer == oplayer):  # view from +1 POV
        board = flipped_agent.flip_board(np.copy(board))
        player = -oplayer
    else:
        player = oplayer

    possible_moves, possible_boards = Backgammon.legal_moves(board, dice, player)
    na = len(possible_boards)
    if (na == 0):
        return []

    xa = np.zeros((na, nx))
    for i in range(0, na):
        xa[i, :] = one_hot_encoding(possible_boards[i], nSecondRoll)
    x = Variable(torch.tensor(xa.transpose(), dtype=torch.float, device=device))

    h = torch.mm(w1, x) + b1
    h_tanh = h.tanh()
    y = torch.mm(w2, h_tanh) + b2
    va = y.sigmoid().detach().cpu().numpy()
    action = possible_moves[np.argmax(va)]
    if (flippedplayer == oplayer):
        action = flipped_agent.flip_move(action)
    return action

# -------------------- Episode hooks for train.py --------------------
def episode_start():
    """Reset traces and per-episode caches (faithful)."""
    global Z_w1, Z_b1, Z_w2, Z_b2
    global Zf_w1, Zf_b1, Zf_w2, Zf_b2
    global xold, xold_flipped
    global I, If, moveNumber

    Z_w1 = torch.zeros(w1.size(), device=device, dtype=torch.float)
    Z_b1 = torch.zeros(b1.size(), device=device, dtype=torch.float)
    Z_w2 = torch.zeros(w2.size(), device=device, dtype=torch.float)
    Z_b2 = torch.zeros(b2.size(), device=device, dtype=torch.float)

    Zf_w1 = torch.zeros(w1.size(), device=device, dtype=torch.float)
    Zf_b1 = torch.zeros(b1.size(), device=device, dtype=torch.float)
    Zf_w2 = torch.zeros(w2.size(), device=device, dtype=torch.float)
    Zf_b2 = torch.zeros(b2.size(), device=device, dtype=torch.float)

    xold = None
    xold_flipped = None

    I = 1.0
    If = 1.0
    moveNumber = 0

def end_episode(outcome, final_board, perspective):
    """No extra updates here; original code resets between games."""
    pass

def game_over_update(board, reward):
    """Compatibility hook; original script handled rewards inline."""
    pass

# -------------------- Main action (called by train.py) --------------------
def action(board_copy, dice, player, i, train=False, train_config=None):
    global Z_w1, Z_b1, Z_w2, Z_b2
    global Zf_w1, Zf_b1, Zf_w2, Zf_b2
    global xold, xold_flipped
    global I, If, moveNumber

    nSecondRoll_flag = bool((dice[0] == dice[1]) and (i == 0))

    # Greedy during eval (unchanged)
    if not train or _eval_mode:
        return greedy_action(np.copy(board_copy), dice, player, nSecondRoll_flag)

    # Sample from actor; now also receive chosen after-state in +1 POV
    act, x, target_val, chosen_after_eff, flipped_flag = greedy_policy(
        np.copy(board_copy), dice, player, nRoll=i
    )
    if isinstance(act, list) and len(act) == 0:
        return []

    # ---- terminal check (faithful to original learnit) ----
    # In +1 POV, terminal if all 15 borne off -> after_state[27] == 15
    is_terminal = (chosen_after_eff[27] == 15)
    flippedplayer = -1

    # Set reward signals exactly as in the original loop
    if is_terminal:
        reward  = 1.0 if (player != flippedplayer) else 0.0
        rewardf = 1.0 - reward
        tgt = 0.0  # target <- 0 at terminal
    else:
        reward  = 0.0
        rewardf = 0.0
        tgt = target_val  # target from critic on after-state

    # Start updates after at least one full turn has passed and a move happened
    if (moveNumber > 1) and (len(act) > 0):
        # ---- flipped branch OR terminal ----
        if (flippedplayer == player) or is_terminal:
            if xold_flipped is not None:
                # critic on previous flipped x
                h = torch.mm(w1, xold_flipped) + b1
                h_tanh = h.tanh()
                y = torch.mm(w2, h_tanh) + b2
                y_sigmoid = y.sigmoid()
                y_sigmoid.backward()

                # traces (critic)
                Zf_w1 = gamma * lam * Zf_w1 + w1.grad.data
                Zf_b1 = gamma * lam * Zf_b1 + b1.grad.data
                Zf_w2 = gamma * lam * Zf_w2 + w2.grad.data
                Zf_b2 = gamma * lam * Zf_b2 + b2.grad.data
                w1.grad.data.zero_(); b1.grad.data.zero_()
                w2.grad.data.zero_(); b2.grad.data.zero_()

                # faithful TD error
                delta = rewardf + gamma * tgt - y_sigmoid
                # critic update
                w1.data = w1.data + alpha1 * delta * Zf_w1
                b1.data = b1.data + alpha1 * delta * Zf_b1
                w2.data = w2.data + alpha2 * delta * Zf_w2
                b2.data = b2.data + alpha2 * delta * Zf_b2
                If = If * gamma

        # ---- non-flipped branch OR terminal ----
        if (flippedplayer != player) or is_terminal:
            if xold is not None:
                h = torch.mm(w1, xold) + b1
                h_tanh = h.tanh()
                y = torch.mm(w2, h_tanh) + b2
                y_sigmoid = y.sigmoid()
                y_sigmoid.backward()

                Z_w1 = gamma * lam * Z_w1 + w1.grad.data
                Z_b1 = gamma * lam * Z_b1 + b1.grad.data
                Z_w2 = gamma * lam * Z_w2 + w2.grad.data
                Z_b2 = gamma * lam * Z_b2 + b2.grad.data
                w1.grad.data.zero_(); b1.grad.data.zero_()
                w2.grad.data.zero_(); b2.grad.data.zero_()

                delta = reward + gamma * tgt - y_sigmoid
                w1.data = w1.data + alpha1 * delta * Z_w1
                b1.data = b1.data + alpha1 * delta * Z_b1
                w2.data = w2.data + alpha2 * delta * Z_w2
                b2.data = b2.data + alpha2 * delta * Z_b2
                I = gamma * I

    # cache current side’s x and actor grads (unchanged)
    if len(act) > 0:
        if player == -1:
            xold_flipped = x
        else:
            xold = x

    if not nSecondRoll_flag:
        moveNumber += 1

    return act
