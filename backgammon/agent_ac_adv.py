#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Faithful Actor–Critic TD(λ) for Backgammon (aligned to original script),
adapted to train.py (online updates inside action()).

- Shared trunk (w1,b1) for both actor and critic.
- Critic head (w2,b2) for V(after-state) via sigmoid.
- Actor head is a single row vector theta (1 x nh) over shared h_tanh.
- Two sets of eligibility traces: normal POV (+1) and flipped (-1).
- Same one-hot features (nx = 11*24 + 4 + 1) and update logic/ordering.

API: episode_start(), action(...), set_eval_mode(), save(), load(), end_episode()
"""

from pathlib import Path
import numpy as np
import torch
from torch.autograd import Variable

# ---- engine imports (lower-case backgammon preferred) ----
import backgammon as Backgammon
import flipped_agent as flipped_agent  # your existing flip helpers

# -------------------- Device --------------------
# Original used CPU by default for stability/faithfulness
device = torch.device("cpu")

# -------------------- Hyperparameters (faithful) --------------------
alpha  = 0.1    # actor step size (theta)
alpha1 = 0.001  # critic layer-1 step size (w1,b1)
alpha2 = 0.001  # critic layer-2 step size (w2,b2)
lam    = 0.7    # TD(λ)
gamma  = 1.0    # episodic undiscounted

# -------------------- Features --------------------
nx = 11 * 24 + 4 + 1  # matches your working script
nh = int(nx / 2)

def one_hot_encoding(board, nSecondRoll):
    oneHot = np.zeros(nx, dtype=np.float32)
    # mark where zeros are
    zero_idx = np.where(board[1:25] == 0)[0]
    if zero_idx.size > 0:
        oneHot[zero_idx] = 1.0
    # +1 piles
    for i in range(0, 4):
        idx = np.where(board[1:25] == (i + 1))[0]
        if idx.size > 0:
            oneHot[24 + i * 24 + idx] = 1.0
    # +1 5+ piles (store count-4)
    idx = np.where(board[1:25] >= 5)[0]
    if idx.size > 0:
        oneHot[24 + 4 * 24 + idx] = board[idx + 1] - 4
    # -1 piles
    for i in range(0, 4):
        idx = np.where(board[1:25] == -(i + 1))[0]
        if idx.size > 0:
            oneHot[6 * 24 + i * 24 + idx] = 1.0
    # -1 5+ piles (store count-4)
    idx = np.where(board[1:25] <= -5)[0]
    if idx.size > 0:
        oneHot[6 * 24 + 4 * 24 + idx] = -board[idx + 1] - 4
    # jail/home + second-roll
    oneHot[11 * 24 + 0] = board[25]
    oneHot[11 * 24 + 1] = board[26]
    oneHot[11 * 24 + 2] = board[27]
    oneHot[11 * 24 + 3] = board[28]
    oneHot[11 * 24 + 4] = float(nSecondRoll)
    return oneHot

# -------------------- Parameters (faithful init) --------------------
w1 = Variable(0.1 * torch.randn(nh, nx, device=device, dtype=torch.float), requires_grad=True)
b1 = Variable(torch.zeros((nh, 1), device=device, dtype=torch.float), requires_grad=True)
w2 = Variable(0.1 * torch.randn(1, nh, device=device, dtype=torch.float), requires_grad=True)
b2 = Variable(torch.zeros((1, 1), device=device, dtype=torch.float), requires_grad=True)
theta = Variable(0.1 * torch.randn(1, nh, device=device, dtype=torch.float), requires_grad=True)

# -------------------- Per-episode state --------------------
# critic traces
Z_w1 = Z_b1 = Z_w2 = Z_b2 = None
Zf_w1 = Zf_b1 = Zf_w2 = Zf_b2 = None
# actor traces (theta)
Z_theta = Zf_theta = None

# caches for previous step
xold = xold_flipped = None
gradlnpi = gradlnpi_flipped = None
advantage = 0.0
advantage_flipped = 0.0
I = 1.0
If = 1.0
moveNumber = 0

_eval_mode = False

# -------------------- Save / Load --------------------
CKPT_DEFAULT = Path("checkpoints/td_lambda_ac.pt")

def save(path: str | None = None):
    p = Path(path) if path else CKPT_DEFAULT
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"w1": w1, "w2": w2, "b1": b1, "b2": b2, "theta": theta}, p)

def load(path: str | None = None, map_location: str | torch.device = "cpu"):
    p = Path(path) if path else CKPT_DEFAULT
    state = torch.load(p, map_location=map_location)
    w1.data.copy_(state["w1"].data);  w2.data.copy_(state["w2"].data)
    b1.data.copy_(state["b1"].data);  b2.data.copy_(state["b2"].data)
    theta.data.copy_(state["theta"].data)
    set_eval_mode(True)

def set_eval_mode(is_eval: bool):
    global _eval_mode
    _eval_mode = bool(is_eval)

# -------------------- Policy helpers (faithful) --------------------
def greedy_action(board, dice, oplayer, nSecondRoll):
    flippedplayer = -1
    if flippedplayer == oplayer:
        board = flipped_agent.flip_board(np.copy(board))
        player = -oplayer
    else:
        player = oplayer

    possible_moves, possible_boards = Backgammon.legal_moves(board, dice, player)
    na = len(possible_boards)
    if na == 0:
        return []

    xa = np.zeros((na, nx), dtype=np.float32)
    for i in range(na):
        xa[i, :] = one_hot_encoding(possible_boards[i], nSecondRoll)
    x = Variable(torch.tensor(xa.T, dtype=torch.float, device=device))

    h = torch.mm(w1, x) + b1
    h_tanh = h.tanh()
    y = torch.mm(w2, h_tanh) + b2
    va = y.sigmoid().detach().cpu().numpy()
    action = possible_moves[int(np.argmax(va))]
    if flippedplayer == oplayer:
        action = flipped_agent.flip_move(action)
    return action

def softmax_policy(board, dice, oplayer, nRoll):
    """
    Returns:
      action (list of moves),
      x_selected (nx,1) tensor for chosen after-state (not strictly required for update),
      target (scalar tensor value prediction of chosen after-state),
      advantage (float),
      chosen_after_eff (numpy 1D of after-state in +1 POV),
      flipped_flag (bool)
    """
    flippedplayer = -1
    nSecondRoll = bool((dice[0] == dice[1]) and (nRoll == 0))
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
        return [], None, None, None, 0.0, None, flipped_flag

    xa = np.zeros((na, nx), dtype=np.float32)
    for i in range(na):
        xa[i, :] = one_hot_encoding(possible_boards[i], nSecondRoll)
    x = Variable(torch.tensor(xa.T, dtype=torch.float, device=device))

    # shared trunk
    h = torch.mm(w1, x) + b1
    h_tanh = h.tanh()

    # actor logits via single-row theta (1 x nh) -> (1 x na)
    logits = torch.mm(theta, h_tanh)
    pi = logits.softmax(dim=1)

    # sample action
    m = torch.multinomial(pi, 1)              # shape (1,1)
    m_idx = int(m.item())
    action = possible_moves[m_idx]
    if flipped_flag:
        action = flipped_agent.flip_move(action)

    # critic on after-states
    y = torch.mm(w2, h_tanh) + b2
    va = y.sigmoid()
    target = va.data[0, m_idx]                # scalar tensor

    # advantage as in original
    advantage = (target - torch.sum(pi * va)).item()

    # grad ln pi(a) wrt theta (1 x nh), faithful construction:
    # grad = h_tanh[:,a]^T - sum_j pi_j * h_tanh[:,j]^T
    xtheta_mean = torch.sum(h_tanh * pi, dim=1)                  # (nh,)
    h_a = h_tanh[:, m_idx]                                       # (nh,)
    grad_ln_pi = (h_a - xtheta_mean).view(1, -1).detach()        # (1 x nh)

    x_selected = Variable(torch.tensor(xa[m_idx, :], dtype=torch.float, device=device)).view(nx, 1)
    chosen_after_eff = possible_boards[m_idx].reshape(-1)

    return action, x_selected, target, grad_ln_pi, advantage, chosen_after_eff, flipped_flag
    
# -------------------- Episode hooks --------------------
def episode_start():
    global Z_w1, Z_b1, Z_w2, Z_b2, Zf_w1, Zf_b1, Zf_w2, Zf_b2
    global Z_theta, Zf_theta
    global xold, xold_flipped, gradlnpi, gradlnpi_flipped
    global advantage, advantage_flipped, I, If, moveNumber

    Z_w1 = torch.zeros_like(w1.data)
    Z_b1 = torch.zeros_like(b1.data)
    Z_w2 = torch.zeros_like(w2.data)
    Z_b2 = torch.zeros_like(b2.data)

    Zf_w1 = torch.zeros_like(w1.data)
    Zf_b1 = torch.zeros_like(b1.data)
    Zf_w2 = torch.zeros_like(w2.data)
    Zf_b2 = torch.zeros_like(b2.data)

    Z_theta = torch.zeros_like(theta.data)
    Zf_theta = torch.zeros_like(theta.data)

    xold = None
    xold_flipped = None
    gradlnpi = None
    gradlnpi_flipped = None

    advantage = 0.0
    advantage_flipped = 0.0

    I = 1.0
    If = 1.0
    moveNumber = 0

def end_episode(outcome, final_board, perspective):
    # original resets per episode; no extra terminal bookkeeping needed here
    pass

def game_over_update(board, reward):
    # compatibility hook (not used in this faithful port)
    pass

# -------------------- Main action (called by train.py) --------------------
def action(board_copy, dice, player, i, train=False, train_config=None):
    global Z_w1, Z_b1, Z_w2, Z_b2, Zf_w1, Zf_b1, Zf_w2, Zf_b2
    global Z_theta, Zf_theta
    global xold, xold_flipped, gradlnpi, gradlnpi_flipped
    global advantage, advantage_flipped, I, If, moveNumber

    nSecondRoll_flag = bool((dice[0] == dice[1]) and (i == 0))
    flippedplayer = -1

    # Greedy during eval
    if (not train) or _eval_mode:
        return greedy_action(np.copy(board_copy), dice, player, nSecondRoll_flag)

    # Sample action + get targets/grad
    out = softmax_policy(np.copy(board_copy), dice, player, nRoll=i)
    act, x, target_val, grad_ln_pi, A, chosen_after_eff, flipped_flag = out
    if isinstance(act, list) and len(act) == 0:
        # no legal moves
        if not nSecondRoll_flag:
            moveNumber += 1
        return []

    # Terminal check using chosen after-state in +1 POV
    is_terminal = (chosen_after_eff[27] == 15)

    # Rewards exactly as in original
    if is_terminal:
        reward  = 1.0 if (player != flippedplayer) else 0.0
        rewardf = 1.0 - reward
        tgt = torch.tensor(0.0, device=device, dtype=torch.float)
    else:
        reward  = 0.0
        rewardf = 0.0
        tgt = target_val  # scalar tensor

    # Start updates after at least one full turn and a move happened
    if (moveNumber > 1) and (len(act) > 0):
        # ----- flipped branch OR terminal -----
        if (flippedplayer == player) or is_terminal:
            if xold_flipped is not None:
                # critic forward/backward on previous flipped after-state
                h = torch.mm(w1, xold_flipped) + b1
                h_tanh = h.tanh()
                y = torch.mm(w2, h_tanh) + b2
                y_sigmoid = y.sigmoid()
                y_sigmoid.backward()

                # update critic traces
                Zf_w1 = gamma * lam * Zf_w1 + w1.grad.data
                Zf_b1 = gamma * lam * Zf_b1 + b1.grad.data
                Zf_w2 = gamma * lam * Zf_w2 + w2.grad.data
                Zf_b2 = gamma * lam * Zf_b2 + b2.grad.data
                # zero grads
                w1.grad.data.zero_(); b1.grad.data.zero_()
                w2.grad.data.zero_(); b2.grad.data.zero_()

                # actor traces with stored grad ln pi from previous flipped step
                if gradlnpi_flipped is not None:
                    Zf_theta = gamma * lam * Zf_theta + If * gradlnpi_flipped

                # TD error (scalar tensor broadcast over params)
                delta = torch.tensor(rewardf, device=device) + gamma * tgt - y_sigmoid.detach()
                # critic updates
                w1.data = w1.data + alpha1 * delta * Zf_w1
                b1.data = b1.data + alpha1 * delta * Zf_b1
                w2.data = w2.data + alpha2 * delta * Zf_w2
                b2.data = b2.data + alpha2 * delta * Zf_b2
                # actor update (faithful: uses advantage from previous flipped step)
                theta.data = theta.data + alpha * advantage_flipped * Zf_theta

                If = If * gamma

        # ----- non-flipped branch OR terminal -----
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

                if gradlnpi is not None:
                    Z_theta = gamma * lam * Z_theta + I * gradlnpi

                delta = torch.tensor(reward, device=device) + gamma * tgt - y_sigmoid.detach()
                w1.data = w1.data + alpha1 * delta * Z_w1
                b1.data = b1.data + alpha1 * delta * Z_b1
                w2.data = w2.data + alpha2 * delta * Z_w2
                b2.data = b2.data + alpha2 * delta * Z_b2

                theta.data = theta.data + alpha * advantage * Z_theta

                I = gamma * I

    # cache current side’s features & actor grad ln pi
    if x is not None and len(act) > 0:
        if player == -1:
            xold_flipped = x
            gradlnpi_flipped = grad_ln_pi
            advantage_flipped = A
        else:
            xold = x
            gradlnpi = grad_ln_pi
            advantage = A

    if not nSecondRoll_flag:
        moveNumber += 1

    return act
