from __future__ import annotations
from typing import Optional, List, Tuple
from dataclasses import dataclass


import os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim

from backgammon import (
    init_board,
    roll_dice,
    game_over,
    legal_moves,
)
# ^ same professor backgammon.py API as v1

@dataclass
class EnvState:
    board: np.ndarray
    player: int
    done: bool = False
    moves_in_episode: int = 0  # how many plies in the current game


@dataclass
class Transition:
    board: np.ndarray
    player: int
    reward: float
    next_board: np.ndarray
    next_player: int
    done: bool



# ============================================================
# V2: Conv TD(0) with reward shaping + deeper net + dropout
# ============================================================

# --- Training config for V2 ---
DEFAULT_TOTAL_EPISODES = 1_000_000

# Epsilon schedule (we can tweak later if needed)
E_START = 0.3     # start fairly exploratory
E_END   = 0.01    # end almost greedy

# Discount + learning rate
GAMMA = 0.99
LR    = 1e-4

# Evaluation + checkpointing
EVAL_INTERVAL   = 10_000      # evaluate vs random every N episodes
EVAL_NUM_GAMES  = 500         # games per evaluation
CHECKPOINT_DIR  = "models_v2"
BASE_MODEL_NAME = "conv_td0_v2_bjarni"

# Use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[V2-INFO] Using device: {DEVICE}")


# ============================================================
# Board encoding: 1D conv-friendly representation
# ============================================================

# We keep the same encoding as in v1 so that changes in v2 are
# about the *learning setup* and network depth, not representation.
MAX_STACK = 5                 # cap at 5 and use ">=6" as final bin
NUM_POINT_CHANNELS = 12       # 6 for me, 6 for opponent
NUM_GLOBAL_FEATS = 4          # [my_bar, opp_bar, my_borne, opp_borne]


def encode_board_np(board: np.ndarray, player: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode a single board position from the perspective of `player`.

    Args:
        board: np.ndarray of shape (29,), dtype int32.
               Indices:
                 1..24: points
                 25: bar for +1
                 26: bar for -1
                 27: borne-off +1
                 28: borne-off -1
        player: +1 or -1 (current player to move).

    Returns:
        point_tensor: np.ndarray, shape (24, NUM_POINT_CHANNELS)
            One row per point, channels for my/opp checker counts.
        global_feats: np.ndarray, shape (NUM_GLOBAL_FEATS,)
            [my_bar, opp_bar, my_borne, opp_borne], normalized to [0,1].
    """
    # Make "me" always positive, "opponent" negative.
    signed = board.astype(np.int32) * int(player)

    # 24 points on the board
    points = signed[1:25]  # shape (24,)

    point_tensor = np.zeros((24, NUM_POINT_CHANNELS), dtype=np.float32)

    for i, c in enumerate(points):
        if c > 0:
            # My checkers on this point
            v = int(c)
            idx = min(v, MAX_STACK)  # 1..5 or ">=6"
            # channels 0..4 = exactly 1..5, channel 5 = >=6
            ch = min(idx, MAX_STACK) - 1
            point_tensor[i, ch] = 1.0

        elif c < 0:
            # Opponent checkers on this point
            v = int(-c)
            idx = min(v, MAX_STACK)
            # channels 6..10 = exactly 1..5, channel 11 = >=6
            ch = 6 + (min(idx, MAX_STACK) - 1)
            point_tensor[i, ch] = 1.0

        # if c == 0: leave row as zeros

    # --- Global features (still in "signed" perspective) ---
    my_bar    = max(signed[25], 0) + max(signed[26], 0)
    opp_bar   = max(-signed[25], 0) + max(-signed[26], 0)
    my_borne  = max(signed[27], 0) + max(signed[28], 0)
    opp_borne = max(-signed[27], 0) + max(-signed[28], 0)

    global_feats = np.array(
        [my_bar, opp_bar, my_borne, opp_borne],
        dtype=np.float32
    ) / 15.0  # normalize: 15 checkers total

    return point_tensor, global_feats


def encode_batch(
    boards: np.ndarray,
    players: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode a batch of boards + players into PyTorch tensors.

    Args:
        boards:  np.ndarray, shape (B, 29), int32
        players: np.ndarray, shape (B,), int32 in {+1, -1}

    Returns:
        pts_t: torch.Tensor, shape (B, NUM_POINT_CHANNELS, 24)
        g_t:   torch.Tensor, shape (B, NUM_GLOBAL_FEATS)
    """
    assert boards.ndim == 2 and boards.shape[1] == 29
    assert players.shape[0] == boards.shape[0]

    B = boards.shape[0]
    pts_list = []
    g_list = []

    for i in range(B):
        pt, gf = encode_board_np(boards[i], int(players[i]))
        pts_list.append(pt)   # (24, C)
        g_list.append(gf)     # (G,)

    pts = np.stack(pts_list, axis=0)     # (B, 24, C)
    g   = np.stack(g_list, axis=0)      # (B, G)

    # To DEVICE + channel-first for Conv1d
    pts_t = torch.from_numpy(pts).to(DEVICE)  # (B, 24, C)
    pts_t = pts_t.permute(0, 2, 1)            # -> (B, C, 24)
    g_t   = torch.from_numpy(g).to(DEVICE)    # (B, G)

    return pts_t, g_t


# ============================================================
# ConvValueNetV2: deeper conv + dropout
# ============================================================

class ConvValueNetV2(nn.Module):
    """
    Deeper 1D-convolutional value network for backgammon.

    Differences vs v1:
      - 3 conv layers instead of 2
      - Dropout in the fully-connected part
      - Same input representation, same scalar output in (-1, 1)
    """

    def __init__(
        self,
        point_channels: int = NUM_POINT_CHANNELS,
        global_dim: int = NUM_GLOBAL_FEATS,
        conv_channels: int = 64,
        fc_hidden: int = 128,
        dropout_p: float = 0.3,
    ):
        super().__init__()

        # 1D conv over the 24 points
        self.conv1 = nn.Conv1d(point_channels, conv_channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(conv_channels, conv_channels, kernel_size=5, padding=2)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)

        # After convs we global-average-pool over positions,
        # then concatenate global features and pass through FC layers.
        self.fc1 = nn.Linear(conv_channels + global_dim, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, 1)

    def forward(self, pts: torch.Tensor, gfeats: torch.Tensor) -> torch.Tensor:
        """
        pts:    (B, C, 24)
        gfeats: (B, G)
        returns:
            values: (B,) in (-1, 1)
        """
        x = self.act(self.conv1(pts))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))

        # Global average pool over board positions -> (B, conv_channels)
        x = x.mean(dim=-1)

        # Concatenate global features (bar + borne-off) -> (B, conv_channels + G)
        x = torch.cat([x, gfeats], dim=-1)

        # Fully-connected + dropout for regularization
        x = self.act(self.fc1(x))
        x = self.dropout(x)

        # Final scalar value, squashed to (-1, 1)
        x = torch.tanh(self.fc2(x))
        return x.squeeze(-1)


# ============================================================
# Epsilon-greedy move selection using ConvValueNetV2
# ============================================================

def select_move_eps_greedy_v2(
    board: np.ndarray,
    player: int,
    epsilon: float,
    net: ConvValueNetV2,
) -> tuple[Optional[np.ndarray], np.ndarray, Optional[float]]:
    """
    Choose a move for `player` on `board` via epsilon-greedy one-ply lookahead.

    Args:
        board:   np.ndarray shape (29,), int32
        player:  +1 or -1
        epsilon: exploration rate in [0, 1]
        net:     ConvValueNetV2 on DEVICE

    Returns:
        chosen_move: np.ndarray, shape (k, 2) with k in {1,2}, or None if pass
        next_board:  np.ndarray, shape (29,) resulting board after move
    """
    # Roll dice and get all legal moves + successor boards
    dice = roll_dice()
    moves, boards = legal_moves(board, dice, player)

    # No legal moves: must pass
    if len(moves) == 0:
        return None, board, None

    # --- Exploration step ---
    # With probability epsilon, pick a random move regardless of value.
    if np.random.rand() < epsilon:
        idx = np.random.randint(len(moves))
        return moves[idx], boards[idx], None

    # --- Exploitation step ---
    # Evaluate each successor board from the NEXT player's perspective.
    # This is equivalent to saying:
    #   "If I play this move, then it's their turn. How good is THAT situation?"
    cand_boards = np.stack(boards, axis=0).astype(np.int32)  # (M, 29)
    next_players = np.full((len(boards),), -player, dtype=np.int32)

    # Encode batch of candidate successor states
    pts_t, g_t = encode_batch(cand_boards, next_players)

    # Run through the value network
    was_training = net.training
    net.eval()
    with torch.no_grad():
        values = net(pts_t, g_t).cpu().numpy()
    if was_training:
        net.train()


    # We want to maximize our win chance.
    # Since values are from the NEXT player's viewpoint, "lower is better" for us.
    best_idx = int(values.argmin())
    best_val = float(values[best_idx])

    return moves[best_idx], boards[best_idx], best_val


# ============================================================
# Reward shaping helpers for v2
# ============================================================

def _extract_global_counts(board: np.ndarray, player: int) -> tuple[float, float, float, float]:
    """
    From (board, player) get:
        my_bar, opp_bar, my_borne, opp_borne
    as *unnormalized* counts (0..15), without going through encode_board_np.
    """
    signed = board.astype(np.int32) * int(player)

    my_bar    = max(signed[25], 0) + max(signed[26], 0)
    opp_bar   = max(-signed[25], 0) + max(-signed[26], 0)
    my_borne  = max(signed[27], 0) + max(signed[28], 0)
    opp_borne = max(-signed[27], 0) + max(-signed[28], 0)

    return float(my_bar), float(opp_bar), float(my_borne), float(opp_borne)



def shaped_positional_reward(
    prev_board: np.ndarray,
    next_board: np.ndarray,
    player: int,
    dense_scale: float = 0.05,
) -> float:
    """
    Small dense reward that encourages:
      - bearing off my own checkers
      - keeping opponent on the bar
      - *avoiding* having my own checkers on the bar

    It looks at the change in a simple "material-like" score from the
    current player's point of view.

    material = (my_borne - opp_borne) - 0.5 * (my_bar - opp_bar)

    The reward is proportional to (material_next - material_prev).
    """
    my_bar_p, opp_bar_p, my_borne_p, opp_borne_p = _extract_global_counts(prev_board, player)
    my_bar_n, opp_bar_n, my_borne_n, opp_borne_n = _extract_global_counts(next_board, player)

    material_prev = (my_borne_p - opp_borne_p) - 0.5 * (my_bar_p - opp_bar_p)
    material_next = (my_borne_n - opp_borne_n) - 0.5 * (my_bar_n - opp_bar_n)

    delta = material_next - material_prev

    # Normalize by 15 checkers and scale down so it doesn't overpower the +1 terminal reward
    dense_reward = dense_scale * (delta / 15.0)

    # Optional: clamp to keep it small and stable
    dense_reward = float(np.clip(dense_reward, -0.1, 0.1))
    return dense_reward


# ============================================================
# Value-of-state helper for ConvValueNetV2
# ============================================================

def value_of_state_v2(
    net: ConvValueNetV2,
    board: np.ndarray,
    player: int,
) -> torch.Tensor:
    """
    Compute V(net, board, player) as a scalar tensor on DEVICE.

    Semantics: V(s) is the value *for the player-to-move*.
    """
    boards = np.expand_dims(board.astype(np.int32), axis=0)  # (1, 29)
    players = np.array([player], dtype=np.int32)             # (1,)
    pts, gfeats = encode_batch(boards, players)              # (1, C, 24), (1, G)
    v = net(pts, gfeats)                                     # (1,)
    return v.squeeze(0)                                      # scalar tensor




def td0_update_batch(
    net: ConvValueNetV2,
    optimizer: optim.Optimizer,
    transitions: list[Transition],
    gamma: float,
) -> float:
    """
    Perform one batched TD(0) update on a list of transitions.

    Each transition i has:
        s_i  = (board, player)
        r_i  = reward
        s'_i = (next_board, next_player)
        done_i

    Target: r_i  if done_i else r_i + gamma * V(s'_i).
    Loss:   mean_i (V(s_i) - target_i)^2
    """
    if not transitions:
        return 0.0

    # --- Pack batch ---
    boards = np.stack([t.board for t in transitions], axis=0)      # (B, 29)
    players = np.array([t.player for t in transitions], dtype=np.int32)  # (B,)
    next_boards = np.stack([t.next_board for t in transitions], axis=0)  # (B, 29)
    next_players = np.array([t.next_player for t in transitions], dtype=np.int32)  # (B,)
    rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32, device=DEVICE)  # (B,)
    dones_np = np.array([t.done for t in transitions], dtype=bool)      # (B,)

    # --- V(s) ---
    pts_s, g_s = encode_batch(boards, players)   # (B, C, 24), (B, G)
    v_s = net(pts_s, g_s).view(-1)               # (B,)

    # --- TD targets ---
    targets = rewards.clone()                    # start with r_i

    if not np.all(dones_np):
        # Compute V(s') only for non-terminal transitions
        not_done_idx = np.where(~dones_np)[0]
        nb = next_boards[not_done_idx]          # (B_nd, 29)
        np_players = next_players[not_done_idx] # (B_nd,)

        pts_sp, g_sp = encode_batch(nb, np_players)
        with torch.no_grad():
            v_next = net(pts_sp, g_sp).view(-1)  # (B_nd,)

        targets_nd = rewards[not_done_idx] + gamma * v_next
        targets[not_done_idx] = targets_nd

    # --- MSE loss over batch ---
    loss = (v_s - targets).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
    optimizer.step()

    return float(loss.item())

# ============================================================
# One shaped TD(0) self-play episode
# ============================================================

def play_and_train_shaped_td0_episode_v2(
    net: ConvValueNetV2,
    optimizer: optim.Optimizer,
    gamma: float,
    epsilon: float,
    verbose: bool = False,
) -> tuple[int, int]:
    """
    Play one self-play game and perform *online* TD(0) updates after each move,
    using a combination of:
      - sparse terminal reward (+1 for win, 0 otherwise)
      - small dense shaped reward based on bearing off + bar status

    Returns:
        winner:     +1 or -1 (who bore off first)
        move_count: number of plies played
    """
    net.train()
    board = init_board().astype(np.int32)
    player = 1   # +1 starts
    move_count = 0

    while True:
        if verbose:
            print(f"\n[TD0-V2] player={player:+d}, move_count={move_count}")

        # --- Current state s ---
        s_board = board.copy()
        s_player = player

        # --- Choose move with epsilon-greedy policy using the same net ---
        move, next_board, v_next_from_policy = select_move_eps_greedy_v2(board, player, epsilon, net)


        # Handle pass (no legal moves)
        if move is None:
            if verbose:
                print("  No legal moves, passing turn.")
            done = False
            next_board = board  # unchanged
            next_player = -player
            terminal_reward = 0.0
        else:
            next_board = next_board.astype(np.int32)
            done = game_over(next_board)
            next_player = -player
            terminal_reward = 1.0 if done else 0.0

        move_count += 1

        # --- Optional hard cap on game length ---
        if (not done) and move_count >= MAX_MOVES_PER_GAME_V2:
            # Treat as a terminal "draw"/no-result: no +1, just shaped reward.
            done = True
            # terminal_reward stays 0.0

        # --- Dense shaped reward from s -> s' from current player's perspective ---
        shaped_r = shaped_positional_reward(s_board, next_board, s_player)
        reward = terminal_reward + shaped_r


        # --- TD(0) update: v(s) -> reward + gamma * v(s') ---
        v_s = value_of_state_v2(net, s_board, s_player)  # scalar tensor

        if done:
            target = torch.tensor(reward, dtype=torch.float32, device=DEVICE)
        else:
            # Reuse V(next_state) from move selection when available (greedy case).
            if v_next_from_policy is not None:
                v_next = torch.tensor(v_next_from_policy, dtype=torch.float32, device=DEVICE)
            else:
                # For exploratory (epsilon) moves, we still need to evaluate V(next_state).
                with torch.no_grad():
                    v_next = value_of_state_v2(net, next_board, next_player)
        
            target = torch.tensor(reward, dtype=torch.float32, device=DEVICE) + gamma * v_next

        loss = (v_s - target) ** 2

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

        if verbose:
            print(
                f"  reward={reward:.4f} "
                f"(terminal={terminal_reward:.1f}, shaped={shaped_r:.4f}), "
                f"done={done}, loss={loss.item():.6f}"
            )

        # --- Step environment ---
        if done:
            if verbose:
                print(f"[TD0-V2 GAME OVER] winner={player:+d}, total_moves={move_count}")
            return player, move_count

        board = next_board
        player = next_player


# ============================================================
# Random baseline policy (same as v1) + v2 evaluation
# ============================================================

def select_move_random(
    board: np.ndarray,
    player: int,
) -> tuple[Optional[np.ndarray], np.ndarray]:
    """
    Select a random legal move for `player`. If no legal moves, pass.
    This is our fixed baseline opponent.
    """
    dice = roll_dice()
    moves, boards = legal_moves(board, dice, player)
    if len(moves) == 0:
        return None, board

    idx = np.random.randint(len(moves))
    return moves[idx], boards[idx]


def play_game_agent_vs_random_v2(
    net: ConvValueNetV2,
    epsilon_eval: float = 0.0,
    verbose: bool = False,
) -> int:
    """
    Play one game: player +1 = our shaped TD(0) value net, player -1 = random policy.

    Args:
        net:          trained ConvValueNetV2
        epsilon_eval: epsilon for our agent (0.0 = greedy)
        verbose:      optional logging

    Returns:
        winner: +1 if our agent wins, -1 if random wins.
    """
    board = init_board().astype(np.int32)
    player = 1
    move_count = 0

    while True:
        if verbose:
            print(f"\n[VS-RAND-V2] player={player:+d}, move_count={move_count}")

        if player == 1:
            # our shaped agent
            move, next_board, _ = select_move_eps_greedy_v2(board, player, epsilon_eval, net)
        else:
            # random opponent
            move, next_board = select_move_random(board, player)
        

        # Handle pass (no legal moves)
        if move is None:
            if verbose:
                print("  No legal moves, passing.")
            player = -player
            move_count += 1
            continue

        next_board = next_board.astype(np.int32)
        move_count += 1

        if game_over(next_board):
            if verbose:
                print(f"[VS-RAND-V2 GAME OVER] winner={player:+d}, total_moves={move_count}")
            return player

        # --- Optional hard cap on game length in evaluation ---
        if move_count >= MAX_MOVES_PER_GAME_V2:
            if verbose:
                print(f"[VS-RAND-V2] Max moves {MAX_MOVES_PER_GAME_V2} reached, treating as draw.")
            return 0  # "draw" (counts as non-win in evaluate_vs_random_v2)

        board = next_board
        player = -player



def evaluate_vs_random_v2(
    net: ConvValueNetV2,
    num_games: int = 100,
    epsilon_eval: float = 0.0,
) -> float:
    # 1) Remember current mode (train or eval)
    was_training = net.training

    # 2) Switch to eval mode for stability (dropout off, etc.)
    net.eval()

    wins = 0
    for _ in range(num_games):
        winner = play_game_agent_vs_random_v2(net, epsilon_eval=epsilon_eval, verbose=False)
        if winner == 1:
            wins += 1

    win_rate = wins / float(num_games)
    print(f"[EVAL-V2] vs random over {num_games} games: win_rate={win_rate:.3f}")

    # 3) Restore previous mode
    if was_training:
        net.train()

    return win_rate



# ============================================================
# Final shaped TD(0) training loop (v2)
# ============================================================

# --- High-level training config for v2 ---
TOTAL_EPISODES_V2   = 1_000_000
E_START_V2          = 0.3
E_END_V2            = 0.01
GAMMA_V2            = 0.99
LR_V2               = 1e-4
MAX_MOVES_PER_GAME_V2 = 450

EVAL_INTERVAL_V2    = 10_000      # evaluate vs random every N episodes
EVAL_NUM_GAMES_V2   = 500         # games per evaluation

CHECKPOINT_DIR_V2   = "models_v2"
BASE_MODEL_NAME_V2  = "conv_td0_shaped_v2"


def train_td0_conv_v2(
    total_episodes: int = TOTAL_EPISODES_V2,
    gamma: float = GAMMA_V2,
    lr: float = LR_V2,
    e_start: float = E_START_V2,
    e_end: float = E_END_V2,
    eval_interval: int = EVAL_INTERVAL_V2,
    eval_num_games: int = EVAL_NUM_GAMES_V2,
    n_envs: int = 8,
    batch_size: int = 64,
    max_buffer_size: int = 50_000,
) -> ConvValueNetV2:
    """
    Final v2 TD(0) training loop with:
      - Multiple self-play games in parallel (n_envs).
      - Shaped rewards on intermediate states.
      - Epsilon annealing across the *completed* episodes.
      - Simple learning-rate schedule.
      - Batched TD(0) updates over recent transitions.
      - Periodic evaluation vs a random baseline.
      - Checkpointing + CSV logging of eval history.
    """
    os.makedirs(CHECKPOINT_DIR_V2, exist_ok=True)

    net = ConvValueNetV2().to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # Optional seeding for partial reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # --- Initialize parallel environments ---
    envs: list[EnvState] = []
    for _ in range(n_envs):
        envs.append(
            EnvState(
                board=init_board().astype(np.int32),
                player=1,
                done=False,
                moves_in_episode=0,
            )
        )

    # --- TD(0) transition buffer ---
    buffer: list[Transition] = []

    best_random_win = 0.0
    eval_history: list[tuple[int, float]] = []  # (episode, winrate_vs_random)

    episodes_done = 0
    next_eval_at = eval_interval

    # Logging helpers for "per 1k episodes"
    last_log_ep = 0
    moves_since_log = 0
    last_log_time = time.time()

    while episodes_done < total_episodes:
        # --- Current epsilon + LR based on completed episodes ---
        t = min(1.0, episodes_done / float(total_episodes))
        epsilon = e_start + t * (e_end - e_start)

        frac = episodes_done / float(total_episodes) if total_episodes > 0 else 1.0
        if frac > 0.85:
            lr_factor = 0.1
        elif frac > 0.60:
            lr_factor = 0.3
        else:
            lr_factor = 1.0

        for g in optimizer.param_groups:
            g["lr"] = lr * lr_factor

        # --- Step each environment once ---
        for env in envs:
            if episodes_done >= total_episodes:
                break

            if env.done:
                # Reset env for a new episode
                env.board = init_board().astype(np.int32)
                env.player = 1
                env.done = False
                env.moves_in_episode = 0

            s_board = env.board.copy()
            s_player = env.player

            # Choose move with epsilon-greedy policy using the same net
            move, next_board, _ = select_move_eps_greedy_v2(
                s_board,
                s_player,
                epsilon,
                net,
            )

            # Handle pass (no legal moves)
            if move is None:
                done = False
                next_board = env.board  # unchanged
                next_player = -env.player
                terminal_reward = 0.0
            else:
                next_board = next_board.astype(np.int32)
                done = game_over(next_board)
                next_player = -env.player
                terminal_reward = 1.0 if done else 0.0

            env.moves_in_episode += 1

            # --- Optional hard cap on game length ---
            if (not done) and env.moves_in_episode >= MAX_MOVES_PER_GAME_V2:
                done = True
                # terminal_reward stays 0.0 (no +1 reward for capped games)

            # Dense shaped reward from s -> s' from current player's perspective
            shaped_r = shaped_positional_reward(s_board, next_board, s_player)
            reward = terminal_reward + shaped_r


            # Store transition for later batched TD update
            buffer.append(
                Transition(
                    board=s_board,
                    player=s_player,
                    reward=float(reward),
                    next_board=next_board,
                    next_player=next_player,
                    done=done,
                )
            )

            # Keep buffer from growing without bound
            if len(buffer) > max_buffer_size:
                # Drop oldest transitions
                excess = len(buffer) - max_buffer_size
                buffer = buffer[excess:]

            # Update env state
            env.board = next_board
            env.player = next_player

            # Episode finished?
            if done:
                episodes_done += 1
                moves_since_log += env.moves_in_episode
                env.done = True  # will be reset on next loop

                # --- Logging every 1k episodes ---
                if episodes_done % 1_000 == 0:
                    now = time.time()
                    elapsed_s = now - last_log_time

                    avg_moves_1k = moves_since_log / 1_000.0
                    print(
                        f"[TRAIN-V2] ep={episodes_done}, "
                        f"epsilon={epsilon:.4f}, "
                        f"lr={optimizer.param_groups[0]['lr']:.6g}, "
                        f"avg_moves_1k={avg_moves_1k:.1f}, "
                        f"time_1k={elapsed_s:.1f}s"
                    )

                    moves_since_log = 0
                    last_log_time = now

                # --- Periodic evaluation vs random + checkpointing ---
                if episodes_done >= next_eval_at:
                    print(
                        f"[TRAIN-V2] Evaluating vs random after ep={episodes_done} "
                        f"({eval_num_games} games, greedy)..."
                    )
                    win_rate = evaluate_vs_random_v2(
                        net,
                        num_games=eval_num_games,
                        epsilon_eval=0.0,
                    )

                    eval_history.append((episodes_done, float(win_rate)))

                    ckpt_path = os.path.join(
                        CHECKPOINT_DIR_V2,
                        f"{BASE_MODEL_NAME_V2}_ep{episodes_done}.pt"
                    )
                    torch.save(net.state_dict(), ckpt_path)
                    print(f"[CKPT-V2] Saved model to {ckpt_path}")

                    if win_rate > best_random_win:
                        best_random_win = win_rate
                        best_path = os.path.join(
                            CHECKPOINT_DIR_V2,
                            f"{BASE_MODEL_NAME_V2}_best_random.pt"
                        )
                        torch.save(net.state_dict(), best_path)
                        print(
                            f"[CKPT-V2] New best vs random={win_rate:.3f}, "
                            f"saved to {best_path}"
                        )

                    next_eval_at += eval_interval

        # --- Batched TD(0) update when we have enough transitions ---
        if len(buffer) >= batch_size:
            batch = buffer[-batch_size:]  # use the most recent transitions
            td0_update_batch(net, optimizer, batch, gamma=gamma)

    # --- Dump eval history to CSV for plotting v1 vs v2 ---
    history_path = os.path.join(
        CHECKPOINT_DIR_V2,
        f"{BASE_MODEL_NAME_V2}_eval_history.csv"
    )
    with open(history_path, "w") as f:
        f.write("episode,winrate_vs_random\n")
        for ep_i, wr in eval_history:
            f.write(f"{ep_i},{wr}\n")
    print(f"[LOG-V2] Saved eval history to {history_path}")

    return net






# ============================================================
# Entry point
# ============================================================
def _smoke_test_v2():
    board = init_board().astype(np.int32)
    player = 1

    boards = np.expand_dims(board, axis=0)
    players = np.array([player], dtype=np.int32)

    pts_t, g_t = encode_batch(boards, players)
    net = ConvValueNetV2().to(DEVICE)

    with torch.no_grad():
        v = net(pts_t, g_t)
    print(f"[SMOKE-V2] v.shape={v.shape}, v={v.item():.4f}")

    # Tiny one-episode train
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    winner, moves = play_and_train_shaped_td0_episode_v2(
        net, optimizer, gamma=0.99, epsilon=0.3, verbose=True
    )
    print(f"[SMOKE-V2] Finished one shaped episode, winner={winner}, moves={moves}")




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Conv TD(0) v2 for Backgammon")
    parser.add_argument(
        "--episodes",
        type=int,
        default=TOTAL_EPISODES_V2,
        help=f"Number of self-play episodes (default: {TOTAL_EPISODES_V2})",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=EVAL_INTERVAL_V2,
        help=f"Evaluate vs random every N episodes (default: {EVAL_INTERVAL_V2})",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=EVAL_NUM_GAMES_V2,
        help=f"Number of games in each eval vs random (default: {EVAL_NUM_GAMES_V2})",
    )

    args = parser.parse_args()

    print(
        f"[TRAIN-V2] Starting shaped TD(0) run (v2)... "
        f"episodes={args.episodes}, eval_interval={args.eval_interval}, "
        f"eval_games={args.eval_games}"
    )

    train_td0_conv_v2(
        total_episodes=args.episodes,
        eval_interval=args.eval_interval,
        eval_num_games=args.eval_games,
    )
