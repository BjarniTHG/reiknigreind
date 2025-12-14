# train_conv_replay_bjarni_baseline.py

from __future__ import annotations
from typing import Optional, List
from dataclasses import dataclass

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from backgammon import (
    init_board,
    roll_dice,
    game_over,
    legal_moves,
)  # from the professor's backgammon.py

# === Training config for final runs ===
DEFAULT_TOTAL_EPISODES = 250_000

E_START = 0.3
E_END   = 0.01

GAMMA = 0.99
LR    = 1e-4

EVAL_INTERVAL   = 5_000      # evaluate vs random every N episodes
EVAL_NUM_GAMES  = 300        # games per evaluation
CHECKPOINT_DIR  = "models"
BASE_MODEL_NAME = "mlp_td0_bjarni_baseline_flat"

# Use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[INFO] Using device: {DEVICE}")


# === Encoding constants ===
# Baseline 2: SAME encoding as conv model.
# Only change is the network: Conv1d ValueNet -> MLP ValueNet.
MAX_STACK = 5
NUM_POINT_CHANNELS = 12
NUM_GLOBAL_FEATS = 4


def encode_board_np(board: np.ndarray, player: int) -> tuple[np.ndarray, np.ndarray]:
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
            idx = min(v, MAX_STACK)  # 1..5 or >=6
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
# End of 2.

#3. Batch encoder -> PyTorch encoders
def encode_batch(
    boards: np.ndarray, players: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode a batch of boards + players into tensors.

    Args:
        boards: np.ndarray, shape (B, 29), int32
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
    g = np.stack(g_list, axis=0)        # (B, G)

    pts_t = torch.from_numpy(pts).to(DEVICE)  # (B,24,C)
    pts_t = pts_t.permute(0, 2, 1)            # -> (B,C,24) for Conv1d
    g_t = torch.from_numpy(g).to(DEVICE)      # (B,G)

    return pts_t, g_t
#End of 3.

class MLPValueNet(nn.Module):
    def __init__(
        self,
        point_channels: int = NUM_POINT_CHANNELS,
        global_dim: int = NUM_GLOBAL_FEATS,
        fc_hidden: int = 128,
    ):
        super().__init__()
        in_dim = point_channels * 24 + global_dim
        self.fc1 = nn.Linear(in_dim, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, 1)
        self.act = nn.ReLU()

    def forward(self, pts: torch.Tensor, gfeats: torch.Tensor) -> torch.Tensor:
        x = pts.reshape(pts.shape[0], -1)      # flatten (B, C*24)
        x = torch.cat([x, gfeats], dim=-1)     # (B, C*24 + G)
        x = self.act(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x.squeeze(-1)


# 2nd iteration(interim), step 1
# === Replay buffer for TD(0) training ===

@dataclass
class Transition:
    board: np.ndarray       # (29,)
    player: int             # +1 or -1
    reward: float
    next_board: np.ndarray  # (29,)
    next_player: int        # +1 or -1
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[Transition] = []
        self.idx = 0

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, tr: Transition) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(tr)
        else:
            self.buffer[self.idx] = tr
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size: int):
        assert len(self.buffer) >= batch_size
        batch = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        boards = []
        players = []
        rewards = []
        next_boards = []
        next_players = []
        dones = []
        for idx in batch:
            tr = self.buffer[int(idx)]
            boards.append(tr.board)
            players.append(tr.player)
            rewards.append(tr.reward)
            next_boards.append(tr.next_board)
            next_players.append(tr.next_player)
            dones.append(tr.done)

        boards = np.stack(boards, axis=0).astype(np.int32)           # (B,29)
        players = np.array(players, dtype=np.int32)                  # (B,)
        rewards = np.array(rewards, dtype=np.float32)                # (B,)
        next_boards = np.stack(next_boards, axis=0).astype(np.int32) # (B,29)
        next_players = np.array(next_players, dtype=np.int32)        # (B,)
        dones = np.array(dones, dtype=np.float32)                    # (B,)

        return boards, players, rewards, next_boards, next_players, dones


#interim step 2: epsilon-greedy move selection
# === Epsilon-greedy move selection using the value network ===

def select_move_eps_greedy(
    board: np.ndarray,
    player: int,
    epsilon: float,
    net: nn.Module,
) -> tuple[Optional[np.ndarray], np.ndarray]:
    """
    Choose a move for `player` on `board` via epsilon-greedy one-ply lookahead.

    Args:
        board:  np.ndarray shape (29,), int32
        player: +1 or -1
        epsilon: exploration rate in [0,1]
        net:   MLPValueNet on DEVICE

    Returns:
        chosen_move: np.ndarray, shape (k, 2) with k in {1,2}
        next_board:  np.ndarray, shape (29,) resulting board after move
    """
    # Roll dice and get all legal moves + successor boards
    dice = roll_dice()
    moves, boards = legal_moves(board, dice, player)
    if len(moves) == 0:
        return None, board

    # Exploration: random move
    if np.random.rand() < epsilon:
        idx = np.random.randint(len(moves))
        return moves[idx], boards[idx]

    # Greedy: evaluate all successor boards from next player's perspective
    cand_boards = np.stack(boards, axis=0).astype(np.int32)  # (M, 29)
    next_players = np.full((len(boards),), -player, dtype=np.int32)

    pts_t, g_t = encode_batch(cand_boards, next_players)
    with torch.no_grad():
        values = net(pts_t, g_t).cpu().numpy()  # (M,)

    best_idx = int(values.argmax())
    return moves[best_idx], boards[best_idx]
#End of interim step 2.
#Interim step 3. Simple self-play game runner(no replay yet)
def play_one_selfplay_game(
    net: nn.Module,
    epsilon: float = 0.2,
    verbose: bool = False,
) -> int:
    """
    Play a single self-play game between two copies of `net`.

    Args:
        net: MLPValueNet used by both players (shared parameters).
        epsilon: exploration rate for both players.
        verbose: if True, prints basic info.

    Returns:
        winner: +1 if player +1 bore off first, -1 if player -1 did.
    """
    board = init_board().astype(np.int32)
    player = 1  # +1 starts

    move_count = 0

    while True:
        if verbose:
            print(f"\n[TURN] player={player}, move_count={move_count}")

        move, next_board = select_move_eps_greedy(board, player, epsilon, net)

        # If no legal move: pass the turn
        if move is None:
            if verbose:
                print("  No legal moves for this player, passing turn.")
            player = -player
            move_count += 1
            continue

        next_board = next_board.astype(np.int32)

        move_count += 1

        if game_over(next_board):
            # Current player just bore off all 15 pieces
            if verbose:
                print(f"[GAME OVER] winner={player}, total moves={move_count}")
            return player

        # Switch to next state / player
        board = next_board
        player = -player
#End of interim step 3.

#iteration 4, 1
# === Pure random baseline policy (for evaluation) ===

def select_move_random(
    board: np.ndarray,
    player: int,
) -> tuple[Optional[np.ndarray], np.ndarray]:
    """
    Select a random legal move for `player`. If no legal moves, pass.
    """
    dice = roll_dice()
    moves, boards = legal_moves(board, dice, player)
    if len(moves) == 0:
        return None, board

    idx = np.random.randint(len(moves))
    return moves[idx], boards[idx]


def play_game_agent_vs_random(
    net: nn.Module,
    epsilon_eval: float = 0.0,
    verbose: bool = False,
) -> int:
    """
    Play one game: player +1 = our value net, player -1 = random policy.

    Args:
        net: trained MLPValueNet
        epsilon_eval: epsilon for our agent (0.0 = greedy)
        verbose: optional logging

    Returns:
        winner: +1 if our agent wins, -1 if random wins.
    """
    board = init_board().astype(np.int32)
    player = 1
    move_count = 0

    while True:
        if verbose:
            print(f"\n[VS-RAND] player={player}, move_count={move_count}")

        if player == 1:
            move, next_board = select_move_eps_greedy(board, player, epsilon_eval, net)
        else:
            move, next_board = select_move_random(board, player)

        # Handle pass
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
                print(f"[VS-RAND GAME OVER] winner={player}, total_moves={move_count}")
            return player

        board = next_board
        player = -player


def evaluate_vs_random(
    net: nn.Module,
    num_games: int = 50,
    epsilon_eval: float = 0.0,
) -> float:
    """
    Evaluate `net` against the random baseline.

    Returns:
        win_rate (fraction of games where our agent wins).
    """
    wins = 0
    for _ in range(num_games):
        winner = play_game_agent_vs_random(net, epsilon_eval=epsilon_eval, verbose=False)
        if winner == 1:
            wins += 1
    win_rate = wins / float(num_games)
    print(f"[EVAL] vs random over {num_games} games: win_rate={win_rate:.3f}")
    return win_rate

#end of iteration 4, 1

#Aborted knob 3, interim 3, step 1
def value_of_state(
    net: nn.Module,
    board: np.ndarray,
    player: int,
) -> torch.Tensor:
    """
    Compute V(net, board, player) as a 0-dim tensor on DEVICE.
    """
    boards = np.expand_dims(board.astype(np.int32), axis=0)  # (1,29)
    players = np.array([player], dtype=np.int32)             # (1,)
    pts, gfeats = encode_batch(boards, players)              # (1,C,24), (1,G)
    v = net(pts, gfeats)                                     # (1,)
    return v.squeeze(0)                                      # scalar tensor

#end of interim 3, step 1

#3rd iteration(interim), step 2
def play_and_train_online_td0_episode(
    net: nn.Module,
    optimizer: optim.Optimizer,
    gamma: float,
    epsilon: float,
    verbose: bool = False,
) -> tuple[int, int]:
    """
    Play one self-play game and perform *online* TD(0) updates after each move.

    Returns:
        winner: +1 or -1
        move_count: number of plies played
    """
    net.train()
    board = init_board().astype(np.int32)
    player = 1
    move_count = 0

    while True:
        if verbose:
            print(f"\n[TD0] player={player}, move_count={move_count}")

        # Current state s
        s_board = board.copy()
        s_player = player

        # Choose move with epsilon-greedy using the *same* net
        move, next_board = select_move_eps_greedy(board, player, epsilon, net)

        # No-legal-move case: pass
        if move is None:
            if verbose:
                print("  No legal moves, passing.")
            reward = 0.0
            done = False
            next_board = board  # unchanged
            next_player = -player
        else:
            next_board = next_board.astype(np.int32)
            done = game_over(next_board)
            reward = 1.0 if done else 0.0
            next_player = -player

        move_count += 1

        # --- TD(0) update for this transition (s -> s') ---
        # v(s)
        v_s = value_of_state(net, s_board, s_player)  # scalar tensor

        # v(s')
        if done:
            target = torch.tensor(reward, dtype=torch.float32, device=DEVICE)
        else:
            with torch.no_grad():
                v_next = value_of_state(net, next_board, next_player)
                target = torch.tensor(reward, dtype=torch.float32, device=DEVICE) + gamma * v_next

        loss = (v_s - target) ** 2

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

        if verbose:
            print(f"  reward={reward}, done={done}, loss={loss.item():.6f}")

        # --- Step environment ---
        if done:
            if verbose:
                print(f"[TD0 GAME OVER] winner={player}, total moves={move_count}")
            return player, move_count

        board = next_board
        player = next_player

#end of 3rd iteration, step 2

#2nd iteration(interim), step 2
def play_game_and_fill_replay(
    net: nn.Module,
    replay: ReplayBuffer,
    epsilon: float,
    gamma: float = 0.99,   # not used yet but kept for symmetry
    verbose: bool = False,
) -> int:
    """
    Play one self-play game and store transitions in replay buffer.

    Reward scheme:
      - 0 for all non-terminal moves
      - +1 for the move that wins the game (from that player's perspective)

    Args:
        net: value network used for move selection (both players share it)
        replay: ReplayBuffer to push transitions into
        epsilon: exploration rate
        gamma: discount factor (for later, here we just use TD(0))
        verbose: optional logging

    Returns:
        winner: +1 or -1
    """
    board = init_board().astype(np.int32)
    player = 1
    move_count = 0

    while True:
        if verbose:
            print(f"\n[TURN] player={player}, move_count={move_count}")

        move, next_board = select_move_eps_greedy(board, player, epsilon, net)

        # Pass move (no legal moves)
        if move is None:
            if verbose:
                print("  No legal moves, passing.")
            replay.push(
                Transition(
                    board=board.copy(),
                    player=player,
                    reward=0.0,
                    next_board=board.copy(),
                    next_player=-player,
                    done=False,
                )
            )
            player = -player
            move_count += 1
            continue

        next_board = next_board.astype(np.int32)
        move_count += 1

        done = game_over(next_board)
        reward = 1.0 if done else 0.0

        # Store transition from current player's perspective
        replay.push(
            Transition(
                board=board.copy(),
                player=player,
                reward=reward,
                next_board=next_board.copy(),
                next_player=-player,
                done=done,
            )
        )

        if done:
            if verbose:
                print(f"[GAME OVER] winner={player}, total moves={move_count}")
            return player

        board = next_board
        player = -player
#end of 2nd iteration, step 2

#2nd iteration(interim), step 3
def make_networks() -> tuple[nn.Module, nn.Module]:
    online = MLPValueNet().to(DEVICE)
    target = MLPValueNet().to(DEVICE)
    target.load_state_dict(online.state_dict())
    target.eval()
    return online, target


def train_step_td0(
    online_net: nn.Module,
    target_net: nn.Module,
    optimizer: optim.Optimizer,
    replay: ReplayBuffer,
    batch_size: int,
    gamma: float = 0.99,
) -> Optional[float]:
    """
    One TD(0) update using replay and target network.

    Returns:
        loss value as float, or None if not enough data yet.
    """
    if len(replay) < batch_size:
        return None

    boards, players, rewards, next_boards, next_players, dones = replay.sample(batch_size)

    # Encode
    pts, gfeats = encode_batch(boards, players)              # (B,C,24), (B,G)
    next_pts, next_gfeats = encode_batch(next_boards, next_players)

    rewards_t = torch.from_numpy(rewards).to(DEVICE)         # (B,)
    dones_t   = torch.from_numpy(dones).to(DEVICE)           # (B,)

    # Current value predictions
    v_s = online_net(pts, gfeats)                            # (B,)

    with torch.no_grad():
        v_next = target_net(next_pts, next_gfeats)           # (B,)
        targets = rewards_t + gamma * (1.0 - dones_t) * v_next

    loss = ((v_s - targets) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.item())

#end of 2nd iteration, step 3

#5. Smoke test
def _smoke_test():
    # Same as before: encoder + forward sanity check
    board = init_board().astype(np.int32)
    player = 1

    boards = np.expand_dims(board, axis=0)      # (1,29)
    players = np.array([player], dtype=np.int32)

    pts_t, g_t = encode_batch(boards, players)

    print(f"pts_t shape: {pts_t.shape}")   # expect (1, 12, 24)
    print(f"g_t shape:   {g_t.shape}")     # expect (1, 4)

    net = MLPValueNet().to(DEVICE)
    with torch.no_grad():
        v = net(pts_t, g_t)

    print(f"Value output shape: {v.shape}")
    print(f"Value output: {v.item():.4f}")

    # Also run a short self-play game with high exploration
    print("\n[SMOKE] Running one self-play game with epsilon=0.9")
    winner = play_one_selfplay_game(net, epsilon=0.9, verbose=False)
    print(f"[SMOKE] Game finished, winner={winner}")


def debug_train():
    """
    Small training run to verify replay + TD(0) wiring.
    """
    num_episodes = 10
    replay_capacity = 10000
    batch_size = 64
    gamma = 0.99
    lr = 1e-4
    epsilon = 0.2
    target_update_interval = 5  # episodes

    online, target = make_networks()
    optimizer = optim.Adam(online.parameters(), lr=lr)
    replay = ReplayBuffer(replay_capacity)

    for ep in range(1, num_episodes + 1):
        winner = play_game_and_fill_replay(online, replay, epsilon, gamma, verbose=False)

        # Do a few gradient steps per episode
        last_loss = None
        for _ in range(10):
            loss = train_step_td0(online, target, optimizer, replay, batch_size, gamma)
            if loss is not None:
                last_loss = loss

        # Periodically sync target with online
        if ep % target_update_interval == 0:
            target.load_state_dict(online.state_dict())

        print(
            f"[DEBUG] ep={ep:02d}, "
            f"winner={winner:+d}, "
            f"replay_size={len(replay)}, "
            f"last_loss={last_loss}"
        )
#iteration 3, step 3
def train_online_td0(
    num_episodes: int = 20000,
    gamma: float = 0.99,
    lr: float = 1e-4,
    epsilon_start: float = 0.3,
    epsilon_end: float = 0.05,
    eval_interval: int = 1000,
    eval_games: int = 100,
    save_path: Optional[str] = "conv_td0_bjarni.pt",
) -> nn.Module:
    """
    Main training loop for the conv TD(0) agent.

    - Online TD(0) updates every move.
    - Epsilon decays linearly from epsilon_start to epsilon_end.
    - Periodically evaluates vs random baseline and optionally saves best model.

    Returns:
        The trained ConvValueNet.
    """
    net = MLPValueNet().to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    best_winrate = -1.0

    for ep in range(1, num_episodes + 1):
        # linear epsilon schedule
        t = min(1.0, ep / float(num_episodes))
        epsilon = epsilon_start + t * (epsilon_end - epsilon_start)

        winner, moves = play_and_train_online_td0_episode(
            net, optimizer, gamma, epsilon, verbose=False
        )

        # simple per-episode log
        print(
            f"[TRAIN] ep={ep:05d}, "
            f"epsilon={epsilon:.3f}, "
            f"winner={winner:+d}, "
            f"moves={moves}"
        )

        # periodic evaluation vs random
        if (ep % eval_interval) == 0:
            print(
                f"\n[TRAIN] Evaluating vs random after ep={ep} "
                f"({eval_games} games, greedy)..."
            )
            winrate = evaluate_vs_random(net, num_games=eval_games, epsilon_eval=0.0)

            # optional checkpointing
            if winrate > best_winrate:
                best_winrate = winrate
                if save_path is not None:
                    torch.save(net.state_dict(), save_path)
                    print(
                        f"[TRAIN] New best winrate={winrate:.3f}, "
                        f"saved to {save_path}"
                    )
            print("")

    return net

#end of iteration 3, step 3

#iteration 5, step 1
def train_td0_conv_main(
    total_episodes: int = DEFAULT_TOTAL_EPISODES,
    gamma: float = GAMMA,
    lr: float = LR,
    e_start: float = E_START,
    e_end: float = E_END,
    eval_interval: int = EVAL_INTERVAL,
    eval_num_games: int = EVAL_NUM_GAMES,
) -> None:
    """
    Final TD(0) training loop:
      - online TD(0) self-play
      - epsilon annealing
      - periodic evaluation vs random
      - periodic checkpointing
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    net = MLPValueNet().to(DEVICE)
    print(f"[INFO] Training model: {net.__class__.__name__}")
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # Optional seed for partial reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    best_random_win = 0.0
    eval_history: list[tuple[int, float]] = []  # (episode, winrate_vs_random)

    for ep in range(1, total_episodes + 1):
        # Linear epsilon decay
        t = min(1.0, ep / float(total_episodes))
        epsilon = e_start + t * (e_end - e_start)

        # --- Cheap tweak 2: LR schedule ---
        # Simple LR schedule: decay at 60% and 85% of training
        frac = ep / float(total_episodes)
        if frac > 0.85:
            lr_factor = 0.1
        elif frac > 0.60:
            lr_factor = 0.3
        else:
            lr_factor = 1.0

        for g in optimizer.param_groups:
            g["lr"] = lr * lr_factor
        # --- end LR schedule ---

        winner, moves = play_and_train_online_td0_episode(
            net, optimizer, gamma, epsilon, verbose=False
        )

        if ep % 1000 == 0:
            print(
                f"[TRAIN] ep={ep}, epsilon={epsilon:.4f}, "
                f"lr={optimizer.param_groups[0]['lr']:.6g}, "
                f"last_winner={winner:+d}, last_moves={moves}"
            )

        # Periodic evaluation + checkpointing
        if ep % eval_interval == 0:
            print(f"[TRAIN] Evaluating vs random after ep={ep} ({eval_num_games} games, greedy)...")
            win_rate = evaluate_vs_random(net, num_games=eval_num_games, epsilon_eval=0.0)

            # Track eval history for plotting later
            eval_history.append((ep, float(win_rate)))

            # Save checkpoint for this episode
            ckpt_path = os.path.join(
                CHECKPOINT_DIR,
                f"{BASE_MODEL_NAME}_ep{ep}.pt"
            )
            torch.save(net.state_dict(), ckpt_path)
            print(f"[CKPT] Saved model to {ckpt_path}")

            # Track best vs random
            if win_rate > best_random_win:
                best_random_win = win_rate
                best_path = os.path.join(
                    CHECKPOINT_DIR,
                    f"{BASE_MODEL_NAME}_best_random.pt"
                )
                torch.save(net.state_dict(), best_path)
                print(f"[CKPT] New best vs random={win_rate:.3f}, saved to {best_path}")

    
    history_path = os.path.join(
        CHECKPOINT_DIR,
        f"{BASE_MODEL_NAME}_eval_history.csv"
    )
    with open(history_path, "w") as f:
        f.write("episode,winrate_vs_random\n")
        for ep_i, wr in eval_history:
            f.write(f"{ep_i},{wr}\n")
    print(f"[LOG] Saved eval history to {history_path}")

#end of iteration 5, 1

if __name__ == "__main__":

    _smoke_test()

    print("\n[TRAIN] Starting final TD(0) run...")
    train_td0_conv_main()