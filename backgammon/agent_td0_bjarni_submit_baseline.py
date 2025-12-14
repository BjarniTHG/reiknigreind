# agent_td0_bjarni_submit.py
import numpy as np
import torch

import backgammon
from train_conv_replay_bjarni import ConvValueNet, encode_batch, DEVICE

# Path to your best checkpoint from the 250k run
CHECKPOINT_PATH = "models/conv_td0_bjarni_best_random.pt"

# --- Load network once on import ---
_net = ConvValueNet().to(DEVICE)
_state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
_net.load_state_dict(_state)
_net.eval()


def _select_move_greedy_with_dice(board, dice, player):
    """
    Greedy one-ply move selection with *given dice*, matching your
    training logic (value from NEXT player's perspective, pick argmin).
    """
    moves, boards = backgammon.legal_moves(board, dice, player)

    # No legal moves -> pass
    if len(moves) == 0:
        return []

    # Evaluate successor boards from NEXT player's perspective
    cand_boards = np.stack(boards, axis=0).astype(np.int32)  # (M, 29)
    next_players = np.full((len(boards),), -player, dtype=np.int32)

    pts_t, g_t = encode_batch(cand_boards, next_players)

    with torch.no_grad():
        values = _net(pts_t, g_t).cpu().numpy()  # (M,)

    # In your v1/v2 design: values are from NEXT player's POV,
    # so "lower is better for us" -> argmin
    best_idx = int(values.argmin())
    return moves[best_idx]


def action(board, dice, player, i=0, train=False):
    """
    Tournament entry point:
    - board: current 29-length board
    - dice: (d1, d2)
    - player: +1 or -1
    Must return a move or [] to pass.
    """
    move = _select_move_greedy_with_dice(
        np.array(board, dtype=np.int32),
        dice,
        int(player),
    )

    # The tournament treats [] / None as "no move"
    return move
