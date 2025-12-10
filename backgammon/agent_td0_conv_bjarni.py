# agent_td0_conv_bjarni.py
import numpy as np
import torch

import backgammon
from train_conv_replay_bjarni import ConvValueNet, encode_batch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to your chosen checkpoint (e.g., best vs random)
MODEL_PATH = "models/conv_td0_bjarni_best_random.pt"

# Load network once, globally
_net = ConvValueNet().to(DEVICE)
_state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
_net.load_state_dict(_state_dict)
_net.eval()


def _select_move_greedy(board: np.ndarray, dice, player: int) -> np.ndarray:
    """
    Greedy move selection for tournament play.
    board: np.ndarray shape (29,)
    dice:  (d1, d2) or np.ndarray shape (2,)
    player: +1 or -1
    returns: np.ndarray shape (k,2) with k in {0,1,2}
             (empty (0,2) if no legal moves)
    """
    board = np.asarray(board, dtype=np.int32)
    dice  = np.asarray(dice, dtype=np.int32)

    moves, boards = backgammon.legal_moves(board, dice, player)
    if len(moves) == 0:
        # no legal move -> give back an empty move
        return np.empty((0, 2), dtype=np.int32)

    # Evaluate successor boards from NEXT player's perspective
    cand_boards = np.stack(boards, axis=0).astype(np.int32)  # (M,29)
    next_players = np.full((len(boards),), -player, dtype=np.int32)

    pts_t, g_t = encode_batch(cand_boards, next_players)
    with torch.no_grad():
        values = _net(pts_t, g_t).cpu().numpy()  # (M,)

    best_idx = int(values.argmax())
    return moves[best_idx]


def action(board, dice, player, i=0, train=False):
    """
    API expected by tournament.py:
    - board: np.ndarray (len 29)
    - dice: (d1, d2)
    - player: +1 or -1
    - i: move index in game (unused)
    - train: if True, you *could* add exploration; for submit agent we ignore it.
    """
    return _select_move_greedy(board, dice, player)
