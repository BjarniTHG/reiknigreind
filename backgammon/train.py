#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import backgammon
import pubeval_player as pubeval         # baseline
import random_player as randomAgent       # baseline
import flipped_agent as flipped_util
import agent_td_lambda as agent                            # student agent (this repo's agent.py)

from pathlib import Path
CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

def plot_perf(perf, title="Training progress (win-rate vs baseline)"):
    if not perf:
        return
    xs = np.arange(len(perf))
    plt.plot(xs, perf)
    plt.xlabel("Evaluation checkpoints")
    plt.ylabel("Win rate (%)")
    plt.title(title)
    plt.grid(True)
    plt.show()

def _is_empty_move(move):
    if move is None: return True
    if isinstance(move, (list, tuple)): return len(move) == 0
    if isinstance(move, np.ndarray): return move.size == 0
    return False

def _apply_move_sequence(board, move_seq, player):
    mv = np.asarray(move_seq, dtype=np.int32)
    return backgammon.update_board(board, mv, player)

def play_one_game(agent1, agent2, training=False, commentary=False):
    board = backgammon.init_board()
    player = np.random.randint(2) * 2 - 1  # +1 or -1

    if hasattr(agent1, "episode_start"): agent1.episode_start()
    if hasattr(agent2, "episode_start"): agent2.episode_start()

    while not backgammon.game_over(board) and not backgammon.check_for_error(board):
        dice = backgammon.roll_dice()
        if commentary:
            print(f"player {player}, dice {dice}")

        for r in range(1 + int(dice[0] == dice[1])):  # doubles -> two applications
            board_copy = board.copy()
            if player == 1:
                move = agent1.action(board_copy, dice, player, i=r, train=training)
            else:
                move = agent2.action(board_copy, dice, player, i=r, train=training)

            if _is_empty_move(move):
                # (optional) per-step observer hook could be placed here
                continue

            board = _apply_move_sequence(board, move, player)

        player = -player

    winner = -player
    final_board = board

    if hasattr(agent1, "end_episode"): agent1.end_episode(+1 if winner == 1 else -1, final_board, perspective=+1)
    if hasattr(agent2, "end_episode"): agent2.end_episode(+1 if winner == -1 else -1, final_board, perspective=-1)

    return winner, final_board

def evaluate(agent_mod, evaluation_agent, n_eval, label=""):
    wins = 0
    # alternate who starts to reduce bias
    for i in range(n_eval):
        if i % 2 == 0:
            w, _ = play_one_game(agent_mod, evaluation_agent, training=False, commentary=False)
        else:
            w, _ = play_one_game(evaluation_agent, agent_mod, training=False, commentary=False)
            w = -w
        wins += int(w == 1)
    winrate = round(wins / n_eval * 100.0, 3)
    print(f"[Eval] {label or 'checkpoint'}: win-rate = {winrate}% over {n_eval} games")
    return winrate

def train(n_games=200_000, n_epochs=5_000, n_eval=500, eval_vs="pubeval"):
    baseline = pubeval if eval_vs == "pubeval" else randomAgent

    best_wr = -1.0
    winrates = []

    print("Training agent with self-play...")
    print(f"Baseline for eval: {baseline.__name__ if hasattr(baseline, '__name__') else baseline}")

    for g in range(1, n_games + 1):
        winner, final_board = play_one_game(agent, agent, training=True, commentary=False)

        # legacy compatibility hook
        if hasattr(agent, "game_over_update"):
            agent.game_over_update(final_board, int(winner == 1))
            flipped_final = flipped_util.flip_board(final_board)
            agent.game_over_update(flipped_final, int(winner == -1))

        if (g % n_epochs) == 0:
            if hasattr(agent, "set_eval_mode"): agent.set_eval_mode(True)
            wr = evaluate(agent, baseline, n_eval, label=f"after {g} games")
            winrates.append(wr)

            # ---- always save this eval checkpoint, plus best if improved ----
            if hasattr(agent, "save"):
                # 1) save per-eval checkpoint with epoch suffix
                epoch_ckpt = CKPT_DIR / f"epoch_{g}.pt"
                agent.save(str(epoch_ckpt))
                print(f"[Checkpoint] Saved {epoch_ckpt}")

                # 2) also update best.pt if improved
                if wr > best_wr:
                    best_wr = wr
                    best_ckpt = CKPT_DIR / "best.pt"
                    agent.save(str(best_ckpt))  # overwrite best
                    print(f"[Best] New best: {best_wr:.3f}% — saved {best_ckpt}")

            if hasattr(agent, "set_eval_mode"): agent.set_eval_mode(False)

    plot_perf(winrates)

if __name__ == "__main__":
    # quick defaults; tweak as needed
    train(n_games=1_000_000, n_epochs=5000, n_eval=100, eval_vs="pubeval")
