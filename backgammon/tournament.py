#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import importlib
import time
import numpy as np

import backgammon

# ---------- utilities ----------
def _is_empty_move(move):
    if move is None:
        return True
    if isinstance(move, (list, tuple)):
        return len(move) == 0
    if isinstance(move, np.ndarray):
        return move.size == 0
    return False

def _opening_toss():
    """Return (starting_player, opening_dice). Reroll on ties.
    starting_player is +1 or -1. opening_dice is (d1, d2) with d1!=d2.
    """
    while True:
        d1 = np.random.randint(1, 7)
        d2 = np.random.randint(1, 7)
        if d1 == d2:
            continue
        starter = 1 if d1 > d2 else -1
        return starter, (d1, d2)

# ---------- single game ----------
def play_one_game(agent_plus, agent_minus, commentary=False):
    """
    agent_plus: module for the +1 player (must expose .action)
    agent_minus: module for the -1 player
    returns winner as +1 or -1
    """
    board = backgammon.init_board()

    # Opening toss decides who starts; those dice are used for the first move.
    player, dice = _opening_toss()
    first_turn = True

    while not backgammon.game_over(board) and not backgammon.check_for_error(board):
        if commentary:
            print(f"player {player}, dice {dice}")

        # choose agent by side
        agent_mod = agent_plus if player == +1 else agent_minus

        # ask for a move (engine expects the whole turn’s sequence once)
        move = agent_mod.action(board.copy(), dice, player, i=0, train=False)

        if not _is_empty_move(move):
            mv = np.asarray(move, dtype=np.int32)
            board = backgammon.update_board(board, mv, player)

        # next turn: alternate player and roll dice normally
        player = -player
        dice = backgammon.roll_dice()
        first_turn = False

    # winner is the one who just made the bearing-off move
    return -player

# ---------- round robin ----------
def run_round_robin(agent_names, games_per_pair=1000, commentary_every=None, seed=None):
    """
    agent_names: list of module names to import (e.g., ["random_player", "pubeval_player", "agent_ac_submit"])
    games_per_pair: number of games for each unordered pair
    commentary_every: if set, prints commentary every N games within a pairing
    seed: optional RNG seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)

    # import agents
    agents = {name: importlib.import_module(name) for name in agent_names}

    names = list(agents.keys())
    n = len(names)

    # results containers
    wins_total = {name: 0 for name in names}
    results = {a: {b: {"wins": 0, "losses": 0} for b in names if b != a} for a in names}

    t0 = start = time.time()
    pair_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            a_name, b_name = names[i], names[j]
            a_mod, b_mod = agents[a_name], agents[b_name]

            pair_count += 1
            print(f"\n=== Pair {pair_count}: {a_name} vs {b_name} — {games_per_pair} games ===")
            p_start = time.time()

            for g in range(1, games_per_pair + 1):
                # We pass A as +1 side and B as -1 side here; opening toss still decides first mover,
                # but sides (+1/-1) are fixed per call so that flip logic is exercised for both agents across many games.
                winner = play_one_game(a_mod, b_mod, commentary=False)

                if winner == +1:
                    wins_total[a_name] += 1
                    results[a_name][b_name]["wins"] += 1
                    results[b_name][a_name]["losses"] += 1
                else:
                    wins_total[b_name] += 1
                    results[b_name][a_name]["wins"] += 1
                    results[a_name][b_name]["losses"] += 1

                if commentary_every and (g % commentary_every == 0):
                    wa = results[a_name][b_name]["wins"]
                    wb = results[b_name][a_name]["wins"]
                    print(f"  [{g}/{games_per_pair}] {a_name} vs {b_name}: {wa}-{wb} "
                          f"({wa/(wa+wb):.3f} / {wb/(wa+wb):.3f})")

            dt = time.time() - p_start
            wa = results[a_name][b_name]["wins"]
            wb = results[b_name][a_name]["wins"]
            print(f"Pair result: {a_name} {wa} — {wb} {b_name}  |  {dt:.2f}s  ({dt/games_per_pair:.4f}s/game)")

    # leaderboard
    dur = time.time() - t0
    total_games = games_per_pair * (n * (n - 1) // 2)
    print("\n===== FINAL LEADERBOARD =====")
    leaderboard = sorted(wins_total.items(), key=lambda kv: kv[1], reverse=True)
    for rank, (name, w) in enumerate(leaderboard, 1):
        print(f"{rank:2d}. {name:20s}  wins: {w:6d}")

    # pretty matrix
    print("\n===== HEAD-TO-HEAD MATRIX (wins) =====")
    header = " " * 12 + " ".join(f"{nm:>12s}" for nm in names)
    print(header)
    for a in names:
        row = [f"{a:>12s}"]
        for b in names:
            if a == b:
                cell = "—"
            else:
                cell = str(results[a][b]["wins"])
            row.append(f"{cell:>12s}")
        print(" ".join(row))

    print(f"\nTotal games: {total_games}  |  Wall time: {dur:.2f}s  |  {dur/max(1,total_games):.5f}s/game")
    return results, wins_total

# ---------- example entry point ----------
if __name__ == "__main__":
    # List the agents you want to include:
    AGENTS = [
        "random_player",
        "pubeval_player",
        "agent_ac_submit",   # your AC submit-only agent
        "agent_td0_bjarni_submit",
        # "agent_dqn_submit",  # add if you have a DQN submit-only agent
    ]
    run_round_robin(AGENTS, games_per_pair=100, commentary_every=None, seed=123)

