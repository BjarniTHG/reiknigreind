# smoke_test.py
import torch
import backgammon

def main():
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    # Basic environment sanity: create a game and print initial state
    game = backgammon.Backgammon()
    observation = game.get_obs()
    print("Initial observation shape:", observation.shape)
    print("Legal moves at start:", len(game.legal_moves()))

if __name__ == "__main__":
    main()
