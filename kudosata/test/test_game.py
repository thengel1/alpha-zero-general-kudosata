import unittest
import sys
import numpy as np

sys.path.append('..')
sys.path.append('.')

from kudosata import openxum_kudosata as k
from kudosata.KudosataGame import KudosataGame


# def log_eng_game_infos(
#     game_id, turn_count, r, manual_reserves, sum_valid_actions, 
#     e, expected_winner, actual_winner, history, 
#     sorted_actual_board_placements, sorted_expected_board_placements
#     ):
#     print(f"Game n°{game_id} ended after {turn_count} moves. Result : {r}")
#     print(f"Final reserves : {manual_reserves}")
#     print(f"{sum_valid_actions} valid actions left")
#     print(f"is game really finished: {e.is_finished()}")
#     print(f"Expected winner: {expected_winner}, actual: {actual_winner}\n")
    
#     print("history:")
#     for (c, x, y, d, t) in history:
#         c1 = "RED" if c == k.Color.RED else "YELLOW"
#         d1 = "N" if d == 0 else "E" if d == 1 else "W" if d == 2 else "S"
#         t1 = "NORMAL" if t == 0 else "STAR" if t == 1 else "CON_R" if t == 2 else "CON_L"
#         print(f"{c1}: x:{x}, y:{y}, d:{d1}, t:{t1}")
#     print("\n")
    
#     print("actual board: ")
#     print("\n".join(sorted_actual_board_placements))
#     print("expected board: ")
#     print("\n".join(sorted_expected_board_placements))
class TestKudosataGame(unittest.TestCase):
    def test_end_to_end_game_loop(self):
        np.random.seed(0)

        game = KudosataGame(k.BoardSize.SMALL)

        for game_idx in range(100):
            board = game.getInitBoard()
            player = 1
            history = []

            for turn in range(25):
                r = game.getGameEnded(board, player)
                if r != 0:
                    break

                valids = game.getValidMoves(board, player)
                valid_actions = np.where(valids == 1)[0]

                self.assertGreater(
                    len(valid_actions),
                    0,
                    f"No valid actions before game end at game {game_idx}, turn {turn}"
                )

                reserves_before = game.retrieve_reserves(board)

                action = int(np.random.choice(valid_actions))
                x, y, dir_idx, type_idx = game.decode_action(action)

                color = k.Color.RED if player == 1 else k.Color.YELLOW
                t_type = game.types[type_idx]

                self.assertGreater(
                    reserves_before[color][t_type],
                    0,
                    f"Played type {t_type} with empty reserve at game {game_idx}, turn {turn}"
                )

                board, next_player = game.getNextState(board, player, action)

                reserves_after = game.retrieve_reserves(board)

                self.assertEqual(
                    reserves_after[color][t_type],
                    reserves_before[color][t_type] - 1,
                    f"Reserve did not decrease correctly at game {game_idx}, turn {turn}"
                )

                for c in game.colors:
                    for t in game.types:
                        self.assertGreaterEqual(
                            reserves_after[c][t],
                            0,
                            f"Negative reserve for {c}, {t}, game {game_idx}, turn {turn}"
                        )

                history.append((color, x, y, dir_idx, type_idx))
                player = next_player

            r = game.getGameEnded(board, player)

            self.assertNotEqual(
                r,
                0,
                f"Game {game_idx} did not end naturally"
            )

            self.assertLessEqual(
                len(history),
                20,
                f"SMALL game should end in <= 20 moves, got {len(history)}"
            )

            self.assertEqual(
                int(np.sum(game.getValidMoves(board, player))),
                0,
                f"Game {game_idx} ended but still has valid moves"
            )


if __name__ == "__main__":
    unittest.main()