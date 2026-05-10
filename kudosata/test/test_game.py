import unittest
import sys
import numpy as np
import copy

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
        for game_id in range(1500) : 
            game = KudosataGame(k.BoardSize.SMALL)
            board = game.getInitBoard()
            player = 1
            
            manual_reserves = {
                1: {t: game.max_remaining[k.Color.RED][t] for t in game.types},
                -1: {t: game.max_remaining[k.Color.YELLOW][t] for t in game.types}
            }
            
            history = []
            
            turn_count = 0
            
            while True:
                current_color = k.Color.RED if player == 1 else k.Color.YELLOW
                r = game.getGameEnded(board, player)
                
                if r != 0: # End of the game : logging info and final assertions
                    sum_valid_actions = sum(game.getValidMoves(board, player))
                    sum_reserve_end = sum(
                        value
                        for player_reserves in manual_reserves.values()
                        for value in player_reserves.values()
                    )
                    
                    print(f"Game n°{game_id} ended after {turn_count} moves. Result : {r}")
                    print(f"Final reserves : {manual_reserves}")
                    print(f"{sum_valid_actions} valid actions left")
                    
                    e = k.Engine(k.BoardSize.SMALL, k.Color.RED)
                    for (c, x, y, d, t) in history: # Replaying the game to check if the engine produce the same result
                        d1 = game.directions[d]
                        c1 = game.colors[c]
                        t1 = game.types[t]
                        e.apply_single_move(x, y, d1, c1, t1)                         
                    
                    print(f"is game really finished: {e.is_finished()}")
                    
                    actual_winner = "NO WINNER" if r == 0.01 else ("RED" if player == r else "YELLOW")
                    expected_winner = e.winner_is()
                    expected_winner_str = "RED" if expected_winner == 0 else ("YELLOW" if expected_winner == 1 else "NO WINNER")
                    
                    print(f"Expected winner: {expected_winner_str}, actual: {actual_winner}\n")
                    
                    print("history:")
                    for (c, x, y, d, t) in history:
                        c1 = "RED" if c == k.Color.RED else "YELLOW"
                        d1 = "N" if d == 0 else "E" if d == 1 else "W" if d == 2 else "S"
                        t1 = "NORMAL" if t == 0 else "STAR" if t == 1 else "CON_R" if t == 2 else "CON_L"
                        print(f"{c1}: x:{x}, y:{y}, d:{d1}, t:{t1}")
                    print("\n")
                    
                    actual_board = game.translate_matrix_to_board(board).to_string()
                    actual_board_placements = str.split(actual_board,"\n")
                    sorted_actual_board_placements = sorted(actual_board_placements)
                    
                    expected_board = e.board().to_string()
                    expected_board_placements = str.split(expected_board,"\n")
                    sorted_expected_board_placements = sorted(expected_board_placements)
                    
                    print("actual board: ")
                    print("\n".join(sorted_actual_board_placements))
                    print("expected board: ")
                    print("\n".join(sorted_expected_board_placements))
                    
                    # log_eng_game_infos(
                    #     game_id, turn_count, r, manual_reserves, sum_valid_actions, e, 
                    #     expected_winner_str, actual_winner, history, 
                    #     sorted_actual_board_placements, expected_board_placements
                    # )
                    
                    self.assertTrue(sum_reserve_end == 0 or sum_valid_actions == 0.0)
                    self.assertTrue(e.is_finished())
                    self.assertEqual(actual_winner, expected_winner_str)
                    self.assertEqual(sorted_actual_board_placements, sorted_expected_board_placements)
                    
                    break
                    
                valids = game.getValidMoves(board, player)
                valid_actions = np.where(valids == 1)[0]
                
                self.assertGreater(len(valid_actions), 0)
                
                action = int(np.random.choice(valid_actions))
                
                x, y, dir_idx, type_idx = game.decode_action(action)
                direction = game.directions[dir_idx]
                t_type = game.types[type_idx]     
                
                duplicates_moves = [
                    (x1, y1, t_type1, direction1) 
                    for (_, x1, y1, t_type1, direction1) in history 
                    if x1 == x and y1 == y and t_type1 == t_type and direction1 == direction
                ]
                np.array_equiv(duplicates_moves, [])
                
                history.append((current_color, x, y, dir_idx, type_idx))
                
                engine_board = game.translate_matrix_to_board(board)
                t_id = k.TriangleID(k.SquareCoord(int(x), int(y)), direction)

                self.assertTrue(engine_board.is_valid_to_place(t_id, current_color, t_type, False))
                
                reserve_before = game.retrieve_reserves(board)[current_color]
                
                sum_reserve_before = sum([reserve_before[t] for t in game.types])
                
                self.assertGreater(sum_reserve_before, 0)
                self.assertEqual(reserve_before[t_type], manual_reserves[player][t_type])
                
                next_board, next_player = game.getNextState(board, player, action)
                
                reserve_after = game.retrieve_reserves(next_board)[current_color]
                
                sum_reserve_after = sum([reserve_after[t1] for t1 in game.types]) 
                self.assertEqual(sum_reserve_after, sum_reserve_before - 1)
                
                manual_reserves[player][t_type] -= 1
                
                board = next_board
                player = next_player
                turn_count += 1
                    
                self.assertLessEqual(turn_count, 20)

if __name__ == "__main__":
    unittest.main()