import unittest
import sys

sys.path.append('..')
sys.path.append('.')

import openxum_kudosata as k
import numpy as np

from kudosata.KudosataGame import KudosataGame


class TestKudosataGame(unittest.TestCase):
    def setUp(self):
        self.game = KudosataGame(k.BoardSize.SMALL)
        self.board_size = int(k.BoardSize.SMALL)
        self.action_size = 144

        self.init_board = np.zeros((33, self.board_size, self.board_size), dtype=np.float32)
        self.init_board[32][:, :] = 1

    def test_init_custom_board_size(self):
        game = KudosataGame(board_size=k.BoardSize.MEDIUM)
        self.assertEqual(game.board_size, k.BoardSize.MEDIUM)

    def test_getBoardSize(self):
        game = KudosataGame(board_size=k.BoardSize.MEDIUM)
        self.assertEqual(game.getBoardSize(), (33, 6, 6))

        game = KudosataGame(board_size=k.BoardSize.SMALL)
        self.assertEqual(game.getBoardSize(), (33, 3, 3))

        game = KudosataGame(board_size=k.BoardSize.LARGE)
        self.assertEqual(game.getBoardSize(), (33, 9, 9))

    def test_action_size(self):
        game = KudosataGame(board_size=k.BoardSize.SMALL)
        self.assertEqual(game.getActionSize(), 144)

        game = KudosataGame(board_size=k.BoardSize.MEDIUM)
        self.assertEqual(game.getActionSize(), 576)
        
        game = KudosataGame(board_size=k.BoardSize.LARGE)
        self.assertEqual(game.getActionSize(), 1296)

    def test_get_init_board(self):
        board = self.game.getInitBoard()
        n = int(self.game.board_size)
        self.assertEqual(board.shape, (33, n, n))
        self.assertFalse(board[:32].any())
        self.assertTrue(np.all(board[32] == 1))

    def test_getValidMoves_len(self):
        
        valid_moves = self.game.getValidMoves(self.init_board, 1)

        self.assertEqual(len(valid_moves), self.action_size)

    def test_getValidMoves_init_board(self):
        player = 1
        
        unexisting_corners_cells = 8
        n_types = 4
        expected_illigal_count = unexisting_corners_cells * n_types

        valid_moves = self.game.getValidMoves(self.init_board, player)
        
        illegal_moves_count = len(valid_moves) - np.count_nonzero(valid_moves)
        
        self.assertEqual(illegal_moves_count, expected_illigal_count)

    def test_getValidMoves_in_game(self):
        player = 1
        valid_moves_before = self.game.getValidMoves(self.init_board, player)

        action = 56
        self.assertEqual(int(valid_moves_before[action]), 1)

        board_after_move, _ = self.game.getNextState(self.init_board, player, action)
        valid_moves_after = self.game.getValidMoves(board_after_move, -1)

        self.assertEqual(valid_moves_after.sum(), valid_moves_before.sum() - 4) # one triangle tile = 4 actions
        self.assertEqual(valid_moves_after[action], 0)

    def test_getNextState(self):
        player = 1
        direction = int(k.Direction.EAST)
        type = int(k.TriangleType.NORMAL)
        action = 1 * (self.board_size * 16) + 0 * 16 + direction * 4 + type # = 48 + 0 + 8 + 0 = 56
        
        next_board, next_player = self.game.getNextState(self.init_board, player, action)
        
        self.assertEqual(next_player, -1)
        self.assertTrue(np.all(next_board[32] == -1))
        
        # color_idx(RED)=0, type_idx(NORMAL)=0, dir_idx(EAST)=2
        # expected_layer = 0*16 + 0*4 + 2 = 2
        expected_layer = 2
        
        self.assertEqual(next_board[expected_layer, 1, 0], 1.0)
        self.assertEqual(np.sum(next_board[:32]), 1.0)

    def test_canonical_form_consistency(self):
        board = self.game.getInitBoard()
        board[0][0][0] = 1.0

        canonical = self.game.getCanonicalForm(board, -1)
        self.assertEqual(canonical[16][0][0], 1.0)
        self.assertEqual(canonical[0][0][0], 0.0)

    def test_game_not_ended(self):
        board = self.game.getInitBoard()
        self.assertEqual(self.game.getGameEnded(board, 1), 0)
    

if __name__ == "__main__":
    unittest.main()
