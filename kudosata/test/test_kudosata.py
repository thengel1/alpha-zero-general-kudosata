import unittest
import sys

sys.path.append('..')
sys.path.append('.')

import openxum_kudosata as k
import numpy as np

from kudosata.KudosataGame import KudosataGame


class TestKudosataGame(unittest.TestCase):
    def setUp(self):
        self.game = KudosataGame(k.BoardSize.MEDIUM)

    def test_init_default_board_size(self):
        self.assertEqual(self.game.board_size, k.BoardSize.MEDIUM)

    def test_init_custom_board_size(self):
        game = KudosataGame(board_size=k.BoardSize.SMALL)
        self.assertEqual(game.board_size, k.BoardSize.SMALL)

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

    def test_getNextState_and_captures(self):
        init_board = self.game.getInitBoard()
        next_board, next_player = self.game.getNextState(init_board, 1, 0)

        self.assertEqual(next_player, -1)
        self.assertTrue(next_board[:16].any())
        self.assertEqual(next_board[32][0][0], -1)

    def test_getValidMoves(self):
        board = self.game.getInitBoard()
        valids = self.game.getValidMoves(board, 1)

        self.assertEqual(len(valids), self.game.getActionSize())
        self.assertTrue(valids.any())

    def test_canonical_form_consistency(self):
        board = self.game.getInitBoard()
        board[0][0][0] = 1.0

        canonical = self.game.getCanonicalForm(board, -1)
        self.assertEqual(canonical[16][0][0], 1.0)
        self.assertEqual(canonical[0][0][0], 0.0)

    def test_game_ended(self):
        board = self.game.getInitBoard()
        self.assertEqual(self.game.getGameEnded(board, 1), 0)
    

if __name__ == "__main__":
    unittest.main()
