import unittest
import sys
import numpy as np

sys.path.append('..')
sys.path.append('.')

from kudosata import openxum_kudosata as k
from kudosata.KudosataGame import KudosataGame


class TestKudosataGame(unittest.TestCase):
    def setUp(self):
        self.game = KudosataGame(k.BoardSize.SMALL)
        self.board_init = k.Board(k.BoardSize.SMALL)
        self.initial_pi = np.zeros(self.game.getActionSize(), dtype=np.float32)
        
    def test_getSymmetries_board_flip_horizontaly(self):
        
        # Initial : RED, NORMAL, EAST, left side (x=1, y=0)
        t_id_init = k.TriangleID(k.SquareCoord(1, 0), k.Direction.EAST)
        self.board_init.place_triangle(t_id_init, k.Color.RED, k.TriangleType.NORMAL, True)

        # Expected : RED, NORMAL, WEST, right side (x=1, y=2)
        p_expected, t_expected, d_expected, x_expected, y_expected = 0, 0, 2, 1, 2
        
        # layer = 16 * player + 4 * type + dir
        layer_expected = 2

        state_init = self.game.getEncodedState(self.board_init, 1)

        pi_dummy = np.zeros(self.game.getActionSize(), dtype=np.float32)

        symm_forms = self.game.getSymmetries(state_init, pi_dummy)
        
        state_original, _ = symm_forms[0]
        state_symm, _ = symm_forms[1]

        np.testing.assert_array_equal(state_original, state_init)
        
        layer_symm, x_symm, y_symm  = np.argwhere(state_symm[:32] == 1.0)[0]
        p_symm = layer_symm // 16
        t_symm = (layer_symm % 16) // 4
        d_symm = layer_symm % 4
        
        self.assertEqual(x_symm, x_expected)
        self.assertEqual(y_symm, y_expected)
        self.assertEqual(layer_symm, layer_expected)
        
        self.assertEqual(p_symm, p_expected)
        self.assertEqual(t_symm, t_expected)
        self.assertEqual(d_symm, d_expected)
        
    def test_getSymmetries_pi_flip_horizontaly(self):
       
        # A : left side, EAST (1)
        # should end up on the right side, and WEST direction (2)
        a_x, a_y, a_dir, a_type = 0, 0, 1, 0
        exp_a_x, exp_a_y, exp_a_dir, exp_a_type = 0, 2, 2, 0
        idx_a = self.game.encode_action(a_x, a_y, a_dir, a_type)
        exp_idx_a = self.game.encode_action(exp_a_x, exp_a_y, exp_a_dir, exp_a_type)
        
        a_prob = 0.5 # Arbitrary value
        self.initial_pi[idx_a] = a_prob
        
        # B : right side, WEST (2)
        # should end up on the left side, and EAST direction (1)
        b_x, b_y, b_dir, b_type = 2, 2, 2, 0
        exp_b_x, exp_b_y, exp_b_dir, exp_b_type = 2, 0, 1, 0
        idx_b = self.game.encode_action(b_x, b_y, b_dir, b_type)
        exp_idx_b = self.game.encode_action(exp_b_x, exp_b_y, exp_b_dir, exp_b_type)
        
        b_prob = 0.3 # arbitrary value
        self.initial_pi[idx_b] = b_prob
        
        #C : centered, NORTH (0)
        # should stay in the same position
        c_x, c_y, c_dir, c_type = 1, 1, 0, 0
        idx_c = self.game.encode_action(c_x, c_y, c_dir, c_type)
        
        c_prob = 0.2 # arbitrary value
        self.initial_pi[idx_c] = c_prob
        

        dummy_board = self.game.getInitBoard()
        symm_forms = self.game.getSymmetries(dummy_board, self.initial_pi)
        
        _, original_pi = symm_forms[0]
        _, sym_pi = symm_forms[1]
        
        np.testing.assert_array_equal(original_pi, self.initial_pi)
        
        self.assertAlmostEqual(sym_pi[exp_idx_a], a_prob)
        self.assertEqual(sym_pi[idx_a], 0.0)
        
        self.assertAlmostEqual(sym_pi[exp_idx_b], b_prob)
        self.assertEqual(sym_pi[idx_b], 0.0)
        
        self.assertAlmostEqual(sym_pi[idx_c], c_prob)
        
        exp_sum = a_prob + b_prob + c_prob
        self.assertAlmostEqual(sum(sym_pi), exp_sum)
        
    def test_getSymmetries_board_flip_vertical(self):
        
        # Initial : RED, NORMAL, NORTH, top side (x=0, y=1)
        t_id_init = k.TriangleID(k.SquareCoord(0, 1), k.Direction.NORTH)
        self.board_init.place_triangle(t_id_init, k.Color.RED, k.TriangleType.NORMAL, True)

        # Expected : RED, NORMAL, SOUTH, bottom side (x=2, y=1)
        p_expected, t_expected, d_expected, x_expected, y_expected = 0, 0, 3, 2, 1
        
        # layer = 16 * player + 4 * type + dir
        layer_expected = 3

        state_init = self.game.getEncodedState(self.board_init, 1)

        pi_dummy = np.zeros(self.game.getActionSize(), dtype=np.float32)

        symm_forms = self.game.getSymmetries(state_init, pi_dummy)
        
        state_original, _ = symm_forms[0]
        state_symm, _ = symm_forms[2]

        np.testing.assert_array_equal(state_original, state_init)
        
        layer_symm, x_symm, y_symm  = np.argwhere(state_symm[:32] == 1.0)[0]
        p_symm = layer_symm // 16
        t_symm = (layer_symm % 16) // 4
        d_symm = layer_symm % 4
        
        self.assertEqual(x_symm, x_expected)
        self.assertEqual(y_symm, y_expected)
        self.assertEqual(layer_symm, layer_expected)
        
        self.assertEqual(p_symm, p_expected)
        self.assertEqual(t_symm, t_expected)
        self.assertEqual(d_symm, d_expected)
        
    def test_getSymmetries_pi_vector_vertical(self):        
        # A : top side, NORTH (0)
        # should end up on the bottom side, and SOUTH (3)
        a_x, a_y, a_dir, a_type = 0, 0, 0, 0
        exp_a_x, exp_a_y, exp_a_dir, exp_a_type = 2, 0, 3, 0
        idx_a = self.game.encode_action(a_x, a_y, a_dir, a_type)
        exp_idx_a = self.game.encode_action(exp_a_x, exp_a_y, exp_a_dir, exp_a_type)
        
        a_prob = 0.5 # Arbitrary value
        self.initial_pi[idx_a] = a_prob
        
        # B : bottom side, SOUTH (3)
        # should end up on the top side, and NORTH (0)
        b_x, b_y, b_dir, b_type = 2, 2, 3, 0
        exp_b_x, exp_b_y, exp_b_dir, exp_b_type = 0, 2, 0, 0
        idx_b = self.game.encode_action(b_x, b_y, b_dir, b_type)
        exp_idx_b = self.game.encode_action(exp_b_x, exp_b_y, exp_b_dir, exp_b_type)
        
        b_prob = 0.3 # arbitrary value
        self.initial_pi[idx_b] = b_prob
        
        # C : centered, EAST (1)
        # should stay the same
        c_x, c_y, c_dir, c_type = 1, 1, 1, 0
        idx_c = self.game.encode_action(c_x, c_y, c_dir, c_type)
        
        c_prob = 0.2 # arbitrary value
        self.initial_pi[idx_c] = c_prob
        
        dummy_board = self.game.getInitBoard()
        symm_forms = self.game.getSymmetries(dummy_board, self.initial_pi)
        
        _, original_pi = symm_forms[0]
        _, sym_pi = symm_forms[2]
        
        np.testing.assert_array_equal(original_pi, self.initial_pi)
        
        self.assertAlmostEqual(sym_pi[exp_idx_a], a_prob)
        self.assertEqual(sym_pi[idx_a], 0.0)
        
        self.assertAlmostEqual(sym_pi[exp_idx_b], b_prob)
        self.assertEqual(sym_pi[idx_b], 0.0)
        
        self.assertAlmostEqual(sym_pi[idx_c], c_prob)
        
        exp_sum = a_prob + b_prob + c_prob
        self.assertAlmostEqual(sum(sym_pi), exp_sum)
        

if __name__ == "__main__":
    unittest.main()