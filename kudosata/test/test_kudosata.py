import unittest
import sys
import numpy as np

sys.path.append('..')
sys.path.append('.')

from kudosata import openxum_kudosata as k
from kudosata.KudosataGame import KudosataGame


class TestKudosataGame(unittest.TestCase):
    def setUp(self):
        self.game = KudosataGame(k.BoardSize.MEDIUM)

    def test_init_default_board_size(self):
        self.assertEqual(self.game.board_size, k.BoardSize.MEDIUM)
        self.assertEqual(self.game.n, 6)

    def test_init_custom_board_size(self):
        small = KudosataGame(k.BoardSize.SMALL)
        medium = KudosataGame(k.BoardSize.MEDIUM)
        large = KudosataGame(k.BoardSize.LARGE)

        self.assertEqual(small.n, 3)
        self.assertEqual(medium.n, 6)
        self.assertEqual(large.n, 9)

    def test_getBoardSize(self):
        self.assertEqual(KudosataGame(k.BoardSize.SMALL).getBoardSize(), (41, 3, 3))
        self.assertEqual(KudosataGame(k.BoardSize.MEDIUM).getBoardSize(), (41, 6, 6))
        self.assertEqual(KudosataGame(k.BoardSize.LARGE).getBoardSize(), (41, 9, 9))

    def test_action_size(self):
        self.assertEqual(KudosataGame(k.BoardSize.SMALL).getActionSize(), 3 * 3 * 4 * 4)
        self.assertEqual(KudosataGame(k.BoardSize.MEDIUM).getActionSize(), 6 * 6 * 4 * 4)
        self.assertEqual(KudosataGame(k.BoardSize.LARGE).getActionSize(), 9 * 9 * 4 * 4)

    def test_get_init_board_shape_and_layers(self):
        board = self.game.getInitBoard()

        self.assertEqual(board.shape, (41, 6, 6))

        self.assertFalse(board[:32].any())

        self.assertTrue(np.all(board[32] == 1.0))

        self.assertTrue(np.all(board[33:41] == 1.0))

    def test_encoded_state_after_one_piece_updates_board_and_reserve(self):
        engine_board = k.Board(k.BoardSize.MEDIUM)

        t_id = k.TriangleID(3, 3, k.Direction.NORTH)
        engine_board.place_triangle(
            t_id,
            k.Color.RED,
            k.TriangleType.NORMAL,
            True
        )

        state = self.game.getEncodedState(engine_board, 1)

        red_normal_layer = 0
        red_normal_reserve_layer = 33

        self.assertEqual(state.shape, (41, 6, 6))
        self.assertEqual(state[red_normal_layer, 3, 3], 1.0)

        expected = 31 / 32
        self.assertAlmostEqual(
            float(state[red_normal_reserve_layer, 0, 0]),
            expected,
            places=5
        )

    def test_translate_matrix_to_board_roundtrip(self):
        engine_board = k.Board(k.BoardSize.MEDIUM)

        samples = [
            (0, 0, k.Direction.NORTH, k.Color.RED, k.TriangleType.NORMAL),
            (0, 1, k.Direction.EAST, k.Color.RED, k.TriangleType.STAR),
            (1, 0, k.Direction.WEST, k.Color.YELLOW, k.TriangleType.CONNECT_R),
            (1, 1, k.Direction.SOUTH, k.Color.YELLOW, k.TriangleType.CONNECT_L),
        ]

        for x, y, direction, color, t_type in samples:
            engine_board.place_triangle(
                k.TriangleID(k.SquareCoord(x, y), direction),
                color,
                t_type,
                True
            )

        encoded = self.game.getEncodedState(engine_board, 1)
        decoded = self.game.translate_matrix_to_board(encoded)

        self.assertEqual(engine_board.to_string(), decoded.to_string())

    def test_encode_decode_action_consistency(self):
        for x in range(self.game.n):
            for y in range(self.game.n):
                for dir_idx in range(4):
                    for type_idx in range(4):
                        action = self.game.encode_action(x, y, dir_idx, type_idx)
                        decoded = self.game.decode_action(action)
                        self.assertEqual(decoded, (x, y, dir_idx, type_idx))

    def test_getValidMoves_returns_correct_shape(self):
        board = self.game.getInitBoard()
        valids = self.game.getValidMoves(board, 1)

        self.assertEqual(valids.shape, (self.game.getActionSize(),))
        self.assertTrue(np.any(valids == 1))
        self.assertTrue(np.all((valids == 0) | (valids == 1)))

    def test_getNextState_changes_player_and_state(self):
        board = self.game.getInitBoard()
        valids = self.game.getValidMoves(board, 1)

        valid_actions = np.where(valids == 1)[0]
        self.assertGreater(len(valid_actions), 0)

        action = int(valid_actions[0])
        next_board, next_player = self.game.getNextState(board, 1, action)

        self.assertEqual(next_player, -1)
        self.assertEqual(next_board.shape, (41, 6, 6))
        self.assertTrue(next_board[:32].any())
        self.assertTrue(np.all(next_board[32] == -1.0))

    def test_reserve_decreases_after_getNextState(self):
        board = self.game.getInitBoard()
        valids = self.game.getValidMoves(board, 1)

        valid_actions = np.where(valids == 1)[0]
        action = int(valid_actions[0])

        x, y, dir_idx, type_idx = self.game.decode_action(action)

        next_board, _ = self.game.getNextState(board, 1, action)

        reserve_layer = 33 + type_idx

        self.assertLess(
            float(next_board[reserve_layer, 0, 0]),
            float(board[reserve_layer, 0, 0])
        )

    def test_canonical_form_swaps_players_and_reserves(self):
        board = self.game.getInitBoard()

        board[0, 0, 0] = 1.0

        board[33, :, :] = 0.5
        board[37, :, :] = 0.25

        canonical = self.game.getCanonicalForm(board, -1)

        self.assertEqual(canonical[16, 0, 0], 1.0)
        self.assertEqual(canonical[0, 0, 0], 0.0)

        self.assertTrue(np.all(canonical[32] == 1.0))

        self.assertAlmostEqual(float(canonical[33, 0, 0]), 0.25)
        self.assertAlmostEqual(float(canonical[37, 0, 0]), 0.5)

    def test_string_representation_distinguishes_reserves(self):
        board1 = self.game.getInitBoard()
        board2 = np.copy(board1)

        board2[33, :, :] = 0.5

        self.assertNotEqual(
            self.game.stringRepresentation(board1),
            self.game.stringRepresentation(board2)
        )

    def test_50_mcts_episodes_full_debug(self):
        from MCTS import MCTS
        from kudosata.NNetWrapper import NNetWrapper
        from utils import dotdict

        game = KudosataGame(k.BoardSize.SMALL)
        nnet = NNetWrapper(game)

        args = dotdict({
            'numMCTSSims': 50,
            'cpuct': 1,
            'tempThreshold' : 30,
        })

        for episode in range(50):
            board = game.getInitBoard()
            player = 1

            seen_states = {}
            history = []
            last_action = None

            for step in range(1, 501):
                state_key = game.stringRepresentation(board)

                repeated_state = state_key in seen_states
                seen_states[state_key] = step

                if game.getGameEnded(board, player) != 0:
                    break

                canonical = game.getCanonicalForm(board, player)

                mcts = MCTS(game, nnet, args)
                pi = np.array(mcts.getActionProb(canonical, temp=0))

                if repeated_state:
                    valids = game.getValidMoves(canonical,1)
                    valid_actions = np.where(valids == 1)[0]

                    self.assertGreater(len(valid_actions),0)

                    if last_action is not None:
                        valid_actions = [a for a in valid_actions if a != last_action["action"]]

                    action = int(np.random.choice(valid_actions))
                else:
                    action = int(np.random.choice(len(pi), p=pi))

                x, y, dir_idx, type_idx = game.decode_action(action)

                history.append({
                    "step": step,
                    "player_before": player,
                    "action": action,
                    "x": x,
                    "y": y,
                    "dir_idx": dir_idx,
                    "type_idx": type_idx,
                })

                last_action = history[-1]

                board, player = game.getNextState(board, player, action)

            self.assertLess(
                step,
                500,
                msg=(
                    f"Episode reached 500 steps | episode={episode}\n"
                    f"player={player}\n"
                    f"valid_moves={game.getValidMoves(board, player).sum()}\n"
                    f"reserve_layers={board[33:41, 0, 0]}\n"
                    f"history_last_20={history[-20:]}"
                )
            )

    def test_game_ended_initial_board(self):
        board = self.game.getInitBoard()
        self.assertEqual(self.game.getGameEnded(board, 1), 0)

if __name__ == "__main__":
    unittest.main()