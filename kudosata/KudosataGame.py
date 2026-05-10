import sys
import numpy as np
import time
import copy

from Coach import log

sys.path.append('..')
sys.path.append('.')
from Game import Game

# from openxum_kudosata import Engine, BoardSize, Color
from . import openxum_kudosata as k


class KudosataGame(Game):
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.

    See othello/OthelloGame.py for an example implementation.
    """
    def __init__(self, board_size: k.BoardSize = k.BoardSize.SMALL):
        self.board_size = board_size
        self.n = 3 if board_size == k.BoardSize.SMALL else 6 if board_size == k.BoardSize.MEDIUM else 9

        self.colors = [k.Color.RED, k.Color.YELLOW]
        self.color_idx = {c: i for i, c in enumerate(self.colors)}

        self.types = [k.TriangleType.NORMAL, k.TriangleType.STAR, k.TriangleType.CONNECT_R, k.TriangleType.CONNECT_L]
        self.type_idx = {t: i for i, t in enumerate(self.types)}

        self.directions = [k.Direction.NORTH, k.Direction.EAST, k.Direction.WEST, k.Direction.SOUTH]
        self.dir_idx = { d: i for i, d in enumerate(self.directions)}

        empty_board = k.Board(self.board_size)
        engine = k.Engine(int(self.board_size), int(k.Color.RED))
        self.max_remaining = engine.remaining_triangle_count(empty_board)

    
    """
    Utility function
    """
    def getEncodedState(self, engine_board, state_player, current_reserves=None):
        state = np.zeros((41, self.n, self.n), dtype=np.float32)

        for x in range(self.n):
            for y in range(self.n):
                squareCoord = k.SquareCoord(x, y)
                for direction in self.directions:

                    triangle_id = k.TriangleID(squareCoord, direction)
                    color = engine_board.get_triangle_color(triangle_id)

                    if color != k.Color.NONE:
                        triangle_type = engine_board.get_triangle_type(triangle_id)

                        layer = (self.color_idx[color] * 16) + (self.type_idx[triangle_type] * 4) + self.dir_idx[direction]
                        state[layer][x][y] = 1.0
                        
        state[32][:, :] = state_player

        if current_reserves is None:
            current_reserves = copy.deepcopy(self.max_remaining)

        for color in self.colors:
            c_idx = self.color_idx[color]
            
            for t_type in self.types:
                
                t_idx = self.type_idx[t_type]
                layer = 33 + c_idx * 4 + t_idx

                max_count = self.max_remaining[color][t_type]
                count = current_reserves[color][t_type]

                value = 0.0 if max_count == 0 else count / max_count
                state[layer, :, :] = value

        return state
    
    """
    Utility function
    """
    def translate_matrix_to_board(self, board_matrix):
        board_obj = k.Board(self.board_size)
        layers, xs, ys = np.where(board_matrix[:32] == 1.0)

        for l, x, y in zip(layers, xs, ys):
            p_idx = l // 16
            t_idx = (l % 16) // 4
            d_idx = l % 4

            color = self.colors[p_idx]
            direction = self.directions[d_idx]
            t_type = self.types[t_idx]

            board_obj.place_triangle(k.TriangleID(k.SquareCoord(int(x), int(y)), direction), color, t_type, True)
        return board_obj
    

    """
    Utility function
    """
    def getEngineTriangleID(self, x, y, orientation_idx):
        return k.TriangleID(
            k.SquareCoord(int(x), int(y)),
            self.directions[orientation_idx]
        )
    
    """
    Utility function
    """
    def update_reserves(self, current_reserves, t_type, player):
        new_reserves = copy.deepcopy(current_reserves)
        color = k.Color.RED if player == 1 else k.Color.YELLOW
        new_reserves[color][t_type] -= 1
        # else:
            # raise ValueError(f"INVALID ACTION : {t_type} for player {player} was played but its reserve is empty")

        return new_reserves
    
    def retrieve_reserves(self, state_board):
        reserves = {k.Color.RED: {}, k.Color.YELLOW: {}}
        for c in self.colors:
            c_idx = self.color_idx[c]
            for t in self.types:
                t_idx = self.type_idx[t]
                layer = 33 + c_idx * 4 + t_idx
                
                max_count = self.max_remaining[c][t]
                current_count = int(round(float(state_board[layer, 0, 0]) * max_count))
                reserves[c][t] = current_count
        
        return reserves
        


    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        engine_board = k.Board(self.board_size)
        return self.getEncodedState(engine_board, 1)

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return (41, self.n, self.n)

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return self.n * self.n * 4 * 4
    
    def getNextState(self, state_board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        current_board_obj = self.translate_matrix_to_board(state_board)
        x, y, dir_idx, type_idx = self.decode_action(action)

        color = k.Color.RED if player == 1 else k.Color.YELLOW
        direction = self.directions[dir_idx]
        t_type = self.types[type_idx]

        current_reserves = self.retrieve_reserves(state_board)

        if current_reserves[color][t_type] <= 0:
            raise ValueError(f"INVALID ACTION: no reserve left for {color} {t_type}")
        
        t_id = k.TriangleID(k.SquareCoord(int(x), int(y)), direction)
        
        if not current_board_obj.is_valid_to_place(t_id, color, t_type):
            raise ValueError(
                f"INVALID ACTION: cannot place | x={x}, y={y}, dir={direction}, type={t_type}, color={color}, action={action}"
            )
            
        new_reserves = self.update_reserves(current_reserves, t_type, player)

        # 4. Gameplay normal
       
        next_board_obj = current_board_obj.get_next_state(
            int(x), int(y), direction, t_type, color
        )
        
        # before = current_board_obj.to_string()
        # after = next_board_obj.to_string()

        # if before == after:
        #     raise ValueError(
        #         f"BOARD DID NOT CHANGE after action : x={x}, y={y}, dir={direction}, type={t_type}, color={color} |\n previous board: {before} \n board after: {after}"
            # )

        next_state = self.getEncodedState(next_board_obj, -player, current_reserves=new_reserves)

        return next_state, -player

    def getValidMoves(self, state_board, state_player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        valid_moves = np.zeros(self.getActionSize(), dtype=np.int8)

        engine_board = self.translate_matrix_to_board(state_board)
        current_color = k.Color.RED if state_player == 1 else k.Color.YELLOW

        for x in range(self.n):
            for y in range(self.n):
                for dir_idx, direction in enumerate(self.directions):
                    for type_idx, t_type in enumerate(self.types):

                        reserve_layer = 33 + type_idx if state_player == 1 else 37 + type_idx
                        reserve_value = float(state_board[reserve_layer, 0, 0])

                        if reserve_value <= 0:
                            continue

                        t_id = k.TriangleID(
                            k.SquareCoord(int(x), int(y)),
                            direction
                        )

                        if engine_board.is_valid_to_place(t_id, current_color, t_type):
                            action_idx = self.encode_action(x, y, dir_idx, type_idx)
                            valid_moves[action_idx] = 1

        return valid_moves


    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               
        """
        current_color = k.Color.RED if player == 1 else k.Color.YELLOW
        remaining = self.retrieve_reserves(board)

        valid_moves = self.getValidMoves(board, player)

        current_reserve_total = sum(
            remaining[current_color][t] for t in self.types
        )

        game_should_end = (
                current_reserve_total <= 0
                or sum(valid_moves) == 0
        )

        if not game_should_end:
            return 0
        
        engine_board = self.translate_matrix_to_board(board)
        engine = k.Engine(int(self.board_size), int(k.Color.RED))
        engine.parse(engine_board.to_string())


        red_gain = engine.gain(k.Color.RED, True)
        yellow_gain = engine.gain(k.Color.YELLOW, True)

        if red_gain == yellow_gain:
            return 0.01

        winner_color = k.Color.RED if red_gain > yellow_gain else k.Color.YELLOW

        if winner_color == current_color:
            return 1
        else:
            return -1


    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        if player == 1:
            return board

        canonical_board = np.copy(board)

        player1_layers = np.copy(board[0:16])
        player2_layers = np.copy(board[16:32])

        canonical_board[0:16] = player2_layers
        canonical_board[16:32] = player1_layers

        canonical_board[32][:, :] = 1.0

        red_reserves = np.copy(board[33:37])
        yellow_reserves = np.copy(board[37:41])

        canonical_board[33:37] = yellow_reserves
        canonical_board[37:41] = red_reserves

        return canonical_board

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector.
        """
        pi_np = np.array(pi)
        n_dirs = 4
        n_types = 4
        expected_size = self.n * self.n * n_dirs * n_types
        if pi_np.size != expected_size:
            log.error(f"Taille incorrecte pour pi: {pi_np.size}, attendu {expected_size}")
            return [(board, pi)]
            
        pi_reshaped = pi_np.reshape((self.n, self.n, n_dirs, n_types))
        l = []

        # original
        l += [(board, pi_np.flatten())]
        
        # horizontal flip
        board_h = np.flip(board, axis=2).copy() 
        
        east_layers_idx = [i for i in range(32) if i % 4 == 1]
        west_layers_idx = [i for i in range(32) if i % 4 == 2]
        temp_east_layers = np.copy(board_h[east_layers_idx])
        board_h[east_layers_idx] = board_h[west_layers_idx]
        board_h[west_layers_idx] = temp_east_layers
        
        pi_h = np.flip(pi_reshaped, axis=1).copy()
        east_pi = np.copy(pi_h[:, :, 1, :])
        west_pi = np.copy(pi_h[:, :, 2, :])
        pi_h[:, :, 1, :] = west_pi
        pi_h[:, :, 2, :] = east_pi
        
        l += [(board_h, pi_h.flatten())]

        # vertical flip
        board_v = np.flip(board, axis=1).copy()
        
        north_layers_idx = [i for i in range(32) if i % 4 == 0]
        south_layers_idx = [i for i in range(32) if i % 4 == 3]
        temp_north_layers = np.copy(board_v[north_layers_idx])
        board_v[north_layers_idx] = board_v[south_layers_idx]
        board_v[south_layers_idx] = temp_north_layers
        
        pi_v = np.flip(pi_reshaped, axis=0).copy()
        north_pi = np.copy(pi_v[:, :, 0, :])
        south_pi = np.copy(pi_v[:, :, 3, :])
        pi_v[:, :, 0, :] = south_pi
        pi_v[:, :, 3, :] = north_pi
        
        l += [(board_v, pi_v.flatten())]
        
        return l

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return board.astype(np.float32).tobytes()

    def encode_action(self, x, y, dir_idx, type_idx):
        return (((x * self.n + y) * 4) + dir_idx) * 4 + type_idx

    def decode_action(self, action):
        type_idx = action % 4
        remaining = action // 4

        dir_idx = remaining % 4
        remaining //= 4

        y = remaining % self.n
        x = remaining // self.n

        return x, y, dir_idx, type_idx