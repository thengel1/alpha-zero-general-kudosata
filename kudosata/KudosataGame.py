import sys
import numpy as np

sys.path.append('..')
sys.path.append('.')
from Game import Game

# from openxum_kudosata import Engine, BoardSize, Color
import openxum_kudosata as k


class KudosataGame(Game):
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.

    See othello/OthelloGame.py for an example implementation.
    """
    def __init__(self, board_size: k.BoardSize = k.BoardSize.MEDIUM):
        self.board_size = board_size

        self.color_idx = {k.Color.RED: 0, k.Color.YELLOW: 1}
        self.type_idx = {
            k.TriangleType.NORMAL: 0,
            k.TriangleType.STAR: 1,
            k.TriangleType.CONNECT_R: 2,
            k.TriangleType.CONNECT_L: 3
        }
        self.dir_idx = {
            k.Direction.NORTH: 0,
            k.Direction.EAST: 1,
            k.Direction.WEST: 2,
            k.Direction.SOUTH: 3
        }
        self.directions = [k.Direction.NORTH, k.Direction.SOUTH, k.Direction.EAST, k.Direction.WEST]
        self.colors = [k.Color.RED, k.Color.YELLOW]
        self.types = [k.TriangleType.NORMAL, k.TriangleType.STAR, k.TriangleType.CONNECT_R, k.TriangleType.CONNECT_L]


    """
    Utility function
    """
    def getEncodedState(self, engine_board, player):
        
        state = np.zeros((33, self.board_size, self.board_size), dtype=np.float32)

        for x in range(self.board_size):
            for y in range(self.board_size):
                squareCoord = k.SquareCoord(x, y)
                for direction in self.directions:
                    
                    triangle_id = k.TriangleID(squareCoord, direction)
                    color = engine_board.get_triangle_color(triangle_id)
                    
                    if color != k.Color.NONE:
                        triangle_type = engine_board.get_triangle_type(triangle_id)

                        layer = (self.color_idx[color] * 16) + (self.type_idx[triangle_type] * 4) + self.dir_idx[direction]
                        state[layer][x][y] = 1.0

        if player == self.color_idx[k.Color.YELLOW]:
            state[32][:, :] = 1.0
            
        return state
    
    """
    Utility function
    """
    def translate_matrix_to_board(self, board_matrix):
        board_obj = k.Board(self.board_size)

        for layer in range(32):
            if np.any(board_matrix[layer] == 1.0):
                p_idx = layer // 16
                t_idx = (layer % 16) // 4
                d_idx = layer % 4

                color = k.Color.RED if p_idx == 0 else k.Color.YELLOW
                direction = self.directions[d_idx]
                t_type = self.types[t_idx]

                xs, ys = np.where(board_matrix[layer] == 1.0)
                for x, y in zip(xs, ys):
                    board_obj.place_triangle(k.TriangleID(k.SquareCoord(int(x), int(y)), direction),color, t_type, True)
        return board_obj
    
    # def encodedStateToBoard(board):
    #     return None

    """
    Utility function
    """
    def getEngineTriangleID(self, x, y, orientation_idx):
        squareCoord = k.SquareCoord(x, y)
        direction = self.directions[orientation_idx]

        return k.TriangleID(squareCoord, direction)

    
    # def getActionIdx(self, x, y, t_orientation, t_type):
    #     board_size = int(self.board_size)
    #     n_action_per_square = 16
    #     n_action_per_row = board_size * n_action_per_square

    #     return x * n_action_per_row + y * n_action_per_square + t_type * 4 + t_orientation
    

    # def getActionFromIdx(self, idx):
    #     board_size = int(self.board_size)
    #     n_action_per_square = 16
    #     n_action_per_row = board_size * n_action_per_square

    #     t_orientation = idx % 4
    #     t_type = (idx // 4) % 4
    #     y = (idx // n_action_per_square) % board_size
    #     x = (idx // n_action_per_row)

    #    return (x, y, t_orientation, t_type)
    
    # def getEngineAction(self, x, y, orientation_idx, type_idx, player_idx):
    #     t_id = self.getEngineTriangleID(x, y, orientation_idx)
    #     color = self.colors[player_idx]
    #     type = self.type[type_idx]

    #     return (t_id, type, color)



    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        board_obj = k.Board(self.board_size)
        return self.getEncodedState(board_obj, int(k.Color.RED))

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        n_squares = int(self.board_size)
        return (n_squares, n_squares)

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        n_squares = int(self.board_size)
        n_squares_coord = n_squares * n_squares
        n_triangles = n_squares_coord * 4
        n_triangle_types = 4
        return n_triangles * n_triangle_types


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
        n_types = 4
        n_dirs = 4

        type_idx = action % n_types
        remaining = action // n_types
        dir_idx = remaining % n_dirs
        remaining = remaining // n_dirs
        y = remaining % self.board_size
        x = remaining // self.board_size

        color = k.Color.RED if player == 1 else k.Color.YELLOW

        next_board_obj = current_board_obj.get_next_state(x, y, dir_idx, type_idx, color)

        return self.getEncodedState(next_board_obj, -player), -player

    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        n_actions = self.getActionSize()
        valid_moves = np.zeros(n_actions, dtype=np.int8)

        engine_board = self.translate_matrix_to_board(board)
        color = self.colors[player] 

        action_idx = 0
        for x in range(self.board_size):
            for y in range(self.board_size):

                if board[:32, x, y].any(): # if the cell is occupied, avoid unecessary loops
                    action_idx += 16
                    continue

                for t_type in self.types:
                    for dir in self.directions:

                        t_id = self.getEngineTriangleID(x, y, dir)
                        if engine_board.is_valid_to_place(t_id, color, t_type):
                            valid_moves[action_idx] = 1
                            
                        action_idx += 1

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
        engine_board = self.translate_matrix_to_board(board)
        engine = k.Engine()

        if not engine.is_finished(engine_board):
            return 0
        
        engine_winner = engine.winner_is(engine_board) #0 for RED, 1 for YELLOW, -1 for draw
        if engine_winner == -1:
            return 0.01
        
        current_player_color = k.Color.RED if player == 1 else k.Color.YELLOW
        
        if engine_winner == self.color_idx[current_player_color]:
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
        pass

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        pass

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        pass