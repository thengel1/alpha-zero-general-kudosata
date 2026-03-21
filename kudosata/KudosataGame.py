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
        self.engine = k.Engine(int(board_size), int(k.Color.RED))

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


    def getEncodedState(self, engine):
        
        state = np.zeros((33, self.board_size, self.board_size), dtype=np.float32)
        board = self.engine.board()

        for x in range(self.board_size):
            for y in range(self.board_size):
                squareCoord = k.SquareCoord(x, y)
                for direction in self.directions:
                    
                    triangle_id = k.TriangleID(squareCoord, direction)
                    color = board.get_triangle_color(triangle_id)
                    
                    if color != k.Color.NONE:
                        triangle_type = board.get_triangle_type(triangle_id)

                        layer = (self.color_idx[color] * 16) + (self.type_idx[triangle_type] * 4) + self.dir_idx[direction]
                        state[layer][x][y] = 1.0

        if self.engine.current_color() == self.color_idx[k.Color.YELLOW]:
            state[32][:, :] = 1.0
            
        return state
    
    def getActionIdx(self, triangle_id, triangle_type):
        pass


    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        e = k.Engine(int(self.board_size), int(k.Color.RED))
        return self.getEncodedState(e)

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


    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        pass

    def isActionValid(self, action):
        pass

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
        valid_moves = np.ones(n_actions)
        action_idx = 0
        for x in range(self.board_size):
            for y in range(self.board_size):
                for dir in self.directions:
                    for type in self.type_idx:
                        action = (x, y, dir, type)
                        if not self.isActionValid(board, player, action):
                            valid_moves[action_idx] = 0
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
        pass

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