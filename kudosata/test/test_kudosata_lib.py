
import sys

sys.path.append('..')
sys.path.append('.')

from kudosata.KudosataGame import KudosataGame
import openxum_kudosata as k

game = KudosataGame(k.BoardSize.SMALL)
e = game.engine
board =  e.board()
print(board.to_string())
squareCoord = k.SquareCoord(0, 1)
triangle_id = k.TriangleID(squareCoord, k.Direction.NORTH)
c = board.get_triangle_color(triangle_id)
print(c)
print(len(board.get_empty_triangle_ids()))
board.place_triangle(triangle_id, k.Color.RED, k.TriangleType.NORMAL)
print(len(board.get_empty_triangle_ids()))
print(board.to_string())
print(e.possible_decisions())