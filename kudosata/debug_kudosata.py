from kudosata.KudosataGame import KudosataGame
from kudosata import openxum_kudosata as k


def debug_roundtrip():
    game = KudosataGame(k.BoardSize.MEDIUM)
    board = k.Board(game.board_size)

    samples = [
        (0, 0, k.Direction.NORTH, k.Color.RED,    k.TriangleType.NORMAL),
        (0, 1, k.Direction.EAST,  k.Color.RED,    k.TriangleType.STAR),
        (1, 0, k.Direction.WEST,  k.Color.YELLOW, k.TriangleType.CONNECT_R),
        (1, 1, k.Direction.SOUTH, k.Color.YELLOW, k.TriangleType.CONNECT_L),
    ]

    for x, y, d, c, t in samples:
        board.place_triangle(k.TriangleID(k.SquareCoord(x, y), d), c, t, True)

    encoded = game.getEncodedState(board, 1)
    decoded = game.translate_matrix_to_board(encoded)

    print("=== ORIGINAL ===")
    print(board.to_string())
    print("=== DECODED ===")
    print(decoded.to_string())

    assert board.to_string() == decoded.to_string(), "Roundtrip encoding/decoding FAILED"
    print("Roundtrip OK")


if __name__ == "__main__":
    debug_roundtrip()