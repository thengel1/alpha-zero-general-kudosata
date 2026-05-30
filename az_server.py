from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np

from kudosata import openxum_kudosata as k
from kudosata.KudosataGame import KudosataGame
from kudosata.NNetWrapper import NNetWrapper

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "temp")
CHECKPOINT_FILE = "best.pth.tar"

BOARD_SIZE = k.BoardSize.SMALL

game = KudosataGame(BOARD_SIZE)
nnet = NNetWrapper(game)

try:
    nnet.load_checkpoint(CHECKPOINT_DIR, CHECKPOINT_FILE)
    MODEL_LOADED = True
    print("AlphaZero model loaded.")
except Exception as e:
    MODEL_LOADED = False
    print("Model not loaded, fallback random:", e)


def js_move_to_action(move_str):
    x = ord(move_str[1])
    y = ord(move_str[2])
    direction_char = move_str[3]
    type_idx = int(move_str[4:])

    direction_map = {
        "N": 0,
        "E": 1,
        "W": 2,
        "S": 3,
    }

    dir_idx = direction_map[direction_char]
    return game.encode_action(x, y, dir_idx, type_idx)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL_LOADED,
        "board_size": "SMALL"
    })


@app.route("/move", methods=["POST"])
def move():
    data = request.json or {}

    possible_moves = data.get("possible_moves", [])
    engine_state = data.get("engine_state")

    if not possible_moves:
        return jsonify({"error": "no possible moves"}), 400

    if not MODEL_LOADED or engine_state is None:
        selected = np.random.choice(possible_moves)
        return jsonify({
            "move": selected,
            "source": "fallback-random"
        })

    try:
        engine = k.Engine(int(BOARD_SIZE), int(k.Color.RED))
        engine.parse(engine_state)

        board_obj = engine.board()
        current_color = engine.current_color()
        player = 1 if current_color == k.Color.RED else -1

        encoded_board = game.getEncodedState(board_obj, player)

        pi, v = nnet.predict(encoded_board)

        masked_scores = []

        for move_str in possible_moves:
            try:
                action = js_move_to_action(move_str)

                if 0 <= action < len(pi):
                    score = float(pi[action])
                    masked_scores.append((score, move_str))
            except Exception:
                continue

        if not masked_scores:
            selected = np.random.choice(possible_moves)
            return jsonify({
                "move": selected,
                "source": "fallback-random-no-mapped-action"
            })

        masked_scores.sort(reverse=True, key=lambda x: x[0])

        top_k = min(30, len(masked_scores))
        candidates = masked_scores[:top_k]

        temperature = 1.5

        scores = np.array([max(s, 1e-8) for s, _ in candidates], dtype=np.float64)
        scores = np.power(scores, 1.0 / temperature)
        scores = scores / scores.sum()

        idx = np.random.choice(len(candidates), p=scores)
        best_score, selected = candidates[idx]

        return jsonify({
            "move": selected,
            "source": "alphazero-policy-topk",
            "value": float(v),
            "policy_score": float(best_score),
            "nb_legal_moves": len(possible_moves),
            "nb_mapped_moves": len(masked_scores),
            "top_k": top_k,
            "temperature": temperature
        })

    except Exception as e:
        selected = np.random.choice(possible_moves)
        return jsonify({
            "move": selected,
            "source": "fallback-random-exception",
            "error": str(e)
        })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)