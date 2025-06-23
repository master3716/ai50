import os
import numpy as np
import chess
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from typing import List, Tuple
import matplotlib.pyplot as plt
import sys

EPOCHS = 100
TEST_SIZE = 0.2
DATA_SIZE = 1_000_000
MATE_VAL = 30 #not centipawns


def fen_to_board(fen: str) -> List[int]:
    piece_map = {
        'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
        'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6,
        '.': 0
    }

    board = []
    rows = fen.split(' ')[0].split('/')
    for row in rows:
        for char in row:
            if char.isdigit():
                board.extend([0] * int(char))
            else:
                board.append(piece_map[char])
    return board



def main():
    if(len(sys.argv) < 2):
        print("usage: python3 test2.py FEN STRING")
        return
    
    model = tf.keras.models.load_model("chess_model_improved.h5")
    fen = sys.argv[1]
    board = fen_to_board(fen=fen)
    board = np.array(board, dtype=np.float32).reshape(1, 64) / 6 
    prediction = model.predict(board)[0][0] * 10
    print(prediction)

main()