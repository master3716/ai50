import os
import numpy as np
import chess
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from typing import List, Tuple
import matplotlib.pyplot as plt

EPOCHS = 100
TEST_SIZE = 0.2
DATA_SIZE = 1_000_000
MATE_VAL = 30 #not centipawns

def main():
    print("Loading data...")
    x, y = load_data("chessData.csv")

   
    x = np.array(x, dtype=np.float32) / 6.0  
    y = np.array(y, dtype=np.float32)

    x, y = shuffle(x, y, random_state=42)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=42, stratify=None
    )

    print("Building model...")
    model = get_model()

  
    callbacks = [
        EarlyStopping(
            monitor='val_mae',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_mae',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print("Training model...")
    history = model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=128,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=1
    )

    print("Evaluating on test set...")
    loss, mae = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test MAE: {mae:.4f}")

    filename = "chess_model_improved.h5"
    model.save(filename)
    print(f"Model saved to {filename}.")


def load_data(path: str) -> Tuple[List[List[int]], List[float]]:

    position = []
    eval = []

    with open(path, 'r') as f:
        next(f)
        for i, line in enumerate(f):
            if i % 10000 == 0:
                print(f"{(i/DATA_SIZE) * 100}%")
            if i > DATA_SIZE:
                break

            line = line.split(",")
            fen = line[0]
            val = line[1]

            board = fen_to_board(fen)
            val = val_to_eval(val)
            val = max(-10, min(val, 10)) 
            val = val / 10 

            position.append(board)
            eval.append(val)
    
    return (position, eval)


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


def val_to_eval(val: str) -> float:
    if "#" not in val:
        return float(val) / 100
    else:
        val = val.replace("#", "")
        val = float(val)
        sign = -1 if abs(val) > val else 1
        val = (MATE_VAL - val) * sign
        return val

def get_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(64,)),
        tf.keras.layers.Dense(64, activation="elu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation="elu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation="elu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512, activation="elu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation="elu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='tanh')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="huber",
        metrics=["mae"]
    )
    return model


main()