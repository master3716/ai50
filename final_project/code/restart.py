import os
import numpy as np
import chess
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from typing import List, Tuple
import matplotlib.pyplot as plt


# we are aiming for a val_mae of 0.08 or less to be considered good (0.8 pawns off stockfish)



EPOCHS = 150
TEST_SIZE = 0.2
DATA_SIZE = 1_000_000
MATE_VAL = 30 #not centipawns

def main():
    print("Loading data...")
    x, y = load_data("chessData.csv")

   
    x = np.array(x, dtype=np.float32)
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
            patience=3,
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

    filename = "chess_model_improved_v_6.h5"
    model.save(filename)
    print(f"Model saved to {filename}.")

    plt.plot(history.history["mae"])
    plt.plot(history.history["val_mae"])
    plt.legend(["Train MAE", "Val MAE"])
    plt.show()


def load_data(path: str) -> Tuple[List[List[float]], List[float]]:
    position = []
    evals = []

    with open(path, 'r') as f:
        next(f)
        for i, line in enumerate(f):
            if i % (DATA_SIZE // 100) == 0:
                print(f"{(i / DATA_SIZE) * 100:.1f}%")
            if i > DATA_SIZE:
                break

            fen, val = line.split(",")[:2]

            board = fen_to_board(fen)
            board = np.array(board, dtype=np.float32)

         
            board[:64] /= 6.0

           
            eval_value = val_to_eval(val)
            eval_value = max(-10, min(eval_value, 10)) / 10.0

            position.append(board.tolist())
            evals.append(eval_value)

    return position, evals



# def fen_to_board(fen: str) -> List[float]:
#     piece_map = {
#         'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
#         'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6
#     }

#     board = []
#     parts = fen.split()
#     rows = parts[0].split('/')

#     for row in rows:
#         for char in row:
#             if char.isdigit():
#                 board.extend([0] * int(char))
#             else:
#                 board.append(piece_map[char])

#     # Now add extra game info
#     turn = 1 if parts[1] == 'w' else -1
#     board.append(turn)

#     rights = parts[2]
#     board.append(1 if 'K' in rights else 0)
#     board.append(1 if 'Q' in rights else 0)
#     board.append(1 if 'k' in rights else 0)
#     board.append(1 if 'q' in rights else 0)

#     ep_square = parts[3]
#     if ep_square == '-':
#         board.append(-1)
#     else:
#         file = ord(ep_square[0]) - ord('a')
#         board.append((file / 3.5) - 1)

#     # NEW: halfmove clock and fullmove number
#     halfmove = min(int(parts[4]), 100) / 100.0
#     fullmove = min(int(parts[5]), 200) / 200.0
#     board.append(halfmove)
#     board.append(fullmove)

#     return board
def fen_to_board(fen: str) -> List[float]:
    """Convert FEN to feature vector with proper normalization"""
    
    # Piece values for normalization
    piece_values = {
        'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 4,
        'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9, 'k': -4
    }

    features = []
    parts = fen.split()
    
    # Parse board position (64 squares)
    board_str = parts[0]
    board = []
    
    for row in board_str.split('/'):
        for char in row:
            if char.isdigit():
                board.extend([0] * int(char))
            else:
                board.append(piece_values[char])
    
    # Normalize piece values to [-1, 1]
    board = [x / 9.0 for x in board]  # Queen is strongest piece (9)
    features.extend(board)
    
    # Side to move
    features.append(1.0 if parts[1] == 'w' else -1.0)
    
    # Castling rights (4 features)
    castling = parts[2]
    features.append(1.0 if 'K' in castling else 0.0)
    features.append(1.0 if 'Q' in castling else 0.0)
    features.append(1.0 if 'k' in castling else 0.0)
    features.append(1.0 if 'q' in castling else 0.0)
    
    # En passant square
    ep_square = parts[3]
    if ep_square == '-':
        features.extend([0.0, 0.0])  # No en passant
    else:
        # Convert to file and rank (normalized)
        file = (ord(ep_square[0]) - ord('a')) / 7.0  # 0-1
        rank = (int(ep_square[1]) - 1) / 7.0  # 0-1
        features.extend([file, rank])
    
    # Halfmove clock (normalized)
    halfmove = min(int(parts[4]), 100) / 100.0
    features.append(halfmove)
    
    # Fullmove number (normalized)
    fullmove = min(int(parts[5]), 300) / 300.0
    features.append(fullmove)
    
    return features

def val_to_eval(val: str) -> float:
    if "#" not in val:
        return float(val) / 100
    else:
        val = val.replace("#", "")
        val = float(val)
        sign = -1 if abs(val) > val else 1
        val = (MATE_VAL - val) * sign
        return val

# def get_model():
#     model = tf.keras.models.Sequential([
#         tf.keras.layers.Input(shape=(72,)),

#         tf.keras.layers.Dense(512, activation="gelu"),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dropout(0.2),

#         tf.keras.layers.Dense(256, activation="gelu"),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dropout(0.2),

#         tf.keras.layers.Dense(128, activation="gelu"),
#         tf.keras.layers.BatchNormalization(),

#         tf.keras.layers.Dense(64, activation="gelu"),
#         tf.keras.layers.BatchNormalization(),

#         tf.keras.layers.Dense(1, activation="tanh")
#     ])
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
#         loss="huber",
#         metrics=["mae"]
#     )
#     return model
def get_model():
    """Enhanced model architecture"""
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(73,)),
        
        # First block - larger
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        # Second block
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        # Third block
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        # Fourth block
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        # Fifth block
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        
        # Output layer - no activation for regression
        tf.keras.layers.Dense(1)
    ])
    
    # Use a more aggressive learning rate schedule
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        ),
        loss="huber",
        metrics=["mae"]
    )
    
    return model
main()