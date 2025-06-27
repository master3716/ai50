import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import chess
import tensorflow as tf
import numpy as np
from typing import List, Optional, Tuple
import threading
import time


# Unicode chess pieces
PIECE_SYMBOLS = {
    'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
    'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
}

class ChessAI:
    def __init__(self, model_path: str):
        """Initialize the chess AI with a trained model"""
        self.model = tf.keras.models.load_model(model_path)
        
    def fen_to_features(self, fen: str) -> List[float]:
        """Convert FEN to feature vector (same as training)"""
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
        board = [x / 9.0 for x in board]
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
            features.extend([0.0, 0.0])
        else:
            file = (ord(ep_square[0]) - ord('a')) / 7.0
            rank = (int(ep_square[1]) - 1) / 7.0
            features.extend([file, rank])
        
        # Halfmove clock (normalized)
        halfmove = min(int(parts[4]), 100) / 100.0
        features.append(halfmove)
        
        # Fullmove number (normalized)
        fullmove = min(int(parts[5]), 300) / 300.0
        features.append(fullmove)
        
        return features
    
    def evaluate_position(self, board: chess.Board) -> float:
        """Evaluate a chess position using the neural network"""
        features = self.fen_to_features(board.fen())
        features_array = np.array([features], dtype=np.float32)
        evaluation = self.model.predict(features_array, verbose=0)[0][0]
        return float(evaluation)
    
    def get_best_move(self, board: chess.Board, depth: int = 3) -> Optional[chess.Move]:
        """Find the best move using optimized search"""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        # For depth 1, just evaluate all moves directly
        if depth == 1:
            return self.get_best_move_simple(board)
        
        # For deeper search, use iterative deepening with move ordering
        return self.get_best_move_iterative(board, depth)
    
    def get_best_move_simple(self, board: chess.Board) -> Optional[chess.Move]:
        """Simple move evaluation for depth 1"""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
            
        move_scores = []
        
        for move in legal_moves:
            board.push(move)
            score = self.evaluate_position(board)
            board.pop()
            move_scores.append((move, score))
        
        # Sort by score (best for current player)
        if board.turn:  # White maximizing
            move_scores.sort(key=lambda x: x[1], reverse=True)
        else:  # Black minimizing
            move_scores.sort(key=lambda x: x[1])
            
        return move_scores[0][0]
    
    def get_best_move_iterative(self, board: chess.Board, max_depth: int) -> Optional[chess.Move]:
        """Iterative deepening search with move ordering"""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        best_move = legal_moves[0]
        
        # Start with simple evaluation to order moves
        move_scores = []
        for move in legal_moves[:min(20, len(legal_moves))]:  # Limit to top 20 moves
            board.push(move)
            score = self.evaluate_position(board)
            board.pop()
            move_scores.append((move, score))
        
        # Sort moves by evaluation (move ordering for better pruning)
        if board.turn:  # White
            move_scores.sort(key=lambda x: x[1], reverse=True)
        else:  # Black
            move_scores.sort(key=lambda x: x[1])
        
        ordered_moves = [move for move, _ in move_scores]
        
        # Iterative deepening
        for current_depth in range(2, max_depth + 1):
            try:
                best_move = self.minimax_root(board, current_depth, ordered_moves)
            except:
                break  # If we run into issues, return the best move so far
                
        return best_move
    
    def minimax_root(self, board: chess.Board, depth: int, ordered_moves: List[chess.Move]) -> chess.Move:
        """Root minimax with ordered moves"""
        best_move = ordered_moves[0]
        best_eval = float('-inf') if board.turn else float('inf')
        
        for move in ordered_moves[:min(15, len(ordered_moves))]:  # Limit search width
            board.push(move)
            eval_score = self.minimax(board, depth - 1, float('-inf'), float('inf'), not board.turn)
            board.pop()
            
            if board.turn:  # White (maximizing)
                if eval_score > best_eval:
                    best_eval = eval_score
                    best_move = move
            else:  # Black (minimizing)
                if eval_score < best_eval:
                    best_eval = eval_score
                    best_move = move
                    
        return best_move
    
    def minimax(self, board: chess.Board, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
        """Optimized minimax with early termination"""
        if depth == 0 or board.is_game_over():
            return self.evaluate_position(board)
        
        legal_moves = list(board.legal_moves)
        
        # Limit the number of moves considered at deeper levels
        if depth > 1:
            legal_moves = legal_moves[:min(25, len(legal_moves))]
        
        if maximizing:
            max_eval = float('-inf')
            for move in legal_moves:
                board.push(move)
                eval_score = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
                    
            return max_eval
        else:
            min_eval = float('inf')
            for move in legal_moves:
                board.push(move)
                eval_score = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
                    
            return min_eval


class ChessGameGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chess vs Neural Network")
        self.root.geometry("800x650")
        self.root.resizable(False, False)
        
        # Game state
        self.board = chess.Board()
        self.ai = None
        self.human_is_white = True
        self.ai_depth = 3
        self.selected_square = None
        self.legal_moves = []
        self.thinking = False
        
        # Colors
        self.light_color = "#F0D9B5"
        self.dark_color = "#B58863"
        self.highlight_color = "#FFFF00"
        self.move_color = "#90EE90"
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - board
        board_frame = ttk.Frame(main_frame, relief=tk.RAISED, borderwidth=2)
        board_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        # Chess board
        self.canvas = tk.Canvas(board_frame, width=480, height=480, bg="white")
        self.canvas.pack(padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_square_click)
        
        # Right panel - controls and info
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Model loading
        model_frame = ttk.LabelFrame(control_frame, text="Model", padding=10)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(model_frame, text="Load Model", command=self.load_model).pack(fill=tk.X)
        self.model_label = ttk.Label(model_frame, text="No model loaded", foreground="red")
        self.model_label.pack(pady=(5, 0))
        
        # Game settings
        settings_frame = ttk.LabelFrame(control_frame, text="Game Settings", padding=10)
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Side selection
        ttk.Label(settings_frame, text="Play as:").pack(anchor=tk.W)
        self.side_var = tk.StringVar(value="White")
        side_frame = ttk.Frame(settings_frame)
        side_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Radiobutton(side_frame, text="White", variable=self.side_var, value="White").pack(side=tk.LEFT)
        ttk.Radiobutton(side_frame, text="Black", variable=self.side_var, value="Black").pack(side=tk.LEFT)
        
        # Difficulty
        ttk.Label(settings_frame, text="AI Difficulty:").pack(anchor=tk.W)
        self.difficulty_var = tk.IntVar(value=3)
        difficulty_scale = ttk.Scale(settings_frame, from_=1, to=5, variable=self.difficulty_var, orient=tk.HORIZONTAL)
        difficulty_scale.pack(fill=tk.X, pady=(0, 5))
        self.difficulty_label = ttk.Label(settings_frame, text="Depth: 3")
        self.difficulty_label.pack(anchor=tk.W)
        difficulty_scale.configure(command=self.update_difficulty_label)
        
        # Game controls
        controls_frame = ttk.LabelFrame(control_frame, text="Game Controls", padding=10)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(controls_frame, text="New Game", command=self.new_game).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(controls_frame, text="Undo Move", command=self.undo_move).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(controls_frame, text="Flip Board", command=self.flip_board).pack(fill=tk.X)
        
        # Game info
        info_frame = ttk.LabelFrame(control_frame, text="Game Info", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.turn_label = ttk.Label(info_frame, text="Turn: White", font=("Arial", 10, "bold"))
        self.turn_label.pack(anchor=tk.W)
        
        self.eval_label = ttk.Label(info_frame, text="Evaluation: 0.00")
        self.eval_label.pack(anchor=tk.W)
        
        self.status_label = ttk.Label(info_frame, text="Load a model to start")
        self.status_label.pack(anchor=tk.W)
        
        # Move history
        history_frame = ttk.LabelFrame(control_frame, text="Move History", padding=10)
        history_frame.pack(fill=tk.BOTH, expand=True)
        
        # Scrollable text widget for moves
        history_scroll_frame = ttk.Frame(history_frame)
        history_scroll_frame.pack(fill=tk.BOTH, expand=True)
        
        self.history_text = tk.Text(history_scroll_frame, height=8, width=25, state=tk.DISABLED)
        scrollbar = ttk.Scrollbar(history_scroll_frame, orient=tk.VERTICAL, command=self.history_text.yview)
        self.history_text.configure(yscrollcommand=scrollbar.set)
        
        self.history_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Initial board draw
        self.flipped = False
        self.draw_board()
        
    def load_model(self):
        """Load the neural network model"""
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("H5 files", "*.h5"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.ai = ChessAI(file_path)
                self.model_label.config(text=f"Model loaded: {file_path.split('/')[-1]}", foreground="green")
                self.status_label.config(text="Ready to play!")
                self.update_evaluation()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
                self.model_label.config(text="Failed to load model", foreground="red")
    
    def update_difficulty_label(self, value):
        """Update the difficulty label"""
        depth = int(float(value))
        self.difficulty_label.config(text=f"Depth: {depth}")
        self.ai_depth = depth
    
    def new_game(self):
        """Start a new game"""
        if not self.ai:
            messagebox.showwarning("Warning", "Please load a model first!")
            return
            
        self.board = chess.Board()
        self.human_is_white = self.side_var.get() == "White"
        self.selected_square = None
        self.legal_moves = []
        self.thinking = False
        
        # Clear move history
        self.history_text.config(state=tk.NORMAL)
        self.history_text.delete(1.0, tk.END)
        self.history_text.config(state=tk.DISABLED)
        
        self.draw_board()
        self.update_status()
        self.update_evaluation()
        
        # If human is black, AI makes first move
        if not self.human_is_white:
            self.root.after(500, self.make_ai_move)
    
    def flip_board(self):
        """Flip the board view"""
        self.flipped = not self.flipped
        self.draw_board()
    
    def undo_move(self):
        """Undo the last move (or two moves to undo both human and AI)"""
        if len(self.board.move_stack) >= 2 and not self.thinking:
            self.board.pop()  # Undo AI move
            self.board.pop()  # Undo human move
            self.draw_board()
            self.update_status()
            self.update_evaluation()
            self.update_move_history()
    
    def square_to_coords(self, square: int) -> Tuple[int, int]:
        """Convert chess square to canvas coordinates"""
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        
        if self.flipped:
            x = (7 - file) * 60 + 30
            y = rank * 60 + 30
        else:
            x = file * 60 + 30
            y = (7 - rank) * 60 + 30
            
        return x, y
    
    def coords_to_square(self, x: int, y: int) -> Optional[int]:
        """Convert canvas coordinates to chess square"""
        file = x // 60
        rank = 7 - (y // 60)
        
        if self.flipped:
            file = 7 - file
            rank = 7 - rank
            
        if 0 <= file <= 7 and 0 <= rank <= 7:
            return chess.square(file, rank)
        return None
    
    def draw_board(self):
        """Draw the chess board and pieces"""
        self.canvas.delete("all")
        
        # Draw squares
        for rank in range(8):
            for file in range(8):
                x1, y1 = file * 60, rank * 60
                x2, y2 = x1 + 60, y1 + 60
                
                color = self.light_color if (rank + file) % 2 == 0 else self.dark_color
                
                # Highlight selected square
                square = chess.square(file, 7 - rank) if not self.flipped else chess.square(7 - file, rank)
                if square == self.selected_square:
                    color = self.highlight_color
                elif square in [move.to_square for move in self.legal_moves]:
                    color = self.move_color
                
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")
                
                # Add file and rank labels
                if rank == 7:  # Bottom row
                    file_label = chr(ord('a') + file) if not self.flipped else chr(ord('a') + (7 - file))
                    self.canvas.create_text(x1 + 50, y1 + 50, text=file_label, font=("Arial", 8), fill="black")
                if file == 0:  # Left column
                    rank_label = str(8 - rank) if not self.flipped else str(rank + 1)
                    self.canvas.create_text(x1 + 10, y1 + 10, text=rank_label, font=("Arial", 8), fill="black")
        
        # Draw pieces
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                x, y = self.square_to_coords(square)
                symbol = PIECE_SYMBOLS[piece.symbol()]
                self.canvas.create_text(x, y, text=symbol, font=("Arial", 32), fill="black")
    
    def on_square_click(self, event):
        """Handle square clicks"""
        if self.thinking or not self.ai:
            return
            
        square = self.coords_to_square(event.x, event.y)
        if square is None:
            return
        
        # If it's not the human's turn, ignore clicks
        if self.board.turn != self.human_is_white:
            return
        
        if self.selected_square is None:
            # Select a piece
            piece = self.board.piece_at(square)
            if piece and piece.color == self.human_is_white:
                self.selected_square = square
                self.legal_moves = [move for move in self.board.legal_moves if move.from_square == square]
                self.draw_board()
        else:
            # Try to make a move
            move = None
            for legal_move in self.board.legal_moves:
                if legal_move.from_square == self.selected_square and legal_move.to_square == square:
                    # Handle pawn promotion
                    if legal_move.promotion:
                        move = legal_move  # Default to queen promotion
                        break
                    else:
                        move = legal_move
                        break
            
            if move:
                self.make_human_move(move)
            
            # Deselect
            self.selected_square = None
            self.legal_moves = []
            self.draw_board()
    
    def make_human_move(self, move: chess.Move):
        """Make a human move"""
        self.board.push(move)
        self.draw_board()
        self.update_status()
        self.update_evaluation()
        self.update_move_history()
        
        if not self.board.is_game_over():
            # AI's turn
            self.root.after(500, self.make_ai_move)
    
    def make_ai_move(self):
        """Make an AI move in a separate thread"""
        if self.board.is_game_over() or self.thinking:
            return
            
        self.thinking = True
        self.status_label.config(text="AI is thinking...")
        
        # Show progress bar
        progress_window = tk.Toplevel(self.root)
        progress_window.title("AI Thinking")
        progress_window.geometry("300x100")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        ttk.Label(progress_window, text="AI is analyzing position...").pack(pady=10)
        progress_bar = ttk.Progressbar(progress_window, mode='indeterminate')
        progress_bar.pack(pady=10, padx=20, fill=tk.X)
        progress_bar.start()
        
        def ai_move_thread():
            try:
                start_time = time.time()
                move = self.ai.get_best_move(self.board, self.ai_depth)
                elapsed_time = time.time() - start_time
                
                self.root.after(0, lambda: self.complete_ai_move(move, progress_window, elapsed_time))
            except Exception as e:
                self.root.after(0, lambda: self.handle_ai_error(str(e), progress_window))
        
        threading.Thread(target=ai_move_thread, daemon=True).start()
        
        # Add timeout (30 seconds)
        def timeout_check():
            if self.thinking and progress_window.winfo_exists():
                self.root.after(30000, lambda: self.handle_ai_timeout(progress_window))
        
        self.root.after(100, timeout_check)
    
    def complete_ai_move(self, move: Optional[chess.Move], progress_window, elapsed_time: float):
        """Complete the AI move on the main thread"""
        self.thinking = False
        
        # Close progress window
        if progress_window.winfo_exists():
            progress_window.destroy()
        
        if move:
            self.board.push(move)
            self.draw_board()
            self.update_status()
            self.update_evaluation()
            self.update_move_history()
            
            # Show thinking time
            self.status_label.config(text=f"AI played {move} ({elapsed_time:.1f}s)")
        else:
            self.status_label.config(text="AI has no legal moves")
        
        # Check for game over
        if self.board.is_game_over():
            result = self.board.result()
            if result == "1-0":
                messagebox.showinfo("Game Over", "White wins!")
            elif result == "0-1":
                messagebox.showinfo("Game Over", "Black wins!")
            else:
                messagebox.showinfo("Game Over", "It's a draw!")
    
    def handle_ai_error(self, error_msg: str, progress_window):
        """Handle AI errors"""
        self.thinking = False
        if progress_window.winfo_exists():
            progress_window.destroy()
        messagebox.showerror("AI Error", f"AI move failed: {error_msg}")
        self.status_label.config(text="AI error - try reducing depth")
    
    def handle_ai_timeout(self, progress_window):
        """Handle AI timeout"""
        if self.thinking:
            self.thinking = False
            if progress_window.winfo_exists():
                progress_window.destroy()
            messagebox.showwarning("Timeout", "AI took too long. Try reducing the depth.")
            self.status_label.config(text="AI timeout - reduce depth")
    
    def update_status(self):
        """Update the game status"""
        if self.board.turn:
            self.turn_label.config(text="Turn: White")
        else:
            self.turn_label.config(text="Turn: Black")
        
        if not self.thinking:
            if self.board.turn == self.human_is_white:
                self.status_label.config(text="Your turn")
            else:
                self.status_label.config(text="AI's turn")
    
    def update_evaluation(self):
        """Update the position evaluation"""
        if self.ai:
            try:
                eval_score = self.ai.evaluate_position(self.board)
                self.eval_label.config(text=f"Evaluation: {eval_score:.2f}")
            except:
                self.eval_label.config(text="Evaluation: N/A")
    
    def update_move_history(self):
        """Update the move history display"""
        self.history_text.config(state=tk.NORMAL)
        self.history_text.delete(1.0, tk.END)
        
        moves = self.board.move_stack
        for i in range(0, len(moves), 2):
            move_num = (i // 2) + 1
            white_move = moves[i]
            black_move = moves[i + 1] if i + 1 < len(moves) else None
            
            line = f"{move_num}. {white_move}"
            if black_move:
                line += f" {black_move}"
            line += "\n"
            
            self.history_text.insert(tk.END, line)
        
        self.history_text.config(state=tk.DISABLED)
        self.history_text.see(tk.END)


def main():
    root = tk.Tk()
    app = ChessGameGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()