import random
import matplotlib.pyplot as plt
from collections import deque
def check_winner(board):
    for a, b, c in [(0,1,2), (3,4,5), (6,7,8),(0,3,6), (1,4,7), (2,5,8),(0,4,8), (2,4,6)]:
        if board[a] == board[b] == board[c] != 0:
            return board[a]
    if 0 not in board:
        return 0.5
    return None
def print_board(board):
    symbols = {1: 'X', -1: 'O', 0: '_'}
    for i in range(0, 9, 3):
        print(symbols[board[i]], symbols[board[i+1]], symbols[board[i+2]])
class QAgent:
    def __init__(self, alpha=0.9, gamma=0.6, epsilon=0.97):
        self.q = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    def get_q(self, state, action):
        return self.q.get((state, action), 0.0)
    def choose_action(self, board):
        state = tuple(board)
        moves = [i for i in range(9) if board[i] == 0]

        if random.random() < self.epsilon:
            action = random.choice(moves)
        else:
            qs = [(self.get_q(state, a), a) for a in moves]
            action = max(qs)[1]

        self.epsilon *= 0.999996
        return action
    def update(self, board, action, reward, next_board):
        state = tuple(board)
        next_state = tuple(next_board)
        next_moves = [i for i in range(9) if next_board[i] == 0]
        future = max(self.get_q(next_state, a) for a in next_moves) if next_moves else 0
        old_q = self.get_q(state, action)
        self.q[(state, action)] = old_q + self.alpha * (
            reward + self.gamma * future - old_q
        )
def train(episodes=100000, ma_window=1000):
    agent = QAgent()
    rewards = []
    ma_rewards = []
    window = deque(maxlen=ma_window)
    for _ in range(episodes):
        board = [0] * 9
        episode_reward = 0
        while True:
            action = agent.choose_action(board)
            new_board = board[:]
            new_board[action] = 1
            result = check_winner(new_board)
            if result is not None:
                reward = 1 if result == 1 else 0.5
                agent.update(board, action, reward, new_board)
                episode_reward += reward
                break
            opp_moves = [i for i in range(9) if new_board[i] == 0]
            opp_action = random.choice(opp_moves)
            new_board[opp_action] = -1
            result = check_winner(new_board)
            if result is not None:
                reward = -1 if result == -1 else 0.5
                agent.update(board, action, reward, new_board)
                episode_reward += reward
                break
            agent.update(board, action, -0.01, new_board)
            episode_reward -= 0.01
            board = new_board
        rewards.append(episode_reward)
        window.append(episode_reward)
        ma_rewards.append(sum(window) / len(window))
    plt.figure(figsize=(10,5))
    plt.plot(ma_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Moving Avg Reward")
    plt.title(f"Tic-Tac-Toe Q-Learning (Moving Avg, window={ma_window})")
    plt.grid(True)
    plt.show()
    return agent
def play(agent):
    board = [0] * 9
    print("You are O (positions 0-8)")
    print_board(board)
    while True:
        a = agent.choose_action(board)
        board[a] = 1
        print("Agent move:")
        print_board(board)
        r = check_winner(board)
        if r is not None:
            print("Agent wins!" if r == 1 else "Draw!")
            break
        while True:
            try:
                m = int(input("Your move: "))
                if m in range(9) and board[m] == 0:
                    break
            except:
                pass
            print("Invalid move! Choose an empty position (0-8).")

        board[m] = -1
        print("Your move:")
        print_board(board)
        r = check_winner(board)
        if r is not None:
            print("You win!" if r == -1 else "Draw!")
            break
agent = train()
play(agent)
