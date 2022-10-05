from modules.utils import save_to_json, read_from_json
import numpy as np


class QAgent:
    def __init__(self, num_actions, gamma, learning_rate, epsilon, qfunction=None) -> None:
        self.n_actions = num_actions
        self.qfunction = {} if qfunction is None else qfunction if isinstance(
            qfunction, dict) else read_from_json(qfunction)
        self.gamma = gamma
        self.alpha = learning_rate
        self.epsilon = epsilon

    def act(self, state, eval=False):
        if state not in self.qfunction:
            self.qfunction[state] = np.zeros(self.n_actions)
        u = np.random.uniform()
        if eval or self.epsilon < u:
            action = np.argmax(self.qfunction[state])
        else:
            action = np.random.randint(self.n_actions)
        return action

    def update(self, state, action, reward, next_state, done):
        if state not in self.qfunction:
            self.qfunction[state] = np.zeros(self.n_actions)

        if done:
            self.qfunction[state][action] = (
                1 - self.alpha)*self.qfunction[state][action] + self.alpha*reward
        else:
            if next_state not in self.qfunction:
                self.qfunction[next_state] = np.zeros(self.n_actions)
            self.qfunction[state][action] = (1 - self.alpha)*self.qfunction[state][action] + \
                self.alpha*(reward + self.gamma *
                            self.qfunction[next_state].max())

    def save_qfunction(self, path="./qfunction.json"):
        save_to_json(self.qfunction, path)

    def load_qfunction(self, path="./qfunction.json"):
        self.qfunction = read_from_json(path)
