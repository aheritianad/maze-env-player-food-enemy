from modules.utils import save_to_json, read_from_json
import numpy as np
from typing import *


class QAgent:
    def __init__(self, num_actions: int, gamma: float, learning_rate: float, epsilon: float, qfunction: str = None) -> None:
        """Free tabular Q-agent.

        Args:
            num_actions (int): number of action that an agent can take.
            gamma (float): discount factor for training (size of horizon).
            learning_rate (float): learning rate for training (exploitation).
            epsilon (float): probability for the agent will act uniformely during training (exploration).
            qfunction (str, optional): path for a json file for the `qfunction`. If None is given, `qfunction`
                                        will be iniatilized by an empty dictionary. Defaults to None.
        """
        self.n_actions = num_actions
        self.qfunction = {} if qfunction is None else qfunction if isinstance(
            qfunction, dict) else read_from_json(qfunction)
        self.gamma = gamma
        self.alpha = learning_rate
        self.epsilon = epsilon

    def act(self, state: str, eval: bool = False) -> int:
        """Act function will allow agent to chose an `action` depending on its current `Q-function` at a given `state`.

        Args:
            state (str): current state.
            eval (bool, optional): flag saying agent will work greedily `(eval = True)` or `epsilon`-greedy `(eval = False)`.
                                    Defaults to False.

        Returns:
            int: the chosen `action` by the agent.
        """
        if state not in self.qfunction:
            self.qfunction[state] = np.zeros(self.n_actions)
        u = np.random.uniform()
        if eval or self.epsilon < u:
            action = np.argmax(self.qfunction[state])
        else:
            action = np.random.randint(self.n_actions)
        return action

    def update(self, state: str, action: int, reward: int, next_state: str, done: bool) -> None:
        """Update of the `Q-function` for an agent at a given `state` and applying a given `action`.

        Args:
            state (str): current state.
            action (int): applied `action`.
            reward (int): `reward` obtained by applying the `action` at the given `state`.
            next_state (str): next state after applying the `action` at the given `state`.
            done (bool): flag saying that the `next state` is a terminal state or not.
        """
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

    def save_qfunction(self, path: str = "./qfunction.json") -> None:
        """Method for saving the `Q-function` in a json file.

        Args:
            path (str, optional): path where the generated json file will be located. Defaults to "./qfunction.json".
        """
        save_to_json(self.qfunction, path)

    def load_qfunction(self, path: str = "./qfunction.json") -> None:
        """_summary_

        Args:
            path (str, optional): path where the json file of the `Q-function` is located. Defaults to "./qfunction.json".
        """
        self.qfunction = read_from_json(path)
