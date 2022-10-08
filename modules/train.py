from modules.environment import Environment
from modules.agent import QAgent
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import *


def run_episode(env: Environment, agent: QAgent, max_step: int, eval: bool, show: bool = False) -> int:
    """Run an episode.

    Args:
        env (Environment): environment for the episode.
        agent (QAgent): agent to be used during the episode.
        max_step (int): maximum number of steps for the episode.
        eval (bool): flag for update. Update for the agent will not be applied if `eval` is ste to False.
        show (bool, optional): flag for showing all moves during the episode. Defaults to False.

    Returns:
        int: reward the agent obtain at the end of the episode.
    """
    state = env.reset()
    actions_name = {0: "halt", 1: "left", 2: "right", 3: "up", 4: "down"}
    n_step = 1
    if show:
        print(
            "Symbols meaning [P: Player/Agent, E: enemy, F: food/goal, ?: E+F, @: P+F, X: P+E, #: P+E+F]")
        print("New Environment :")
        print(env)
    while True:
        action = agent.act(state, eval)
        reward, next_state, done = env.step(action)
        if show:
            print(
                f"action : {actions_name[action]} is taken at step: {n_step}\n")
            print(env)
        if not eval:
            agent.update(state, action, reward, next_state, done)
        state = next_state
        n_step += 1
        if done or n_step > max_step:
            break

    return reward


def train(environment: Environment, agent: QAgent, num_episode: int, max_step: int = 1000, eval_every: int = 10, n_eval: int = 5) -> tuple[list, list]:
    """Training function for an agent.

    Args:
        environment (Environment): _description_
        agent (QAgent): the agent that will be trained.
        num_episode (int): number of episodes during training.
        max_step (int, optional): maximum number of steps for each episode. Defaults to 1000.
        eval_every (int, optional): periode for evaluations. Defaults to 10.
        n_eval (int, optional): number of episodes to run for each evaluation. Defaults to 5.

    Returns:
        tuple[list, list]: lists of episode number and average rewards of each evaluation.
    """
    rewards = []
    episodes = []
    try:
        for ep in tqdm(range(1, num_episode+1)):
            run_episode(environment, agent, max_step, eval=False, show=False)
            if ep % eval_every == 0:
                episodes.append(ep)
                mean_reward = np.mean([run_episode(
                    environment, agent, max_step, eval=True, show=False) for _ in range(n_eval)])
                rewards.append(mean_reward)
    except KeyboardInterrupt:
        print(f"Training interrupted at {ep} episodes.")
        input("Press enter to continue ...")

    return episodes, rewards


def plot_rewards(episodes: list, rewards: list) -> None:
    """Visualization of average rewards during training.

    Args:
        episodes (list): list of episode number of each evaluation.
        rewards (list): list of average rewards of each evaluation.
    """
    plt.figure(figsize=(10, 5))
    plt.title("Average reward during training.")
    plt.plot(episodes, rewards, "g:", label="average reward")
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.legend()
    plt.show()
