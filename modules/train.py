from modules.environment import Environment
from modules.agent import QAgent
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def run_episode(env: Environment, agent: QAgent, max_step: int, eval: bool, show=False):
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


def train(environment: Environment, agent: QAgent, num_episode: int, max_step: int = 1000):
    rewards = []
    episodes = []
    eval_every = 10
    n_eval = 5
    try:
        for ep in tqdm(range(1, num_episode+1)):
            run_episode(environment, agent, max_step, eval=False, show=False)
            if ep % 10 == 0:
                episodes.append(ep)
                mean_reward = np.mean([run_episode(
                    environment, agent, max_step, eval=True, show=False) for _ in range(n_eval)])
                rewards.append(mean_reward)
    except KeyboardInterrupt:
        print(f"Training interrupted at {ep} episodes.")
        input("Press enter to continue ...")

    return episodes, rewards


def plot_rewards(episodes, rewards):
    plt.figure(figsize=(10, 5))
    plt.title("Average reward during training.")
    plt.plot(episodes, rewards, "g:", label="average reward")
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.legend()
    plt.show()
