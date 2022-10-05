from environment import Environment
from agent import QAgent
from train import train, plot_rewards, run_episode

from argparse import ArgumentParser

parser = ArgumentParser(
    description="Generate a qvalue function in a json file.")

parser.add_argument("-n", "--env-size", type=int,
                    help="size of the environment")
parser.add_argument("-g", "--gamma", type=float,
                    help="discount factor (horizon-size)")
parser.add_argument("-lr", "--learning-rate", type=float,
                    help="learning rate/alpha (exploitation)")
parser.add_argument("-eps", "--epsilon", type=float,
                    help="probability to act non-greedy (exploration)")
parser.add_argument("-epi", "--n-episode", type=int,
                    help="number of episodes for training")
parser.add_argument("-max", "--max-step", type=int,
                    help="maximum number of steps for each episode")
parser.add_argument("-l", "--path-to-load-qfunction", type=str,
                    help="path for the json file of qvalue function to load")
parser.add_argument("-s", "--path-to-save-qfunction", type=str,
                    help="path for the json file of qvalue function to save")
parser.add_argument("-show", "--show", type=bool,
                    help="flag either user want to show the reward during training")


args = vars(parser.parse_args())

env_size = args["env_size"]
num_actions = 5
gamma = args["gamma"]
learning_rate = args["learning_rate"]
epsilon = args["epsilon"]
n_episode = args["n_episode"]
max_step = args["max_step"]
path_load_qfunction = args.get("path_to_load_qfunction", None)
path_save_qfunction = args.get(
    "path_to_save_qfunction", f"./qfunction-{env_size}.json")
show_plot = args.get("show", False)


env = Environment(env_size=env_size)
agent = QAgent(num_actions=num_actions, gamma=gamma,
               learning_rate=learning_rate, epsilon=epsilon, qfunction=path_load_qfunction)
episodes, rewards = train(environment=env, agent=agent,
                          num_episode=n_episode, max_step=max_step)
agent.save_qfunction(path_save_qfunction)
if show_plot:
    plot_rewards(episodes, rewards)
