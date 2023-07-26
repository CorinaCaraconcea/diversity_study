import argparse
import time
import torch
import matplotlib.pyplot as plt
import wandb

from torch_ac.utils.penv import ParallelEnv


import utils
from utils import device


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--episodes", type=int, default=100,
                    help="number of episodes of evaluation (default: 100)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected")
parser.add_argument("--worst-episodes-to-show", type=int, default=10,
                    help="how many worst episodes to show")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")
parser.add_argument("--wadb-project-name", default="RL_evals",
                    help="name of the project for Weights and Biases")
parser.add_argument("--model-flag", default="",
                    help="name of the method (default: "")")

if __name__ == "__main__":
    args = parser.parse_args()

    # Set seed for all randomness sources

    utils.seed(args.seed)

    # Set device

    print(f"Device: {device}\n")

    # Load Weights & Biases
    wandb.login()

    # Load environments

    envs = []
    for i in range(args.procs):
        env = utils.make_env(args.env, args.seed + 10000 * i)
        envs.append(env)
    env = ParallelEnv(envs)
    print("Environments loaded\n")

    # Load agent

    model_dir = utils.get_model_dir(args.model)
    agent = utils.Agent(env.observation_space, env.action_space,args.model_flag, model_dir,
                        argmax=args.argmax, num_envs=args.procs,
                        use_memory=args.memory, use_text=args.text)
    print("Agent loaded\n")

    # Initialize logs

    logs = {"num_frames_per_episode": [], "return_per_episode": []}

    # Run agent

    start_time = time.time()

    obss = env.reset()

    log_done_counter = 0
    # args.procs is the number of parallel environments
    # placeholder for episode return and the episode ends when the episode ends or after a number of steps
    log_episode_return = torch.zeros(args.procs, device=device)
    log_episode_num_frames = torch.zeros(args.procs, device=device)

    # Initialize run
    run = wandb.init(
    # Set the entity
    entity = "cori-caraconcea",
    # Set the project where this run will be logged
    project="test for RL",
    # Track hyperparameters and run metadata
    config={
        "model": args.model_flag    
        })

    # essentially run this until you have a certain number of done episodes
    while log_done_counter < args.episodes:
        actions = agent.get_actions(obss)
        obss, rewards, terminateds, truncateds, _ = env.step(actions)
        dones = tuple(a | b for a, b in zip(terminateds, truncateds))
        agent.analyze_feedbacks(rewards, dones)

        log_episode_return += torch.tensor(rewards, device=device, dtype=torch.float)
        log_episode_num_frames += torch.ones(args.procs, device=device)

        for i, done in enumerate(dones):
            if done:
                log_done_counter += 1
                logs["return_per_episode"].append(log_episode_return[i].item())
                logs["num_frames_per_episode"].append(log_episode_num_frames[i].item())
                # wandb.log({"return_per_episode": log_episode_return[i].item(), "num_frames_per_episode": log_episode_num_frames[i].item()})

        mask = 1 - torch.tensor(dones, device=device, dtype=torch.float)
        log_episode_return *= mask
        log_episode_num_frames *= mask

    end_time = time.time()

    # Print logs

    num_frames = sum(logs["num_frames_per_episode"])
    # calculate the frames per second
    fps = num_frames / (end_time - start_time)
    # calculate the total duration for all the episodes
    duration = int(end_time - start_time)
    # return per episodes and no of frames per episode -> the synthesize function will return mean, std, min and max
    return_per_episode = utils.synthesize(logs["return_per_episode"])
    num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

    print("F {} | FPS {:.0f} | D {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {}"
          .format(num_frames, fps, duration,
                  *return_per_episode.values(),
                  *num_frames_per_episode.values()))


    print("return per episode", return_per_episode['mean'])

    # Print worst episodes

    n = args.worst_episodes_to_show
    if n > 0:
        print("\n{} worst episodes:".format(n))

        indexes = sorted(range(len(logs["return_per_episode"])), key=lambda k: logs["return_per_episode"][k])
        for i in indexes[:n]:
            print("- episode {}: R={}, F={}".format(i, logs["return_per_episode"][i], logs["num_frames_per_episode"][i]))

    # # Plot return per episode
    # plt.figure(figsize=(10, 5))
    # plt.plot(logs["return_per_episode"])
    # plt.xlabel('Episode')
    # plt.ylabel('Return')
    # plt.title('Return per Episode')
    # plt.show()
    # plt.savefig('return_per_episode.png')