import argparse
import time
import datetime
# import torch_ac
import tensorboardX
import sys
import wandb

sys.path.insert(0, '/cluster/project7/diversity_rl/diversity_study/rl-starter-files')

import utils 
from utils import device
from model import ACModel

import sys
import os

# Add folder1 to system path
# sys.path.insert(0, os.path.abspath('/Users/corinacaraconcea/Downloads/diversity_study/torch_ac_v2/torch_ac_v2/algos'))

sys.path.insert(0, '/cluster/project7/diversity_rl/diversity_study/torch_ac_v2/torch_ac_v2/algos')


import a2c
import ppo

from Minigrid.minigrid.__init__ import register_minigrid_envs

register_minigrid_envs()

# Parse arguments

#The argparse module is a standard Python library for writing user-friendly command-line interfaces.
parser = argparse.ArgumentParser()

# General parameters
# which algorithm
parser.add_argument("--algo", required=True,
                    help="algorithm to use: a2c | ppo (REQUIRED)")
# which environment
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
# which model
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=10,
                    help="number of updates between two saves (default: 10, 0 means no saving)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 1e7)")

# Parameters for main algorithm
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO (default: 256)")
parser.add_argument("--frames-per-proc", type=int, default=None,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.0001,
                    help="learning rate (default: 0.0001)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.0005,
                    help="entropy term coefficient (default: 0.0005)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--recurrence", type=int, default=2,
                    help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model to handle text input")

parser.add_argument("--intrinsic-coef", type=float, default=0.005,
                    help="beta coefficient for the intrinsic reward (default: 0.005)")

parser.add_argument("--intrinsic-reward-model", default=None,
                    help="ntrinsic-reward-model, pick from count, RND , DIAYN, TrajectoryCount (default: count)")

parser.add_argument("--number-skills", type=int, default=10,
                    help="number of skills for DIAYN (default: 10)")

parser.add_argument("--window-size", type=int, default=5,
                    help="window size (default: 5)")


if __name__ == "__main__":


    args = parser.parse_args()

    args.mem = args.recurrence > 1

    diayn_flag = args.intrinsic_reward_model == "DIAYN"

    if args.intrinsic_reward_model == None:
        model_flag = ""
    else:
        model_flag = args.intrinsic_reward_model
        print(model_flag)

    # Set run dir
    # set te directory where you save the model and the logs

    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"

    model_name = args.model or default_model_name
    model_dir = utils.get_model_dir(model_name)
    # Load loggers and Tensorboard writer

    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)

    # Log command and all script arguments
    # setup loggers to log information to the terminal, a text file, a CSV file and tensorboard

    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources

    utils.seed(args.seed)

    # Set device

    txt_logger.info(f"Device: {device}\n")

    # Load environments

    envs = []
    for i in range(args.procs):
        envs.append(utils.make_env(args.env, args.seed + 10000 * i))
    txt_logger.info("Environments loaded\n")

    # Load training status for ACmodel from the intrinsic reward directory
    try:
        status_base = utils.get_status(model_flag, model_dir)
    except OSError:
        status_base = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    # print("status_base: ", status_base)

    # Load observations preprocessor

    obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
    action_space = envs[0].action_space
    print("actions space", action_space)

    if "vocab" in status_base:
        preprocess_obss.vocab.load_vocab(status_base["vocab"])
    txt_logger.info("Observations preprocessor loaded")

    # Load model

    acmodel = ACModel(obs_space, envs[0].action_space, args.mem, args.text,diayn_flag,args.number_skills)
    if "model_state" in status_base:
        acmodel.load_state_dict(status_base["model_state"])
    acmodel = acmodel.to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(acmodel))

    
    # Load algo

    if args.algo == "a2c":
        if args.rnd_model_flag is not None:
            algo = a2c.A2CAlgo(envs, acmodel,rndmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                            args.optim_alpha, args.optim_eps, preprocess_obss,args.wrapper, args.intrinsic_coef)
        else:
            algo = a2c.A2CAlgo(envs, acmodel,None, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                args.optim_alpha, args.optim_eps, preprocess_obss,args.wrapper, args.intrinsic_coef)

    elif args.algo == "ppo":
        if args.intrinsic_reward_model == "count":
            algo = ppo.PPOAlgo(envs, obs_space,action_space, acmodel,None, args.intrinsic_reward_model, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss, args.intrinsic_coef, args.number_skills,args.window_size)
        elif args.intrinsic_reward_model == "RND":
            algo = ppo.PPOAlgo(envs, obs_space, action_space, acmodel,None, args.intrinsic_reward_model, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss, args.intrinsic_coef,args.number_skills,args.window_size)
        elif args.intrinsic_reward_model == "TrajectoryCount":
            algo = ppo.PPOAlgo(envs,obs_space, action_space, acmodel,None, args.intrinsic_reward_model, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss, args.intrinsic_coef,args.number_skills,args.window_size)
        elif args.intrinsic_reward_model == "TrajectoryWindowCount":
            algo = ppo.PPOAlgo(envs,obs_space, action_space, acmodel,None, args.intrinsic_reward_model, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss, args.intrinsic_coef,args.number_skills,args.window_size)
        elif args.intrinsic_reward_model == "DIAYN":
            algo = ppo.PPOAlgo(envs,obs_space, action_space, acmodel, None, args.intrinsic_reward_model, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss, args.intrinsic_coef,args.number_skills,args.window_size)
        elif args.intrinsic_reward_model == "TrajectoryRND":
            algo = ppo.PPOAlgo(envs,obs_space, action_space, acmodel, None, args.intrinsic_reward_model, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss, args.intrinsic_coef,args.number_skills,args.window_size)
        elif args.intrinsic_reward_model == "TrajectoryModel":
            algo = ppo.PPOAlgo(envs,obs_space, action_space, acmodel, None, args.intrinsic_reward_model, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss, args.intrinsic_coef,args.number_skills,args.window_size)
        elif args.intrinsic_reward_model == "TrajectoryAutoencoder":
            algo = ppo.PPOAlgo(envs,obs_space, action_space, acmodel, None, args.intrinsic_reward_model, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss, args.intrinsic_coef,args.number_skills,args.window_size)  
        elif args.intrinsic_reward_model == "None":
            algo = ppo.PPOAlgo(envs, obs_space, action_space, acmodel, None, None, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss, args.intrinsic_coef,args.number_skills,args.window_size)     
        else:
            raise ValueError("Incorrect algorithm name: {}".format(args.algo))

    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))

    if "optimizer_state" in status_base:
        algo.optimizer.load_state_dict(status_base["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")

    # Initialize the Weights and Biases run
    run = wandb.init(
    # Set the entity
    entity = "cori-caraconcea",
    # Set the project where this run will be logged
    project="Empty Minigrid 16x16",
    # Track hyperparameters and run metadata
    config={
        "model": model_flag,
        "env": args.env,
        "seed": args.seed,
        "intr_coeff": args.intrinsic_coef,
        "window_size": args.window_size,
        "no_skills": args.number_skills
        })

    # Train model

    num_frames = status_base["num_frames"]
    update = status_base["update"]
    start_time = time.time()

    while num_frames < args.frames:
        # Update model parameters
        update_start_time = time.time()
        # collect experiece and logs
        exps, logs1 = algo.collect_experiences()
        # update parameters and get logs
        logs2 = algo.update_parameters(exps)
        # print("parameters update")
        logs = {**logs1, **logs2}
        update_end_time = time.time()

        # increase the update count
        num_frames += logs["num_frames"]
        update += 1

        # Print logs

        if update % args.log_interval == 0:
            # number of frames per second
            fps = logs["num_frames"] / (update_end_time - update_start_time)
            # duration
            duration = int(time.time() - start_time)
            # return per episode
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            # reshaped retun per episode
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            # number of frames per episode
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])
            # external return per episode
            external_return_per_episode = utils.synthesize(logs["ext_return_per_episode"])
            # batch policies entropy 
            entropy = utils.synthesize(logs["entropy"])            


            # "mean_rreturn_per_episode_mean": rreturn_per_episode['mean'],
            wandb.log({"mean_rreturn_per_episode_mean": rreturn_per_episode['mean'],
                       "batch_entropy_mean": entropy['mean'],
                       "intrinsic_reward": sum(list(exps.intrinsic_rewards))/len(list(exps.intrinsic_rewards))})

            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
            data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

            txt_logger.info(
                "U {} | F {:010} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
                .format(*data))

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()

            if status_base["num_frames"] == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)

        # Save status

        # if args.save_interval > 0 and update % args.save_interval == 0:
        #     status_base = {"model_name": "ACmodel","num_frames": num_frames, "update": update,
        #               "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
        #     if hasattr(preprocess_obss, "vocab"):
        #         status_base["vocab"] = preprocess_obss.vocab.vocab
        #     utils.save_status(status_base,model_flag, model_dir)
        #     txt_logger.info("Status saved")

        # Save status rn

        # if args.save_interval > 0 and update % args.save_interval == 0:
        #     status_rnd = {"model_name": "RNDmodel","num_frames": num_frames, "update": update,
        #               "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
        #     if hasattr(preprocess_obss, "vocab"):
        #         status_rnd["vocab"] = preprocess_obss.vocab.vocab
        #     utils.save_status_rnd(status_rnd,model_flag, model_dir)
        #     txt_logger.info("Status saved")
