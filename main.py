#!/usr/bin/env python3

import argparse
import datetime
import copy
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas

from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_device, set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
import torch
from wandb.integration.sb3 import WandbCallback
import wandb

from callbacks import RecordAndEvalCallback
from rl_playground.envs.pyboy.register import createPyboyEnv


def vecNormalizePath(checkpointPath: str) -> str:
    dir, file = os.path.split(checkpointPath)
    return os.path.join(dir, f"{os.path.splitext(file)[0]}_vn.pkl")


def train(args):
    print("Training mode")

    envSettings, _, orchClass = createPyboyEnv(args.rom)

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    runName = now + "_" + args.algorithm
    if args.run_name:
        runName += "_" + args.run_name

    gameName = os.path.basename(os.path.splitext(args.rom)[0])
    saveDir = Path("checkpoints", "train", gameName, runName)
    saveDir.mkdir(parents=True)

    def makeEnv(seed: int, rank: int, isEval: bool = False, envKwargs: Dict = {}):
        def _init():
            _, env, _ = createPyboyEnv(
                args.rom,
                args.render,
                speed=args.emulation_speed,
                isEval=isEval,
                outputDir=saveDir,
                envKwargs=envKwargs,
            )
            env.reset(seed=seed + rank)
            return env

        set_random_seed(seed)
        return _init

    seed = args.seed
    if seed == -1:
        seed = np.random.randint(2**32 - 1, dtype="int64").item()
    print(f"seed: {seed}")

    config = envSettings.hyperParameters(args.algorithm)
    envKwargs = {}
    if "env_kwargs" in config:
        envKwargs = config["env_kwargs"]
        del config["env_kwargs"]

    numTrainingEnvs = os.cpu_count() if args.parallel_envs == 0 else args.parallel_envs
    trainingVec = SubprocVecEnv
    if numTrainingEnvs == 1:
        trainingVec = DummyVecEnv
    trainingEnv = trainingVec([makeEnv(seed, i, envKwargs=envKwargs) for i in range(numTrainingEnvs)])
    evalEnv = DummyVecEnv([makeEnv(seed, 0, isEval=True, envKwargs=envKwargs)])

    orchestrator = orchClass(trainingEnv)

    projectName = args.project_name
    if projectName == "":
        projectName = gameName.replace("_", " ").title()

    config["device"] = args.device
    if "policy_kwargs" in config and "features_extractor_kwargs" in config["policy_kwargs"]:
        config["policy_kwargs"]["features_extractor_kwargs"]["device"] = get_device(args.device)

    # copy config so wandb callback doesn't include extraneous keys set
    # by the algorithm constructor
    wabConfig = copy.deepcopy(config)
    wabConfig["algorithm"] = args.algorithm
    wabConfig["parallel_envs"] = numTrainingEnvs

    wabRun = None
    if not args.disable_wandb:
        wabRun = wandb.init(
            project=projectName,
            name=args.run_name,
            notes=args.notes,
            config=wandb.helper.parse_config(wabConfig, include=wabConfig.keys()),
            sync_tensorboard=True,
        )

    normalizeObs, normalizeRewards = envSettings.normalize()
    if args.model_checkpoint:
        if normalizeObs or normalizeRewards:
            path = vecNormalizePath(args.model_checkpoint)
            trainingEnv = VecNormalize.load(path, trainingEnv)
            evalEnv = VecNormalize.load(path, evalEnv)
            evalEnv.training = False
            evalEnv.norm_reward = False

        model = args.algo.load(args.model_checkpoint, env=trainingEnv)
    else:
        if normalizeObs or normalizeRewards:
            trainingEnv = VecNormalize(trainingEnv, norm_obs=normalizeObs, norm_reward=normalizeRewards)
            evalEnv = VecNormalize(evalEnv, training=False, norm_obs=normalizeObs, norm_reward=False)
            if "gamma" in config:
                trainingEnv.gamma = config["gamma"]
                evalEnv.gamma = config["gamma"]

        config["env"] = trainingEnv
        config["device"] = args.device
        config["tensorboard_log"] = "tensorboard"
        config["verbose"] = 1
        model = args.algo(**config)

    exception = None
    try:
        callbackFreq = max(args.eval_freq // numTrainingEnvs, 1)
        callbacks = [
            RecordAndEvalCallback(
                evalEnv,
                orchestrator,
                wabRun,
                n_eval_episodes=envSettings.evalEpisodes(),
                eval_freq=callbackFreq,
                model_save_path=saveDir,
                save_replay_buffer=args.save_replay_buffers,
                save_vecnormalize=True,
                verbose=2,
            ),
        ]
        if not args.disable_wandb:
            callbacks.append(WandbCallback(log="parameters"))

        # model.collect_rollouts = torch.compile(model.collect_rollouts, mode="reduce-overhead")
        # model.train = torch.compile(model.train, mode="reduce-overhead")

        model.learn(
            total_timesteps=args.session_length,
            reset_num_timesteps=True,
            callback=callbacks,
            tb_log_name=runName,
            progress_bar=True,
        )
    except Exception as e:
        # TODO: doesn't work
        if e is not KeyboardInterrupt:
            exception = e
        print(f"Quitting: {e}")

    model.save(saveDir / "rl_model_done.zip")
    if args.save_replay_buffers and hasattr(model, "replay_buffer") and model.replay_buffer is not None:
        model.save_replay_buffer(saveDir / "rl_model_done_rb.pkl")
    if model.get_vec_normalize_env() is not None:
        model.get_vec_normalize_env().save(saveDir / "rl_model_done_vn.pkl")

    trainingEnv.close()
    evalEnv.close()
    if not args.disable_wandb:
        wabRun.finish()

    # raise exception after training is gracefully stopped
    if exception is not None:
        raise exception


def evaluate(args):
    if args.interactive_eval:
        print("Interactive evaluation mode")
        isEval = False
        isInteractiveEval = True
    else:
        print("Evaluation mode")
        isEval = True
        isInteractiveEval = False

    _, env, _ = createPyboyEnv(
        args.rom,
        render=args.render,
        speed=args.emulation_speed,
        isEval=isEval,
        isInteractiveEval=isInteractiveEval,
    )
    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load(vecNormalizePath(args.model_checkpoint), env)
    env.training = False
    env.norm_reward = False

    model = args.algo.load(args.model_checkpoint, env=env, device=args.device)

    episodeRewards = []
    obs = env.reset()
    try:
        while True:
            done = False
            rewards = []
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                rewards.append(reward)
    except KeyboardInterrupt:
        pass

    episodeRewards.append(sum(rewards))

    print(episodeRewards)
    print(f"Mean reward: {np.mean(episodeRewards)}")
    env.close()


def replay(args):
    print("Replay mode")

    envSettings, env, _ = createPyboyEnv(args.rom, args.render, args.emulation_speed)

    replay = pandas.read_csv(args.replay_episode, compression="zstd")
    for i in range(len(replay["step"])):
        stats = envSettings.gameState().stats()
        if (
            stats["x_area"] != replay["x_area"][i]
            or stats["y_area"] != replay["y_area"][i]
            or stats["x_coord"] != replay["x_coord"][i]
            or stats["y_coord"] != replay["y_coord"][i]
        ):
            print("area or coordinate mismatch")
            print(stats)
            print(replay["x_coord"][i], replay["y_coord"][i])

        actions = [int(a) for a in replay["actions"][i] if a not in "(,)[] "]

        env.step(actions)


def playtest(args):
    print("Playtest mode")

    _, env, _ = createPyboyEnv(args.rom, args.render, args.emulation_speed, isPlaytest=True)

    rewards = []
    recentRewards = []

    try:
        while True:
            env.reset()
            done = False
            while not done:
                _, reward, term, trunc, _ = env.step(0)
                done = term or trunc
                if reward != 0:
                    rewards.append(reward)
                    if len(recentRewards) == 3:
                        recentRewards.pop()
                    recentRewards.insert(0, reward)

                # print(f"Recent rewards: {recentRewards}", flush=True)
    except KeyboardInterrupt:
        pass

    print(reward)
    print(f"Reward sum: {sum(rewards)}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("rl-playground")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("-t", "--train", action="store_true", help="train AI to play a game")
    mode_group.add_argument("--evaluate", action="store_true", help="evaluate a model")
    mode_group.add_argument("-r", "--replay", action="store_true", help="replay a training episode")
    mode_group.add_argument(
        "--playtest",
        action="store_true",
        help="manually test state and reward tracking",
    )
    parser.add_argument(
        "--interactive-eval",
        action="store_true",
        help="evaluate a model while also allowing for manual inputs",
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        type=str,
        help="reinforcement learning algorithm to train with",
    )
    parser.add_argument(
        "-l",
        "--session-length",
        type=int,
        default=10_000_000,
        help="number of steps the training session will run",
    )
    parser.add_argument(
        "-c",
        "--model-checkpoint",
        type=Path,
        help="path to model checkpoint to resume training from",
    )
    parser.add_argument(
        "-p", "--project-name", type=str, default="", help="name of the Weights and Biases project to use"
    )
    parser.add_argument(
        "-n",
        "--run-name",
        type=str,
        default="",
        help="name of this training run for organization purposes",
    )
    parser.add_argument("--notes", type=str, default="", help="notes about this training run to store")
    parser.add_argument(
        "--save-replay-buffers",
        action="store_true",
        help="save replay buffers of best and final models so training can be continued for off-policy algorithms",
    )
    parser.add_argument(
        "--replay-episode",
        type=Path,
        help="path to compressed CSV of episode to replay",
    )
    parser.add_argument(
        "--emulation-speed",
        type=int,
        default=0,
        help="multiple of real-time to emulate; ignored if training",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="display live AI progress in emulator windows (will slow training)",
    )
    parser.add_argument(
        "--parallel-envs",
        type=int,
        default=0,
        help="number of environments to run in parallel",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=500_000,
        help="evaluate the model every n steps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="device to use to train, cpu, cuda or auto",
    )
    parser.add_argument("--seed", type=int, default=-1, help="seed to make randomness reproducible")
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="disable Weights & Biases integration",
    )
    parser.add_argument("rom", help="ROM to train against")

    args = parser.parse_args()

    if not args.train and not args.evaluate and not args.replay and not args.playtest:
        print("no mode specified")
        parser.print_help()
        exit(1)
    if args.evaluate and not args.model_checkpoint:
        print("--model-checkpoint must be specified when --evaluate is specified")
        parser.print_help()
        exit(1)
    if args.interactive_eval and not args.evaluate:
        print("--evaluate must be specified when --interactive-eval is specified")
        parser.print_help()
        exit(1)
    if args.replay and not args.replay_episode:
        print("--replay-episode must be specified when --replay is specified")
        parser.print_help()
        exit(1)

    if args.train or args.evaluate:
        args.algorithm = args.algorithm.lower()
        match args.algorithm:
            case "ppo":
                args.algo = PPO

                if args.save_replay_buffers:
                    print("PPO doesn't have a replay buffer")
                    exit(1)
            case _:
                print(f"invalid algorithm {args.algorithm}")
                exit(1)

    if args.evaluate or args.replay or args.playtest:
        args.render = True

    if args.emulation_speed == 0:
        if args.playtest or args.interactive_eval:
            args.emulation_speed = 1
        elif args.evaluate or args.replay:
            args.emulation_speed = 3

    if args.train:
        train(args)
    elif args.replay:
        replay(args)
    elif args.evaluate:
        evaluate(args)
    elif args.playtest:
        playtest(args)
