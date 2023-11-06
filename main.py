#!/usr/bin/env python3

import argparse
import datetime
import os
from pathlib import Path

import pandas
from sb3_contrib import QRDQN
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from wandb.integration.sb3 import WandbCallback
import wandb

from callbacks import RecordAndEvalCallback
from rl_playground.envs.pyboy.register import createPyboyEnv


def train(args):
    print("Training mode")

    envSettings, _ = createPyboyEnv(args.rom)

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    runName = now + "_" + args.algorithm
    if args.run_name:
        runName += "_" + args.run_name

    gameName = os.path.basename(os.path.splitext(args.rom)[0])
    saveDir = Path("checkpoints") / gameName / runName
    saveDir.mkdir(parents=True)

    def makeEnv(rank: int, isEval: bool = False, seed: int = 0):
        def _init():
            _, env = createPyboyEnv(
                args.rom,
                args.render,
                speed=args.emulation_speed,
                isEval=isEval,
                outputDir=saveDir,
            )
            env.reset(seed=seed + rank)
            return env

        set_random_seed(seed)
        return _init

    numTrainingEnvs = os.cpu_count() if args.parallel_envs == 0 else args.parallel_envs
    trainingEnv = SubprocVecEnv([makeEnv(i) for i in range(numTrainingEnvs)])
    evalEnv = DummyVecEnv([makeEnv(0, isEval=True)])

    projectName = gameName.replace("_", " ").title()

    config = envSettings.hyperParameters(args.algorithm)
    # copy config so wandb callback doesn't include extraneous keys set
    # by the algorithm constructor
    wabConfig = config.copy()
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

    if args.model_checkpoint:
        if envSettings.normalizeObservation():
            dir, file = os.path.split(args.model_checkpoint)
            path = os.path.join(dir, f"{os.path.splitext(file)[0]}_vn.pkl")

            trainingEnv = VecNormalize.load(path, trainingEnv)
            evalEnv = VecNormalize.load(path, evalEnv)
            evalEnv.training = False
            evalEnv.norm_reward = False

        model = args.algo.load(args.model_checkpoint, env=trainingEnv)
    else:
        if envSettings.normalizeObservation():
            trainingEnv = VecNormalize(trainingEnv)
            evalEnv = VecNormalize(evalEnv, training=False, norm_reward=False)
            if "gamma" in config:
                trainingEnv.gamma = config["gamma"]
                evalEnv.gamma = config["gamma"]

        config["env"] = trainingEnv
        config["tensorboard_log"] = "tensorboard"
        config["verbose"] = 1
        model = args.algo(**config)

    exception = None
    try:
        callbackFreq = max(args.eval_freq // numTrainingEnvs, 1)
        callbacks = [
            RecordAndEvalCallback(
                evalEnv,
                n_eval_episodes=envSettings.evalEpisodes(),
                eval_freq=callbackFreq,
                best_model_save_path=saveDir,
                best_model_save_prefix="rl_model_best",
                save_replay_buffer=args.save_replay_buffers,
                save_vecnormalize=True,
                verbose=2,
            ),
        ]
        if not args.disable_wandb:
            callbacks.append(WandbCallback(log="parameters"))

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
    if (
        args.save_replay_buffers
        and hasattr(model, "replay_buffer")
        and model.replay_buffer is not None
    ):
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
    print("Evaluation mode")

    envSettings, env = createPyboyEnv(
        args.rom, render=args.render, speed=args.emulation_speed, isEval=True
    )
    env = DummyVecEnv([lambda: env])
    model = args.algo.load(args.model_checkpoint, env=env)

    obs = env.reset()
    for _ in range(envSettings.evalEpisodes()):
        done = False
        try:
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _ = env.step(action)
        except KeyboardInterrupt:
            pass

    env.close()


def replay(args):
    print("Replay mode")

    envSettings, env = createPyboyEnv(args.rom, args.render, args.emulation_speed)

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

    _, env = createPyboyEnv(
        args.rom, args.render, args.emulation_speed, isPlaytest=True
    )
    env = DummyVecEnv([lambda: env])
    model = args.algo("MlpPolicy", env)
    obs = env.reset()

    rewards = []
    recentRewards = []
    done = False

    try:
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            if reward != 0:
                rewards.append(reward)
                if len(recentRewards) == 3:
                    recentRewards.pop()
                recentRewards.insert(0, reward)

            print(f"Recent rewards: {recentRewards}")
    except KeyboardInterrupt:
        pass

    print(reward)
    print(f"Reward sum: {sum(rewards)}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("rl-playground")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "-t", "--train", action="store_true", help="train AI to play a game"
    )
    mode_group.add_argument(
        "--evaluate", action="store_true", help="evaluate a network"
    )
    mode_group.add_argument(
        "-r", "--replay", action="store_true", help="replay a training episode"
    )
    mode_group.add_argument(
        "--playtest",
        action="store_true",
        help="manually test state and reward tracking",
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
        "-n",
        "--run-name",
        type=str,
        default="",
        help="name of this training run for organization purposes",
    )
    parser.add_argument(
        "--notes", type=str, default="", help="notes about this training run to store"
    )
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
    if args.replay and not args.replay_episode:
        print("--replay-episode must be specified when --replay is specified")
        parser.print_help()
        exit(1)

    args.algorithm = args.algorithm.lower()
    match args.algorithm:
        case "ppo":
            args.algo = PPO

            if args.save_replay_buffers:
                print("PPO doesn't have a replay buffer")
                exit(1)
        case "qrdqn":
            args.algo = QRDQN
        case _:
            print(f"invalid algorithm {args.algorithm}")
            exit(1)

    if args.evaluate or args.replay or args.playtest:
        args.render = True

    if args.emulation_speed == 0:
        if args.evaluate or args.replay:
            args.emulation_speed = 3
        if args.playtest:
            args.emulation_speed = 1

    if args.train:
        train(args)
    elif args.replay:
        replay(args)
    elif args.evaluate:
        evaluate(args)
    elif args.playtest:
        playtest(args)
