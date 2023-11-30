#!/usr/bin/env/python3


import argparse
import ast
import os
from datetime import datetime
from os import listdir
from os.path import isfile, join
from typing import Any, Callable, Dict, Union
from pathlib import Path

import numpy as np
import optuna
import flax.linen as nn
from optuna.pruners import BasePruner, MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState

# from sb3_contrib import QRDQN
from sbx import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from callbacks import TrialEvalCallback
from rl_playground.envs.pyboy.register import createPyboyEnv

# TODO: make dynamic
N_EVAL_EPISODES = 8

DEFAULT_QRDQN_HYPERPARAMS = {
    "gradient_steps": -1,
    "policy_kwargs": {
        "net_arch": [256, 256],
    },
}

DEFAULT_PPO_HYPERPARAMS = {
    "policy": "MlpPolicy",
    "device": "cpu",
}


# def sample_qrdqn_params(trial: optuna.Trial) -> Dict[str, Any]:
#     batch_size = trial.suggest_categorical(
#         "batch_size", [16, 32, 64, 100, 128, 256, 512]
#     )
#     buffer_size = trial.suggest_categorical(
#         "buffer_size", [int(1e4), int(5e4), int(1e5), int(1e6)]
#     )
#     exploration_final_eps = trial.suggest_float("exploration_final_eps", 0, 0.2)
#     exploration_fraction = trial.suggest_float("exploration_fraction", 0, 0.5)
#     gamma = trial.suggest_categorical(
#         "gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
#     )
#     learning_rate = trial.suggest_float("learning_rate", 5e-6, 1, log=True)
#     learning_starts = trial.suggest_categorical(
#         "learning_starts", [0, 1000, 5000, 10000, 20000]
#     )
#     target_update_interval = trial.suggest_categorical(
#         "target_update_interval", [100, 1000, 5000, 10000, 15000, 20000]
#     )
#     train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 128, 256, 1000])
#     # make higher values more likely
#     tau = 1 - trial.suggest_float("tau", low=0, high=1, log=True)

#     # net_arch = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])
#     # net_arch = {"tiny": [64], "small": [64, 64], "medium": [256, 256]}[net_arch]
#     n_quantiles = trial.suggest_int("n_quantiles", 5, 200)

#     hyperparams = {
#         "batch_size": batch_size,
#         "buffer_size": buffer_size,
#         "exploration_final_eps": exploration_final_eps,
#         "exploration_fraction": exploration_fraction,
#         "gamma": gamma,
#         "learning_rate": learning_rate,
#         "learning_starts": learning_starts,
#         "target_update_interval": target_update_interval,
#         "train_freq": train_freq,
#         "tau": tau,
#         "policy_kwargs": dict(
#             # net_arch=net_arch,
#             n_quantiles=n_quantiles,
#         ),
#     }

#     # use_her = trial.suggest_categorical("use_her_replay_buffer", [False, True])
#     # if use_her:
#     #     hyperparams = sample_her_params(trial, hyperparams)

#     return hyperparams


# def sample_her_params(
#     trial: optuna.Trial, hyperparams: Dict[str, Any]
# ) -> Dict[str, Any]:
#     her_kwargs = {}
#     her_kwargs["n_sampled_goal"] = trial.suggest_int("n_sampled_goal", 1, 5)
#     her_kwargs["goal_selection_strategy"] = trial.suggest_categorical(
#         "goal_selection_strategy", ["final", "episode", "future"]
#     )
#     hyperparams["replay_buffer_class"] = HerReplayBuffer
#     hyperparams["replay_buffer_kwargs"] = her_kwargs
#     return hyperparams


def sample_ppo_params(trial: optuna.Trial, numTrainingEnvs: int) -> Dict[str, Any]:
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512, 1024])
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    gae_lambda = trial.suggest_categorical(
        "gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0]
    )
    gamma = trial.suggest_categorical(
        "gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
    )
    max_grad_norm = trial.suggest_categorical(
        "max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5]
    )
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)

    lr_schedule = trial.suggest_categorical("lr_schedule", ["linear", "constant"])
    if lr_schedule == "linear":
        learning_rate = linear_schedule(learning_rate)

    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    n_steps = trial.suggest_categorical("n_steps", [64, 128, 256, 512, 1024, 2048])

    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    activation_fn = {
        "tanh": nn.tanh,
        "relu": nn.relu,
        "elu": nn.elu,
        "leaky_relu": nn.leaky_relu,
    }[activation_fn]

    net_arch = trial.suggest_categorical(
        "net_arch",
        [
            "small",
            "medium",
            "large",
        ],
    )
    # Independent networks usually work best
    # when not working with images
    net_arch = {
        "tiny": dict(pi=[64], vf=[64]),
        "small": dict(pi=[64, 64], vf=[64, 64]),
        "medium": dict(pi=[256, 256], vf=[256, 256]),
        "large": dict(pi=[512, 512], vf=[512, 512]),
    }[net_arch]

    vf_coef = trial.suggest_float("vf_coef", 0, 1)

    bufferSize = numTrainingEnvs * n_steps
    if bufferSize % batch_size > 0:
        batch_size = n_steps

    return {
        "batch_size": batch_size,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "gae_lambda": gae_lambda,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "max_grad_norm": max_grad_norm,
        "n_epochs": n_epochs,
        "n_steps": n_steps,
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            activation_fn=activation_fn,
            net_arch=net_arch,
            # ortho_init=False,
        ),
        "vf_coef": vf_coef,
    }


def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    # Force conversion to float
    _initial_value = float(initial_value)

    def _schedule(progress_remaining: float) -> float:
        return progress_remaining * _initial_value

    return _schedule


class SlowTrialPruner(BasePruner):
    def __init__(
        self, wrappedPruner: BasePruner, enqueuedTrials: int, evalsPerTrial: int
    ) -> None:
        self.curTrialNum = 0
        self.lastCalled: datetime.datetime | None = None
        self.enqueuedTrials = enqueuedTrials
        self.evalsPerTrial = evalsPerTrial
        self.wrappedPruner = wrappedPruner

    def prune(
        self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial"
    ) -> bool:
        # reset lastCalled when a new trial is started
        if self.curTrialNum != trial.number:
            self.curTrialNum = trial.number
            self.lastCalled = None

        # if no values have been reported yet just note when this trial started
        if len(trial.intermediate_values) == 0:
            self.lastCalled = datetime.now()
            return False

        # skip the enqueued trials, if any
        if trial.number < self.enqueuedTrials:
            return self.wrappedPruner.prune(study, trial)

        trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        if len(trials) != 0:
            durations = [
                (t.datetime_complete - t.datetime_start).total_seconds() for t in trials
            ]
            avgDuration = np.mean(durations)

            lastCalled = self.lastCalled
            now = datetime.now()
            self.lastCalled = now

            # if this trial is on track to take over twice as long as
            # the average trial, prune the trial
            if lastCalled is not None:
                sinceLastCalled = now - lastCalled
                if sinceLastCalled.total_seconds() >= avgDuration // (
                    self.evalsPerTrial // 2.5
                ):
                    return True

            # if this trial has already run for over twice as long of
            # the average trial, prune the trial
            trialDuration = now - trial.datetime_start
            if trialDuration.total_seconds() >= 2.0 * avgDuration:
                return True

        return self.wrappedPruner.prune(study, trial)


def makeEnv(rank: int, isEval=False, seed: int = 0):
    def _init():
        _, env = createPyboyEnv(
            "games/super_mario_land.gb", isEval=isEval, isHyperparamOptimize=True
        )
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


def createObjective(saveDir: Path, numTrainingEnvs: int, evalFreq: int, steps: int):
    def _objective(trial: optuna.Trial) -> float:
        kwargs = DEFAULT_PPO_HYPERPARAMS.copy()
        # Sample hyperparameters
        hyperparams = sample_ppo_params(trial, numTrainingEnvs)
        kwargs.update(hyperparams)

        trainingEnv = SubprocVecEnv([makeEnv(i) for i in range(numTrainingEnvs)])
        trainingEnv = VecNormalize(trainingEnv)
        evalEnv = DummyVecEnv([makeEnv(0, isEval=True)])
        evalEnv = VecNormalize(evalEnv, training=False, norm_reward=False)
        if "gamma" in hyperparams:
            trainingEnv.gamma = hyperparams["gamma"]
            evalEnv.gamma = hyperparams["gamma"]

        kwargs["env"] = trainingEnv
        model = PPO(**kwargs)

        eval_callback = TrialEvalCallback(
            evalEnv,
            trial,
            best_model_save_path=saveDir,
            n_eval_episodes=N_EVAL_EPISODES,
            eval_freq=evalFreq,
            deterministic=True,
        )

        # let the SlowTrialPruner know a trial has started
        trial.should_prune()

        try:
            model.learn(steps, callback=eval_callback)
            # Free memory
            model.env.close()
            evalEnv.close()
        except (AssertionError, ValueError) as e:
            # Sometimes, random hyperparams can generate NaN
            # Free memory
            model.env.close()
            evalEnv.close()
            # Prune hyperparams that generate NaNs
            print(e)
            print("============")
            print("Sampled hyperparams:")
            print(hyperparams)
            raise optuna.exceptions.TrialPruned() from e

        del model.env, evalEnv
        del model

        if eval_callback.is_pruned:
            raise optuna.exceptions.TrialPruned()

        # Report best reward to prevent the trail being judged by a regression
        return eval_callback.best_mean_reward

    return _objective


if __name__ == "__main__":
    parser = argparse.ArgumentParser("optimize")
    parser.add_argument("-n", "--study-name", type=str, help="name of optuna study")
    parser.add_argument(
        "-t", "--trials", type=int, default=100, help="total number of trials to run"
    )
    parser.add_argument(
        "-s",
        "--startup-trials",
        type=int,
        default=5,
        help="number of initial random trials to run",
    )
    parser.add_argument(
        "-l",
        "--trial-steps",
        type=int,
        default=5e6,
        help="maximun number of steps per trial",
    )
    parser.add_argument(
        "--parallel-envs",
        type=int,
        default=0,
        help="number of environments to run in parallel",
    )
    parser.add_argument(
        "-e",
        "--evaluations-per-trial",
        type=int,
        default=20,
        help="number of evaluations to run per trial",
    )
    parser.add_argument(
        "--trials-dir",
        type=str,
        default="./trials",
        help="directory that contains trials to start the study with",
    )
    parser.add_argument("--resume", action="store_true", help="resume a canceled study")
    parser.add_argument(
        "--no-last-trial-restart",
        action="store_false",
        help="when resuming if the last trial failed, don't attempt it again",
    )
    args = parser.parse_args()

    study = optuna.create_study(
        storage="mysql://root@localhost/optuna",
        study_name=args.study_name,
        direction="maximize",
        load_if_exists=True,
    )

    trials = study.get_trials(deepcopy=False)
    newStudy = len(trials) == 0

    startupTrials = args.startup_trials
    enqueuedTrials = 0
    if newStudy:
        trialFiles = sorted(
            [
                join(args.trials_dir, f)
                for f in listdir(args.trials_dir)
                if isfile(join(args.trials_dir, f))
            ]
        )
        for trialFile in trialFiles:
            with open(trialFile, "r") as f:
                trial = ast.literal_eval(f.read())
                study.enqueue_trial(trial, skip_if_exists=True)

        # Don't count enqueued trials as random startup trials
        enqueuedTrials = len(trialFiles)
        startupTrials += enqueuedTrials

    nTrials = args.trials
    if args.resume:
        nTrials -= len(trials)
        startupTrials -= len(trials)

        # restart last trial if it failed
        if not args.no_last_trial_restart:
            lastTrial = trials[-1]
            if lastTrial.state == TrialState.FAIL:
                nTrials += 1
                startupTrials += 1

                study.enqueue_trial(lastTrial.params)

        if startupTrials < 0:
            startupTrials = 0

    study.sampler = TPESampler(n_startup_trials=startupTrials, multivariate=True)
    # Do not prune before 1/2 of the max budget is used.
    pruner = MedianPruner(
        n_startup_trials=startupTrials, n_warmup_steps=args.evaluations_per_trial // 2
    )
    study.pruner = SlowTrialPruner(pruner, enqueuedTrials, args.evaluations_per_trial)

    numTrainingEnvs = (
        os.cpu_count() * 2 if args.parallel_envs == 0 else args.parallel_envs
    )
    evalFreq = (args.trial_steps // args.evaluations_per_trial) // numTrainingEnvs

    saveDir = Path("checkpoints", "optimize", args.study_name)
    saveDir.mkdir(parents=True, exist_ok=True)

    try:
        study.optimize(
            createObjective(saveDir, numTrainingEnvs, evalFreq, args.trial_steps),
            n_trials=args.trials,
            gc_after_trial=True,
        )
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))
