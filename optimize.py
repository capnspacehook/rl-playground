#!/usr/bin/env/python3

import os
from typing import Any, Callable, Dict, Union

from datetime import datetime
import gymnasium
import numpy as np
import optuna
import torch.nn as nn
from optuna.pruners import BasePruner, PercentilePruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState

# from sb3_contrib import QRDQN
from stable_baselines3 import PPO, HerReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from rl_playground.envs.pyboy.register import createPyboyEnv

N_TRIALS = 100
N_STARTUP_TRIALS = 5
N_EVALUATIONS = 20
N_TIMESTEPS = int(5e6)
N_TRAINING_ENVS = 48
EVAL_FREQ = (N_TIMESTEPS // N_EVALUATIONS) // N_TRAINING_ENVS
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


def sample_qrdqn_params(trial: optuna.Trial) -> Dict[str, Any]:
    batch_size = trial.suggest_categorical(
        "batch_size", [16, 32, 64, 100, 128, 256, 512]
    )
    buffer_size = trial.suggest_categorical(
        "buffer_size", [int(1e4), int(5e4), int(1e5), int(1e6)]
    )
    exploration_final_eps = trial.suggest_float("exploration_final_eps", 0, 0.2)
    exploration_fraction = trial.suggest_float("exploration_fraction", 0, 0.5)
    gamma = trial.suggest_categorical(
        "gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
    )
    learning_rate = trial.suggest_float("learning_rate", 5e-6, 1, log=True)
    learning_starts = trial.suggest_categorical(
        "learning_starts", [0, 1000, 5000, 10000, 20000]
    )
    target_update_interval = trial.suggest_categorical(
        "target_update_interval", [100, 1000, 5000, 10000, 15000, 20000]
    )
    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 128, 256, 1000])
    # make higher values more likely
    tau = 1 - trial.suggest_float("tau", low=0, high=1, log=True)

    # net_arch = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])
    # net_arch = {"tiny": [64], "small": [64, 64], "medium": [256, 256]}[net_arch]
    n_quantiles = trial.suggest_int("n_quantiles", 5, 200)

    hyperparams = {
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "exploration_final_eps": exploration_final_eps,
        "exploration_fraction": exploration_fraction,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "learning_starts": learning_starts,
        "target_update_interval": target_update_interval,
        "train_freq": train_freq,
        "tau": tau,
        "policy_kwargs": dict(
            # net_arch=net_arch,
            n_quantiles=n_quantiles,
        ),
    }

    # use_her = trial.suggest_categorical("use_her_replay_buffer", [False, True])
    # if use_her:
    #     hyperparams = sample_her_params(trial, hyperparams)

    return hyperparams


def sample_her_params(
    trial: optuna.Trial, hyperparams: Dict[str, Any]
) -> Dict[str, Any]:
    her_kwargs = {}
    her_kwargs["n_sampled_goal"] = trial.suggest_int("n_sampled_goal", 1, 5)
    her_kwargs["goal_selection_strategy"] = trial.suggest_categorical(
        "goal_selection_strategy", ["final", "episode", "future"]
    )
    hyperparams["replay_buffer_class"] = HerReplayBuffer
    hyperparams["replay_buffer_kwargs"] = her_kwargs
    return hyperparams


def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
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
    n_steps = trial.suggest_categorical(
        "n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    )

    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    activation_fn = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
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

    bufferSize = N_TRAINING_ENVS * n_steps
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
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    # Force conversion to float
    initial_value_ = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value_

    return func


class TimeLimitPruner(BasePruner):
    def __init__(self, wrappedPruner: BasePruner) -> None:
        self.curTrialNum = 0
        self.lastCalled = None
        self.wrappedPruner = wrappedPruner

    def prune(
        self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial"
    ) -> bool:
        # reset lastCalled when a new trial is started
        if self.curTrialNum != trial.number:
            self.curTrialNum = trial.number
            self.lastCalled = None

        trials = study.get_trials(states=[TrialState.COMPLETE])
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
                    N_EVALUATIONS // 2.5
                ):
                    return True

            # if this trial has already run for over twice as long of
            # the average trial, prune the trial
            trialDuration = now - trial.datetime_start
            if trialDuration.total_seconds() >= 2.0 * avgDuration:
                return True

        return self.wrappedPruner.prune(study, trial)


class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: gymnasium.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1

            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need.
            if self.trial.should_prune():
                self.is_pruned = True
                return False

        return True


def makeEnv(rank: int, seed: int = 0, isEval=False):
    def _init():
        _, env = createPyboyEnv(
            "games/super_mario_land.gb", isEval=isEval, isHyperparamOptimize=True
        )
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


def objective(trial: optuna.Trial) -> float:
    kwargs = DEFAULT_PPO_HYPERPARAMS.copy()
    # Sample hyperparameters.
    hyperparams = sample_ppo_params(trial)
    kwargs.update(hyperparams)

    trainingEnv = SubprocVecEnv([makeEnv(i) for i in range(N_TRAINING_ENVS)])
    trainingEnv = VecNormalize(trainingEnv)
    evalEnv = DummyVecEnv([makeEnv(0, True)])
    evalEnv = VecNormalize(evalEnv, training=False, norm_reward=False)
    if "gamma" in hyperparams:
        trainingEnv.gamma = hyperparams["gamma"]
        evalEnv.gamma = hyperparams["gamma"]

    kwargs["env"] = trainingEnv
    model = PPO(**kwargs)

    eval_callback = TrialEvalCallback(
        evalEnv,
        trial,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq=EVAL_FREQ,
        deterministic=True,
    )

    try:
        model.learn(N_TIMESTEPS, callback=eval_callback)
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


if __name__ == "__main__":
    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS, multivariate=True)
    # Prune the bottom 25% performing trials
    # Do not prune before 1/2 of the max budget is used.
    pruner = PercentilePruner(
        percentile=75,
        n_startup_trials=N_STARTUP_TRIALS,
        n_warmup_steps=N_EVALUATIONS // 2,
    )
    pruner = TimeLimitPruner(pruner)

    study = optuna.create_study(
        storage="mysql://root@localhost/optuna",
        study_name="mario_land_ppo_longer2",
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )

    # TODO: load these from filrs in specified dir
    # add default PPO hyperparams
    study.enqueue_trial(
        {
            "batch_size": 64,
            "clip_range": 0.2,
            "ent_coef": 0.00000001,
            "gae_lambda": 0.95,
            "gamma": 0.99,
            "learning_rate": 3e-4,
            "lr_schedule": "constant",
            "max_grad_norm": 0.5,
            "n_epochs": 10,
            "n_steps": 2048,
            "activation_fn": "tanh",
            "net_arch": "small",
            "vf_coef": 0.5,
        },
        skip_if_exists=True,
    )

    study.enqueue_trial(
        {
            "batch_size": 512,
            "clip_range": 0.2,
            "ent_coef": 1.080365148093321e-05,
            "gae_lambda": 0.8,
            "gamma": 0.99,
            "learning_rate": 6.160438419274751e-05,
            "lr_schedule": "constant",
            "max_grad_norm": 0.9,
            "n_epochs": 10,
            "n_steps": 256,
            "activation_fn": "tanh",
            "net_arch": "medium",
            "vf_coef": 0.21730023144009505,
        },
        skip_if_exists=True,
    )

    try:
        study.optimize(
            objective,
            n_trials=N_TRIALS,
            show_progress_bar=True,
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
