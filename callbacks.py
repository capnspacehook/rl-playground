from pathlib import Path
from typing import Any, Dict, Optional
from matplotlib.pyplot import waitforbuttonpress

import numpy as np
import optuna
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import TensorBoardOutputFormat, Video
from stable_baselines3.common.vec_env import sync_envs_normalization
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
import wandb
from wandb.sdk.wandb_run import Run

from rl_playground.env_settings.env_settings import Orchestrator


# Copied from rl-zoo3 to avoid dependency
class RawStatisticsCallback(BaseCallback):
    """
    Callback used for logging raw episode data (return and episode length).
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        # Custom counter to reports stats
        # (and avoid reporting multiple values for the same step)
        self._timesteps_counter = 0
        self._tensorboard_writer = None

    def _init_callback(self) -> None:
        assert self.logger is not None
        # Retrieve tensorboard writer to not flood the logger output
        for out_format in self.logger.output_formats:
            if isinstance(out_format, TensorBoardOutputFormat):
                self._tensorboard_writer = out_format
        assert (
            self._tensorboard_writer is not None
        ), "You must activate tensorboard logging when using RawStatisticsCallback"

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" in info:
                logger_dict = {
                    "raw/rollouts/episodic_return": info["episode"]["r"],
                    "raw/rollouts/episodic_length": info["episode"]["l"],
                }
                exclude_dict = {key: None for key in logger_dict.keys()}
                self._timesteps_counter += info["episode"]["l"]
                self._tensorboard_writer.write(logger_dict, exclude_dict, self._timesteps_counter)

        return True


class RecordAndEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env: VecEnv,
        orchestrator: Orchestrator,
        wabRun: Run,
        eval_freq: int = 0,
        n_eval_episodes: int = 1,
        model_save_path: Optional[Path] = None,
        model_save_prefix: str = "rl_model",
        save_vecnormalize: bool = False,
        save_replay_buffer: bool = False,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param eval_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__(verbose)

        self.eval_env = eval_env
        self.orchestrator = orchestrator
        self.wabRun = wabRun
        self._eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self._n_eval_episodes = n_eval_episodes
        self.save_vecnormalize = save_vecnormalize
        self.save_replay_buffer = save_replay_buffer
        self._deterministic = deterministic

        if model_save_path is not None:
            self.latest_model_save_path = str(model_save_path / f"{model_save_prefix}_latest")
            self.best_model_save_path = str(model_save_path / f"{model_save_prefix}_best")

    def _on_step(self) -> bool:
        if self._eval_freq > 0 and self.n_calls % self._eval_freq == 0:
            screens = []

            def after_step(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                if _locals["done"]:
                    info = _locals["info"]

                    self.orchestrator.processEvalInfo(info)

                    logEntries = self.orchestrator.evalInfoLogEntries(info)
                    for entry in logEntries:
                        key, value = entry
                        self.logger.record(f"eval/{key.lower()}", value, exclude=("stdout"))

            # sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, both should be wrapped with VecNormalize"
                    ) from e

            if self.verbose >= 1:
                print("Evaluating model checkpoint")

            # tell the base env that an evaluation is starting
            self.eval_env.set_options({"_eval_starting": True})
            self.eval_env.reset()

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
                callback=after_step,
                return_episode_rewards=True,
            )

            self.orchestrator.postEval()

            if self.wabRun is not None:
                self.wabRun.log({"eval_video": wandb.Video("/tmp/eval.mp4", fps=30)})

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, "
                    f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                )
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if self.latest_model_save_path is not None:
                saveModel(
                    self.model,
                    self.latest_model_save_path,
                    self.save_replay_buffer,
                    self.save_vecnormalize,
                )

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    saveModel(
                        self.model,
                        self.best_model_save_path,
                        self.save_replay_buffer,
                        self.save_vecnormalize,
                    )

                self.best_mean_reward = float(mean_reward)

        return True


class TrialEvalCallback(EvalCallback):
    def __init__(
        self,
        eval_env: VecEnv,
        trial: optuna.Trial,
        best_model_save_path: Path,
        n_eval_episodes: int,
        eval_freq: int,
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
        self._best_reward: float | None = None
        self.modelSavePath = str(best_model_save_path / f"rl_model_{trial.number}")
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # tell the base env that an evaluation is starting
            self.eval_env.set_options({"_eval_starting": True})
            self.eval_env.reset()
            self.eval_env._reset_options()

            super()._on_step()
            self.eval_idx += 1

            if self._best_reward is None or self.last_mean_reward > self._best_reward:
                self._best_reward = self.last_mean_reward
                saveModel(
                    self.model,
                    self.modelSavePath,
                    saveReplayBuffer=False,
                    saveVecNormalize=True,
                )

            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if needed
            if self.trial.should_prune():
                self.is_pruned = True
                return False

        return True


def saveModel(model: BaseAlgorithm, savePath: str, saveReplayBuffer: bool, saveVecNormalize: bool):
    model.save(savePath + ".zip")

    # save replay buffer
    if saveReplayBuffer and hasattr(model, "replay_buffer") and model.replay_buffer is not None:
        model.save_replay_buffer(savePath + "_rb.pkl")

    # save VecNormalize settings
    if saveVecNormalize and model.get_vec_normalize_env() is not None:
        model.get_vec_normalize_env().save(savePath + "_vn.pkl")
