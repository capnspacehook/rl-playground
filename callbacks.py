from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import optuna
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import TensorBoardOutputFormat, Video
from stable_baselines3.common.vec_env import sync_envs_normalization
from stable_baselines3.common.vec_env.base_vec_env import VecEnv


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
                self._tensorboard_writer.write(
                    logger_dict, exclude_dict, self._timesteps_counter
                )

        return True


class RecordAndEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env: VecEnv,
        eval_freq: int = 0,
        n_eval_episodes: int = 1,
        best_model_save_path: Optional[str] = None,
        best_model_save_prefix: str = "rl_model",
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
        self._eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self._n_eval_episodes = n_eval_episodes
        self.best_model_save_path = best_model_save_path
        self.save_vecnormalize = save_vecnormalize
        self.save_replay_buffer = save_replay_buffer
        self._deterministic = deterministic

        if self.best_model_save_path is not None:
            self.best_model_save_path = str(
                self.best_model_save_path / best_model_save_prefix
            )

    def _on_step(self) -> bool:
        if self._eval_freq > 0 and self.n_calls % self._eval_freq == 0:
            screens = []

            def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, recording the screen in the captured `screens` list

                :param _locals: A dictionary containing all local variables of the callback's scope
                :param _globals: A dictionary containing all global variables of the callback's scope
                """
                screen = self.eval_env.render()
                # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                screens.append(screen.transpose(2, 0, 1))

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

            # tell the base envs that an evaluation is starting
            self.eval_env.set_options({"_eval_starting": True})
            self.eval_env.reset()
            self.eval_env._reset_options()

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )

            # Add dimension to array so it's a 5-D array (the video encoder
            # requires this for some reason)
            screens = np.expand_dims(np.stack(screens), axis=0)
            self.logger.record(
                "eval/video",
                Video(screens, fps=12),
                exclude=("stdout", "log", "json", "csv"),
            )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(
                episode_lengths
            )
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
            self.logger.record(
                "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
            )
            self.logger.dump(self.num_timesteps)

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
        eval_env: gym.Env,
        trial: optuna.Trial,
        best_model_save_path: str,
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
        self._best_reward = 0
        self.modelSavePath = str(best_model_save_path / f"rl_model_{trial.number}")
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1

            if self.last_mean_reward > self._best_reward:
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


def saveModel(
    model: BaseAlgorithm, savePath: str, saveReplayBuffer: bool, saveVecNormalize: bool
):
    model.save(savePath + ".zip")

    # save replay buffer
    if (
        saveReplayBuffer
        and hasattr(model, "replay_buffer")
        and model.replay_buffer is not None
    ):
        model.save_replay_buffer(savePath + "_rb.pkl")

    # save VecNormalize settings
    if saveVecNormalize and model.get_vec_normalize_env() is not None:
        model.get_vec_normalize_env().save(savePath + "_vn.pkl")
