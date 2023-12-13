from typing import Callable, Union

import torch.nn as nn

from rl_playground.env_settings.super_mario_land.extractor import MarioLandExtractor

# Observation settings
N_OBS_STACK = 6  # number of observations to stack
N_STATE_STACK = 6  # number of games states to use to calculate mean speeds

# General env settings
STARTING_LIVES = 2
STARTING_TIME = 400

# Reward values
# TODO: punish more for game over?
DEATH_PUNISHMENT = -25
HIT_PUNISHMENT = -5
# TODO: linearly increase to encourage progress over speed
# earlier, then after game mechanics and levels are learned
# encourage speed more
CLOCK_PUNISHMENT = -0.1
# TODO: weight forward and backward movement differently so moving
# backwards isn't punished as much, and maybe also add extra reward
# when max level progress increases
MOVEMENT_REWARD_COEF = 1
MUSHROOM_REWARD = 20
FLOWER_REWARD = 20
STAR_REWARD = 30
HEART_REWARD = 25
BOULDER_REWARD = 3
HIT_BOSS_REWARD = 10
KILL_BOSS_REWARD = 25
# TODO: is this necessary?
CHECKPOINT_REWARD = 15
LEVEL_CLEAR_REWARD = 35

# Random env settings
RANDOM_NOOP_FRAMES = 60
RANDOM_POWERUP_CHANCE = 25

# Training level selection settings
N_WARMUP_EVALS = 10
EVAL_WINDOW = 15
STD_COEF = 1.5
# probabilities before normalization
MIN_PROB = 0.01
MAX_PROB = 0.99


def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    # Force conversion to float
    _initial_value = float(initial_value)

    def _schedule(progress_remaining: float) -> float:
        return progress_remaining * _initial_value

    return _schedule


PPO_HYPERPARAMS = {
    "policy": "MultiInputPolicy",
    "batch_size": 512,
    "clip_range": 0.2,
    "ent_coef": 9.513020308749457e-06,
    "gae_lambda": 0.98,
    "gamma": 0.995,
    "learning_rate": 3e-05,
    "max_grad_norm": 5,
    "n_epochs": 5,
    "n_steps": 512,
    "vf_coef": 0.33653746631712467,
    "policy_kwargs": dict(
        activation_fn=nn.ReLU,
        features_extractor_class=MarioLandExtractor,
        features_extractor_kwargs=dict(
            # will be changed later
            device="auto",
        ),
        net_arch=dict(pi=[512, 512], vf=[512, 512]),
        normalize_images=False,
    ),
}
