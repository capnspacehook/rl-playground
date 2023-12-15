from typing import Callable, Union

import torch.nn as nn

from rl_playground.env_settings.super_mario_land.extractor import MarioLandExtractor

# Observation settings
N_OBS_STACK = 6  # number of observations to stack
N_STATE_STACK = 6  # number of games states to use to calculate mean speeds

# General env settings
STARTING_TIME = 400

# Reward values
HIT_PUNISHMENT = -5.0
DEATH_PUNISHMENT = -30.0
GAME_OVER_PUNISHMENT = -50
# TODO: linearly increase to encourage progress over speed
# earlier, then after game mechanics and levels are learned
# encourage speed more
CLOCK_PUNISHMENT = -0.01
FORWARD_REWARD_COEF = 0.25
BACKWARD_PUNISHMENT_COEF = 0.1
PROGRESS_REWARD_COEF = 0.75
MUSHROOM_REWARD = 20
# TODO: add reward for killing enemies with fireballs
FLOWER_REWARD = 20
STAR_REWARD = 25
HEART_REWARD = 30
BOULDER_REWARD = 3
HIT_BOSS_REWARD = 5
KILL_BOSS_REWARD = 25
LEVEL_CLEAR_REWARD = 30
LEVEL_CLEAR_LIVES_COEF_REWARD = 10
LEVEL_CLEAR_BIG_REWARD = 5
LEVEL_CLEAR_FIRE_REWARD = 10
LEVEL_CLEAR_TIME_COEF_REWARD = 0.00125

# Random env settings
RANDOM_NOOP_FRAMES = 60
RANDOM_POWERUP_CHANCE = 25
STARTING_LIVES_MIN = 1
STARTING_LIVES_MAX = 3

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
    "batch_size": 512,  # TODO: try 256
    "clip_range": 0.2,
    "ent_coef": 1e-02,
    "gae_lambda": 0.98,
    "gamma": 0.995,
    "learning_rate": linear_schedule(3e-05),
    "max_grad_norm": 1,
    "n_epochs": 5,
    "n_steps": 1024,
    "vf_coef": 0.5,
    "policy_kwargs": dict(
        activation_fn=nn.ReLU,
        features_extractor_class=MarioLandExtractor,
        features_extractor_kwargs=dict(
            # will be changed later
            device="auto",
        ),
        net_arch=dict(pi=[512, 512], vf=[512, 512]),
        normalize_images=False,
        share_features_extractor=True,
    ),
}
