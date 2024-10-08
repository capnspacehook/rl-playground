from typing import Callable, Union

import torch.nn as nn

from rl_playground.env_settings.super_mario_land.extractor import MarioLandExtractor

# Observation settings
N_GAME_AREA_STACK = 6  # number of game area observaions to stack
N_MARIO_OBS_STACK = 6  # number of mario observations to stack
N_ENTITY_OBS_STACK = 4  # number of entity observations to stack
N_SCALAR_OBS_STACK = 2  # number of scalar observations to stack
N_STATE_STACK = 6  # number of games states to use to calculate mean speeds

# Time settings
MIN_TIME = 10
STARTING_TIME = 400
MIN_RANDOM_TIME = 60
DEATH_TIME_LOSS = 10

# Reward values
HIT_PUNISHMENT = -10
DEATH_SCALE = -10
DEATH_PUNISHMENT = -30
GAME_OVER_PUNISHMENT = -50
# TODO: linearly increase to encourage progress over speed
# earlier, then after game mechanics and levels are learned
# encourage speed more
CLOCK_PUNISHMENT = -0.01
SCORE_REWARD_COEF = 0.01
COIN_REWARD = 2  # +100 score when getting a coin must be factored in
FORWARD_REWARD_COEF = 1
BACKWARD_PUNISHMENT_COEF = 0.25
MUSHROOM_REWARD = 15  # 1000 score
# TODO: add reward for killing enemies with fireballs
FLOWER_REWARD = 15  # 1000 score
# TODO: make reward when star bug is fixed/mitigated
STAR_REWARD = -35  # 1000 score
HEART_REWARD = 30  # 1000 score
HEART_FARM_PUNISHMENT = -60
MOVING_PLATFORM_DISTANCE_REWARD_MAX = 0.5
MOVING_PLATFORM_X_REWARD_COEF = 0.15
MOVING_PLATFORM_Y_REWARD_COEF = 1.25
BOULDER_REWARD = 5
HIT_BOSS_REWARD = 5
KILL_BOSS_REWARD = 25
# TODO: handle score getting updated
LEVEL_CLEAR_REWARD = 35
LEVEL_CLEAR_TOP_REWARD = 20
LEVEL_CLEAR_LIVES_COEF_REWARD = 5
LEVEL_CLEAR_BIG_REWARD = 5
LEVEL_CLEAR_FIRE_REWARD = 10

# Random env settings
RANDOM_NOOP_FRAMES = 60
RANDOM_NOOP_FRAMES_WITH_ENEMIES = 10
RANDOM_POWERUP_CHANCE = 25
STARTING_LIVES_MIN = 1
STARTING_LIVES_MAX = 3

# Heart farming detection settings
HEART_FARM_X_POS_MULTIPLE = 15

# Cell selection settings
X_POS_MULTIPLE = 150
Y_POS_MULTIPLE = 20
FRAME_CELL_CHECK = 8


def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    # Force conversion to float
    _initial_value = float(initial_value)

    def _schedule(progress_remaining: float) -> float:
        return progress_remaining * _initial_value

    return _schedule


PPO_HYPERPARAMS = {
    "policy": "MultiInputPolicy",
    "batch_size": 512,  # minibatch size
    "clip_range": 0.2,
    "ent_coef": 7e-03,
    "gae_lambda": 0.98,
    "gamma": 0.995,
    "learning_rate": 3e-05,
    "max_grad_norm": 1,
    "n_epochs": 5,
    "n_steps": 2048,  # horizon = n_steps * n_envs
    "vf_coef": 0.5,
    "policy_kwargs": dict(
        activation_fn=nn.ReLU,
        features_extractor_class=MarioLandExtractor,
        features_extractor_kwargs=dict(
            # will be changed later
            device="auto",
        ),
        net_arch=dict(
            # policy NN size and layers
            pi=[2048, 2048],
            # value NN size and layers
            vf=[2048, 2048],
        ),
        normalize_images=False,
        share_features_extractor=True,
    ),
}
