# rl-playground

A collection of my reinforcement learning projects and experiments.

## Patching pyboy

Small modifications to pyboy (game boy emulator) have been made to accomplish tasks that aren't currently possible with pyboy v1. When v2 is released I will probably be able to drop my modified version. For now though you'll simply have to patch and compile pyboy yourself:

```sh
# in directory of cloned pyboy
cp path_to_rl_playground/pyboy.patch .
git apply pyboy.patch
make install
```

## Getting started

Tested with Python 3.10 and 3.11 on Linux, I'm unsure if another Python version or OS will work.

`main.py` can train and evaluate models and test an environment.

To start training a model run:

`python3 main.py -t -a ppo -l <RUN_STEP_LENGTH> -n <RUN_NAME> <PATH_TO_ROM>`

`optimize.py` can find optimized hyperparameters.

Note that currently only PPO is supported.

# Current projects

- [Super Mario Land](rl_playground/env_settings/super_mario_land/)
