import logging
from pathlib import Path

from gymnasium.wrappers.frame_stack import FrameStack
from pyboy import PyBoy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from rl_playground.env_settings.metroid2 import Metroid2Settings
from rl_playground.env_settings.super_mario_land import MarioLandSettings
from rl_playground.envs.pyboy.pyboy_env import PyBoyEnv
from rl_playground.envs.pyboy.wrappers import SkipFrame


def createPyboyEnv(
    rom: Path,
    render: bool = False,
    speed: int = 0,
    isEval: bool = False,
    isPlaytest: bool = False,
    isHyperparamOptimize: bool = False,
    outputDir: Path = "",
):
    # silence useless pyboy logs
    logging.getLogger("pyboy.pyboy").setLevel(logging.ERROR)
    logging.getLogger("pyboy.plugins.window_headless").setLevel(logging.ERROR)
    if isPlaytest:
        logging.getLogger("pyboy.plugins.debug").setLevel(logging.INFO)

    shouldRender = not isHyperparamOptimize and (render or isEval)
    pyboy = PyBoy(
        rom,
        window_type="SDL2" if render else "headless",
        scale=4,
        debug=isPlaytest,
        game_wrapper=True,
    )
    pyboy._rendering(shouldRender)
    pyboy.set_emulation_speed(speed)

    envSettings = None
    if pyboy.cartridge_title() == "METROID2":
        envSettings = Metroid2Settings(pyboy, isEval)
    else:
        envSettings = MarioLandSettings(pyboy, isEval)

    env = PyBoyEnv(
        pyboy,
        envSettings,
        render == render,
        isEval=isEval,
        isPlaytest=isPlaytest,
        outputDir=outputDir,
    )

    env = Monitor(env)
    env = SkipFrame(env, skip=4)
    env = FrameStack(env, num_stack=4)

    # check_env(env)

    return envSettings, env
