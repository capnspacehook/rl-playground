import logging
from pathlib import Path
from typing import Dict

from gymnasium.experimental.wrappers.stateful_action import StickyActionV0
from pyboy import PyBoy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from rl_playground.env_settings.metroid2.metroid2 import Metroid2Settings
from rl_playground.env_settings.super_mario_land.super_mario_land import MarioLandSettings
from rl_playground.env_settings.super_mario_land.orchestrator import (
    MarioLandOrchestrator,
)
from rl_playground.envs.pyboy.pyboy_env import PyBoyEnv
from rl_playground.envs.pyboy.wrappers import FrameSkip


def createPyboyEnv(
    rom: Path,
    render: bool = False,
    speed: int = 0,
    isEval: bool = False,
    isPlaytest: bool = False,
    isInteractiveEval: bool = False,
    isHyperparamOptimize: bool = False,
    outputDir: Path = "",
    envKwargs: Dict = {},
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
        debug=False,
        game_wrapper=True,
    )
    pyboy._rendering(shouldRender)
    pyboy.set_emulation_speed(speed)

    envSettings = None
    orchestrator = None
    if pyboy.cartridge_title() == "METROID2":
        envSettings = Metroid2Settings(pyboy, isEval, **envKwargs)
    else:
        envSettings = MarioLandSettings(pyboy, isEval, **envKwargs)
        orchestrator = MarioLandOrchestrator

    env = PyBoyEnv(
        pyboy,
        envSettings,
        render == render,
        isEval=isEval,
        isPlaytest=isPlaytest,
        isInteractiveEval=isInteractiveEval,
        outputDir=outputDir,
    )

    env = Monitor(env)
    env = FrameSkip(env, skip=4)
    # make evaluations strictly deterministic
    if not isEval:
        env = StickyActionV0(env, 0.25)

    # check_env(env)

    return envSettings, env, orchestrator
