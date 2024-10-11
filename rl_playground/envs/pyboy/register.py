from pathlib import Path
from typing import Dict

from gymnasium.experimental.wrappers.stateful_action import StickyActionV0
from pyboy import PyBoy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

# from rl_playground.env_settings.metroid2.metroid2 import Metroid2Settings
from rl_playground.env_settings.super_mario_land.super_mario_land import MarioLandSettings
from rl_playground.env_settings.super_mario_land.orchestrator import (
    MarioLandOrchestrator,
)
from rl_playground.envs.pyboy.pyboy_env import PyBoyEnv
from rl_playground.envs.pyboy.wrappers import FrameSkip, Recorder


def createPyboyEnv(
    rom: Path,
    envID: int | None = None,
    render: bool = False,
    speed: int = 0,
    isEval: bool = False,
    isPlaytest: bool = False,
    isInteractiveEval: bool = False,
    isHyperparamOptimize: bool = False,
    outputDir: Path = "",
    envKwargs: Dict = {},
):
    debug = False
    logLvl = "ERROR"
    if isPlaytest:
        debug = True
        logLvl = "INFO"

    pyboy = PyBoy(
        rom,
        window="SDL2" if render else "null",
        scale=4,
        debug=debug,
        log_level=logLvl,
    )
    pyboy.set_emulation_speed(speed)

    envSettings = None
    orchestrator = None

    envSettings = MarioLandSettings(pyboy, envID, isEval, **envKwargs)
    orchestrator = MarioLandOrchestrator

    env = PyBoyEnv(
        pyboy,
        envSettings,
        render=render,
        isEval=isEval,
        isPlaytest=isPlaytest,
        isInteractiveEval=isInteractiveEval,
        outputDir=outputDir,
    )

    env = Monitor(env)
    if isEval:
        env = Recorder(env, episode_num=envSettings.evalEpisodes())
    env = FrameSkip(env, skip=4)
    # make evaluations strictly deterministic
    # if not isEval:
    #     env = StickyActionV0(env, 0.25)

    # check_env(env)

    return envSettings, env, orchestrator
