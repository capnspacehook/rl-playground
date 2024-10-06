from collections import deque
from typing import Any, Dict, List, Tuple

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from rl_playground.env_settings.env_settings import Orchestrator
from rl_playground.env_settings.super_mario_land.settings import *


class MarioLandOrchestrator(Orchestrator):
    def __init__(self, env: VecEnv) -> None:
        self.newMaxLevel = False
        self.maxLevel = "1-1"

        super().__init__(env)

    def preEval(self):
        self.newMaxLevel = False

    def processEvalInfo(self, info: Dict[str, Any]):
        level = info["worldLevel"]
        if level > self.maxLevel:
            print(f"New max level: {level}")
            self.newMaxLevel = True
            self.maxLevel = level

    def evalInfoLogEntries(self, info: Dict[str, Any]) -> List[Tuple[str, Any]]:
        return [
            (f"{info['worldLevel']}_progress", info["levelProgress"]),
            (f"{info['worldLevel']}_deaths", info["deaths"]),
            (f"{info['worldLevel']}_hearts", info["hearts"]),
            (f"{info['worldLevel']}_powerups", info["powerups"]),
            (f"{info['worldLevel']}_coins", info["coins"]),
            (f"{info['worldLevel']}_score", info["score"]),
        ]

    def postEval(self):
        if self.newMaxLevel:
            options = {"_update_max_level": self.maxLevel}
            self.env.set_options(options)
            self.env.reset()
