from collections import deque
from typing import Any, Dict, List, Tuple

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from rl_playground.env_settings.env_settings import Orchestrator
from rl_playground.env_settings.super_mario_land.settings import *


levels = [
    "1-1e",
    "1-1h",
    "1-2e",
    "1-2h",
    "1-3e",
    "1-3h",
    "2-1e",
    "2-1h",
    "2-2e",
    "2-2h",
    "3-1e",
    "3-1h",
    "3-2e",
    "3-2h",
    "3-3e",
    "3-3h",
    "4-1e",
    "4-1h",
    "4-2e",
    "4-2h",
]

levelEndPositions = [
    2600,
    2600,
    2440,
    2440,
    2588,
    2588,
    2760,
    2760,
    2440,
    2440,
    3880,
    3880,
    2760,
    2760,
    2588,
    2588,
    3880,
    3880,
    3400,
    3400,
]


class MarioLandOrchestrator(Orchestrator):
    def __init__(self, env: VecEnv) -> None:
        self.levelProgress = [None] * len(levelEndPositions)

        self.warmup = N_WARMUP_EVALS
        self.window = EVAL_WINDOW
        self.stdCoef = STD_COEF
        self.minProb = MIN_PROB
        self.maxProb = MAX_PROB

        super().__init__(env)

    def processEvalInfo(self, info: Dict[str, Any]):
        level = info["worldLevel"]
        progress = info["levelProgress"]
        idx = levels.index(level)
        if self.levelProgress[idx] is not None:
            self.levelProgress[idx].addProgress(progress)
        else:
            self.levelProgress[idx] = LevelProgress(self.window, progress)

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
        if self.n_called >= self.warmup:
            probs = [0] * len(levelEndPositions)
            for idx, progress in enumerate(self.levelProgress):
                if progress == None:
                    continue

                p = progress.average
                if progress.stdDeviation > 0.0:
                    p -= progress.stdDeviation / self.stdCoef

                consistentProgress = 0.0
                if p != 0.0:
                    consistentProgress = p / levelEndPositions[idx]
                    consistentProgress = np.clip(1 - consistentProgress, self.minProb, self.maxProb)
                    print(f"{idx}: {consistentProgress}")

                probs[idx] = consistentProgress

            # normalize probabilities
            totalProb = sum(probs)
            probs = [prob / totalProb for prob in probs]
            print(probs)

            options = {"_update_level_choose_probs": probs}
            self.env.set_options(options)
            self.env.reset()

        super().postEval()


class LevelProgress:
    def __init__(self, window: int, progress: int) -> None:
        self.progresses = deque([progress], maxlen=window)
        self.average = float(progress)
        self.stdDeviation = 0.0

    def addProgress(self, progress: int):
        self.progresses.append(progress)
        self.average = np.mean(self.progresses)
        self.stdDeviation = np.std(self.progresses)
