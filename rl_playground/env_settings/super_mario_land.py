import itertools
from typing import Any, Dict
from os import listdir
from os.path import isfile, join
import random
from pathlib import Path

import numpy as np
import torch.nn as nn
from gymnasium.spaces import Box, Discrete, Space
from pyboy import PyBoy, WindowEvent
from pyboy.botsupport.constants import TILES

from rl_playground.env_settings.env_settings import EnvSettings, GameState


qrdqnConfig = {
    "policy": "MlpPolicy",
    "batch_size": 100,
    "buffer_size": 100000,
    "exploration_final_eps": 0.010561494547433554,
    "exploration_fraction": 0.1724714484114384,
    "gamma": 0.95,
    "gradient_steps": -1,
    "learning_rate": 1.7881224306668102e-05,
    "learning_starts": 10000,
    "target_update_interval": 20000,
    "train_freq": 256,
    "tau": 1.0,
    "policy_kwargs": {
        "net_arch": [256, 256],
        "n_quantiles": 166,
    },
}

ppoConfig = {
    "policy": "MlpPolicy",
    "batch_size": 512,
    "clip_range": 0.2,
    "ent_coef": 1.080365148093321e-05,
    "gae_lambda": 0.8,
    "gamma": 0.98,
    "learning_rate": 6.160438419274751e-05,
    "max_grad_norm": 0.9,
    "n_epochs": 10,
    "n_steps": 256,
    "vf_coef": 0.21730023144009505,
    "policy_kwargs": dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    ),
}

STATUS_TIMER_MEM_VAL = 0xFFA6
DEAD_JUMP_TIMER_MEM_VAL = 0xC0AC
POWERUP_STATUS_MEM_VAL = 0xFF99
HAS_FIRE_FLOWER_MEM_VAL = 0xFFB5
STAR_TIMER_MEM_VAL = 0xC0D3

STATUS_SMALL = 0
STATUS_BIG = 1
STATUS_FIRE = 2
STATUS_STAR = 3
STATUS_INVINCIBLE = 4

TIMER_DEATH = 0x90
TIMER_LEVEL_CLEAR = 0xF0


class MarioLandGameState(GameState):
    def __init__(self, pyboy: PyBoy):
        self.pyboy = pyboy
        self.gameWrapper = pyboy.game_wrapper()

        # Find the real level progress x
        levelBlock = pyboy.get_memory_value(0xC0AB)
        # C202 Mario's X position relative to the screen
        marioX = pyboy.get_memory_value(0xC202)
        scx = pyboy.botsupport_manager().screen().tilemap_position_list()[16][0]
        real = (scx - 7) % 16 if (scx - 7) % 16 != 0 else 16

        self.realXPos = levelBlock * 16 + real + marioX
        self.timeLeft = self.gameWrapper.time_left
        self.livesLeft = self.gameWrapper.lives_left
        self.score = self.gameWrapper.score
        self.levelProgressMax = max(self.gameWrapper._level_progress_max, self.realXPos)
        self.world = self.gameWrapper.world
        self.statusTimer = self.pyboy.get_memory_value(STATUS_TIMER_MEM_VAL)
        self.deadJumpTimer = self.pyboy.get_memory_value(DEAD_JUMP_TIMER_MEM_VAL)

        powerupStatus = self.pyboy.get_memory_value(POWERUP_STATUS_MEM_VAL)
        hasFireFlower = self.pyboy.get_memory_value(HAS_FIRE_FLOWER_MEM_VAL)
        starTimer = self.pyboy.get_memory_value(STAR_TIMER_MEM_VAL)
        if starTimer != 0:
            self.powerupStatus = STATUS_STAR
        elif powerupStatus == 0 or powerupStatus == 3:
            self.powerupStatus = STATUS_SMALL
        elif powerupStatus == 1:
            self.powerupStatus = STATUS_BIG
        elif powerupStatus == 2:
            if hasFireFlower:
                self.powerupStatus = STATUS_FIRE
            else:
                self.powerupStatus = STATUS_BIG
        elif powerupStatus == 4:
            self.powerupStatus = STATUS_INVINCIBLE


class MarioLandSettings(EnvSettings):
    def __init__(
        self,
        pyboy: PyBoy,
        isEval: bool,
        stateDir: Path = Path("states", "super_mario_land"),
    ):
        self.pyboy = pyboy
        self.gameWrapper = self.pyboy.game_wrapper()
        self.isEval = isEval
        self.evalStateCounter = 0
        self.evalNoProgress = 0
        self.stateFiles = sorted(
            [join(stateDir, f) for f in listdir(stateDir) if isfile(join(stateDir, f))]
        )

    def reset(self, options: dict[str, Any] | None = None):
        # this will be passed before evals are started, reset the eval
        # state counter so all evals will start at the same state
        if options is not None and options["_eval_starting"]:
            self.evalStateCounter = 0
            return

        self.evalNoProgress = 0

        # reset game state
        state = random.choice(self.stateFiles)
        if self.isEval:
            state = self.stateFiles[self.evalStateCounter]
            self.evalStateCounter += 1
            if self.evalStateCounter == len(self.stateFiles):
                self.evalStateCounter = 0
        with open(state, "rb") as f:
            self.pyboy.load_state(f)

        # seed randomizer
        self.gameWrapper._set_timer_div(None)

    def reward(self, prevState: MarioLandGameState) -> (float, MarioLandGameState):
        curState = self.gameState()

        # return flat punishment on mario's death
        if self._isDead(curState):
            return -50, curState

        # handle level clear
        if curState.statusTimer == TIMER_LEVEL_CLEAR:
            # advance until time left starts getting added to score
            for _ in range(curState.statusTimer + 64):
                self.pyboy.tick()
            # advance until time left has been depleted
            for _ in range(curState.timeLeft):
                self.pyboy.tick()
            # advance until next level loads
            # get updated timer value
            for _ in range(self.pyboy.get_memory_value(STATUS_TIMER_MEM_VAL) + 11):
                self.pyboy.tick()

            curState = self.gameState()

        # reset level progress max on new level
        if self._levelCleared(prevState, curState):
            self.gameWrapper._level_progress_max = curState.realXPos
            curState._levelProgressMax = curState.realXPos
            return 50, curState

        # add time punishment every step to encourage speed more
        clock = -0.25
        movement = curState.realXPos - prevState.realXPos

        if self.isEval:
            if curState.levelProgressMax - prevState.levelProgressMax == 0:
                self.evalNoProgress += 1
            else:
                self.evalNoProgress = 0

        powerup = 0
        # if mario is briefly invincible after getting hit set his status
        # to what it was before
        if (
            curState.powerupStatus != prevState.powerupStatus
            and curState.powerupStatus == STATUS_INVINCIBLE
        ):
            curState.powerupStatus = prevState.powerupStatus

        # don't punish or reward the star invincibility running out
        if (
            curState.powerupStatus != prevState.powerupStatus
            and prevState.powerupStatus != STATUS_STAR
        ):
            if curState.powerupStatus == STATUS_SMALL:
                powerup = -10
            elif curState.powerupStatus == STATUS_BIG:
                powerup = 10
            elif curState.powerupStatus == STATUS_FIRE:
                powerup = 20
            elif curState.powerupStatus == STATUS_STAR:
                powerup = 30

        reward = clock + movement + powerup

        return reward, curState

    def _levelCleared(
        self, prevState: MarioLandGameState, curState: MarioLandGameState
    ) -> bool:
        return max(
            curState.world[0] - prevState.world[0],
            curState.world[1] - prevState.world[1],
        )

    def observation(self, gameState: MarioLandGameState) -> Any:
        obs = self.gameWrapper._game_area_np()
        # make 20x16 array a 1x320 array so it's Box compatible
        flatObs = np.concatenate(obs.tolist(), axis=None, dtype=np.int32)
        # add powerup status
        return np.append(flatObs, [gameState.powerupStatus])

    def terminated(
        self, prevState: MarioLandGameState, curState: MarioLandGameState
    ) -> bool:
        return self._isDead(curState)

    def truncated(self, prevState: GameState, curState: GameState) -> bool:
        # if no forward progress has been made in 15s, stop the eval episode
        return self.isEval and self.evalNoProgress == 900

    def _isDead(self, curState: MarioLandGameState) -> bool:
        return curState.deadJumpTimer != 0 or curState.statusTimer == TIMER_DEATH

    def actionSpace(self):
        baseActions = [
            WindowEvent.PASS,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
        ]

        totalActionsWithRepeats = list(itertools.permutations(baseActions, 2))
        withoutRepeats = []

        for combination in totalActionsWithRepeats:
            # remove useless action combinations
            if (
                isinstance(combination, tuple)
                and WindowEvent.PASS in combination
                or combination
                == (
                    WindowEvent.PRESS_ARROW_LEFT,
                    WindowEvent.PRESS_ARROW_RIGHT,
                )
                or combination
                == (
                    WindowEvent.PRESS_ARROW_RIGHT,
                    WindowEvent.PRESS_ARROW_LEFT,
                )
            ):
                continue
            reversedCombination = combination[::-1]
            if reversedCombination not in withoutRepeats:
                withoutRepeats.append(combination)

        filteredActions = [[action] for action in baseActions] + withoutRepeats

        return filteredActions, Discrete(len(filteredActions))

    def observationSpace(self) -> Space:
        # game area
        size = 20 * 16
        b = Box(low=0, high=TILES, shape=(size,), dtype=np.int32)
        # add space for powerup status
        low = np.append(b.low, [0])
        high = np.append(b.high, [3])
        return Box(low=low, high=high, dtype=np.int32)

    def normalizeObservation(self) -> bool:
        return True

    def hyperParameters(self, algo: str) -> Dict[str, Any]:
        match algo:
            case "qrdqn":
                return qrdqnConfig
            case "ppo":
                return ppoConfig
            case _:
                return {}

    def evalEpisodes(self) -> int:
        return len(self.stateFiles)

    def gameState(self):
        return MarioLandGameState(self.pyboy)

    def printGameState(self, state: MarioLandGameState):
        print(f"Fake level progress: {self.gameWrapper.level_progress}")
        print(f"Real level progress: {state.realXPos}")
        print(f"Max level progress: {state.levelProgressMax}")
        print(f"Lives left: {state.livesLeft}")
        print(f"Powerup: {state.powerupStatus}")
        print(f"World: {state.world}")
        print(f"Time left: {state.timeLeft}")
        print(f"Time respawn: {state.statusTimer}")

    def render(self):
        return self.pyboy.screen_image()
