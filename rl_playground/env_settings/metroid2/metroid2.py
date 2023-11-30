import itertools
import random
from os import listdir
from os.path import isfile, join
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from gymnasium.spaces import Box, Discrete, Space
from pyboy import PyBoy, WindowEvent
from pyboy.botsupport.constants import TILES

from rl_playground.env_settings.env_settings import EnvSettings, GameState

AREA_X_MEM_VAL = 0xD028
AREA_Y_MEM_VAL = 0xD02A
COORD_X_MEM_VAL = 0xD027
COORD_Y_MEM_VAL = 0xD029
HEALTH_MEM_VAL = 0xD051
E_TANKS_MEM_VAL = 0xD050
CURRENT_MISSLES_MEM_VAL = 0xD053
TOTAL_MISSLES_MEM_VAL = 0xD081
UPGRADES_MEM_VAL = 0xD045
METROIDS_REMAINING_MEM_VAL = 0xD09A

config = {
    "policy": "MlpPolicy",
    "gradient_steps": -1,
}


class Metroid2GameState(GameState):
    def __init__(self, pyboy: PyBoy):
        self.pyboy = pyboy

        self.area = (
            self._get_mem_value(AREA_X_MEM_VAL),
            self._get_mem_value(AREA_Y_MEM_VAL),
        )
        self.coords = (
            self._get_mem_value(COORD_X_MEM_VAL),
            self._get_mem_value(COORD_Y_MEM_VAL),
        )
        self.health = self._get_mem_value(HEALTH_MEM_VAL)
        self.eTanks = self._get_mem_value(E_TANKS_MEM_VAL)
        self.currentMissles = self._get_mem_value(CURRENT_MISSLES_MEM_VAL)
        self.totalMissles = self._get_mem_value(TOTAL_MISSLES_MEM_VAL)
        self.upgrades = self._get_mem_value(UPGRADES_MEM_VAL)
        self.metroidsRemaining = self._get_mem_value(METROIDS_REMAINING_MEM_VAL)

    def _get_mem_value(self, addr: int) -> int:
        return self.pyboy.get_memory_value(addr)

    def stats(self) -> dict:
        return {
            "x_area": self.area[0],
            "y_area": self.area[1],
            "x_coord": self.coords[0],
            "y_coord": self.coords[1],
            "health": self.health,
            "e_tanks": self.eTanks,
            "current_missles": self.currentMissles,
            "total_missles": self.totalMissles,
            "upgrades": self.upgrades,
            "metroids": self.metroidsRemaining,
        }


class Metroid2Settings(EnvSettings):
    def __init__(
        self, pyboy: PyBoy, isEval: bool, stateDir: Path = Path("states", "metroid2")
    ) -> None:
        self.pyboy = pyboy
        self.isEval = isEval
        # TODO: figure out how to get game area without this
        self.gameWrapper = self.pyboy.game_wrapper()
        self.evalStateCounter = 0
        self.stateFiles = sorted(
            [join(stateDir, f) for f in listdir(stateDir) if isfile(join(stateDir, f))]
        )

        self.reset()

    def reset(self, options: dict[str, Any] | None = None):
        self.seenAreas = {(7, 7)}
        self.ticks = 0
        self.secsSinceNewArea = 0
        self.curReward = 0
        self.maxReward = 0
        self.stagnantTicks = 0

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
        # ensure player has full health and missles
        self.pyboy.set_memory_value(HEALTH_MEM_VAL, 153)
        self.pyboy.set_memory_value(CURRENT_MISSLES_MEM_VAL, 45)

    def reward(self, prevState: Metroid2GameState) -> (float, Metroid2GameState):
        curState = self.gameState()

        # previous state was before the game started, don't set a reward
        if prevState.coords == (0, 0) and prevState.area == (0, 0):
            return 0, curState

        # Encourage finding new areas faster by subtracting the amount
        # of seconds it took to find the area. Ensure the reward for
        # finding a new area is never less than 5.
        self.ticks += 1
        if self.ticks % 60 == 0:
            if self.secsSinceNewArea < 45:
                self.secsSinceNewArea += 1
            # avoid overhead of bignum addition, only needs to count to 60
            self.ticks = 0

        newArea = 0
        if curState.area not in self.seenAreas:
            self.seenAreas.add(curState.area)
            newArea = 50 - (self.secsSinceNewArea)
            self.secsSinceNewArea = 0

        # punish losing health more than replenishing missles is rewarded
        health = 2 * (curState.health - prevState.health)
        # punish dying
        if curState.health == 0:
            health += -100

        missles = 0
        # reward replenished missles, don't punish using missles
        if curState.currentMissles > prevState.currentMissles:
            missles = curState.currentMissles - prevState.currentMissles
        minorUpgrades = 75 * (curState.eTanks - prevState.eTanks) - (
            curState.totalMissles - prevState.totalMissles
        )
        majorUpgrades = 200 * (curState.upgrades - prevState.upgrades)
        metroids = 500 * (prevState.metroidsRemaining - curState.metroidsRemaining)

        self.curReward = (
            newArea + health + missles + minorUpgrades + majorUpgrades + metroids
        )

        # update max reward if necessary
        maxReward = max(self.maxReward, self.curReward)
        if maxReward > self.maxReward:
            self.maxReward = maxReward

        # reset stagnant ticks if progress is still being made
        if self.curReward >= self.maxReward:
            self.stagnantTicks = 0
        else:
            self.stagnantTicks += 1

        return self.curReward, curState

    def observation(self, state: Metroid2GameState) -> Any:
        obs = self.gameWrapper._game_area_np()
        # make 20x16 array a 1x320 array so it's Box compatible
        obs = np.concatenate(obs.tolist(), axis=None, dtype=np.int32)
        # add player status, the array is now 1x324
        obs = np.append(
            obs,
            [
                state.health,
                state.currentMissles,
                state.metroidsRemaining,
                state.upgrades,
            ],
        )
        return obs

    def info(self, state: Metroid2GameState) -> dict[str, Any]:
        return {
            "health": state.health,
            "current_missles": state.currentMissles,
            "areas_found": len(self.seenAreas),
        }

    def terminated(self, gameState: Metroid2GameState) -> bool:
        return gameState.health == 0

    def truncated(self, gameState: Metroid2GameState) -> bool:
        # simulated 3m
        timeout = 10800

        if self.isEval:
            # simulated 30s
            timeout = 1800

        # return true if it's been a long time with no progress
        if (self.stagnantTicks) == timeout:
            return True
        return False

    def actionSpace(self) -> (List[WindowEvent], Discrete):
        baseActions = [
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_BUTTON_SELECT,
        ]

        totalActionsWithRepeats = list(itertools.permutations(baseActions, 2))
        withoutRepeats = []

        for combination in totalActionsWithRepeats:
            # remove useless action combinations
            if combination == (
                WindowEvent.PRESS_ARROW_UP,
                WindowEvent.PRESS_ARROW_DOWN,
            ) or combination == (
                WindowEvent.PRESS_ARROW_LEFT,
                WindowEvent.PRESS_ARROW_RIGHT,
            ):
                continue
            reversedCombination = combination[::-1]
            if reversedCombination not in withoutRepeats:
                withoutRepeats.append(combination)

        filteredActions = [[action] for action in baseActions] + withoutRepeats

        return filteredActions, Discrete(len(filteredActions))

    def observationSpace(self) -> Space:
        elements = (20 * 16) + 4
        return Box(low=0, high=TILES, shape=(elements,), dtype=np.int32)

    def hyperParameters(self, algo: str) -> Dict[str, Any]:
        return config

    def evalEpisodes(self) -> int:
        return len(self.stateFiles)

    def gameState(self) -> Metroid2GameState:
        return Metroid2GameState(self.pyboy)

    def printGameState(self, state: Metroid2GameState) -> None:
        print(
            f"health: {state.health} - ",
            f"e-tanks: {state.eTanks} - ",
            f"missles: {state.currentMissles} - ",
            f"total missles: {state.totalMissles} - ",
            f"upgrades: {state.upgrades} - ",
            f"metroids: {state.metroidsRemaining} - ",
            f"stagnant ticks: {self.stagnantTicks} - ",
            f"current reward: {self.curReward} - ",
            f"max reward: {self.maxReward}",
        )

    def render(self):
        return self.pyboy.screen_image()
