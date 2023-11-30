from typing import Any, Callable, Dict, List, Union
from os import listdir
from os.path import basename, isfile, join, splitext
import random
from pathlib import Path

import numpy as np
import torch.nn as nn
from gymnasium.spaces import Box, Discrete, Space
from pyboy import PyBoy, WindowEvent
from pyboy.botsupport.constants import TILES

from rl_playground.env_settings.env_settings import EnvSettings
from rl_playground.env_settings.super_mario_land.game_area import (
    bouncing_boulder_tiles,
    worldTilesets,
)
from rl_playground.env_settings.super_mario_land.ram import *


# Reward values
DEATH_PUNISHMENT = -25
HIT_PUNISHMENT = -5
# TODO: linearly increase to encourage progress over speed
# earlier, then after game mechanics and levels are learned
# encourage speed more
CLOCK_PUNISHMENT = -0.01
MOVEMENT_REWARD_COEF = 1
MUSHROOM_REWARD = 20
FLOWER_REWARD = 20
STAR_REWARD = 30
MOVING_PLATFORM_REWARD = 7.5
BOULDER_REWARD = 3
HIT_BOSS_REWARD = 10
KILL_BOSS_REWARD = 25
CHECKPOINT_REWARD = 25

# Random env settings
RANDOM_NOOP_FRAMES = 60
RANDOM_POWERUP_CHANCE = 25


def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    # Force conversion to float
    _initial_value = float(initial_value)

    def _schedule(progress_remaining: float) -> float:
        return progress_remaining * _initial_value

    return _schedule


ppoConfig = {
    "policy": "MlpPolicy",
    "batch_size": 512,
    "clip_range": 0.2,
    "ent_coef": 9.513020308749457e-06,
    "gae_lambda": 0.98,
    "gamma": 0.995,
    "learning_rate": 3e-05,
    "max_grad_norm": 5,
    "n_epochs": 5,
    "n_steps": 512,
    "vf_coef": 0.33653746631712467,
    "policy_kwargs": dict(
        activation_fn=nn.ReLU,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    ),
}


# Game area dimensions
GAME_AREA_HEIGHT = 16
GAME_AREA_WIDTH = 20


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
        self.tileSet = None
        self.stateIdx = 0
        self.evalStateCounter = 0
        self.evalNoProgress = 0
        self.invincibilityTimer = 0

        self.stateFiles = sorted(
            [join(stateDir, f) for f in listdir(stateDir) if isfile(join(stateDir, f))]
        )
        self.stateCheckpoint = 0
        self.currentCheckpoint = 0
        self.nextCheckpointRewardAt = 0
        # TODO: number of checkpoints should be able to differ between levels
        self.levelCheckpointRewards = {
            (1, 1): [950, 1605],
            (1, 2): [1150, 1840],
            (1, 3): [1065, 1927],
            (2, 1): [855, 1610],
            (2, 2): [870, 1975],
            (3, 1): [2130, 2784],
            (3, 2): [930, 1980],
            (3, 3): [705, 1775],
            (4, 1): [865, 1970],
            (4, 2): [980, 2150],
        }

        chooseProb = len(self.levelCheckpointRewards) / 100.0
        # all levels are equally likely to be trained on at the start
        self.levelChooseProbs = [
            chooseProb for _ in range(len(self.levelCheckpointRewards))
        ]

        self.onGroundFor = 0
        self.movingPlatformJumpState = None

        # so level progress max will be set
        self.gameWrapper.start_game()

    def reset(
        self, options: dict[str, Any] | None = None
    ) -> (MarioLandGameState, bool):
        if options is not None:
            if "_eval_starting" in options:
                # this will be passed before evals are started, reset the eval
                # state counter so all evals will start at the same state
                self.evalStateCounter = 0
            elif "_update_level_choose_probs" in options:
                # an eval has ended, update the level choose probabilities
                self.levelChooseProbs = options["_update_level_choose_probs"]

            return self.gameState(), False

        self.onGroundFor = 0
        self.movingPlatformJumpState = None

        if self.isEval:
            # evaluate levels in order
            self.stateIdx = self.evalStateCounter
            # don't start from checkpoints
            self.evalStateCounter += 3
            if self.evalStateCounter >= len(self.stateFiles):
                self.evalStateCounter = 0
        else:
            # reset game state to a random level
            level = np.random.choice(10, p=self.levelChooseProbs)
            checkpoint = np.random.randint(3)
            self.stateIdx = (3 * level) + checkpoint

        return self._loadLevel(), True

    def _loadLevel(self) -> MarioLandGameState:
        stateFile = self.stateFiles[self.stateIdx]
        with open(stateFile, "rb") as f:
            self.pyboy.load_state(f)

        # get checkpoint number from state filename
        stateFile = basename(splitext(stateFile)[0])
        world = int(stateFile[0])
        level = int(stateFile[2])
        self.stateCheckpoint = int(stateFile[4])
        self.currentCheckpoint = self.stateCheckpoint

        self._setNextCheckpointRewardAt((world, level), self.stateCheckpoint)
        self.tileSet = worldTilesets[world]

        # seed randomizer
        self.gameWrapper._set_timer_div(None)

        # reset max level progress
        self.gameWrapper._level_progress_max = 0
        self.evalNoProgress = 0

        # if we're starting at a level checkpoint do nothing for a random
        # amount of frames to make object placements varied
        if self.stateCheckpoint != 0:
            for _ in range(random.randint(0, RANDOM_NOOP_FRAMES)):
                self.pyboy.tick()

        # occasionally randomly set mario's powerup status so the NN
        # can learn to use the powerups; also makes the environment more
        # stochastic
        curState = self.gameState()
        if not self.isEval:
            if random.randint(0, 100) < RANDOM_POWERUP_CHANCE:
                # 0: small with star
                # 1: big
                # 2: big with star
                # 3: fire flower
                # 4: fire flower with star
                gotStar = False
                randPowerup = random.randint(0, 4)
                # TODO: change back when pyboy bug is fixed
                if False:  # randPowerup in (0, 2, 4):
                    gotStar = True
                    self.pyboy.set_memory_value(STAR_TIMER_MEM_VAL, 0xF8)
                    # set star song so timer functions correctly
                    self.pyboy.set_memory_value(0xDFE8, 0x0C)
                if randPowerup != STATUS_SMALL:
                    self.pyboy.set_memory_value(POWERUP_STATUS_MEM_VAL, STATUS_BIG)
                    if randPowerup > 2:
                        self.pyboy.set_memory_value(HAS_FIRE_FLOWER_MEM_VAL, 1)

                prevState = curState
                curState = self.gameState()
                if gotStar:
                    curState.gotStar = True
                    curState.hasStar = True
                    curState.isInvincible = True
                self._handlePowerup(prevState, curState)

        # level checkpoints get less time
        timerHundreds = 4 - self.stateCheckpoint

        # set level timer
        self.pyboy.set_memory_value(0xDA00, 0x28)
        self.pyboy.set_memory_value(0xDA01, 0)
        self.pyboy.set_memory_value(0xDA02, timerHundreds)

        return curState

    def _setNextCheckpointRewardAt(self, world, currentCheckpoint):
        self.nextCheckpointRewardAt = 0
        if currentCheckpoint < 2:
            self.nextCheckpointRewardAt = self.levelCheckpointRewards[world][
                self.stateCheckpoint
            ]

    def reward(self, prevState: MarioLandGameState) -> (float, MarioLandGameState):
        curState = self.gameState()

        # return flat punishment on mario's death
        if self._isDead(curState):
            return DEATH_PUNISHMENT, curState

        # handle level clear
        if curState.statusTimer == TIMER_LEVEL_CLEAR:
            # if we're evaluating and the level is cleared return, the
            # episode is over; otherwise load the next level directly
            # to avoid processing unnecessary frames and the AI playing
            # levels we don't want it to
            if self.isEval:
                return CHECKPOINT_REWARD, curState
            else:
                # load start of next level, not a level checkpoint
                self.stateIdx += (2 - self.stateCheckpoint) + 1
                if self.stateIdx >= len(self.stateFiles):
                    self.stateIdx = 0

                curState = self._loadLevel()

            # reset level progress max on new level
            self.gameWrapper._level_progress_max = curState.xPos
            curState.levelProgressMax = curState.xPos

            return CHECKPOINT_REWARD, curState

        # add time punishment every step to encourage speed more
        clock = CLOCK_PUNISHMENT
        movement = (curState.xPos - prevState.xPos) * MOVEMENT_REWARD_COEF

        # the game registers mario as on the ground 1 or 2 frames before
        # he actually is to change his pose
        if not curState.onGround:
            self.onGroundFor = 0
        elif self.onGroundFor < 2:
            self.onGroundFor += 1
        onGround = self.onGroundFor == 2

        # reward jumping from moving platform to another block mario
        # can stand on, but don't reward jumping on the same moving platform
        movingPlatform = self._handleMovingPlatform(onGround, prevState, curState)

        # in world 3 reward standing on bouncing boulders to encourage
        # waiting for them to fall and ride on them instead of immediately
        # jumping into spikes, but only if the boulders are moving to the
        # right
        standingOnBoulder = 0
        if (
            curState.world[0] == 3
            and onGround
            and movement > 0
            and self._standingOnTiles(bouncing_boulder_tiles)
        ):
            standingOnBoulder = BOULDER_REWARD

        # reward for passing checkpoints
        checkpoint = 0
        if (
            curState.levelProgressMax != prevState.levelProgressMax
            and self.nextCheckpointRewardAt != 0
            and curState.levelProgressMax >= self.nextCheckpointRewardAt
        ):
            self.currentCheckpoint += 1
            self._setNextCheckpointRewardAt(curState.world, self.currentCheckpoint)
            checkpoint = CHECKPOINT_REWARD

        # keep track of how long the agent is idle so we can end early
        # in an evaluation
        if self.isEval:
            if curState.levelProgressMax - prevState.levelProgressMax == 0:
                self.evalNoProgress += 1
            else:
                self.evalNoProgress = 0

        # reward getting powerups and manage powerup related bookkeeping
        powerup = self._handlePowerup(prevState, curState)

        # reward damaging or killing a boss
        boss = 0
        if curState.bossActive and curState.bossHealth < prevState.bossHealth:
            if prevState.bossHealth - curState.bossHealth == 1:
                boss = HIT_BOSS_REWARD
            elif curState.bossHealth == 0:
                boss = KILL_BOSS_REWARD

        reward = (
            clock
            + movement
            + movingPlatform
            + standingOnBoulder
            + checkpoint
            + powerup
            + boss
        )

        return reward, curState

    def _handleMovingPlatform(
        self,
        onGround: bool,
        prevState: MarioLandGameState,
        curState: MarioLandGameState,
    ) -> float:
        if onGround and self.movingPlatformJumpState != None:
            jumpPlatformObj = self.movingPlatformJumpState.movingPlatformObj
            landPlatformObj, onMovingPlatform = self._standingOnMovingPlatform(curState)
            jumpPlatformWidth = 30
            if jumpPlatformObj.typeID in (58, 59):
                jumpPlatformWidth = 20

            # only reward landing from a moving platform if mario made
            # forward progress to prevent rewarding from jumping backwards
            # or continually jumping from the same platform
            if (
                curState.levelProgressMax
                > self.movingPlatformJumpState.levelProgressMax
            ):
                # Only reward jumping from a moving platform and landing
                # on one if it's a different one, or more than 30 units
                # were traveled. If mario jumps fast enough the platform
                # he jumped off of may go off screen and be removed from
                # the object list, and if he lands on a platform of the
                # same type it'll have the same object index and type.
                # In this case ensure more distance has been traveled than
                # the width of the platform.
                if onMovingPlatform and (
                    (
                        jumpPlatformObj.index != landPlatformObj.index
                        or jumpPlatformObj.typeID != landPlatformObj.typeID
                    )
                    or curState.xPos - self.movingPlatformJumpState.xPos
                    > jumpPlatformWidth
                ):
                    self.movingPlatformJumpState = None
                    return MOVING_PLATFORM_REWARD
                # reward landing on ground
                elif not onMovingPlatform:
                    self.movingPlatformJumpState = None
                    return MOVING_PLATFORM_REWARD
        elif onGround and self.movingPlatformJumpState == None:
            platformObj, onMovingPlatform = self._standingOnMovingPlatform(curState)
            if onMovingPlatform:
                curState.movingPlatformObj = platformObj
        elif (
            not curState.onGround
            and self.movingPlatformJumpState is None
            and prevState.movingPlatformObj is not None
        ):
            self.movingPlatformJumpState = curState
            self.movingPlatformJumpState.movingPlatformObj = prevState.movingPlatformObj

        return 0

    def _standingOnMovingPlatform(
        self, curState: MarioLandGameState
    ) -> (MarioLandObject | None, bool):
        for o in curState.objects:
            if (
                o.typeID in OBJ_TYPES_MOVING_PLATFORM
                and curState.relYPos + 10 == o.relYPos
            ):
                return o, True
        return None, False

    def _standingOnTiles(self, tiles: List[int]) -> bool:
        sprites = self.pyboy.botsupport_manager().sprite_by_tile_identifier(
            tiles, on_screen=True
        )
        if len(sprites) == 0:
            return False

        leftMarioLeg = self.pyboy.botsupport_manager().sprite(5)
        leftMarioLegXPos = leftMarioLeg.x
        rightMarioLegXPos = leftMarioLegXPos + 8
        marioLegsYPos = leftMarioLeg.y
        if leftMarioLeg.attr_x_flip:
            rightMarioLegXPos = leftMarioLegXPos
            leftMarioLegXPos -= 8

        for spriteIdxs in sprites:
            for spriteIdx in spriteIdxs:
                sprite = self.pyboy.botsupport_manager().sprite(spriteIdx)
                # y positions are inverted for some reason
                if (
                    marioLegsYPos + 6 <= sprite.y and marioLegsYPos + 10 >= sprite.y
                ) and (
                    (
                        leftMarioLegXPos >= sprite.x - 4
                        and leftMarioLegXPos <= sprite.x + 4
                    )
                    or (
                        rightMarioLegXPos >= sprite.x - 4
                        and rightMarioLegXPos <= sprite.x + 4
                    )
                ):
                    return True

        return False

    def _handlePowerup(
        self, prevState: MarioLandGameState, curState: MarioLandGameState
    ) -> int:
        powerup = 0
        if curState.gotStar:
            self.invincibilityTimer = STAR_TIME
            # The actual star timer is set to 248 and only ticks down
            # when the frame counter is a one greater than a number
            # divisible by four. Don't ask me why. This accounts for
            # extra invincibility frames depending on what the frame
            # counter was at when the star was picked up
            frames = self.pyboy.get_memory_value(FRAME_COUNTER_MEM_VAL)
            extra = (frames - 1) % 4
            self.invincibilityTimer += extra
            powerup += STAR_REWARD
        if curState.hasStar:
            # current powerup status will be set to star, so set it to
            # the powerup of the last frame so the base powerup is accurate
            curState.powerupStatus = prevState.powerupStatus

        # big reward for acquiring powerups, small punishment for
        # loosing them but not too big a punishment so abusing
        # invincibility frames isn't discouraged
        if curState.powerupStatus != prevState.powerupStatus:
            if prevState.powerupStatus == STATUS_SMALL:
                # mario got a mushroom
                powerup = MUSHROOM_REWARD
            elif prevState.powerupStatus == STATUS_BIG:
                if curState.powerupStatus == STATUS_FIRE:
                    powerup = FLOWER_REWARD
                elif curState.powerupStatus == STATUS_SMALL:
                    self.invincibilityTimer = SHRINK_TIME
                    powerup = HIT_PUNISHMENT
            elif prevState.powerupStatus == STATUS_FIRE:
                # mario got hit and lost the fire flower
                self.invincibilityTimer = SHRINK_TIME
                powerup = HIT_PUNISHMENT

        if self.invincibilityTimer != 0:
            curState.invincibleTimer = self.invincibilityTimer
            self.invincibilityTimer -= 1

        return powerup

    def observation(
        self, prevState: MarioLandGameState, curState: MarioLandGameState
    ) -> Any:
        obs = self.gameWrapper._game_area_np(self.tileSet)

        # if mario is invincible his sprites will periodically flash by
        # cycling between being visible and not visible, ensure they are
        # always visible in the game area
        if curState.isInvincible:
            self._drawMario(obs)

        # flatten the game area array so it's Box compatible
        flatObs = np.concatenate(obs.tolist(), axis=None, dtype=np.int32)

        # add other features
        return np.append(
            flatObs,
            [
                curState.powerupStatus,
                curState.hasStar,
                curState.invincibleTimer,
                curState.xPos,
                curState.yPos,
                curState.xPos - prevState.xPos,
                curState.yPos - prevState.yPos,
            ],
        )

    def _drawMario(self, obs):
        # convert relative to screen y pos to sprite pos
        relYPos = self.pyboy.get_memory_value(0xC201) - 22
        marioLeftHead = self.pyboy.botsupport_manager().sprite(3)
        x1 = marioLeftHead.x // 8
        x2 = x1 + 1
        if marioLeftHead.attr_x_flip:
            x2 = x1 - 1

        y1 = (marioLeftHead.y // 8) - 1
        if y1 >= GAME_AREA_HEIGHT:
            # sprite is not visible so y pos is off screen, set it to
            # correct pos where mario is
            y1 = (relYPos // 8) - 1
        y2 = y1 - 1

        if y1 >= 0 and y1 < GAME_AREA_HEIGHT:
            obs[y1][x1] = 1
            obs[y1][x2] = 1
        if y2 >= 0 and y2 < GAME_AREA_HEIGHT:
            obs[y2][x1] = 1
            obs[y2][x2] = 1

    def terminated(
        self, prevState: MarioLandGameState, curState: MarioLandGameState
    ) -> bool:
        return self._isDead(curState)

    def truncated(
        self, prevState: MarioLandGameState, curState: MarioLandGameState
    ) -> bool:
        # if no forward progress has been made in 15s, end the eval episode
        # if the level is completed end this episode so the next level
        # isn't played twice
        return self.isEval and (
            self.evalNoProgress == 900 or curState.statusTimer == TIMER_LEVEL_CLEAR
        )

    def _isDead(self, curState: MarioLandGameState) -> bool:
        return curState.deadJumpTimer != 0 or curState.statusTimer == TIMER_DEATH

    def actionSpace(self):
        actions = [
            [WindowEvent.PASS],
            [WindowEvent.PRESS_ARROW_LEFT],
            [WindowEvent.PRESS_ARROW_RIGHT],
            [WindowEvent.PRESS_BUTTON_B],
            [WindowEvent.PRESS_BUTTON_A],
            [WindowEvent.PRESS_BUTTON_B, WindowEvent.PRESS_BUTTON_A],
            [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_BUTTON_B],
            [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_B],
            [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_BUTTON_A],
            [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_A],
            [
                WindowEvent.PRESS_ARROW_LEFT,
                WindowEvent.PRESS_BUTTON_B,
                WindowEvent.PRESS_BUTTON_A,
            ],
            [
                WindowEvent.PRESS_ARROW_RIGHT,
                WindowEvent.PRESS_BUTTON_B,
                WindowEvent.PRESS_BUTTON_A,
            ],
        ]

        return actions, Discrete(len(actions))

    def observationSpace(self) -> Space:
        # game area
        size = GAME_AREA_HEIGHT * GAME_AREA_WIDTH
        b = Box(low=0, high=TILES, shape=(size,))
        # add space for powerup status, is invincible, star timer,
        # x and y pos, x and y speed
        # TODO: add x, y acceleration
        low = np.append(b.low, [0, 0, 0, 0, 0, -2, -4])
        high = np.append(b.high, [3, 1, 1000, 5000, 255, 2, 4])
        return Box(low=low, high=high, dtype=np.int32)

    def hyperParameters(self, algo: str) -> Dict[str, Any]:
        match algo:
            case "ppo":
                return ppoConfig
            case _:
                return {}

    def evalEpisodes(self) -> int:
        return len(self.stateFiles) // 3

    def gameState(self):
        return MarioLandGameState(self.pyboy)

    def info(self, curState: MarioLandGameState) -> Dict[str, Any]:
        return {
            "worldLevel": curState.world,
            "levelProgress": curState.levelProgressMax,
        }

    def printGameState(
        self, prevState: MarioLandGameState, curState: MarioLandGameState
    ):
        objects = ""
        for i, o in enumerate(curState.objects):
            objects += f"{i}: {o.typeID} {o.relXPos} {o.relYPos}\n"

        s = f"""
Max level progress: {curState.levelProgressMax}
Powerup: {curState.powerupStatus}
Status timer: {curState.statusTimer} {self.pyboy.get_memory_value(STAR_TIMER_MEM_VAL)} {self.pyboy.get_memory_value(0xDA00)}
X, Y: {curState.xPos}, {curState.yPos}
Rel X, Y {curState.relXPos} {curState.relYPos}
Speeds: {curState.xPos - prevState.xPos} {curState.yPos - prevState.yPos}
Invincibility: {curState.gotStar} {curState.hasStar} {curState.isInvincible} {curState.invincibleTimer}
Object type: {self.pyboy.get_memory_value(PROCESSING_OBJECT_MEM_VAL)}
Boss: {curState.bossActive} {curState.bossHealth}
Moving platform: {curState.movingPlatformObj is not None}
{objects}
"""
        print(s[1:], flush=True)

    def render(self):
        return self.pyboy.screen_image()
