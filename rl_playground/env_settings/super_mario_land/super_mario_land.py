from collections import deque
import hashlib
from math import floor
from io import BytesIO
from typing import Any, Deque, Dict, List, Tuple, Tuple
from os import listdir
from os.path import basename, isfile, join, splitext
import random
from pathlib import Path

import numpy as np
from gymnasium.spaces import Discrete, Space
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from sqlalchemy import create_engine

from rl_playground.env_settings.env_settings import EnvSettings
from rl_playground.env_settings.super_mario_land.constants import (
    LEVEL_END_X_POS,
    MARIO_MAX_X_SPEED,
    MARIO_MAX_Y_SPEED,
    MAX_EUCLIDEAN_DISTANCE,
)
from rl_playground.env_settings.super_mario_land.game_area import bouncing_boulder_tiles, worldTilesets
from rl_playground.env_settings.super_mario_land.observation import (
    combineObservations,
    getObservations,
    getStackedObservation,
    observationSpace,
)
from rl_playground.env_settings.super_mario_land.ram import *
from rl_playground.env_settings.super_mario_land.settings import *
from rl_playground.go_explore.state_manager import StateManager


worldToNextLevelState = {
    (1, 1): 3,
    (1, 2): 6,
    (1, 3): 9,
    (2, 1): 12,
    (2, 2): 15,
    (3, 1): 18,
    (3, 2): 21,
    (3, 3): 24,
    (4, 1): 27,
}


class MarioLandSettings(EnvSettings):
    def __init__(
        self,
        pyboy: PyBoy,
        envID: int | None,
        isEval: bool,
        stateDir: Path = Path("states", "super_mario_land"),
    ):
        self.pyboy = pyboy
        self.envID = envID

        self.gameStateCache: Deque[MarioLandGameState] = deque(maxlen=N_STATE_STACK)
        self.observationCaches = [
            deque(maxlen=N_GAME_AREA_STACK),  # game area
            deque(maxlen=N_MARIO_OBS_STACK),  # mario features
            deque(maxlen=N_ENTITY_OBS_STACK),  # entity IDs
            deque(maxlen=N_ENTITY_OBS_STACK),  # entity features
            deque(maxlen=N_SCALAR_OBS_STACK),  # scalar features
        ]

        self.isEval = isEval
        self.maxLevel = "1-1"
        self.cellScore = 0
        self.cellID = 0
        self.cellCheckCounter = 0
        self.levelStr = ""
        self.evalStuck = 0
        self.evalNoProgress = 0
        self.invincibilityTimer = 0
        self.underground = False
        self.heartGetXPos = None
        self.heartFarming = False

        self.deathCounter = 0
        self.heartCounter = 0
        self.powerupCounter = 0
        self.coinCounter = 0

        self.levelProgressMax = 0
        self.onGroundFor = 0

        self.stateFiles = sorted([join(stateDir, f) for f in listdir(stateDir) if isfile(join(stateDir, f))])

        engine = create_engine("postgresql+psycopg://postgres:password@localhost/postgres")
        self.stateManager = StateManager(engine)
        with open(self.stateFiles[1], "rb") as f:
            self.pyboy.load_state(f)
            curState = self.gameState()
            cellHash, hashInput = self.cellHash(curState, isInitial=True)
            if self.stateManager.cell_exists(cellHash):
                return

            f.seek(0)
            state = memoryview(f.read())
            self.stateManager.insert_initial_cell(cellHash, hashInput, RANDOM_NOOP_FRAMES, "1-1", state)

    def reset(self, options: dict[str, Any]) -> Tuple[Any, MarioLandGameState, bool, Dict[str, Any]]:
        if options is not None:
            if "_update_max_level" in options:
                # a level was cleared in the eval that hasn't been cleared
                # before, update the max level
                self.maxLevel = options["_update_max_level"]

            curState = self.gameState()
            return self.observation(options["_prevState"], curState), curState, False, {}

        # delete old cell score entries so querying the DB doesn't
        # slow too much
        if self.isEval:
            self.stateManager.delete_old_cell_scores()

        # reset counters
        self.cellScore = 0
        self.cellCheckCounter = 0
        self.deathCounter = 0
        self.heartCounter = 0
        self.powerupCounter = 0
        self.coinCounter = 0

        if self.isEval:
            self.cellID, prevAction, maxNOOPs, initial, state = self.stateManager.get_first_cell()
        else:
            self.cellID, prevAction, maxNOOPs, initial, state = self.stateManager.get_random_cell(
                self.maxLevel
            )
            # print(f"{self.envID}: loaded cell {self.cellID}")

        curState = self._loadLevel(state, maxNOOPs, initial=initial)

        timer = STARTING_TIME
        if not self.isEval:
            # set the timer to a random time to make the environment more
            # stochastic; set it to lower values depending on where mario
            # is in the level is it's completable
            minTime = STARTING_TIME - int((curState.xPos / LEVEL_END_X_POS[self.levelStr]) * STARTING_TIME)
            minTime = max(MIN_RANDOM_TIME, minTime)
            timer = random.randint(minTime, STARTING_TIME)

        return self._reset(curState, True, timer), curState, True, {"_prev_action": prevAction}

    def _loadLevel(
        self,
        state: memoryview,
        maxNOOPs: int,
        initial: bool = False,
        prevState: MarioLandGameState | None = None,
        transferState: bool = False,
    ) -> MarioLandGameState:
        with BytesIO(state) as bs:
            self.pyboy.load_state(bs)

        livesLeft = 2
        coins = 0
        score = 0

        if transferState:
            livesLeft = prevState.livesLeft
            coins = prevState.coins
            score = prevState.score
            if prevState.powerupStatus != STATUS_SMALL:
                self.pyboy.memory[POWERUP_STATUS_MEM_VAL] = 1
            if prevState.powerupStatus == STATUS_FIRE:
                self.pyboy.memory[HAS_FIRE_FLOWER_MEM_VAL] = 1

            curState = self.gameState()
            self._handlePowerup(prevState, curState)
        elif not self.isEval:
            # make starting lives random so NN can learn to strategically
            # handle lives
            livesLeft = random.randint(STARTING_LIVES_MIN, STARTING_LIVES_MAX) - 1

            # make starting coins random so NN can learn that collecting
            # 100 coins means a level up
            coins = random.randint(0, 99)

            # occasionally randomly set mario's powerup status so the NN
            # can learn to use the powerups; also makes the environment more
            # stochastic
            curState = self.gameState()
            if initial and random.randint(0, 100) < RANDOM_POWERUP_CHANCE:
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
                    self.pyboy.memory[STAR_TIMER_MEM_VAL] = 0xF8
                    # set star song so timer functions correctly
                    self.pyboy.memory[0xDFE8] = 0x0C
                if randPowerup != STATUS_SMALL:
                    self.pyboy.memory[POWERUP_STATUS_MEM_VAL] = 1
                    if randPowerup > 2:
                        self.pyboy.memory[HAS_FIRE_FLOWER_MEM_VAL] = 1

                prevState = curState
                curState = self.gameState()
                if gotStar:
                    curState.gotStar = True
                    curState.hasStar = True
                    curState.isInvincible = True
                self._handlePowerup(prevState, curState)
        else:
            curState = self.gameState()

        # set lives left
        livesTens = livesLeft // 10
        livesOnes = livesLeft % 10
        self.pyboy.memory[LIVES_LEFT_MEM_VAL] = (livesTens << 4) | livesOnes
        self.pyboy.memory[LIVES_LEFT_DISPLAY_MEM_VAL] = livesTens
        self.pyboy.memory[LIVES_LEFT_DISPLAY_MEM_VAL + 1] = livesOnes
        curState.livesLeft = livesLeft

        # set coins
        self.pyboy.memory[COINS_MEM_VAL] = dec_to_bcm(coins)
        self.pyboy.memory[COINS_DISPLAY_MEM_VAL] = coins // 10
        self.pyboy.memory[COINS_DISPLAY_MEM_VAL + 1] = coins % 10

        # set score
        if not self.isEval or score == 0:
            # set score to 0
            for i in range(3):
                self.pyboy.memory[SCORE_MEM_VAL + i] = 0
            for i in range(5):
                self.pyboy.memory[SCORE_DISPLAY_MEM_VAL + i] = 44
            self.pyboy.memory[SCORE_DISPLAY_MEM_VAL + 5] = 0
        else:
            scoreHundreds = score // 100
            if scoreHundreds > 100:
                scoreHundreds -= 100
            scoreTenThousands = score // 10000
            scoreTens = score - ((scoreTenThousands * 10000) + (scoreHundreds * 100))
            self.pyboy.memory[SCORE_MEM_VAL] = dec_to_bcm(scoreTens)
            self.pyboy.memory[SCORE_MEM_VAL + 1] = dec_to_bcm(scoreHundreds)
            self.pyboy.memory[SCORE_MEM_VAL + 2] = dec_to_bcm(scoreTenThousands)

            paddedScore = f"{score:06}"
            leadingZerosReplaced = True
            for i in range(6):
                digit = paddedScore[i]
                if leadingZerosReplaced and digit == "0":
                    self.pyboy.memory[SCORE_DISPLAY_MEM_VAL + i] = 44
                else:
                    leadingZerosReplaced = False
                    self.pyboy.memory[SCORE_DISPLAY_MEM_VAL + i] = int(digit)

        if not self.isEval:
            # do nothing for a random amount of frames to make entity
            # placements varied and the environment more stochastic
            nopFrames = random.randint(0, maxNOOPs)
            self.pyboy.tick(count=nopFrames, render=False)
            curState = self.gameState()
            # set cell as invalid just in case a cell was added that
            # will cause mario to die almost immediately
            if self._isDead(curState):
                self.stateManager.set_cell_invalid(self.cellID)

        # reset max level progress
        self.levelProgressMax = curState.xPos
        curState.levelProgressMax = curState.xPos

        # reset death counter
        if not self.isEval:
            self.deathCounter = 0

        self.heartGetXPos = None
        self.heartFarming = False

        # set game area mapping
        self.levelStr = f"{curState.world[0]}-{curState.world[1]}"
        self.pyboy.game_area_mapping(worldTilesets[curState.world[0]])

        return curState

    def _reset(self, curState: MarioLandGameState, resetCaches: bool, timer: int) -> Dict[str, Any]:
        self.evalStuck = 0
        self.evalNoProgress = 0
        self.underground = False
        self.onGroundFor = 0

        # reset the game state cache
        if resetCaches:
            [self.gameStateCache.append(curState) for _ in range(N_STATE_STACK)]

        gameArea, marioInfo, entityID, entityInfo, scalar = getObservations(self.pyboy, self.gameStateCache)

        if resetCaches:
            # reset the observation cache
            [self.observationCaches[0].append(gameArea) for _ in range(N_GAME_AREA_STACK)]
            [self.observationCaches[1].append(marioInfo) for _ in range(N_MARIO_OBS_STACK)]
            [self.observationCaches[2].append(entityID) for _ in range(N_ENTITY_OBS_STACK)]
            [self.observationCaches[3].append(entityInfo) for _ in range(N_ENTITY_OBS_STACK)]
            [self.observationCaches[4].append(scalar) for _ in range(N_SCALAR_OBS_STACK)]
        else:
            curState.posReset = True
            self.gameStateCache.append(curState)

        # set level timer
        timerHundreds = timer // 100
        timerTens = timer - (timerHundreds * 100)
        self.pyboy.memory[TIMER_HUNDREDS] = timerHundreds
        self.pyboy.memory[TIMER_TENS] = dec_to_bcm(timerTens)
        self.pyboy.memory[TIMER_FRAMES] = 0x28

        return combineObservations(self.observationCaches)

    def reward(self, prevState: MarioLandGameState) -> Tuple[float, MarioLandGameState]:
        curState = self.gameState()
        self.levelProgressMax = max(self.levelProgressMax, curState.xPos)
        curState.levelProgressMax = self.levelProgressMax
        self.gameStateCache.append(curState)

        # return flat punishment on mario's death
        if self._isDead(curState):
            consecutiveDeaths = self.deathCounter * DEATH_SCALE
            self.deathCounter += 1

            if curState.livesLeft == 0:
                # no lives left, just return so this episode can be terminated
                return GAME_OVER_PUNISHMENT + consecutiveDeaths, curState

            # don't let the game set the timer back to max time
            timer = np.clip(curState.timeLeft - DEATH_TIME_LOSS, MIN_TIME, STARTING_TIME)

            # skip frames where mario is dying
            statusTimer = curState.statusTimer
            gameState = curState.gameState
            while gameState in (3, 4) or (gameState == 1 and statusTimer != 0):
                self.pyboy.tick(render=False)
                gameState = self.pyboy.memory[GAME_STATE_MEM_VAL]
                statusTimer = self.pyboy.memory[STATUS_TIMER_MEM_VAL]

            self.pyboy.tick(count=5, render=False)

            curState = self.gameState()
            # don't reset state and observation caches so the agent can
            # see that it died
            self._reset(curState, False, timer)

            return DEATH_PUNISHMENT + consecutiveDeaths, curState

        # handle level clear
        if curState.statusTimer == TIMER_LEVEL_CLEAR:
            levelClear = LEVEL_CLEAR_REWARD
            # reward clearing a level through the top spot
            if curState.yPos > 60:
                levelClear += LEVEL_CLEAR_TOP_REWARD
            # reward clearing a level with extra lives
            levelClear += curState.livesLeft * LEVEL_CLEAR_LIVES_COEF_REWARD
            # reward clearing a level while powered up
            if curState.powerupStatus == STATUS_BIG:
                levelClear += LEVEL_CLEAR_BIG_REWARD
            elif curState.powerupStatus == STATUS_FIRE:
                levelClear += LEVEL_CLEAR_FIRE_REWARD

            if curState.world == (4, 2):
                self.cellScore += levelClear
                return levelClear, curState

            # load the next level directly to avoid processing
            # unnecessary frames and the AI playing levels we
            # don't want it to
            stateFile = self.stateFiles[worldToNextLevelState[curState.world] + 1]
            with open(stateFile, "rb") as f:
                state = memoryview(f.read())
                # keep lives and powerup in new level
                curState = self._loadLevel(state, RANDOM_NOOP_FRAMES, prevState=prevState, transferState=True)
            # don't reset state and observation caches so the agent can
            # see that it started a new level
            self._reset(curState, False, STARTING_TIME)

            return levelClear, curState

        # add time punishment every step to encourage speed more
        clock = CLOCK_PUNISHMENT

        xSpeed = np.clip(curState.xPos - prevState.xPos, -MARIO_MAX_X_SPEED, MARIO_MAX_X_SPEED)
        movement = 0
        if xSpeed > 0:
            movement = xSpeed * FORWARD_REWARD_COEF
        elif xSpeed < 0:
            movement = xSpeed * BACKWARD_PUNISHMENT_COEF

        score = (curState.score - prevState.score) * SCORE_REWARD_COEF
        coins = (curState.coins - prevState.coins) * COIN_REWARD
        self.coinCounter += curState.coins - prevState.coins

        # the game registers mario as on the ground 1 or 2 frames before
        # he actually is to change his pose
        if not curState.onGround:
            self.onGroundFor = 0
        elif self.onGroundFor < 2:
            self.onGroundFor += 1
        onGround = self.onGroundFor == 2

        movingPlatform = 0
        movPlatObj, onMovingPlatform = self._standingOnMovingPlatform(curState)
        if onGround and onMovingPlatform:
            ySpeed = np.clip(curState.yPos - prevState.yPos, -MARIO_MAX_Y_SPEED, MARIO_MAX_Y_SPEED)
            movingPlatform += max(0, xSpeed) * MOVING_PLATFORM_X_REWARD_COEF
            movingPlatform += max(0, ySpeed) * MOVING_PLATFORM_Y_REWARD_COEF

            curPlatPos = np.array((curState.xPos, curState.yPos))
            platDistances = []
            for obj in curState.objects:
                if movPlatObj == obj or obj.typeID != MOVING_PLATFORM_TYPE_ID or curState.xPos > obj.xPos:
                    continue
                platDistances.append(np.linalg.norm(curPlatPos - np.array((obj.xPos, obj.yPos))))

            if len(platDistances) > 0:
                minDistance = min(platDistances)
                movingPlatform += MOVING_PLATFORM_DISTANCE_REWARD_MAX - (
                    minDistance * (MOVING_PLATFORM_DISTANCE_REWARD_MAX / MAX_EUCLIDEAN_DISTANCE)
                )

        # in world 3 reward standing on bouncing boulders to encourage
        # waiting for them to fall and ride on them instead of immediately
        # jumping into spikes, but only if the boulders are moving to the
        # right
        standingOnBoulder = 0
        if (
            curState.world[0] == 3
            and onGround
            and xSpeed > 0
            and self._standingOnTiles(bouncing_boulder_tiles)
        ):
            standingOnBoulder = BOULDER_REWARD

        # keep track of how long the agent is idle so we can end early
        # in an evaluation
        if self.isEval:
            if curState.xPos == prevState.xPos:
                self.evalStuck += 1
            else:
                self.evalStuck = 0

            if curState.levelProgressMax == prevState.levelProgressMax:
                self.evalNoProgress += 1
            else:
                self.evalNoProgress = 0

        # reward getting powerups and manage powerup related bookkeeping
        powerup = self._handlePowerup(prevState, curState)
        if powerup > 0:
            self.powerupCounter += 1

        # discourage heart farming
        heart = 0
        if curState.livesLeft - prevState.livesLeft != 0:
            if (
                self.heartGetXPos is not None
                and curState.xPos + HEART_FARM_X_POS_MULTIPLE >= self.heartGetXPos
                and curState.xPos - HEART_FARM_X_POS_MULTIPLE <= self.heartGetXPos
            ):
                self.heartFarming = True
                heart = HEART_FARM_PUNISHMENT
            else:
                # reward getting 1-up
                heart = (curState.livesLeft - prevState.livesLeft) * HEART_REWARD
                self.heartCounter += curState.livesLeft - prevState.livesLeft

            self.heartGetXPos = curState.xPos

        # reward damaging or killing a boss
        boss = 0
        if curState.bossActive and curState.bossHealth < prevState.bossHealth:
            if prevState.bossHealth - curState.bossHealth == 1:
                boss = HIT_BOSS_REWARD
            elif curState.bossHealth == 0:
                boss = KILL_BOSS_REWARD

        reward = (
            clock + movement + score + coins + movingPlatform + standingOnBoulder + powerup + heart + boss
        )

        # handle going in pipe
        if curState.pipeWarping:
            gameState = curState.gameState
            while gameState != 0:
                self.pyboy.tick(render=False)
                gameState = self.pyboy.memory[GAME_STATE_MEM_VAL]

            curState = self.gameState()

        return reward, curState

    def postStep(
        self, prevState: MarioLandGameState, curState: MarioLandGameState, action: int, reward: float
    ):
        if self.isEval:
            return

        self.cellScore += reward

        if self.terminated(prevState, curState) or self.truncated(prevState, curState):
            self.stateManager.record_score(self.cellID, self.cellScore)
            return

        # only check if this cell is new every N frames to avoid
        # making DB queries every frame
        if self.cellCheckCounter != FRAME_CELL_CHECK:
            self.cellCheckCounter += 1
            return
        self.cellCheckCounter = 0

        cellHash, hashInput = self.cellHash(curState)
        if cellHash is not None and not self.stateManager.cell_exists(cellHash):
            with BytesIO() as state:
                self.pyboy.save_state(state)
                state.seek(0)

                maxNOOPs = RANDOM_NOOP_FRAMES

                # ensure loading from the state won't instantly kill mario
                if any((obj.typeID in ENEMY_TYPE_IDS for obj in curState.objects)):
                    self.pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT)
                    self.pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)

                    unsafeState = False
                    # if mario dies without moving in less than 2 seconds
                    # don't save the state; we don't want to load states
                    # that will result in an unpreventable death
                    for _ in range(20):
                        self.pyboy.tick(count=6, render=False)
                        if self.pyboy.memory[GAME_STATE_MEM_VAL] in GAME_STATES_DEAD:
                            unsafeState = True
                            break

                    self.pyboy.load_state(state)
                    state.seek(0)
                    if unsafeState:
                        return

                    maxNOOPs = RANDOM_NOOP_FRAMES_WITH_ENEMIES

                try:
                    section = f"{curState.world[0]}-{curState.world[1]}"
                    self.stateManager.insert_cell(
                        cellHash, hashInput, action, maxNOOPs, section, state.getbuffer()
                    )
                    # print(f"added cell: {hashInput}")
                except Exception as e:
                    print(e)

    def _standingOnMovingPlatform(self, curState: MarioLandGameState) -> Tuple[MarioLandObject | None, bool]:
        for obj in curState.objects:
            if obj.typeID == MOVING_PLATFORM_TYPE_ID and curState.yPos - 10 == obj.yPos:
                return obj, True
        return None, False

    def _standingOnTiles(self, tiles: List[int]) -> bool:
        sprites = self.pyboy.get_sprite_by_tile_identifier(tiles, on_screen=True)
        if len(sprites) == 0:
            return False

        leftMarioLeg = self.pyboy.get_sprite(5)
        leftMarioLegXPos = leftMarioLeg.x
        rightMarioLegXPos = leftMarioLegXPos + 8
        marioLegsYPos = leftMarioLeg.y
        if leftMarioLeg.attr_x_flip:
            rightMarioLegXPos = leftMarioLegXPos
            leftMarioLegXPos -= 8

        for spriteIdxs in sprites:
            for spriteIdx in spriteIdxs:
                sprite = self.pyboy.get_sprite(spriteIdx)
                # y positions are inverted for some reason
                if (marioLegsYPos + 6 <= sprite.y and marioLegsYPos + 10 >= sprite.y) and (
                    (leftMarioLegXPos >= sprite.x - 4 and leftMarioLegXPos <= sprite.x + 4)
                    or (rightMarioLegXPos >= sprite.x - 4 and rightMarioLegXPos <= sprite.x + 4)
                ):
                    return True

        return False

    def _handlePowerup(self, prevState: MarioLandGameState, curState: MarioLandGameState) -> int:
        powerup = 0
        if curState.gotStar:
            self.invincibilityTimer = STAR_TIME
            # The actual star timer is set to 248 and only ticks down
            # when the frame counter is a one greater than a number
            # divisible by four. Don't ask me why. This accounts for
            # extra invincibility frames depending on what the frame
            # counter was at when the star was picked up
            frames = self.pyboy.memory[FRAME_COUNTER_MEM_VAL]
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
            elif prevState.powerupStatus == STATUS_GROWING and curState.powerupStatus == STATUS_SMALL:
                # mario got hit while growing from a mushroom
                powerup = HIT_PUNISHMENT
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

    def observation(self, prevState: MarioLandGameState, curState: MarioLandGameState) -> Any:
        return getStackedObservation(self.pyboy, self.observationCaches, self.gameStateCache)

    def terminated(self, prevState: MarioLandGameState, curState: MarioLandGameState) -> bool:
        return self._isDead(curState) and curState.livesLeft == 0

    def truncated(self, prevState: MarioLandGameState, curState: MarioLandGameState) -> bool:
        # If Mario has not moved in 10s, end the eval episode.
        # If no forward progress has been made in 20s, end the eval episode.
        # If the level is completed end this episode so the next level
        # isn't played twice. If 4-2 is completed end the episode, that's
        # final normal level so there's no level to start after it.
        return (
            self.heartFarming
            or curState.hasStar  # TODO: remove once star bug has been fixed
            or (self.isEval and (self.evalStuck == 600 or self.evalNoProgress == 1200))
            or (curState.statusTimer == TIMER_LEVEL_CLEAR and curState.world == (4, 2))
        )

    def cellHash(self, curState: MarioLandGameState, isInitial=False) -> Tuple[str | None, str | None]:
        if not isInitial and self.onGroundFor != 2:
            return None, None

        roundedXPos = X_POS_MULTIPLE * floor(curState.xPos / X_POS_MULTIPLE)
        roundedYPos = Y_POS_MULTIPLE * floor(curState.yPos / Y_POS_MULTIPLE)

        objectTypes = ""
        for obj in curState.objects:
            objRoundedXPos = ENTITY_X_POS_MULTIPLE * floor(obj.xPos / ENTITY_X_POS_MULTIPLE)
            objRoundedYPos = 0
            if obj.typeID not in ENTITIES_IGNORE_Y_POS:
                objRoundedYPos = ENTITY_Y_POS_MULTIPLE * floor(obj.yPos / ENTITY_Y_POS_MULTIPLE)
            objectTypes += f"{obj.typeID}|{objRoundedXPos}|{objRoundedYPos}/"

        input = f"{curState.world}|{roundedXPos}|{roundedYPos}|{curState.powerupStatus}/{objectTypes}"
        return hashlib.md5(input.encode("utf-8")).hexdigest(), input

    def _isDead(self, curState: MarioLandGameState) -> bool:
        return curState.gameState in GAME_STATES_DEAD

    def actionSpace(self):
        actions = [
            [WindowEvent.PASS],
            [WindowEvent.PRESS_ARROW_LEFT],
            [WindowEvent.PRESS_ARROW_RIGHT],
            # [WindowEvent.PRESS_ARROW_DOWN],
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
        return observationSpace()

    def normalize(self) -> Tuple[bool, bool]:
        return False, True

    def hyperParameters(self, algo: str) -> Dict[str, Any]:
        match algo:
            case "ppo":
                return PPO_HYPERPARAMS
            case _:
                return {}

    def evalEpisodes(self) -> int:
        # TODO: return 2/3 if normal/hard levels should be evaled
        return 1

    def gameState(self):
        return MarioLandGameState(self.pyboy)

    def info(self, curState: MarioLandGameState) -> Dict[str, Any]:
        return {
            "worldLevel": self.levelStr,
            "levelProgress": curState.levelProgressMax,
            "deaths": self.deathCounter,
            "hearts": self.heartCounter,
            "powerups": self.powerupCounter,
            "coins": self.coinCounter,
            "score": curState.score,
        }

    def printGameState(self, prevState: MarioLandGameState, curState: MarioLandGameState):
        objects = ""
        for i, o in enumerate(curState.objects):
            objects += f"{i}: {o.typeID} {o.xPos} {o.yPos} {round(o.meanXSpeed,3)} {round(o.meanYSpeed,3)} {round(o.xAccel,3)} {round(o.yAccel,3)}\n"

        s = f"""
Max level progress: {curState.levelProgressMax}
Powerup: {curState.powerupStatus}
Lives left: {curState.livesLeft}
Score: {curState.score}
Status timer: {curState.statusTimer} {self.pyboy.memory[STAR_TIMER_MEM_VAL]} {self.pyboy.memory[0xDA00]}
X, Y: {curState.xPos}, {curState.yPos}
Rel X, Y {curState.relXPos} {curState.relYPos}
Speeds: {round(curState.meanXSpeed, 3)} {round(curState.meanYSpeed, 3)} {round(curState.xAccel, 3)} {round(curState.yAccel, 3)}
Invincibility: {curState.gotStar} {curState.hasStar} {curState.isInvincible} {curState.invincibleTimer}
Boss: {curState.bossActive} {curState.bossHealth}
Game state: {curState.gameState}
{objects}
"""
        print(s[1:], flush=True)

    def render(self):
        return self.pyboy.screen.image.copy()
