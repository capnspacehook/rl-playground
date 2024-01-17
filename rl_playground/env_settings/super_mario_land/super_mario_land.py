from collections import deque
from typing import Any, Deque, Dict, List
from os import listdir
from os.path import basename, isfile, join, splitext
import random
from pathlib import Path

import numpy as np
from gymnasium.spaces import Discrete, Space
from pyboy import PyBoy, WindowEvent

from rl_playground.env_settings.env_settings import EnvSettings
from rl_playground.env_settings.super_mario_land.constants import MARIO_MAX_X_SPEED
from rl_playground.env_settings.super_mario_land.game_area import bouncing_boulder_tiles, worldTilesets
from rl_playground.env_settings.super_mario_land.observation import (
    combineObservations,
    getObservations,
    getStackedObservation,
    observationSpace,
)
from rl_playground.env_settings.super_mario_land.ram import *
from rl_playground.env_settings.super_mario_land.settings import *


class MarioLandSettings(EnvSettings):
    def __init__(
        self,
        pyboy: PyBoy,
        isEval: bool,
        stateDir: Path = Path("states", "super_mario_land"),
    ):
        self.pyboy = pyboy
        self.gameWrapper = self.pyboy.game_wrapper()

        self.gameStateCache: Deque[MarioLandGameState] = deque(maxlen=N_STATE_STACK)
        self.observationCaches = [
            deque(maxlen=N_GAME_AREA_STACK),  # game area
            deque(maxlen=N_MARIO_OBS_STACK),  # mario features
            deque(maxlen=N_ENTITY_OBS_STACK),  # entity IDs
            deque(maxlen=N_ENTITY_OBS_STACK),  # entity features
            deque(maxlen=N_SCALAR_OBS_STACK),  # scalar features
        ]

        self.isEval = isEval
        self.tileSet = None
        self.stateIdx = 0
        self.evalStateCounter = 0
        self.evalEpisodeCounter = 0
        self.evalNoProgress = 0
        self.invincibilityTimer = 0
        self.deathCounter = 0

        self.stateFiles = sorted([join(stateDir, f) for f in listdir(stateDir) if isfile(join(stateDir, f))])

        # all levels are equally likely to be trained on at the start
        self.levelChooseProbs = [0.1 for _ in range(10)]

        self.levelProgressMax = 0
        self.onGroundFor = 0

    def reset(self, options: dict[str, Any] | None = None) -> (Any, MarioLandGameState, bool):
        if options is not None:
            if "_eval_starting" in options:
                # this will be passed before evals are started, reset the eval
                # state counter so all evals will start at the same state
                self.evalStateCounter = 0
                self.evalEpisodeCounter = 0
            elif "_update_level_choose_probs" in options:
                # an eval has ended, update the level choose probabilities
                self.levelChooseProbs = options["_update_level_choose_probs"]

            curState = self.gameState()
            return self.observation(options["_prevState"], curState), curState, False

        if self.isEval:
            # evaluate levels in order, each level in easy and hard mode
            # then the next level
            self.stateIdx = self.evalStateCounter
            # don't start from checkpoints
            if self.evalEpisodeCounter % 2 == 0:
                self.evalStateCounter += 1
            else:
                self.evalStateCounter += 5
            if self.evalStateCounter >= len(self.stateFiles):
                self.evalStateCounter = 0

            self.evalEpisodeCounter += 1
        else:
            # reset game state to a random level
            level = np.random.choice(10, p=self.levelChooseProbs)
            checkpoint = np.random.randint(6)
            self.stateIdx = (6 * level) + checkpoint

        curState = self._loadLevel()

        return self._reset(curState, True, STARTING_TIME), curState, True

    def _loadLevel(
        self, prevState: MarioLandGameState | None = None, transferState: bool = False
    ) -> MarioLandGameState:
        stateFile = self.stateFiles[self.stateIdx]
        with open(stateFile, "rb") as f:
            self.pyboy.load_state(f)

        # get checkpoint number from state filename
        stateFile = basename(splitext(stateFile)[0])
        world = int(stateFile[0])
        self.tileSet = worldTilesets[world]

        # seed randomizer
        self.gameWrapper._set_timer_div(None)

        # set score to 0
        for i in range(3):
            self.pyboy.set_memory_value(SCORE_MEM_VAL + i, 0)
        for i in range(5):
            self.pyboy.set_memory_value(SCORE_DISPLAY_MEM_VAL + i, 44)
        self.pyboy.set_memory_value(SCORE_DISPLAY_MEM_VAL + 5, 0)

        self._setStartingPos()

        livesLeft = 1
        coins = 0
        if transferState:
            livesLeft = prevState.livesLeft
            coins = prevState.coins
            if prevState.powerupStatus != STATUS_SMALL:
                self.pyboy.set_memory_value(POWERUP_STATUS_MEM_VAL, 1)
            if prevState.powerupStatus == STATUS_FIRE:
                self.pyboy.set_memory_value(HAS_FIRE_FLOWER_MEM_VAL, 1)

            curState = self.gameState()
            self._handlePowerup(prevState, curState)
        elif not self.isEval:
            # make starting lives random to NN can learn to strategically
            # handle lives
            livesLeft = random.randint(STARTING_LIVES_MIN, STARTING_LIVES_MAX) - 1

            # make starting coins random so NN can learn that collecting
            # 100 coins means a level up
            coins = random.randint(0, 99)

            # occasionally randomly set mario's powerup status so the NN
            # can learn to use the powerups; also makes the environment more
            # stochastic
            curState = self.gameState()
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
                    self.pyboy.set_memory_value(POWERUP_STATUS_MEM_VAL, 1)
                    if randPowerup > 2:
                        self.pyboy.set_memory_value(HAS_FIRE_FLOWER_MEM_VAL, 1)

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
        self.pyboy.set_memory_value(LIVES_LEFT_MEM_VAL, (livesTens << 4) | livesOnes)
        self.pyboy.set_memory_value(LIVES_LEFT_DISPLAY_MEM_VAL, livesTens)
        self.pyboy.set_memory_value(LIVES_LEFT_DISPLAY_MEM_VAL + 1, livesOnes)
        curState.livesLeft = livesLeft

        # set coins
        self.pyboy.set_memory_value(COINS_MEM_VAL, dec_to_bcm(coins))
        self.pyboy.set_memory_value(COINS_DISPLAY_MEM_VAL, coins // 10)
        self.pyboy.set_memory_value(COINS_DISPLAY_MEM_VAL + 1, coins % 10)

        # if we're starting from a state with entities do nothing for
        # a random amount of frames to make entity placements varied
        if len(curState.objects) != 0:
            for _ in range(random.randint(0, RANDOM_NOOP_FRAMES)):
                self.pyboy.tick()

        # reset max level progress
        self.levelProgressMax = curState.xPos
        curState.levelProgressMax = curState.xPos

        # reset death counter
        self.deathCounter = 0

        return curState

    def _setStartingPos(self):
        if self.isEval:
            return

        # make starting position less deterministic to prevent action memorization
        startingMovement = random.randint(0, 2)
        if startingMovement == 1:
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_LEFT)
            self.pyboy.tick()
            self.pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT)
            self.pyboy.tick()
        elif startingMovement == 2:
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
            self.pyboy.tick()
            self.pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)
            self.pyboy.tick()

    def _reset(self, curState: MarioLandGameState, resetCaches: bool, timer: int) -> Dict[str, Any]:
        self.evalNoProgress = 0
        self.onGroundFor = 0

        # reset the game state cache
        if resetCaches:
            [self.gameStateCache.append(curState) for _ in range(N_STATE_STACK)]

        gameArea, marioInfo, entityID, entityInfo, scalar = getObservations(
            self.pyboy, self.tileSet, self.gameStateCache
        )

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
        self.pyboy.set_memory_value(TIMER_HUNDREDS, timerHundreds)
        self.pyboy.set_memory_value(TIMER_TENS, dec_to_bcm(timerTens))
        self.pyboy.set_memory_value(TIMER_FRAMES, 0x28)

        return combineObservations(self.observationCaches)

    def reward(self, prevState: MarioLandGameState) -> (float, MarioLandGameState):
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
            timer = min(curState.timeLeft + DEATH_EXTRA_TIME, STARTING_TIME)

            # skip frames where mario is dying
            statusTimer = curState.statusTimer
            gameState = curState.gameState
            while gameState in (3, 4) or (gameState == 1 and statusTimer != 0):
                self.pyboy.tick()
                gameState = self.pyboy.get_memory_value(GAME_STATE_MEM_VAL)
                statusTimer = self.pyboy.get_memory_value(STATUS_TIMER_MEM_VAL)

            for _ in range(5):
                self.pyboy.tick()

            self._setStartingPos()

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

            # if we're evaluating and the level is cleared return, the
            # episode is over; otherwise load the next level directly
            # to avoid processing unnecessary frames and the AI playing
            # levels we don't want it to
            if self.isEval:
                return levelClear, curState
            else:
                # load start of next level, not a level checkpoint
                self.stateIdx += 6 - (self.stateIdx % 6)
                if self.stateIdx >= len(self.stateFiles):
                    self.stateIdx = 0

                # keep lives and powerup in new level
                curState = self._loadLevel(prevState=prevState, transferState=True)
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

        # the game registers mario as on the ground 1 or 2 frames before
        # he actually is to change his pose
        if not curState.onGround:
            self.onGroundFor = 0
        elif self.onGroundFor < 2:
            self.onGroundFor += 1
        onGround = self.onGroundFor == 2

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
            if curState.levelProgressMax - prevState.levelProgressMax == 0:
                self.evalNoProgress += 1
            else:
                self.evalNoProgress = 0

        # reward getting powerups and manage powerup related bookkeeping
        powerup = self._handlePowerup(prevState, curState)

        # reward getting 1-up
        heart = (curState.livesLeft - prevState.livesLeft) * HEART_REWARD

        # reward damaging or killing a boss
        boss = 0
        if curState.bossActive and curState.bossHealth < prevState.bossHealth:
            if prevState.bossHealth - curState.bossHealth == 1:
                boss = HIT_BOSS_REWARD
            elif curState.bossHealth == 0:
                boss = KILL_BOSS_REWARD

        reward = clock + movement + score + coins + standingOnBoulder + powerup + heart + boss

        return reward, curState

    def _standingOnTiles(self, tiles: List[int]) -> bool:
        sprites = self.pyboy.botsupport_manager().sprite_by_tile_identifier(tiles, on_screen=True)
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
        return getStackedObservation(self.pyboy, self.tileSet, self.observationCaches, self.gameStateCache)

    def terminated(self, prevState: MarioLandGameState, curState: MarioLandGameState) -> bool:
        return self._isDead(curState) and curState.livesLeft == 0

    def truncated(self, prevState: MarioLandGameState, curState: MarioLandGameState) -> bool:
        # TODO: remove once star bug has been fixed
        # If no forward progress has been made in 15s, end the eval episode.
        # If the level is completed end this episode so the next level
        # isn't played twice. If 4-2 is completed end the episode, that's
        # final normal level so there's no level to start after it.
        return (
            curState.hasStar
            or (self.isEval and (self.evalNoProgress == 900 or curState.statusTimer == TIMER_LEVEL_CLEAR))
            or (curState.statusTimer == TIMER_LEVEL_CLEAR and curState.world == (4, 2))
        )

    def _isDead(self, curState: MarioLandGameState) -> bool:
        return curState.gameState in (1, 3)

    def actionSpace(self):
        actions = [
            [WindowEvent.PASS],
            [WindowEvent.PRESS_ARROW_LEFT],
            [WindowEvent.PRESS_ARROW_RIGHT],
            [WindowEvent.PRESS_ARROW_DOWN],
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

    def normalize(self) -> (bool, bool):
        return False, True

    def hyperParameters(self, algo: str) -> Dict[str, Any]:
        match algo:
            case "ppo":
                return PPO_HYPERPARAMS
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

    def printGameState(self, prevState: MarioLandGameState, curState: MarioLandGameState):
        objects = ""
        for i, o in enumerate(curState.objects):
            objects += f"{i}: {o.typeID} {o.xPos} {o.yPos} {round(o.meanXSpeed,3)} {round(o.meanYSpeed,3)} {round(o.xAccel,3)} {round(o.yAccel,3)}\n"

        s = f"""
Max level progress: {curState.levelProgressMax}
Powerup: {curState.powerupStatus}
Lives left: {curState.livesLeft}
Score: {curState.score}
Status timer: {curState.statusTimer} {self.pyboy.get_memory_value(STAR_TIMER_MEM_VAL)} {self.pyboy.get_memory_value(0xDA00)}
X, Y: {curState.xPos}, {curState.yPos}
Rel X, Y {curState.relXPos} {curState.relYPos}
Speeds: {round(curState.meanXSpeed, 3)} {round(curState.meanYSpeed, 3)} {round(curState.xAccel, 3)} {round(curState.yAccel, 3)}
Invincibility: {curState.gotStar} {curState.hasStar} {curState.isInvincible} {curState.invincibleTimer}
Boss: {curState.bossActive} {curState.bossHealth}
{objects}
"""
        print(s[1:], flush=True)

    def render(self):
        return self.pyboy.screen_image()
