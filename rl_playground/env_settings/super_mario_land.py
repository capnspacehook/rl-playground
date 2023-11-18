import itertools
from typing import Any, Dict
from os import listdir
from os.path import basename, isfile, join, splitext
import random
from pathlib import Path

import numpy as np
import flax.linen as nn
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
    "batch_size": 64,
    "clip_range": 0.2,
    "ent_coef": 7.513020308749457e-06,
    "gae_lambda": 0.98,
    "gamma": 0.99,
    "learning_rate": 3.183807492928217e-05,
    "max_grad_norm": 5,
    "n_epochs": 5,
    "n_steps": 512,
    "vf_coef": 0.33653746631712467,
    "policy_kwargs": dict(
        activation_fn=nn.tanh,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    ),
}

RANDOM_NOOP_FRAMES = 60
RANDOM_POWERUP_CHANCE = 25

GAME_AREA_HEIGHT = 16
GAME_AREA_WIDTH = 20

STATUS_TIMER_MEM_VAL = 0xFFA6
DEAD_JUMP_TIMER_MEM_VAL = 0xC0AC
POWERUP_STATUS_MEM_VAL = 0xFF99
HAS_FIRE_FLOWER_MEM_VAL = 0xFFB5
STAR_TIMER_MEM_VAL = 0xC0D3
FRAME_COUNTER_MEM_VAL = 0xDA00

STATUS_SMALL = 0
STATUS_BIG = 1
STATUS_FIRE = 2

TIMER_DEATH = 0x90
TIMER_LEVEL_CLEAR = 0xF0
STAR_TIME = 956
SHRINK_TIME = 0x50 + 0x40


class MarioLandGameState(GameState):
    def __init__(self, pyboy: PyBoy):
        self.pyboy = pyboy
        self.gameWrapper = pyboy.game_wrapper()

        # Find the real level progress x
        levelBlock = pyboy.get_memory_value(0xC0AB)
        # C202 Mario's X position relative to the screen
        xPos = pyboy.get_memory_value(0xC202)
        scx = pyboy.botsupport_manager().screen().tilemap_position_list()[16][0]
        real = (scx - 7) % 16 if (scx - 7) % 16 != 0 else 16
        self.xPos = levelBlock * 16 + real + xPos

        yPos = self.pyboy.get_memory_value(0xC201)
        if yPos <= 183:
            # 183 is lowest position, y coordinate is flipped, 0 is higher than 1
            self.yPos = 183 - yPos
        else:
            # handle underflow
            self.yPos = 183 + (256 - yPos)

        self.timeLeft = self.gameWrapper.time_left
        self.livesLeft = self.gameWrapper.lives_left
        self.score = self.gameWrapper.score
        self.levelProgressMax = max(self.gameWrapper._level_progress_max, self.xPos)
        self.world = self.gameWrapper.world
        self.statusTimer = self.pyboy.get_memory_value(STATUS_TIMER_MEM_VAL)
        self.deadJumpTimer = self.pyboy.get_memory_value(DEAD_JUMP_TIMER_MEM_VAL)

        powerupStatus = self.pyboy.get_memory_value(POWERUP_STATUS_MEM_VAL)
        hasFireFlower = self.pyboy.get_memory_value(HAS_FIRE_FLOWER_MEM_VAL)
        starTimer = self.pyboy.get_memory_value(STAR_TIMER_MEM_VAL)

        self.powerupStatus = STATUS_SMALL
        self.gotStar = False
        self.isInvincible = False
        self.invincibleTimer = 0
        if starTimer != 0:
            self.gotStar = True
            self.isInvincible = True
        elif powerupStatus == 1:
            self.powerupStatus = STATUS_BIG
        elif powerupStatus == 2:
            if hasFireFlower:
                self.powerupStatus = STATUS_FIRE
            else:
                self.powerupStatus = STATUS_BIG
        if powerupStatus == 3 or powerupStatus == 4:
            self.isInvincible = True

        self.hasStar = self.gotStar


# Mario and Daisy
base_scripts = (1, list(range(81)))
plane = (2, list(range(99, 110)))
submarine = (3, list(range(112, 122)))
mario_fireball = (4, [96, 110, 122])

# Bonuses
coin = (10, [244])
mushroom = (11, [131])
flower = (12, [224, 229])
star = (13, [134])
heart = (14, [132])

# Blocks
pipes = list(range(368, 381))
world_4_extra_pipes = [363, 364, 365, 366]  # are normal blocks on other worlds
common_blocks = (
    [
        142,
        143,
        221,
        222,
        231,
        232,
        233,
        234,
        235,
        236,
        301,
        302,
        303,
        304,
        340,
        352,
        353,
        355,
        356,
        357,
        358,
        359,
        360,
        361,
        362,
        381,
        382,
        383,
    ]
    + pipes
    + world_4_extra_pipes,
)
world_1_2_blocks = (20, [*common_blocks, 319])  # 319 is scenery on worlds 3 and 4
world_3_4_blocks = (20, common_blocks)
moving_block = (21, [239])
lift_block = (22, [230])
falling_block = (23, [238])
bouncing_boulder = (24, [194, 195, 210, 211])
pushable_blocks = (25, [128, 130, 354])  # 354 invisible on 2-2
question_block = (26, [129])
# add pipes here if they should be separate
spike = (28, [237])
lever = (29, [255])  # Lever for level end

# Enemies
goomba = (30, [144])
koopa = (30, [150, 151, 152, 153])
shell = (32, [154, 155])
explosion = (33, [157, 158])
piranha_plant = (34, [146, 147, 148, 149])
bill_launcher = (35, [135, 136])
bullet_bill = (36, [249])
projectiles = (
    37,
    [
        # fireball
        226,
        # spitting plant seed
        227,
    ],
)
flying_moth_arrow = (37, [172, 188])

# Level specific enemies
sharedEnemy1 = [160, 161, 162, 163, 176, 177, 178, 179]
moth = (30, sharedEnemy1)
flying_moth = (30, [192, 193, 194, 195, 208, 209, 210, 211])
sharedEnemy2 = [164, 165, 166, 167, 180, 181, 182, 183]
sphinx = (30, sharedEnemy2)
sharedEnemy3 = [192, 193, 208, 209]
bone_fish = (30, sharedEnemy3)
seahorse = (30, sharedEnemy2)
sharedEnemy4 = [196, 197, 198, 199, 212, 213, 214, 215]
robot = (30, sharedEnemy4)
fist_rock = (30, sharedEnemy2)
flying_rock = (30, [171, 187])
falling_spider = (30, sharedEnemy4)
jumping_spider = (30, sharedEnemy1)
zombie = (30, sharedEnemy1)
fire_worm = (30, sharedEnemy2)
spitting_plant = (30, sharedEnemy3)
fist = (51, [240, 241, 242, 243])

# Bosses
big_sphinx = (60, [198, 199, 201, 202, 203, 204, 205, 206, 214, 215, 217, 218, 219])
big_sphinx_fire = (37, [196, 197, 212, 213])
big_fist_rock = (62, [188, 189, 204, 205, 174, 175, 190, 191, 206, 207])

base_tiles = [
    base_scripts,
    # plane,
    # submarine,
    mario_fireball,
    # coin,
    mushroom,
    flower,
    star,
    # heart,
    moving_block,
    lift_block,
    falling_block,
    pushable_blocks,
    question_block,
    spike,
    lever,
    goomba,
    koopa,
    shell,
    explosion,
    piranha_plant,
    bill_launcher,
    bullet_bill,
    projectiles,
]


def _buildCompressedTileset(tiles) -> np.ndarray:
    compressedTileset = np.zeros(TILES, dtype=np.uint8)

    for t in tiles:
        i, tileList = t
        for tile in tileList:
            compressedTileset[tile] = i

    return compressedTileset


worldTilesets = {
    1: _buildCompressedTileset(
        [
            *base_tiles,
            world_1_2_blocks,
            moth,
            flying_moth,
            flying_moth_arrow,
            sphinx,
            big_sphinx,
            big_sphinx_fire,
        ]
    ),
    2: _buildCompressedTileset(
        [
            *base_tiles,
            world_1_2_blocks,
            bone_fish,
            seahorse,
            robot,
        ]
    ),
    3: _buildCompressedTileset(
        [
            *base_tiles,
            world_3_4_blocks,
            fist_rock,
            flying_rock,
            bouncing_boulder,
            falling_spider,
            jumping_spider,
            big_fist_rock,
        ]
    ),
    4: _buildCompressedTileset(
        [
            *base_tiles,
            world_3_4_blocks,
            zombie,
            fire_worm,
            spitting_plant,
        ]
    ),
}


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
        self.nextCheckpoint = 0
        self.levelCheckpoints = {
            (1, 1): [950, 1605],
            (1, 2): [1150, 1840],
            (2, 1): [855, 1610],
            (2, 2): [870, 1975],
            (3, 1): [2130, 2784],
            (3, 2): [930, 1980],
            (4, 1): [865, 1970],
            (4, 2): [980, 2150],
        }

        # so level progress max will be set
        self.gameWrapper.start_game()

    def reset(self, options: dict[str, Any] | None = None) -> MarioLandGameState:
        # this will be passed before evals are started, reset the eval
        # state counter so all evals will start at the same state
        if options is not None and options["_eval_starting"]:
            self.evalStateCounter = 0
            return self.gameState()

        if self.isEval:
            # evaluate levels in order
            self.stateIdx = self.evalStateCounter
            # don't start from checkpoints
            self.evalStateCounter += 3
            if self.evalStateCounter >= len(self.stateFiles):
                self.evalStateCounter = 0
        else:
            # reset game state to a random level
            self.stateIdx = random.randint(0, len(self.stateFiles) - 1)

        return self._loadLevel()

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

        self._setNextCheckpoint((world, level), self.stateCheckpoint)
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
            if random.randint(0, 100) <= RANDOM_POWERUP_CHANCE:
                # 0: small with star
                # 1: big
                # 2: big with star
                # 3: fire flower
                # 4: fire flower with star
                randPowerup = random.randint(0, 4)
                if randPowerup in (0, 2, 4):
                    self.pyboy.set_memory_value(STAR_TIMER_MEM_VAL, 248)
                if randPowerup != STATUS_SMALL:
                    self.pyboy.set_memory_value(POWERUP_STATUS_MEM_VAL, STATUS_BIG)
                    if randPowerup > 2:
                        self.pyboy.set_memory_value(HAS_FIRE_FLOWER_MEM_VAL, 1)

                prevState = curState
                curState = self.gameState()
                self._handlePowerup(prevState, curState, True)

        # level checkpoints get less time
        timerHundreds = 4 - self.stateCheckpoint

        # set level timer
        self.pyboy.set_memory_value(0x9831, timerHundreds)
        self.pyboy.set_memory_value(0x9832, 0)
        self.pyboy.set_memory_value(0x9833, 0)

        return curState

    def _setNextCheckpoint(self, world, currentCheckpoint):
        self.nextCheckpoint = 0
        if currentCheckpoint < 2:
            self.nextCheckpoint = self.levelCheckpoints[world][self.stateCheckpoint]

    def reward(self, prevState: MarioLandGameState) -> (float, MarioLandGameState):
        curState = self.gameState()

        # return flat punishment on mario's death
        if self._isDead(curState):
            return -50, curState

        # handle level clear
        if curState.statusTimer == TIMER_LEVEL_CLEAR:
            # if we're evaluating and the level is cleared return, the
            # episode is over; otherwise load the next level directly
            # to avoid processing unnecessary frames and the AI playing
            # levels we don't want it to
            if self.isEval:
                return 50, curState
            else:
                # load start of next level, not a level checkpoint
                self.stateIdx += (2 - self.stateCheckpoint) + 1
                if self.stateIdx >= len(self.stateFiles):
                    self.stateIdx = 0

                curState = self._loadLevel()

            # reset level progress max on new level
            self.gameWrapper._level_progress_max = curState.xPos
            curState.levelProgressMax = curState.xPos

            return 50, curState

        # add time punishment every step to encourage speed more
        clock = -0.25
        movement = curState.xPos - prevState.xPos

        # reward for passing checkpoints
        checkpoint = 0
        if (
            curState.levelProgressMax != prevState.levelProgressMax
            and self.nextCheckpoint != 0
            and curState.levelProgressMax >= self.nextCheckpoint
        ):
            self.currentCheckpoint += 1
            self._setNextCheckpoint(curState.world, self.currentCheckpoint)
            checkpoint = 50

        if self.isEval:
            if curState.levelProgressMax - prevState.levelProgressMax == 0:
                self.evalNoProgress += 1
            else:
                self.evalNoProgress = 0

        reward = (
            clock
            + movement
            + checkpoint
            + self._handlePowerup(prevState, curState, False)
        )

        return reward, curState

    def _handlePowerup(
        self,
        prevState: MarioLandGameState,
        curState: MarioLandGameState,
        manuallySet: bool,
    ) -> int:
        # only reward getting star once
        powerup = 0
        if (prevState.gotStar or prevState.hasStar) and curState.gotStar:
            curState.gotStar = False
        if curState.gotStar:
            self.invincibilityTimer = STAR_TIME
            if manuallySet:
                # when a star is given to mario via manipulating RAM
                # the invinciblility lasts a bit longer
                self.invincibilityTimer += 36
            # The actual star timer is set to 248 and only ticks down
            # when the frame counter is a one greater than a number
            # divisible by four. Don't ask me why. This accounts for
            # extra invincibility frames depending on what the frame
            # counter was at when the star was picked up
            frames = self.pyboy.get_memory_value(FRAME_COUNTER_MEM_VAL)
            extra = (frames - 1) % 4
            self.invincibilityTimer += extra
            powerup += 40
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
                powerup = 25
            elif prevState.powerupStatus == STATUS_BIG:
                if curState.powerupStatus == STATUS_FIRE:
                    powerup = 20
                elif curState.powerupStatus == STATUS_SMALL:
                    self.invincibilityTimer = SHRINK_TIME
                    powerup = -10
            elif prevState.powerupStatus == STATUS_FIRE:
                # mario got hit and lost the fire flower
                self.invincibilityTimer = SHRINK_TIME
                powerup = -10

        if self.invincibilityTimer != 0:
            curState.invincibleTimer = self.invincibilityTimer
            self.invincibilityTimer -= 1

        return powerup

    def observation(self, gameState: MarioLandGameState) -> Any:
        obs = self.gameWrapper._game_area_np(self.tileSet)

        # if mario is invincible his sprites will periodically flash by
        # cycling between being visible and not visible, ensure they are
        # always visible in the game area
        if gameState.isInvincible:
            self._drawMario(obs)

        # flatten the game area array so it's Box compatible
        flatObs = np.concatenate(obs.tolist(), axis=None, dtype=np.int32)

        # add other features
        return np.append(
            flatObs,
            [
                gameState.powerupStatus,
                gameState.hasStar,
                gameState.invincibleTimer,
                gameState.xPos,
                gameState.yPos,
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
        size = GAME_AREA_HEIGHT * GAME_AREA_WIDTH
        b = Box(low=0, high=TILES, shape=(size,))
        # TODO: make invincible an accurate timer with frameskips
        # add space for powerup status, is invincible, star timer, x and y pos
        # dunno max x pos so just making it max uint16
        low = np.append(b.low, [0, 0, 0, 0, 0])
        high = np.append(b.high, [3, 1, 1000, 65_535, 255])
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
        return len(self.stateFiles) // 3

    def gameState(self):
        return MarioLandGameState(self.pyboy)

    def printGameState(
        self, prevState: MarioLandGameState, curState: MarioLandGameState
    ):
        s = f"""
Max level progress: {curState.levelProgressMax}
Powerup: {curState.powerupStatus}
Status timer: {curState.statusTimer} {self.pyboy.get_memory_value(STAR_TIMER_MEM_VAL)} {self.pyboy.get_memory_value(0xDA00)}
X, Y: {curState.xPos}, {curState.yPos}
Invincibility: {curState.gotStar} {curState.hasStar} {curState.isInvincible} {curState.invincibleTimer}
"""
        print(s[1:], flush=True)

    def render(self):
        return self.pyboy.screen_image()
