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
    "gamma": 0.98,
    "learning_rate": 3.183807492928217e-05,
    "max_grad_norm": 5,
    "n_epochs": 5,
    "n_steps": 512,
    "vf_coef": 0.33653746631712467,
    "policy_kwargs": dict(
        activation_fn=nn.relu,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    ),
}

RANDOM_NOOP_FRAMES = 60

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
        xPos = pyboy.get_memory_value(0xC202)
        scx = pyboy.botsupport_manager().screen().tilemap_position_list()[16][0]
        real = (scx - 7) % 16 if (scx - 7) % 16 != 0 else 16

        self.realXPos = levelBlock * 16 + real + xPos
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
    ],
)
world_1_2_blocks = (20, [*common_blocks, 319])  # 319 is scenery on worlds 3 and 4
world_3_4_blocks = (20, common_blocks)
moving_blocks = (21, [230, 239])
falling_block = (22, [238])
bouncing_boulder = (23, [194, 195, 210, 211])
pushable_blocks = (24, [128, 130, 354])  # 354 invisible on 2-2
question_block = (25, [129])
pipes = (26, list(range(368, 381)))
spike = (27, [237])
lever = (28, [255])  # Lever for level end

# Enemies
goomba = (30, [144])
koopa = (31, [150, 151, 152, 153])
shell = (32, [154, 155])
explosion = (33, [157, 158])
piranha_plant = (34, [146, 147, 148, 149])
bill_launcher = (35, [135, 136])
bill = (36, [249])

# Level specific enemies
sharedEnemy1 = [160, 161, 162, 163, 176, 177, 178, 179]
moth = (37, sharedEnemy1)
flying_moth = (38, [192, 193, 194, 195, 208, 209, 210, 211])
arrow = (39, [172, 188])
sharedEnemy2 = [164, 165, 166, 167, 180, 181, 182, 183]
sphinx = (40, sharedEnemy2)
sharedEnemy3 = [192, 193, 208, 209]
bone_fish = (41, sharedEnemy3)
seahorse = (42, sharedEnemy2)
fireball = (43, [226])
sharedEnemy4 = [196, 197, 198, 199, 212, 213, 214, 215]
robot = (44, sharedEnemy4)
fist_rock = (45, sharedEnemy2)
flying_rock = (46, [171, 187])
falling_spider = (47, sharedEnemy4)
jumping_spider = (48, sharedEnemy1)
zombie = (49, sharedEnemy1)
fire_worm = (50, sharedEnemy2)
spitting_plant = (51, bone_fish)
seed = (52, [227])
fist = (53, [240, 241, 242, 243])

# Bosses
big_sphinx = (60, [198, 199, 201, 202, 203, 204, 205, 206, 214, 215, 217, 218, 219])
sphinx_fire = (61, [196, 197, 212, 213])
big_fist_rock = (62, [188, 189, 204, 205, 174, 175, 190, 191, 206, 207])

base_tiles = [
    base_scripts,
    plane,
    submarine,
    mario_fireball,
    coin,
    mushroom,
    flower,
    star,
    heart,
    moving_blocks,
    falling_block,
    bouncing_boulder,
    pushable_blocks,
    question_block,
    pipes,
    spike,
    lever,
    goomba,
    koopa,
    shell,
    explosion,
    piranha_plant,
    bill_launcher,
    bill,
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
            arrow,
            sphinx,
            big_sphinx,
            sphinx_fire,
        ]
    ),
    2: _buildCompressedTileset(
        [
            *base_tiles,
            world_1_2_blocks,
            bone_fish,
            seahorse,
            fireball,
            robot,
        ]
    ),
    3: _buildCompressedTileset(
        [
            *base_tiles,
            world_3_4_blocks,
            fist_rock,
            flying_rock,
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
            fireball,
            spitting_plant,
            seed,
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

    def reset(self, options: dict[str, Any] | None = None):
        # this will be passed before evals are started, reset the eval
        # state counter so all evals will start at the same state
        if options is not None and options["_eval_starting"]:
            self.evalStateCounter = 0
            return

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

        self._loadLevel()

    def _loadLevel(self):
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

        # level checkpoints get less time
        timerHundreds = 4 - self.stateCheckpoint

        # set level timer
        self.pyboy.set_memory_value(0x9831, timerHundreds)
        self.pyboy.set_memory_value(0x9832, 0)
        self.pyboy.set_memory_value(0x9833, 0)

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
                self._loadLevel()

                curState = self.gameState()

            # reset level progress max on new level
            self.gameWrapper._level_progress_max = curState.realXPos
            curState.levelProgressMax = curState.realXPos

            return 50, curState

        # add time punishment every step to encourage speed more
        clock = -0.25
        movement = curState.realXPos - prevState.realXPos

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

        reward = clock + movement + checkpoint + powerup

        return reward, curState

    def observation(self, gameState: MarioLandGameState) -> Any:
        obs = self.gameWrapper._game_area_np(self.tileSet)
        # make 20x16 array a 1x320 array so it's Box compatible
        flatObs = np.concatenate(obs.tolist(), axis=None, dtype=np.int32)
        # add powerup status
        return np.append(flatObs, [gameState.powerupStatus])

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
        return len(self.stateFiles) // 3

    def gameState(self):
        return MarioLandGameState(self.pyboy)

    def printGameState(self, state: MarioLandGameState):
        print(f"Level progress: {state.realXPos}")
        print(f"Max level progress: {state.levelProgressMax}")
        print(f"Lives left: {state.livesLeft}")
        print(f"Powerup: {state.powerupStatus}")
        print(f"World: {state.world}")
        print(f"Time left: {state.timeLeft}")
        print(f"Time respawn: {state.statusTimer}")

    def render(self):
        return self.pyboy.screen_image()
