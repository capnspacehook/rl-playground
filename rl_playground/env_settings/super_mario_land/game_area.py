from typing import Any, Deque, Dict
import numpy as np
from pyboy import PyBoy
from pyboy.botsupport.constants import TILES

from rl_playground.env_settings.super_mario_land.constants import *
from rl_playground.env_settings.super_mario_land.ram import MarioLandGameState


mario = (
    1,
    [
        *list(range(0, 11)),
        *list(range(15, 27)),
        *list(range(31, 43)),
        *list(range(48, 59)),
        *list(range(66, 71)),
    ],
)
mario_fireball = (2, [96, 110, 122])

# Bonuses
coin = (12, [244])
mushroom = (13, [131])
flower = (14, [224, 229])
star = (15, [134])
heart = (16, [132])

# Blocks
pipe_tiles = list(range(368, 381))
world_4_extra_pipe_tiles = [363, 364, 365, 366]  # are normal blocks on other worlds
common_block_tiles = [
    142,
    143,
    230,  # lift block
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
    356,
    357,
    358,
    359,
    381,
    382,
    383,
]
world_1_jump_through_block_tiles = [360, 361, 362]
world_2_jump_through_block_tiles = [352, 353, 355]

world_1_2_3_pipes = (16, [*pipe_tiles])
world_4_pipes = (16, [*pipe_tiles, *world_4_extra_pipe_tiles])
world_1_blocks = (
    17,
    [*common_block_tiles, *world_2_jump_through_block_tiles, 319, *world_4_extra_pipe_tiles],
)  # 319 is scenery on worlds 3 and 4
world_2_blocks = (
    17,
    [*common_block_tiles, *world_1_jump_through_block_tiles, 319, *world_4_extra_pipe_tiles],
)  # 319 is scenery on worlds 3 and 4
world_3_blocks = (
    17,
    [
        *common_block_tiles,
        *world_1_jump_through_block_tiles,
        *world_2_jump_through_block_tiles,
        *world_4_extra_pipe_tiles,
    ],
)
world_4_blocks = (
    17,
    [*common_block_tiles, *world_1_jump_through_block_tiles, *world_2_jump_through_block_tiles],
)
world_1_jump_through_blocks = (18, [360, 361, 362])
world_2_jump_through_blocks = (18, [352, 353, 355])
moving_block = (19, [239])
crush_blocks = (20, [221, 222, 223])
falling_block = (21, [238])
pushable_blocks = (22, [128, 130, 354])  # 354 invisible on 2-2
question_block = (23, [129])
# add pipes here if they should be separate
spike = (24, [237])
lever = (25, [225])  # Lever for level end

# Enemies
goomba = (35, [144])
koopa = (35, [150, 151, 152, 153])
shell = (36, [154, 155])
explosion = (37, [157, 158])
piranha_plant = (38, [146, 147, 148, 149])
bill_launcher = (39, [135, 136])
bullet_bill = (35, [249])
projectiles = (
    40,
    [
        # fireball
        226,
        # spitting plant seed
        227,
    ],
)
flying_moth_arrow = (40, [172, 188])

# Level specific enemies
sharedEnemy1 = [160, 161, 162, 163, 176, 177, 178, 179]
moth = (35, sharedEnemy1)
flying_moth = (35, [192, 193, 194, 195, 208, 209, 210, 211])
sharedEnemy2 = [164, 165, 166, 167, 180, 181, 182, 183]
sphinx = (35, sharedEnemy2)
sharedEnemy3 = [192, 193, 208, 209]
bone_fish = (35, sharedEnemy3)
seahorse = (35, sharedEnemy2)
sharedEnemy4 = [196, 197, 198, 199, 212, 213, 214, 215]
robot = (35, sharedEnemy4)
fist_rock = (35, sharedEnemy2)
flying_rock = (35, [171, 187])
falling_spider = (35, sharedEnemy4)
jumping_spider = (35, sharedEnemy1)
zombie = (35, sharedEnemy1)
fire_worm = (35, sharedEnemy2)
spitting_plant = (35, sharedEnemy3)
bouncing_boulder_tiles = [194, 195, 210, 211]
bouncing_boulder = (41, bouncing_boulder_tiles)

# Bosses
big_sphinx = (51, [198, 199, 201, 202, 203, 204, 205, 206, 214, 215, 217, 218, 219])
big_sphinx_fire = (40, [196, 197, 212, 213])
big_fist_rock = (52, [188, 189, 204, 205, 174, 175, 190, 191, 206, 207])

base_tiles = [
    mario,
    mario_fireball,
    coin,
    mushroom,
    flower,
    star,
    heart,
    moving_block,
    crush_blocks,
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


# different worlds use the same tiles for different things so only load
# necessary tiles per world
worldTilesets = {
    1: _buildCompressedTileset(
        [
            *base_tiles,
            world_1_2_3_pipes,
            world_1_blocks,
            world_1_jump_through_blocks,
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
            world_1_2_3_pipes,
            world_2_blocks,
            world_2_jump_through_blocks,
            bone_fish,
            seahorse,
            robot,
        ]
    ),
    3: _buildCompressedTileset(
        [
            *base_tiles,
            world_1_2_3_pipes,
            world_3_blocks,
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
            world_4_pipes,
            world_4_blocks,
            zombie,
            fire_worm,
            spitting_plant,
        ]
    ),
}


def getGameArea(
    pyboy: PyBoy,
    tileSet: np.ndarray,
    curState: MarioLandGameState,
) -> np.ndarray:
    gameArea = pyboy.game_wrapper()._game_area_np(tileSet)
    if curState.isInvincible:
        _drawMario(pyboy, gameArea)

    # _game_area_np returns an ndarray with dtype uint16, pytorch doesn't
    # support uint16 for some reason
    return gameArea.astype(dtype=np.uint8, copy=False)


def _drawMario(pyboy: PyBoy, gameArea: np.ndarray):
    # convert relative to screen y pos to sprite pos
    relYPos = pyboy.get_memory_value(0xC201) - 22
    marioLeftHead = pyboy.botsupport_manager().sprite(3)
    x1 = min(marioLeftHead.x // 8, GAME_AREA_WIDTH - 1)
    x2 = min(x1 + 1, GAME_AREA_WIDTH - 1)
    if marioLeftHead.attr_x_flip:
        x2 = x1 - 1

    y1 = (marioLeftHead.y // 8) - 1
    if y1 >= GAME_AREA_HEIGHT:
        # sprite is not visible so y pos is off screen, set it to
        # correct pos where mario is
        y1 = (relYPos // 8) - 1
    y2 = y1 - 1

    if y1 >= 0 and y1 < GAME_AREA_HEIGHT:
        gameArea[y1][x1] = 1
        gameArea[y1][x2] = 1
    if y2 >= 0 and y2 < GAME_AREA_HEIGHT:
        gameArea[y2][x1] = 1
        gameArea[y2][x2] = 1
