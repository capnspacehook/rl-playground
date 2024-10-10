import numpy as np
from pyboy import PyBoy
from pyboy.api.constants import TILES

from rl_playground.env_settings.super_mario_land.constants import *
from rl_playground.env_settings.super_mario_land.ram import MarioLandGameState

# starts at 1 so 0 can represent empty space
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
coin = (3, [244])
mushroom = (4, [131])
flower = (5, [224, 229])
star = (6, [134])
heart = (7, [132])

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

world_1_2_3_pipes = (8, [*pipe_tiles])
world_4_pipes = (8, [*pipe_tiles, *world_4_extra_pipe_tiles])
world_1_blocks = (
    9,
    [*common_block_tiles, *world_2_jump_through_block_tiles, 319, *world_4_extra_pipe_tiles],
)  # 319 is scenery on all other worlds
world_2_blocks = (
    9,
    [*common_block_tiles, *world_1_jump_through_block_tiles, *world_4_extra_pipe_tiles],
)
world_3_blocks = (
    9,
    [
        *common_block_tiles,
        *world_1_jump_through_block_tiles,
        *world_2_jump_through_block_tiles,
        *world_4_extra_pipe_tiles,
    ],
)
world_4_blocks = (
    9,
    [*common_block_tiles, *world_1_jump_through_block_tiles, *world_2_jump_through_block_tiles],
)
world_1_jump_through_blocks = (10, [360, 361, 362])
world_2_jump_through_blocks = (10, [352, 353, 355])
moving_block = (11, [239])
crush_blocks = (12, [221, 222])
falling_stalactites = (13, [223])
falling_block = (14, [238])
pushable_blocks = (15, [128, 130, 354])  # 354 invisible on 2-2
question_block = (16, [129])
# add pipes here if they should be separate
spike = (17, [237])
lever = (18, [225])  # Lever for level end

# Enemies
goomba = (19, [144])
koopa = (19, [150, 151, 152, 153])
shell = (20, [154, 155])
explosion = (21, [157, 158])
piranha_plant = (22, [146, 147, 148, 149])
bill_launcher = (23, [135, 136])
bullet_bill = (19, [249])
projectiles = (
    24,
    [
        # fireball
        226,
        # spitting plant seed
        227,
    ],
)
flying_moth_arrow = (24, [172, 188])

# Level specific enemies
sharedEnemy1 = [160, 161, 162, 163, 176, 177, 178, 179]
moth = (19, sharedEnemy1)
flying_moth = (19, [192, 193, 194, 195, 208, 209, 210, 211])
sharedEnemy2 = [164, 165, 166, 167, 180, 181, 182, 183]
sphinx = (19, sharedEnemy2)
sharedEnemy3 = [192, 193, 208, 209]
bone_fish = (19, sharedEnemy3)
seahorse = (19, sharedEnemy2)
sharedEnemy4 = [196, 197, 198, 199, 212, 213, 214, 215]
robot = (19, sharedEnemy4)
fist_rock = (19, sharedEnemy2)
flying_rock = (19, [171, 187])
falling_spider = (19, sharedEnemy4)
jumping_spider = (19, sharedEnemy1)
zombie = (19, [*sharedEnemy1, 168, 169])
fire_worm = (19, sharedEnemy2)
spitting_plant = (19, sharedEnemy3)
bouncing_boulder_tiles = [194, 195, 210, 211]
bouncing_boulder = (25, bouncing_boulder_tiles)

# Dead enemies
deadSharedEnemy1 = [168, 169]
deadSharedEnemy2 = [184, 185]
deadSharedEnemy3 = [216, 217]
dead_moth = (26, deadSharedEnemy1)
dead_flying_moth = (26, [200, 201])
dead_sphinx = (26, deadSharedEnemy2)
dead_robot = (26, deadSharedEnemy3)
dead_flying_rock = (26, [173])
dead_fist_rock = (26, deadSharedEnemy2)
dead_falling_spider = (26, deadSharedEnemy3)
dead_jumping_spider = (26, deadSharedEnemy1)
dead_fire_worm = (26, deadSharedEnemy2)

# Bosses
big_sphinx = (27, [171, 187, 198, 199, 202, 203, 204, 205, 206, 214, 215, 218, 219, 220])
big_sphinx_fire = (28, [196, 197, 212, 213])
big_fist_rock = (29, [188, 189, 204, 205, 174, 175, 190, 191, 206, 207])

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
    falling_stalactites,
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
            dead_moth,
            flying_moth,
            dead_flying_moth,
            flying_moth_arrow,
            sphinx,
            dead_sphinx,
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
            dead_robot,
        ]
    ),
    3: _buildCompressedTileset(
        [
            *base_tiles,
            world_1_2_3_pipes,
            world_3_blocks,
            flying_rock,
            dead_flying_rock,
            fist_rock,
            dead_fist_rock,
            bouncing_boulder,
            falling_spider,
            dead_falling_spider,
            jumping_spider,
            dead_jumping_spider,
            big_fist_rock,
        ]
    ),
    4: _buildCompressedTileset(
        [
            *base_tiles,
            world_4_pipes,
            world_4_blocks,
            zombie,
            spitting_plant,
            fire_worm,
            dead_fire_worm,
        ]
    ),
}


def getGameArea(pyboy: PyBoy, curState: MarioLandGameState) -> np.ndarray:
    gameArea = pyboy.game_area()
    gameArea = np.array(gameArea, dtype=np.uint8, copy=False)
    if curState.isInvincible:
        _drawMario(pyboy, gameArea)

    return gameArea


def _drawMario(pyboy: PyBoy, gameArea: np.ndarray):
    # convert relative to screen y pos to sprite pos
    relYPos = pyboy.memory[0xC201] - 22
    marioLeftHead = pyboy.get_sprite(3)
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
