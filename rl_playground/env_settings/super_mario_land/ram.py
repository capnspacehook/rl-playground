from typing import List
from pyboy import PyBoy

from rl_playground.env_settings.env_settings import GameState

# Memory constants
LEVEL_BLOCK_MEM_VAL = 0xC0AB
MARIO_MOVING_DIRECTION_MEM_VAL = 0xC20D
MARIO_X_POS_MEM_VAL = 0xC202
MARIO_Y_POS_MEM_VAL = 0xC201
WORLD_LEVEL_MEM_VAL = 0xFFB4
STATUS_TIMER_MEM_VAL = 0xFFA6
DEAD_JUMP_TIMER_MEM_VAL = 0xC0AC
MARIO_ON_GROUND_MEM_VAL = 0xC20A
POWERUP_STATUS_MEM_VAL = 0xFF99
HAS_FIRE_FLOWER_MEM_VAL = 0xFFB5
STAR_TIMER_MEM_VAL = 0xC0D3
FRAME_COUNTER_MEM_VAL = 0xDA00
PROCESSING_OBJECT_MEM_VAL = 0xFFFB
OBJECTS_START_MEM_VAL = 0xD100

MOVING_LEFT = 0x20
OBJ_TYPE_STAR = 0x34
BOSS1_TYPE = 8
BOSS2_TYPE = 50

STATUS_SMALL = 0
STATUS_BIG = 1
STATUS_FIRE = 2

TIMER_DEATH = 0x90
TIMER_LEVEL_CLEAR = 0xF0
STAR_TIME = 956
SHRINK_TIME = 0x50 + 0x40

OBJ_TYPES_MOVING_PLATFORM = (10, 11, 56, 57, 58, 59)


class MarioLandGameState(GameState):
    def __init__(self, pyboy: PyBoy):
        self.pyboy = pyboy
        self.gameWrapper = pyboy.game_wrapper()

        # Find the real level progress x
        levelBlock = pyboy.get_memory_value(LEVEL_BLOCK_MEM_VAL)
        self.relXPos = pyboy.get_memory_value(MARIO_X_POS_MEM_VAL)
        scx = pyboy.get_memory_value(0xFF43)
        real = (scx - 7) % 16 if (scx - 7) % 16 != 0 else 16
        self.xPos = levelBlock * 16 + real + self.relXPos

        self.relYPos = self.pyboy.get_memory_value(MARIO_Y_POS_MEM_VAL)
        self.yPos = convertYPos(self.relYPos)

        # will be set later
        self.xSpeed = 0
        self.ySpeed = 0

        self.levelProgressMax = max(self.gameWrapper._level_progress_max, self.xPos)
        world = self.pyboy.get_memory_value(WORLD_LEVEL_MEM_VAL)
        self.world = (world >> 4, world & 0x0F)
        self.statusTimer = self.pyboy.get_memory_value(STATUS_TIMER_MEM_VAL)
        self.deadJumpTimer = self.pyboy.get_memory_value(DEAD_JUMP_TIMER_MEM_VAL)
        self.onGround = self.pyboy.get_memory_value(MARIO_ON_GROUND_MEM_VAL) == 1
        self.movingPlatformObj = None

        self.objects: List[MarioLandObject] = []
        self.bossActive = False
        self.bossHealth = 0
        for i in range(10):
            addr = OBJECTS_START_MEM_VAL | (i * 0x10)
            objType = self.pyboy.get_memory_value(addr)
            if objType == 255:
                continue
            relXPos = self.pyboy.get_memory_value(addr + 0x3)
            xPos = levelBlock * 16 + real + relXPos
            relYPos = self.pyboy.get_memory_value(addr + 0x2)
            yPos = convertYPos(relYPos)
            self.objects.append(MarioLandObject(i, objType, relXPos, xPos, yPos))

            if objType == BOSS1_TYPE or objType == BOSS2_TYPE:
                self.bossActive = True
                self.bossHealth = self.pyboy.get_memory_value(addr | 0xC)

        powerupStatus = self.pyboy.get_memory_value(POWERUP_STATUS_MEM_VAL)
        hasFireFlower = self.pyboy.get_memory_value(HAS_FIRE_FLOWER_MEM_VAL)
        starTimer = self.pyboy.get_memory_value(STAR_TIMER_MEM_VAL)

        self.powerupStatus = STATUS_SMALL
        self.gotStar = False
        self.hasStar = False
        self.isInvincible = False
        self.invincibleTimer = 0
        if starTimer != 0:
            self.hasStar = True
            self.isInvincible = True
            if self.pyboy.get_memory_value(PROCESSING_OBJECT_MEM_VAL) == OBJ_TYPE_STAR:
                self.gotStar = True
        elif powerupStatus == 1:
            self.powerupStatus = STATUS_BIG
        elif powerupStatus == 2:
            if hasFireFlower:
                self.powerupStatus = STATUS_FIRE
            else:
                self.powerupStatus = STATUS_BIG
        if powerupStatus == 3 or powerupStatus == 4:
            self.isInvincible = True


def convertYPos(relYPos: int) -> int:
    yPos = 0

    # 185 is lowest y pos before mario is dead, y coordinate is flipped, 0 is higher than 1
    if relYPos <= 185:
        yPos = 185 - relYPos
    else:
        # handle underflow
        yPos = 185 + (256 - relYPos)

    return yPos


class MarioLandObject:
    def __init__(self, index, typeID, relX, x, y) -> None:
        self.index = index
        self.typeID = typeID
        self.relXPos = relX
        self.xPos = x
        self.yPos = y
