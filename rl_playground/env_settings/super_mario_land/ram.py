from typing import List
from pyboy import PyBoy

from rl_playground.env_settings.env_settings import GameState

# Memory constants
LEVEL_BLOCK_MEM_VAL = 0xC0AB
MARIO_MOVING_DIRECTION_MEM_VAL = 0xC20D
MARIO_X_POS_MEM_VAL = 0xC202
MARIO_Y_POS_MEM_VAL = 0xC201
MARIO_POSE_MEM_VAL = 0xC203
WORLD_LEVEL_MEM_VAL = 0xFFB4
NUM_WINS_MEM_VAL = 0xFF9A
LIVES_LEFT_MEM_VAL = 0xDA15
STATUS_TIMER_MEM_VAL = 0xFFA6
GAME_STATE_MEM_VAL = 0xFFB3
MARIO_ON_GROUND_MEM_VAL = 0xC20A
POWERUP_STATUS_MEM_VAL = 0xFF99
HAS_FIRE_FLOWER_MEM_VAL = 0xFFB5
STAR_TIMER_MEM_VAL = 0xC0D3
FRAME_COUNTER_MEM_VAL = 0xDA00
PROCESSING_OBJECT_MEM_VAL = 0xFFFB
OBJECTS_START_MEM_VAL = 0xD100

TIMER_HUNDREDS = 0xDA02
TIMER_TENS = 0xDA01
TIMER_FRAMES = 0xDA00
SCORE_MEM_VAL = 0xC0A0
SCORE_DISPLAY_MEM_VAL = 0x9820
COINS_MEM_VAL = 0xFFFA
COINS_DISPLAY_MEM_VAL = 0x9829
LIVES_LEFT_DISPLAY_MEM_VAL = 0x9806


MOVING_LEFT = 0x20
CROUCHING_POSE = 24
OBJ_TYPE_STAR = 0x34
BOSS1_TYPE = 8
BOSS2_TYPE = 50

STATUS_SMALL = 0
STATUS_GROWING = 1
STATUS_BIG = 2
STATUS_FIRE = 3

TIMER_DEATH = 0x90
TIMER_LEVEL_CLEAR = 0xF0
STAR_TIME = 956
SHRINK_TIME = 0x50 + 0x40


class MarioLandGameState(GameState):
    def __init__(self, pyboy: PyBoy):
        self.pyboy = pyboy

        # Find the real level progress x
        levelBlock = pyboy.memory[LEVEL_BLOCK_MEM_VAL]
        self.relXPos = pyboy.memory[MARIO_X_POS_MEM_VAL]
        scx = pyboy.memory[0xFF43]
        real = (scx - 7) % 16 if (scx - 7) % 16 != 0 else 16
        self.xPos = levelBlock * 16 + real + self.relXPos

        self.relYPos = self.pyboy.memory[MARIO_Y_POS_MEM_VAL]
        self.yPos = convertYPos(self.relYPos)

        # will be set later
        self.levelProgressMax = 0
        self.rawXSpeed = 0
        self.rawYSpeed = 0
        self.meanXSpeed = 0
        self.meanYSpeed = 0
        self.xAccel = 0
        self.yAccel = 0
        self.posReset = False

        world = self.pyboy.memory[WORLD_LEVEL_MEM_VAL]
        self.world = (world >> 4, world & 0x0F)
        # self.hardMode = self.pyboy.memory[NUM_WINS_MEM_VAL] != 0
        self.livesLeft = bcm_to_dec(self.pyboy.memory[LIVES_LEFT_MEM_VAL])
        self.coins = bcm_to_dec(self.pyboy.memory[COINS_MEM_VAL])
        self.score = bcm_to_dec(self.pyboy.memory[SCORE_MEM_VAL])
        self.score += bcm_to_dec(self.pyboy.memory[SCORE_MEM_VAL] + 1) * 100
        self.score += bcm_to_dec(self.pyboy.memory[SCORE_MEM_VAL] + 2) * 1000
        self.statusTimer = self.pyboy.memory[STATUS_TIMER_MEM_VAL]
        self.gameState = self.pyboy.memory[GAME_STATE_MEM_VAL]
        self.onGround = self.pyboy.memory[MARIO_ON_GROUND_MEM_VAL] == 1

        self.pipeWarping = False
        self.underground = False
        if self.gameState in (0x9, 0xA, 0xB, 0xC):
            self.pipeWarping = True
            if self.gameState in (0x9, 0xA):
                self.underground = True

        timerHundreds = self.pyboy.memory[TIMER_HUNDREDS]
        timerTens = bcm_to_dec(self.pyboy.memory[TIMER_TENS])
        self.timeLeft = (timerHundreds * 100) + timerTens

        self.objects: List[MarioLandObject] = []
        self.bossActive = False
        self.bossHealth = 0
        for i in range(10):
            addr = OBJECTS_START_MEM_VAL | (i * 0x10)
            objType = self.pyboy.memory[addr]
            if objType == 255 or objType not in typeIDMap:
                continue

            relXPos = self.pyboy.memory[addr] + 0x3
            xPos = levelBlock * 16 + real + relXPos
            relYPos = self.pyboy.memory[addr] + 0x2
            yPos = convertYPos(relYPos)
            objType = typeIDMap[objType]
            self.objects.append(MarioLandObject(objType, relXPos, xPos, yPos))

            if objType == BOSS1_TYPE or objType == BOSS2_TYPE:
                self.bossActive = True
                self.bossHealth = self.pyboy.memory[addr] | 0xC

        powerupStatus = self.pyboy.memory[POWERUP_STATUS_MEM_VAL]
        hasFireFlower = self.pyboy.memory[HAS_FIRE_FLOWER_MEM_VAL]
        starTimer = self.pyboy.memory[STAR_TIMER_MEM_VAL]
        processingObj = self.pyboy.memory[PROCESSING_OBJECT_MEM_VAL]

        self.powerupStatus = STATUS_SMALL
        self.gotMushroom = False
        self.gotStar = False
        self.hasStar = False
        self.isInvincible = False
        self.invincibleTimer = 0
        if starTimer != 0:
            self.hasStar = True
            self.isInvincible = True
            if processingObj == OBJ_TYPE_STAR:
                self.gotStar = True
        elif powerupStatus == 1:
            self.powerupStatus = STATUS_GROWING
        elif powerupStatus == 2:
            if hasFireFlower:
                self.powerupStatus = STATUS_FIRE
            else:
                self.powerupStatus = STATUS_BIG
        if powerupStatus == 3 or powerupStatus == 4:
            self.isInvincible = True


def bcm_to_dec(value: int) -> int:
    return (value >> 4) * 10 + (value & 0x0F)


def dec_to_bcm(value: int) -> int:
    return ((value // 10) << 4) | (value % 10)


def convertYPos(relYPos: int) -> int:
    yPos = 0

    # 191 is lowest y pos for an entity (that I've seen), y coordinate
    # is flipped, 0 is higher than 1
    if relYPos <= 191:
        yPos = 191 - relYPos
    else:
        # handle underflow
        yPos = 191 + (256 - relYPos)

    return yPos


TYPE_ID_MOVING_PLATFORM = 7
_typeIDs = [
    ((0x0,), 1),  # goomba
    ((0x2, 0x55), 2),  # pirana plant
    ((0x4,), 3),  # koopa
    ((0x5,), 4),  # koopa bomb
    ((0x8,), 5),  # sphinx boss
    ((0x9,), 6),  # spitting plant
    ((0xA, 0xB, 0x38, 0x39, 0x3A, 0x3B), TYPE_ID_MOVING_PLATFORM),  # moving platforms
    ((0xC, 0x35), 8),  # crush blocks
    ((0xE,), 9),  # moth/jumping spider
    ((0x10,), 10),  # fish
    ((0x13, 0x14), 11),  # lift blocks
    ((0x16, 0x17, 0x18), 12),  # robot
    ((0x1E, 0x23, 0x45, 0x51, 0x54), 13),  # projectiles
    ((0x24,), 14),  # seahorse
    ((0x25,), 15),  # falling spider
    ((0x27,), 16),  # explosion?
    ((0x28, 0x29), 17),  # mushroom
    ((0x2D, 0x2E), 18),  # fire flower
    ((0x2A, 0x2B), 19),  # heart (1-up)
    ((0x31,), 20),  # fist rock
    ((0x32,), 21),  # fist rock boss
    ((0x33, 0x47), 22),  # bouncing boulder
    ((0x34,), 23),  # star
    ((0x3C,), 24),  # flying rock
    ((0x3F,), 25),  # sphinx/dragon
    ((0x42,), 26),  # flying moth
    ((0x49,), 27),  # bill launcher
    ((0x4B,), 28),  # bullet bill | and 0x4A?
    ((0x56, 0x57), 29),  # zombie
]

typeIDMap = {}
for gameIDs, obsID in _typeIDs:
    for gameID in gameIDs:
        typeIDMap[gameID] = obsID


class MarioLandObject:
    def __init__(self, typeID: int, relX: int, x: int, y: int) -> None:
        self.typeID = typeID
        self.relXPos = relX
        self.xPos = x
        self.yPos = y

        # will be set later
        self.rawXSpeed = 0
        self.rawYSpeed = 0
        self.meanXSpeed = 0
        self.meanYSpeed = 0
        self.xAccel = 0
        self.yAccel = 0
