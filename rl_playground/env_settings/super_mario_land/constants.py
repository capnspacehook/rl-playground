GAME_AREA_OBS = "gameArea"
ENTITY_ID_OBS = "entityID"
ENTITY_INFO_OBS = "entityInfo"
SCALAR_OBS = "scalar"

ENTITY_INFO_SIZE = 8
SCALAR_SIZE = 6

MARIO_MAX_X_SPEED = 4
MARIO_MAX_Y_SPEED = 4
ENTITY_MAX_RAW_X_SPEED = 10
ENTITY_MAX_RAW_Y_SPEED = 12
ENTITY_MAX_MEAN_X_SPEED = 3
ENTITY_MAX_MEAN_Y_SPEED = 3
# distance between 0, 0 and 160, 210 is 264.007... so rounding up
MAX_EUCLIDEAN_DISTANCE = 265

POWERUP_STATUSES = 4
MAX_INVINCIBILITY_TIME = 960

MAX_ENTITY_ID = 30

# max number of objects that can be on screen at once (excluding mario)
N_OBJECTS = 10
# max number of objects that con be on screen at once
N_ENTITIES = N_OBJECTS + 1

MAX_REL_X_POS = 160
MAX_X_POS = 3880
MAX_Y_POS = 210

# Game area dimensions
GAME_AREA_HEIGHT = 16
GAME_AREA_WIDTH = 20

# update if the maximum tile value changes
MAX_TILE = 61
