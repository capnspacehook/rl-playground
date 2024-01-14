import math
from typing import Any, Deque, Dict, List, Tuple

from gymnasium import spaces
import numpy as np
from pyboy import PyBoy
import torch

from rl_playground.env_settings.super_mario_land.constants import *
from rl_playground.env_settings.super_mario_land.game_area import getGameArea
from rl_playground.env_settings.super_mario_land.ram import (
    MarioLandGameState,
    MarioLandObject,
    CROUCHING_POSE,
)
from rl_playground.env_settings.super_mario_land.settings import *


def observationSpace() -> spaces.Dict:
    return spaces.Dict(
        {
            GAME_AREA_OBS: spaces.Box(
                low=0,
                high=MAX_TILE,
                shape=(N_GAME_AREA_STACK, GAME_AREA_HEIGHT, GAME_AREA_WIDTH),
                dtype=np.uint8,
            ),
            MARIO_INFO_OBS: spaces.Box(
                low=0, high=1, shape=(N_MARIO_OBS_STACK, MARIO_INFO_SIZE), dtype=np.float32
            ),
            ENTITY_ID_OBS: spaces.Box(
                low=0, high=MAX_ENTITY_ID, shape=(N_ENTITY_OBS_STACK, N_ENTITIES), dtype=np.uint8
            ),
            ENTITY_INFO_OBS: spaces.Box(
                low=0, high=1, shape=(N_ENTITY_OBS_STACK, N_ENTITIES, ENTITY_INFO_SIZE), dtype=np.float32
            ),
            SCALAR_OBS: spaces.Box(low=0, high=1, shape=(N_SCALAR_OBS_STACK, SCALAR_SIZE), dtype=np.float32),
        }
    )


def getStackedObservation(
    pyboy: PyBoy,
    tileSet: np.ndarray,
    obsCache: Tuple[Deque[np.ndarray], Deque[np.ndarray], Deque[np.ndarray], Deque[np.ndarray]],
    states: Deque[MarioLandGameState],
) -> Dict[str, Any]:
    gameArea, marioInfo, entityIDs, entityInfos, scalar = getObservations(pyboy, tileSet, states)

    obsCache[0].append(gameArea)
    obsCache[1].append(marioInfo)
    obsCache[2].append(entityIDs)
    obsCache[3].append(entityInfos)
    obsCache[4].append(scalar)

    return combineObservations(obsCache)


def getObservations(
    pyboy: PyBoy,
    tileSet: np.ndarray,
    states: Deque[MarioLandGameState],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        getGameArea(pyboy, tileSet, states[-1]),
        *getEntityIDsAndInfo(states),
        getScalarFeatures(states[-1]),
    )


def combineObservations(
    obsCache: Tuple[
        Deque[np.ndarray], Deque[np.ndarray], Deque[np.ndarray], Deque[np.ndarray], Deque[np.ndarray]
    ],
) -> Dict[str, Any]:
    return {
        GAME_AREA_OBS: np.squeeze(np.array(obsCache[0])),
        MARIO_INFO_OBS: np.squeeze(np.array(obsCache[1])),
        ENTITY_ID_OBS: np.squeeze(np.array(obsCache[2])),
        ENTITY_INFO_OBS: np.squeeze(np.array(obsCache[3])),
        SCALAR_OBS: np.squeeze(np.array(obsCache[4])),
    }


# numpy operations in this function cause compilations errors for some reason
@torch._dynamo.disable
def getEntityIDsAndInfo(
    states: Deque[MarioLandGameState],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    prevState = states[-2]
    curState = states[-1]

    # a level was completed on the last step, discard the last step's
    # state to avoid incorrect speed and acceleration calculations
    if prevState.world != curState.world:
        prevState = curState

    # In rare occasions (jumping on a moving platform in 3-3 for
    # instance) the scroll X value is ~16, while in the next frame
    # the scroll X value is where it was before but the level block
    # increased making the X position wildly jump again. Until I
    # figure out how to properly fix this, clip the X (and Y for
    # good measure) speed to ensure it's always a sane value.
    curState.rawXSpeed = np.clip(curState.xPos - prevState.xPos, -MARIO_MAX_X_SPEED, MARIO_MAX_X_SPEED)
    curState.rawYSpeed = np.clip(curState.yPos - prevState.yPos, -MARIO_MAX_Y_SPEED, MARIO_MAX_Y_SPEED)

    # only consider speeds in states of the same life or level as the
    # current state
    resetIdx = -1
    for i, s in enumerate(states):
        if s.posReset:
            resetIdx = i
            break

    # don't calculate mean speed if the latest state was the reset one
    if resetIdx != N_STATE_STACK - 1:
        curState.meanXSpeed = np.mean([states[i].rawXSpeed for i in range(resetIdx + 1, N_STATE_STACK)])
        curState.meanYSpeed = np.mean([states[i].rawYSpeed for i in range(resetIdx + 1, N_STATE_STACK)])
    curState.xAccel = curState.meanXSpeed - prevState.meanXSpeed
    curState.yAccel = curState.meanYSpeed - prevState.meanYSpeed

    marioInfo = np.array(
        [
            scaledEncoding(curState.relXPos, MAX_REL_X_POS, True),
            scaledEncoding(curState.yPos, MAX_Y_POS, True),
            scaledEncoding(curState.meanXSpeed, MARIO_MAX_X_SPEED, False),
            scaledEncoding(curState.meanYSpeed, MARIO_MAX_Y_SPEED, False),
            scaledEncoding(curState.xAccel, MARIO_MAX_X_SPEED, False),
            scaledEncoding(curState.yAccel, MARIO_MAX_Y_SPEED, False),
            scaledEncoding(math.atan2(curState.meanXSpeed, curState.meanYSpeed), math.pi, False),
        ],
        dtype=np.float32,
    )
    marioPos = np.array((curState.xPos, curState.yPos))

    entities = []
    if len(curState.objects) != 0:
        for i in range(len(curState.objects)):
            obj = curState.objects[i]

            # attempt to find the same object in the previous frame's state
            # so the speed and acceleration can be calculated
            if len(prevState.objects) != 0:
                prevObj = findObjectInPrevState(obj, prevState)
                if prevObj is not None:
                    obj.rawXSpeed = obj.xPos - prevObj.xPos
                    obj.rawYSpeed = obj.yPos - prevObj.yPos
                    rawXSpeeds = [prevObj.rawXSpeed, obj.rawXSpeed]
                    rawYSpeeds = [prevObj.rawYSpeed, obj.rawYSpeed]
                    obj.meanXSpeed, obj.meanYSpeed = calculateMeanSpeeds(states, obj, rawXSpeeds, rawYSpeeds)
                    obj.xAccel = obj.meanXSpeed - prevObj.meanXSpeed
                    obj.yAccel = obj.meanYSpeed - prevObj.meanYSpeed

            # calculate speed for offscreen objects for when they come
            # onscreen but don't add them to the observation
            if obj.relXPos > MAX_REL_X_POS or obj.yPos > MAX_Y_POS:
                continue

            xDistance = obj.xPos - curState.xPos
            yDistance = obj.yPos - curState.yPos
            euclideanDistance = np.linalg.norm(marioPos - np.array((obj.xPos, obj.yPos)))
            entities.append(
                (
                    obj.typeID,
                    np.array(
                        [
                            scaledEncoding(obj.relXPos, MAX_REL_X_POS, True),
                            scaledEncoding(obj.yPos, MAX_Y_POS, True),
                            scaledEncoding(xDistance, MAX_X_DISTANCE, False),
                            scaledEncoding(yDistance, MAX_Y_DISTANCE, False),
                            scaledEncoding(euclideanDistance, MAX_EUCLIDEAN_DISTANCE, True),
                            scaledEncoding(obj.meanXSpeed, ENTITY_MAX_MEAN_X_SPEED, False),
                            scaledEncoding(obj.meanYSpeed, ENTITY_MAX_MEAN_Y_SPEED, False),
                            scaledEncoding(obj.xAccel, ENTITY_MAX_MEAN_X_SPEED, False),
                            scaledEncoding(obj.yAccel, ENTITY_MAX_MEAN_Y_SPEED, False),
                            scaledEncoding(math.atan2(obj.meanXSpeed, obj.meanYSpeed), math.pi, False),
                        ],
                        dtype=np.float32,
                    ),
                )
            )

    # sort entities by euclidean distance to mario
    sortedEntities = sorted(entities, key=lambda o: o[1][2])

    ids = [i[0] for i in sortedEntities]
    paddingIDs = np.zeros((N_ENTITIES - len(ids)), dtype=np.uint8)
    allIDs = np.concatenate((ids, paddingIDs))

    entities = [i[1] for i in sortedEntities]
    paddingEntities = np.zeros((N_ENTITIES - len(entities), ENTITY_INFO_SIZE), dtype=np.float32)
    if len(entities) == 0:
        allEntities = paddingEntities
    else:
        allEntities = np.concatenate((entities, paddingEntities))

    return (marioInfo, allIDs, allEntities)


def findObjectInPrevState(obj: MarioLandObject, prevState: MarioLandGameState) -> MarioLandObject | None:
    prevObj: MarioLandObject = None
    prevObjs = [
        po
        for po in prevState.objects
        if obj.typeID == po.typeID
        and abs(obj.xPos - po.xPos) <= ENTITY_MAX_RAW_X_SPEED
        and abs(obj.yPos - po.yPos) <= ENTITY_MAX_RAW_Y_SPEED
    ]
    if len(prevObjs) == 1:
        prevObj = prevObjs[0]
    if len(prevObjs) > 1:
        prevObj = min(prevObjs, key=lambda po: abs(obj.xPos - po.xPos) + abs(obj.yPos - po.yPos))

    return prevObj


def calculateMeanSpeeds(
    states: Deque[MarioLandGameState], obj: MarioLandObject, rawXSpeeds: List[int], rawYSpeeds: List[int]
) -> (float, float):
    # we already have the previous and current raw speeds
    for i in range(N_STATE_STACK - 2):
        state = states[i]
        xSpeed = 0
        ySpeed = 0
        prevObj = findObjectInPrevState(obj, state)
        if prevObj is not None:
            xSpeed = prevObj.rawXSpeed
            ySpeed = prevObj.rawYSpeed
        rawXSpeeds.append(xSpeed)
        rawYSpeeds.append(ySpeed)

    return np.mean(rawXSpeeds), np.mean(rawYSpeeds)


def getScalarFeatures(curState: MarioLandGameState) -> np.ndarray:
    return np.array(
        np.concatenate(
            (
                oneHotEncoding(curState.powerupStatus, POWERUP_STATUSES),
                np.array([float(curState.pose == CROUCHING_POSE), float(curState.hasStar)], dtype=np.float32),
                np.array(
                    [
                        scaledEncoding(curState.invincibleTimer, MAX_INVINCIBILITY_TIME, True),
                        scaledEncoding(curState.livesLeft, 99, True),
                        scaledEncoding(curState.coins, 99, True),
                        scaledEncoding(curState.timeLeft, 400, True),
                    ],
                    dtype=np.float32,
                ),
            ),
        )
    )


def scaledEncoding(val: int, max: int, minIsZero: bool) -> float:
    # if val > max:
    #     print(f"{val} > {max}")
    # elif minIsZero and val < 0:
    #     print(f"{val} < 0")
    # elif not minIsZero and val < -max:
    #     print(f"{val} < {-max}")

    scaled = 0.0
    if minIsZero:
        scaled = val / max
    else:
        # minimum value is less than zero, ensure scaling minimum is zero
        scaled = (val + max) / (max * 2)

    return np.clip(scaled, 0.0, 1.0)


def oneHotEncoding(val: int, max: int) -> np.ndarray:
    return np.squeeze(np.identity(max, dtype=np.float32)[val : val + 1])
