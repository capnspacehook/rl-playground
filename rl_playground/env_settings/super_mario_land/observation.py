from typing import Any, Deque, Dict, Tuple

from gymnasium import spaces
import numpy as np
from pyboy import PyBoy

from gymnasium.wrappers.frame_stack import FrameStack

from rl_playground.env_settings.super_mario_land.constants import *
from rl_playground.env_settings.super_mario_land.game_area import getGameArea
from rl_playground.env_settings.super_mario_land.ram import MarioLandGameState
from rl_playground.env_settings.super_mario_land.settings import N_STACK


def observationSpace() -> spaces.Space:
    return spaces.Dict(
        {
            GAME_AREA_OBS: spaces.Box(
                low=0, high=MAX_TILE, shape=(N_STACK, GAME_AREA_HEIGHT, GAME_AREA_WIDTH), dtype=np.uint8
            ),
            ENTITY_ID_OBS: spaces.Box(low=0, high=MAX_ENTITY_ID, shape=(N_STACK, N_ENTITIES), dtype=np.uint8),
            ENTITY_INFO_OBS: spaces.Box(low=0, high=1, shape=(N_STACK, N_ENTITIES, 6), dtype=np.float32),
            SCALAR_OBS: spaces.Box(low=0, high=1, shape=(N_STACK, 5), dtype=np.float32),
        }
    )


def getStackedObservation(
    pyboy: PyBoy,
    tileSet: np.ndarray,
    obsCache: Tuple[Deque[np.ndarray], Deque[np.ndarray], Deque[np.ndarray], Deque[np.ndarray]],
    prevState: MarioLandGameState,
    curState: MarioLandGameState,
) -> Dict[str, Any]:
    gameArea, entityIDs, entityInfos, scalar = getObservations(pyboy, tileSet, prevState, curState)

    obsCache[0].append(gameArea)
    obsCache[1].append(entityIDs)
    obsCache[2].append(entityInfos)
    obsCache[3].append(scalar)

    return combineObservations(obsCache)


def getObservations(
    pyboy: PyBoy,
    tileSet: np.ndarray,
    prevState: MarioLandGameState,
    curState: MarioLandGameState,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        getGameArea(pyboy, tileSet, curState),
        *getEntityIDsAndInfo(prevState, curState),
        getScalarFeatures(curState),
    )


def combineObservations(
    obsCache: Tuple[Deque[np.ndarray], Deque[np.ndarray], Deque[np.ndarray], Deque[np.ndarray]]
) -> Dict[str, Any]:
    return {
        GAME_AREA_OBS: np.squeeze(np.array(obsCache[0])),
        ENTITY_ID_OBS: np.squeeze(np.array(obsCache[1])),
        ENTITY_INFO_OBS: np.squeeze(np.array(obsCache[2])),
        SCALAR_OBS: np.squeeze(np.array(obsCache[3])),
    }


def getEntityIDsAndInfo(
    prevState: MarioLandGameState,
    curState: MarioLandGameState,
) -> Tuple[np.ndarray, np.ndarray]:
    # a level was completed on the last step, discard the last step's
    # state to avoid incorrect speed and acceleration calculations
    if prevState.world != curState.world:
        prevState = curState

    ids = np.zeros((N_ENTITIES,), dtype=np.uint8)
    # the first ID is always mario
    ids[0] = 1
    if len(curState.objects) != 0:
        for i in range(1, len(curState.objects)):
            # TODO: coalesce typeID to curated list
            ids[i] = curState.objects[i - 1].typeID

    # TODO: maybe make these an average over multiple frames?
    # pass in a list of game states maybe?
    curState.xSpeed = curState.xPos - prevState.xPos
    curState.ySpeed = curState.yPos - prevState.yPos
    xAccel = curState.xSpeed - prevState.xSpeed
    yAccel = curState.ySpeed - prevState.ySpeed

    entities = np.zeros((N_ENTITIES, 6), dtype=np.float32)
    entities[0] = np.array(
        [
            scaledEncoding(curState.xPos, MAX_X_POS, True),
            scaledEncoding(curState.yPos, MAX_Y_POS, True),
            scaledEncoding(curState.xSpeed, MAX_X_SPEED, False),
            scaledEncoding(curState.ySpeed, MAX_Y_SPEED, False),
            scaledEncoding(xAccel, MAX_X_SPEED * 2, False),
            scaledEncoding(yAccel, MAX_Y_SPEED * 2, False),
        ]
    )
    if len(curState.objects) != 0:
        for i in range(1, len(curState.objects)):
            obj = curState.objects[i - 1]
            if obj.relXPos > MAX_REL_X_POS or obj.yPos > MAX_Y_POS:
                continue

            entities[i] = np.array(
                [
                    scaledEncoding(obj.xPos, MAX_X_POS, True),
                    scaledEncoding(obj.yPos, MAX_Y_POS, True),
                    # TODO: calculate speeds and accels with prevState
                    0,
                    0,
                    0,
                    0,
                ]
            )

    return (ids, entities)


def getScalarFeatures(curState: MarioLandGameState) -> np.ndarray:
    return np.array(
        np.concatenate(
            (
                oneHotEncoding(curState.powerupStatus, 3),
                np.array([float(curState.hasStar)], dtype=np.float32),
                np.array([scaledEncoding(curState.invincibleTimer, 960, True)], dtype=np.float32),
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
