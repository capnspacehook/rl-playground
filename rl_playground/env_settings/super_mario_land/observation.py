from typing import Any, Dict, Tuple
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, TensorDict
import torch as th
from torch import nn

from rl_playground.env_settings.super_mario_land.game_area import MAX_TILE, getGameArea
from rl_playground.env_settings.super_mario_land.ram import MarioLandGameState


# max number of objects that can be on screen at once (excluding mario)
N_OBJECTS = 10
# max number of objects that con be on screen at once
N_ENTITIES = N_OBJECTS + 1

MAX_X_POS = 160
MAX_Y_POS = 200


def getObservation(
    pyboy: PyBoy, tileSet: np.ndarray, prevState: MarioLandGameState, curState: MarioLandGameState
) -> Dict[str, Any]:
    gameArea = getGameArea(pyboy, tileSet, curState)
    entities = getEntityIDsAndInfo(prevState, curState)
    scalar = getScalarFeatures(curState)

    return {
        "gameArea": gameArea,
        "entities": entities,
        "scalar": scalar,
    }


def getEntityIDsAndInfo(
    prevState: MarioLandGameState, curState: MarioLandGameState
) -> Tuple[np.ndarray, np.ndarray]:
    ids = np.zeros((11,), dtype=np.int16)
    # the first ID will be skipped, that will represent mario who will be 0
    for i in range(1, N_OBJECTS):
        # TODO: coalesce typeID to curated list
        ids[i] = curState.objects[i - 1].typeID

    # TODO: maybe make these an average over multiple frames?
    # pass in a list of game states maybe?
    curState.xSpeed = curState.xPos - prevState.xPos
    curState.ySpeed = curState.yPos - prevState.yPos
    xAccel = curState.xSpeed - prevState.xSpeed
    yAccel = curState.ySpeed - prevState.ySpeed

    infos = np.zeros((11, 6), dtype=np.int16)
    infos[0] = np.array(
        [
            scaledEncoding(curState.xPos, MAX_X_POS, True),
            scaledEncoding(curState.yPos, MAX_Y_POS, True),
            scaledEncoding(curState.xSpeed, 2, False),
            scaledEncoding(curState.ySpeed, 4, False),
            scaledEncoding(xAccel, 4, False),
            scaledEncoding(yAccel, 8, False),
        ]
    )
    for i in range(1, N_OBJECTS):
        obj = curState.objects[i - 1]
        if obj.relXPos > MAX_X_POS or obj.relYPos > MAX_Y_POS:
            continue

        infos[i] = np.array(
            [
                scaledEncoding(obj.relXPos, MAX_X_POS, True),
                scaledEncoding(obj.relYPos, MAX_Y_POS, True),
                # TODO: calculate speeds and accels with prevState
                0,
                0,
                0,
                0,
            ]
        )

    return (ids, infos)


def getScalarFeatures(curState: MarioLandGameState) -> np.ndarray:
    return np.array(
        [
            oneHotEncoding(curState.powerupStatus, 3),
            oneHotEncoding(int(curState.hasStar), 1),
            scaledEncoding(curState.invincibleTimer, 960),
        ]
    )


def scaledEncoding(val: int, max: int, minIsZero: bool) -> float:
    if minIsZero:
        return val / max
    # minimum value is less than zero, ensure scaling minimum is zero
    return (val + max) / (max * 2)


def oneHotEncoding(val: int, max: int) -> np.ndarray:
    return np.identity(max)[val : val + 1]


class MarioLandExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        activation_layer: nn.Module = nn.ReLU,
        n_hidden_layers: int = 32,
    ) -> None:
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        # (gameArea (nStack, 16, 20)
        gameArea = observation_space["gameArea"]
        inputChannels, xDim, yDim = gameArea.shape
        self.gameAreaCNN = nn.Sequential(
            nn.Conv2d(inputChannels, 32, kernel_size=2, stride=1, padding=0),
            activation_layer(),
            # max pool to downsample
            nn.AdaptiveMaxPool2d(output_size=(xDim // 2, yDim // 2)),
            nn.Conv2d(32, 32, kernel_size=2, stride=1, padding=0),
            activation_layer(),
            nn.Flatten(),
            # TODO: add 2 FC layers?
        )
        cnnOutputSize = _computeShape(gameArea, self.gameAreaCNN)

        entityIDs, entityInfos = observation_space["entities"]
        # there are single maximum value for these spaces
        assert np.all(entityIDs.high == entityIDs.high[0])
        assert np.all(entityInfos.high == entityInfos.high[0])

        # entityIDs (nStack, 11) -> (nStack, 11, 8)
        self.entityIDEmbedding = nn.Embedding(entityIDs.high[0], 8)

        # entityInfos (nStack, 11, 6) -> (nStack, 11, hiddenLayers)
        self.entityFC = nn.Sequential(
            nn.Linear(entityInfos.shape[2], n_hidden_layers),
            activation_layer(),
            nn.Linear(n_hidden_layers, n_hidden_layers),
            activation_layer(),
        )

        self.entityMaxPool = nn.AdaptiveMaxPool2d(output_size=(1, n_hidden_layers))

        # TODO: change
        self._features_dim = 0

    def forward(self, observations: TensorDict) -> th.Tensor:
        # normalize game area
        gameArea = observations["gameArea"].float() / float(MAX_TILE)
        gameArea = self.gameAreaCNN(gameArea)  # (32, 63)

        entityIDs, entityInfos = observations["entities"]
        embeddedEntityIDs = self.entityIDEmbedding(entityIDs.to(th.int))  # (nStack, 11, 8)
        entityInfos = entityInfos.to(th.int)  # (nStack, 11, 6)
        entities = th.cat((embeddedEntityIDs, entityInfos), dim=-1)  # (nStack, 11, 14)
        entities = self.entityFC(entities)  # (nStack, 11, hiddenLayers)
        entities = self.entityMaxPool(entities).squeeze(-2)  # (nStack, hiddenLayers)

        scalar = observations["scalar"]  # (nStack, 5)

        allFeatures = th.cat((gameArea, entities, scalar), dim=-1)  # (nStack, ?)

        return allFeatures


def _computeShape(observation_space: spaces.Space, mod: nn.Module) -> int:
    with th.no_grad():
        t = th.as_tensor(observation_space.sample()[None])
        return mod(t).shape[1]
