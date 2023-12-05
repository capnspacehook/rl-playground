from typing import Any
from gymnasium import spaces
import numpy as np

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, TensorDict
import torch as th
from torch import nn

from rl_playground.env_settings.super_mario_land.constants import *
from rl_playground.env_settings.super_mario_land.game_area import MAX_TILE


class MarioLandExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observationSpace: spaces.Dict,
        actLayer: nn.Module = nn.ReLU,
        cnnHiddenLayers: int = 64,
        embeddingDimensions: int = 8,
        entityHiddenLayers: int = 64,
    ) -> None:
        gameArea = observationSpace[GAME_AREA_OBS]
        numStack, xDim, yDim = gameArea.shape
        scalar = observationSpace[SCALAR_OBS]

        featuresDim = cnnHiddenLayers + (numStack * (entityHiddenLayers + scalar.shape[1]))
        super().__init__(observationSpace, features_dim=featuresDim)

        # gameArea (nStack, 16, 20)
        self.gameAreaCNN = nn.Sequential(
            nn.Conv2d(numStack, 32, kernel_size=2, stride=1, padding=0),
            actLayer(),
            # max pool to downsample
            nn.AdaptiveMaxPool2d(output_size=(xDim // 2, yDim // 2)),
            nn.Conv2d(32, 32, kernel_size=2, stride=1, padding=0),
            actLayer(),
            nn.Flatten(),
        )
        cnnOutputSize = _computeShape(gameArea, th.float32, self.gameAreaCNN)
        self.gameAreaFC = nn.Sequential(
            nn.Linear(cnnOutputSize, cnnHiddenLayers),
            actLayer(),
        )

        entityIDs = observationSpace[ENTITY_ID_OBS]
        entityInfos = observationSpace[ENTITY_INFO_OBS]

        # entityIDs (nStack, 11) -> (nStack, 11, embeddingDimensions)
        self.entityIDEmbedding = nn.Embedding(entityIDs.high[0][0], embeddingDimensions)
        # entities concat -> (nStack, 11, 8+embeddingDimensions)

        # entityInfos (nStack, 11, 8) -> (nStack, 11, hiddenLayers)
        self.entityFC = nn.Sequential(
            nn.Linear(entityInfos.shape[2] + embeddingDimensions, entityHiddenLayers),
            actLayer(),
            nn.Linear(entityHiddenLayers, entityHiddenLayers),
            actLayer(),
        )

        self.entityMaxPool = nn.AdaptiveMaxPool2d(output_size=(1, entityHiddenLayers))

    def forward(self, observations: TensorDict) -> th.Tensor:
        # normalize game area
        gameArea = observations[GAME_AREA_OBS].to(th.float) / float(MAX_TILE)
        gameArea = self.gameAreaFC(self.gameAreaCNN(gameArea))  # (cnnHiddenLayers,)

        entityIDs = observations[ENTITY_ID_OBS].to(th.int)  # (nStack, 11)
        embeddedEntityIDs = self.entityIDEmbedding(entityIDs)  # (nStack, 11, embeddingDimensions)
        entityInfos = observations[ENTITY_INFO_OBS]  # (nStack, 11, 8)
        entities = th.cat((embeddedEntityIDs, entityInfos), dim=-1)  # (nStack, 11, 8+embeddingDimensions)
        entities = self.entityFC(entities)  # (nStack, 11, entityHiddenLayers)
        entities = self.entityMaxPool(entities).squeeze(-2)  # (nStack, entityHiddenLayers)

        scalar = observations[SCALAR_OBS]  # (nStack, 6)

        entityScalars = th.cat((entities, scalar), dim=-1)  # (nStack, entityHiddenLayers+6)
        # leave the batch dimension intact
        entityScalars = th.flatten(entityScalars, start_dim=-2, end_dim=-1)  # (nStack*entityHiddenLayers+6)
        allFeatures = th.cat(
            (gameArea, entityScalars), dim=-1
        )  # (cnnHiddenLayers+(nStack*(entityHiddenLayers+6)),)

        return allFeatures


def _computeShape(space: spaces.Space, dtype: Any, mod: nn.Module) -> int:
    with th.no_grad():
        t = th.as_tensor(space.sample()[None]).to(dtype)
        return mod(t).shape[1]
