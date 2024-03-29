from typing import Any
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, TensorDict
import torch as th
from torch import nn

from rl_playground.env_settings.super_mario_land.constants import *
from rl_playground.env_settings.super_mario_land.game_area import MAX_TILE


class MarioLandExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observationSpace: spaces.Dict,
        device: str,
        activationFn: nn.Module = nn.ReLU,
        cnnHiddenLayers: int = 128,
        marioHiddenLayers: int = 64,
        embeddingDimensions: int = 8,
        entityHiddenLayers: int = 64,
    ) -> None:
        gameArea = observationSpace[GAME_AREA_OBS]
        gameAreaStack, xDim, yDim = gameArea.shape
        marioInfo = observationSpace[MARIO_INFO_OBS]
        entityIDs = observationSpace[ENTITY_ID_OBS]
        entityInfos = observationSpace[ENTITY_INFO_OBS]
        scalar = observationSpace[SCALAR_OBS]

        featuresDim = (
            cnnHiddenLayers
            + (marioInfo.shape[0] * marioHiddenLayers)
            + (entityInfos.shape[0] * 10 * entityHiddenLayers)
            + (scalar.shape[0] * scalar.shape[1])
        )
        super().__init__(observationSpace, features_dim=featuresDim)

        # gameArea (nStack, 16, 20)
        self.gameAreaCNN = nn.Sequential(
            nn.Conv2d(gameAreaStack, 32, kernel_size=2, stride=1, padding=0, device=device),
            activationFn(),
            # max pool to downsample
            nn.AdaptiveMaxPool2d(output_size=(xDim // 2, yDim // 2)),
            nn.Conv2d(32, 32, kernel_size=2, stride=1, padding=0, device=device),
            activationFn(),
            nn.Flatten(),
        )
        cnnOutputSize = _computeShape(gameArea, th.float32, device, self.gameAreaCNN)
        self.gameAreaFC = nn.Sequential(
            nn.Linear(cnnOutputSize, cnnHiddenLayers, device=device),
            activationFn(),
        )

        self.marioFC = nn.Sequential(
            nn.Linear(marioInfo.shape[1], marioHiddenLayers, device=device),
            activationFn(),
            nn.Linear(marioHiddenLayers, marioHiddenLayers, device=device),
            activationFn(),
        )

        self.entityIDEmbedding = nn.Embedding(entityIDs.high[0][0], embeddingDimensions, device=device)

        self.entityFC = nn.Sequential(
            nn.Linear(entityInfos.shape[2] + embeddingDimensions, entityHiddenLayers, device=device),
            activationFn(),
            nn.Linear(entityHiddenLayers, entityHiddenLayers, device=device),
            activationFn(),
        )

    def forward(self, observations: TensorDict) -> th.Tensor:
        # normalize game area
        gameArea = observations[GAME_AREA_OBS].to(th.float32) / float(MAX_TILE)
        gameArea = self.gameAreaFC(self.gameAreaCNN(gameArea))

        marioInfo = observations[MARIO_INFO_OBS]
        mario = self.marioFC(marioInfo)
        mario = th.flatten(mario, start_dim=-2, end_dim=-1)

        entityIDs = observations[ENTITY_ID_OBS].to(th.int)
        embeddedEntityIDs = self.entityIDEmbedding(entityIDs)
        entityInfos = observations[ENTITY_INFO_OBS]
        entities = th.cat((embeddedEntityIDs, entityInfos), dim=-1)
        entities = self.entityFC(entities)
        entities = th.flatten(entities, start_dim=-3, end_dim=-1)

        scalar = observations[SCALAR_OBS]
        scalar = th.flatten(scalar, start_dim=-2, end_dim=-1)

        allFeatures = th.cat((gameArea, mario, entities, scalar), dim=-1)

        return allFeatures


def _computeShape(space: spaces.Space, dtype: Any, device: Any, mod: nn.Module) -> int:
    with th.no_grad():
        t = th.as_tensor(space.sample()[None], device=device).to(dtype)
        return mod(t).shape[1]
