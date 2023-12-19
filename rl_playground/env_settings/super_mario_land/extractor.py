from typing import Any
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, TensorDict
import torch as th
from torch import nn

from rl_playground.env_settings.super_mario_land.constants import *
from rl_playground.env_settings.super_mario_land.game_area import MAX_TILE


# TODO: update number in comments once NN architecture is stable
class MarioLandExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observationSpace: spaces.Dict,
        device: str,
        activationFn: nn.Module = nn.ReLU,
        cnnHiddenLayers: int = 128,
        marioHiddenLayers: int = 64,
        embeddingDimensions: int = 8,
        entityHiddenLayers: int = 128,
    ) -> None:
        gameArea = observationSpace[GAME_AREA_OBS]
        numStack, xDim, yDim = gameArea.shape
        scalar = observationSpace[SCALAR_OBS]

        featuresDim = cnnHiddenLayers + marioHiddenLayers + entityHiddenLayers + scalar.shape[0]
        super().__init__(observationSpace, features_dim=featuresDim)

        # gameArea (nStack, 16, 20)
        self.gameAreaCNN = nn.Sequential(
            nn.Conv2d(numStack, 32, kernel_size=2, stride=1, padding=0, device=device),
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

        marioInfo = observationSpace[MARIO_INFO_OBS]

        # marioInfo (7) -> (hiddenLayers)
        self.marioFC = nn.Sequential(
            nn.Linear(marioInfo.shape[0], marioHiddenLayers, device=device),
            activationFn(),
            nn.Linear(marioHiddenLayers, marioHiddenLayers, device=device),
            activationFn(),
        )

        entityIDs = observationSpace[ENTITY_ID_OBS]
        entityInfos = observationSpace[ENTITY_INFO_OBS]

        # entityIDs (10) -> (10, embeddingDimensions)
        self.entityIDEmbedding = nn.Embedding(entityIDs.high[0], embeddingDimensions, device=device)
        # entities concat -> (10, 8+embeddingDimensions)

        # entityInfos (10, 8) -> (10, hiddenLayers)
        self.entityFC = nn.Sequential(
            nn.Linear(entityInfos.shape[1] + embeddingDimensions, entityHiddenLayers, device=device),
            activationFn(),
            nn.Linear(entityHiddenLayers, entityHiddenLayers, device=device),
            activationFn(),
        )

        # TODO: try reducing entityHiddenLayers and not max pooling
        # entityInfos (10, hiddenLayers) -> (1, hiddenLayers)
        self.entityMaxPool = nn.AdaptiveMaxPool2d(output_size=(1, entityHiddenLayers))
        # entityInfos squeeze -> (hiddenLayers)

    def forward(self, observations: TensorDict) -> th.Tensor:
        # normalize game area
        gameArea = observations[GAME_AREA_OBS].to(th.float32) / float(MAX_TILE)
        gameArea = self.gameAreaFC(self.gameAreaCNN(gameArea))  # (cnnHiddenLayers,)

        marioInfo = observations[MARIO_INFO_OBS]  # (7)
        mario = self.marioFC(marioInfo)  # (marioHiddenLayers)

        entityIDs = observations[ENTITY_ID_OBS].to(th.int)  # (10)
        embeddedEntityIDs = self.entityIDEmbedding(entityIDs)  # (10, embeddingDimensions)
        entityInfos = observations[ENTITY_INFO_OBS]  # (10, 8)
        entities = th.cat((embeddedEntityIDs, entityInfos), dim=-1)  # (10, 8+embeddingDimensions)
        entities = self.entityFC(entities)  # (10, entityHiddenLayers)
        entities = self.entityMaxPool(entities).squeeze(-2)  # (entityHiddenLayers)

        scalar = observations[SCALAR_OBS]  # (8)

        allFeatures = th.cat((gameArea, mario, entities, scalar), dim=-1)

        return allFeatures


def _computeShape(space: spaces.Space, dtype: Any, device: Any, mod: nn.Module) -> int:
    with th.no_grad():
        t = th.as_tensor(space.sample()[None], device=device).to(dtype)
        return mod(t).shape[1]
