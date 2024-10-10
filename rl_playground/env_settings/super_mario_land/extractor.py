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
        gameAreaEmbeddingDim: int = 4,
        marioHiddenLayers: int = 64,
        entityEmbeddingDim: int = 4,
        entityHiddenLayers: int = 64,
    ) -> None:
        # we will set the actual features_dim later once we know the
        # output size of the CNN
        super().__init__(observationSpace, features_dim=1)

        gameArea = observationSpace[GAME_AREA_OBS]
        marioInfo = observationSpace[MARIO_INFO_OBS]
        entityInfos = observationSpace[ENTITY_INFO_OBS]
        scalar = observationSpace[SCALAR_OBS]

        # account for 0 in number of embeddings
        self.gameAreaEmbedding = nn.Embedding(MAX_TILE + 1, gameAreaEmbeddingDim, device=device)

        self.gameAreaCNN = nn.Sequential(
            nn.Conv2d(
                gameAreaEmbeddingDim * gameArea.shape[0],
                32,
                kernel_size=3,
                stride=1,
                padding=1,
                device=device,
            ),
            activationFn(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, device=device),
            activationFn(),
            nn.Flatten(),
        )
        cnnOutputSize = self._computeCNNShape(gameArea, device)

        self.marioFC = nn.Sequential(
            nn.Linear(marioInfo.shape[1], marioHiddenLayers, device=device),
            activationFn(),
            nn.Linear(marioHiddenLayers, marioHiddenLayers, device=device),
            activationFn(),
        )

        # account for 0 in number of embeddings
        self.entityIDEmbedding = nn.Embedding(MAX_ENTITY_ID + 1, entityEmbeddingDim, device=device)

        self.entityFC = nn.Sequential(
            nn.Linear(entityInfos.shape[2] + entityEmbeddingDim, entityHiddenLayers, device=device),
            activationFn(),
            nn.Linear(entityHiddenLayers, entityHiddenLayers, device=device),
            activationFn(),
        )

        self._features_dim = (
            cnnOutputSize
            + (marioInfo.shape[0] * marioHiddenLayers)
            + (entityInfos.shape[0] * 10 * entityHiddenLayers)
            + (scalar.shape[0] * scalar.shape[1])
        )

    def forward(self, observations: TensorDict) -> th.Tensor:
        gameArea = observations[GAME_AREA_OBS].to(th.int)
        gameArea = self.gameAreaEmbedding(gameArea).to(th.float32)
        # move embedding dimension to be after stacked dimension
        gameArea = gameArea.permute(0, 4, 1, 2, 3)
        # flatten embedding and stack dim
        gameArea = th.flatten(gameArea, start_dim=1, end_dim=2)
        gameArea = self.gameAreaCNN(gameArea)

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

    def _computeCNNShape(self, space: spaces.Space, device: Any) -> int:
        with th.no_grad():
            t = th.as_tensor(space.sample()[None], device=device).to(th.int)
            e = self.gameAreaEmbedding(t).to(th.float32)
            e = e.permute(0, 4, 1, 2, 3)
            e = th.flatten(e, start_dim=1, end_dim=2)
            return self.gameAreaCNN(e).shape[1]
