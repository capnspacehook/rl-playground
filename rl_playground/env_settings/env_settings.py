from typing import Any, Dict, List

from gymnasium import Space


class GameState:
    def __init__(self):
        """Used to hold a copy of the previous game state"""
        raise Exception("GameState init not implemented!")

    def stats(self) -> dict | None:
        return None


class EnvSettings:
    def hyperParameters(self, algo: str) -> Dict[str, Any]:
        """Used to get hyperparameters"""
        return {}

    def normalizeObservation(self) -> bool:
        """Whether observations should be normalized or not"""
        return False

    def evalEpisodes(self) -> int:
        """Number of evaluation episodes that should be preformed"""
        return 1

    def reset(self, options: dict[str, Any] | None = None) -> GameState:
        """Reset state when starting a new training run"""
        pass

    def actionSpace(self) -> (List[Any], Space):
        """Get action space for AI"""
        raise Exception("actionSpace not implemented!")

    def observationSpace(self) -> Space:
        """Get observation space for AI"""
        raise Exception("observationSpace not implemented!")

    def reward(self, prevState: GameState) -> (float, GameState):
        """Reward function for the AI"""
        raise Exception("reward not implemented!")

    def gameState(self) -> GameState:
        """Get game state from pyboy to save important information"""
        raise Exception("gameState not implemented!")

    def observation(self, gameState: GameState) -> Any:
        raise Exception("observation not implemented!")

    def info(self, gameState: GameState) -> dict[str, Any]:
        return {}

    def terminated(self, prevState: GameState, curState: GameState) -> bool:
        """Returns true if the game should end, ie game over"""
        return False

    def truncated(self, prevState: GameState, curState: GameState) -> bool:
        """Returns true if the AI is has not been progressing
        for awhile and training should stop"""
        return False

    def printGameState(self, prevState: GameState, curState: GameState) -> None:
        """Used to print in playtest mode"""
        raise Exception("PrintGameState not implemented!")

    def render(self):
        pass
