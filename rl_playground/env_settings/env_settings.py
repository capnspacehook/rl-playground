from typing import Any, Dict, List, Tuple

from gymnasium import Space
from stable_baselines3.common.vec_env.base_vec_env import VecEnv


class GameState:
    def __init__(self):
        """Used to hold a copy of the previous game state"""
        raise Exception("GameState init not implemented!")

    def stats(self) -> Dict | None:
        return None


class EnvSettings:
    def hyperParameters(self, algo: str) -> Dict[str, Any]:
        """Used to get hyperparameters"""
        return {}

    def normalize(self) -> Tuple[bool, bool]:
        """Whether observations and rewards should be normalized or not"""
        return True, True

    def evalEpisodes(self) -> int:
        """Number of evaluation episodes that should be preformed"""
        return 1

    def reset(self, options: Dict[str, Any] | None = None) -> Tuple[Any, GameState, bool, Dict[str, Any]]:
        """Reset state when starting a new training run"""
        raise Exception("reset not implemented!")

    def actionSpace(self) -> Tuple[List[Any], Space]:
        """Get action space for AI"""
        raise Exception("actionSpace not implemented!")

    def observationSpace(self) -> Space:
        """Get observation space for AI"""
        raise Exception("observationSpace not implemented!")

    def postStep(self, prevState: GameState, curState: GameState, action: int, reward: float):
        """Preform bookeeping after a step"""
        raise Exception("postStep not implemented!")

    def reward(self, prevState: GameState) -> Tuple[float, GameState]:
        """Reward function for the AI"""
        raise Exception("reward not implemented!")

    def gameState(self) -> GameState:
        """Get game state from pyboy to save important information"""
        raise Exception("gameState not implemented!")

    def observation(self, prevState: GameState, curState: GameState) -> Any:
        raise Exception("observation not implemented!")

    def info(self, gameState: GameState) -> Dict[str, Any]:
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


class Orchestrator:
    def __init__(self, env: VecEnv) -> None:
        self.env = env

    def preEval(self):
        pass

    def processEvalInfo(self, info: Dict[str, Any]):
        pass

    def evalInfoLogEntries(self, info: Dict[str, Any]) -> List[Tuple[str, Any]]:
        return ()

    def postEval(self):
        pass
