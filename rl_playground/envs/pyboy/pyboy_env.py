from pathlib import Path
from typing import Any

import pandas
from gymnasium import Env
from pyboy import PyBoy, WindowEvent

from rl_playground.env_settings.env_settings import GameState, EnvSettings


# TODO: make env more generic and more input sending to EnvSettings
class PyBoyEnv(Env):
    def __init__(
        self,
        pyboy: PyBoy,
        envSettings: EnvSettings,
        render: bool = False,
        isEval: bool = False,
        isPlaytest: bool = False,
        outputDir: Path | None = None,
    ) -> None:
        self.pyboy = pyboy
        self.envSettings = envSettings
        self.prevGameState: GameState | None = None
        self._started = False
        self.isEval = isEval
        self.isPlaytest = isPlaytest
        self.episodeNum = 0
        self.curInfo = dict()
        self.agentStats = []
        self.outputDir = outputDir

        # build list of possible inputs
        self._buttons = [
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_SELECT,
            WindowEvent.PRESS_BUTTON_START,
        ]
        self._button_is_pressed = {button: False for button in self._buttons}

        self._buttons_release = [
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_SELECT,
            WindowEvent.RELEASE_BUTTON_START,
        ]
        self._release_button = {
            button: r_button
            for button, r_button in zip(self._buttons, self._buttons_release)
        }

        # set the action and observation spaces
        self.actions, self.action_space = self.envSettings.actionSpace()
        self.observation_space = self.envSettings.observationSpace()

        # compabitility with Env
        self.metadata["render_modes"] = ["rgb_array"]
        self.metadata["render_fps"] = 12
        self.render_mode = None
        if render:
            self.render_mode = "rgb_array"

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)

        self.envSettings.reset(options=options)

        if not self._started:
            self._started = True
        else:
            # write agent stats to disk
            # pandas.DataFrame(self.agentStats).to_csv(
            #     self.outputDir / Path(f"agent_stats_ep{self.episodeNum}.csv.zstd"),
            #     compression="zstd",
            #     mode="x",
            #     index_label="step",
            # )
            # self.agentStats = []

            self.episodeNum += 1

        self.button_is_pressed = {button: False for button in self._buttons}

        self.prevGameState = self.envSettings.gameState()

        return self.envSettings.observation(self.prevGameState), {}

    def step(self, action_idx: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if self.prevGameState is None:
            self.prevGameState = self.envSettings.gameState()

        actions = self.actions[action_idx]
        if not self.isPlaytest:
            self.sendInputs(actions)
        else:
            self.envSettings.printGameState(self.prevGameState)
        pyboyDone = self.pyboy.tick()

        reward, curGameState = self.envSettings.reward(self.prevGameState)

        info = self.envSettings.info(self.prevGameState)
        if info is not None:
            info["actions"] = actions
            info["reward"] = reward
            # self.agentStats.append(info)

        terminated = pyboyDone or self.envSettings.terminated(
            self.prevGameState, curGameState
        )
        truncated = self.envSettings.truncated(self.prevGameState, curGameState)

        self.prevGameState = curGameState

        return (
            self.envSettings.observation(curGameState),
            reward,
            terminated,
            truncated,
            info,
        )

    def sendInputs(self, actions: list[int]):
        # release buttons that were pressed in the past
        for pressedFromBefore in [
            pressed
            for pressed in self._button_is_pressed
            if self._button_is_pressed[pressed] == True
        ]:  # get all buttons currently pressed
            if pressedFromBefore not in actions:
                release = self._release_button[pressedFromBefore]
                self.pyboy.send_input(release)
                self._button_is_pressed[release] = False

        # press buttons we want to press
        for buttonToPress in actions:
            if buttonToPress == WindowEvent.PASS:
                continue
            self.pyboy.send_input(buttonToPress)
            # update status of the button
            self._button_is_pressed[buttonToPress] = True

    def render(self):
        return self.envSettings.render()
