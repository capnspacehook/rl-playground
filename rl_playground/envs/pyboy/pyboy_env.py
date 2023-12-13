from pathlib import Path
import random
from typing import Any

import pandas
from gymnasium import Env
from pyboy import PyBoy, WindowEvent

from rl_playground.env_settings.env_settings import GameState, EnvSettings


class PyBoyEnv(Env):
    def __init__(
        self,
        pyboy: PyBoy,
        envSettings: EnvSettings,
        render: bool = False,
        isEval: bool = False,
        isPlaytest: bool = False,
        isInteractiveEval: bool = False,
        outputDir: Path | None = None,
    ) -> None:
        self.pyboy = pyboy
        self.envSettings = envSettings
        self.prevGameState: GameState | None = None
        self._started = False
        self.isEval = isEval
        self.isPlaytest = isPlaytest
        self.isInteractiveEval = isInteractiveEval
        self.episodeNum = 0
        self.curInfo = dict()
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
            button: r_button for button, r_button in zip(self._buttons, self._buttons_release)
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

        self.interactive = False

        random.seed()

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)

        if options is not None:
            options["_prevState"] = self.prevGameState
        obs, self.prevGameState, envReset = self.envSettings.reset(options=options)

        if envReset:
            if not self._started:
                self._started = True
            else:
                self.episodeNum += 1

            self.button_is_pressed = {button: False for button in self._buttons}

        return obs, {}

    def step(self, action_idx: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if self.prevGameState is None:
            self.prevGameState = self.envSettings.gameState()

        actions = self.actions[action_idx]
        if self.isInteractiveEval:
            with open("agent_enabled.txt", "r") as f:
                if "true" in f.read():
                    self.interactive = False
                    self.sendInputs(actions)
                elif not self.interactive:
                    self.sendInputs([WindowEvent.PASS])
                    self.interactive = True
        elif not self.isPlaytest:
            self.sendInputs(actions)

        pyboyDone = self.pyboy.tick()
        reward, curGameState = self.envSettings.reward(self.prevGameState)

        obs = self.envSettings.observation(self.prevGameState, curGameState)
        terminated = pyboyDone or self.envSettings.terminated(self.prevGameState, curGameState)
        truncated = self.envSettings.truncated(self.prevGameState, curGameState)
        info = self.envSettings.info(self.prevGameState)

        if self.isPlaytest:
            self.envSettings.printGameState(self.prevGameState, curGameState)

        self.prevGameState = curGameState

        return obs, reward, terminated, truncated, info

    def sendInputs(self, actions: list[int]):
        # release buttons that were pressed in the past
        for pressedFromBefore in [
            pressed for pressed in self._button_is_pressed if self._button_is_pressed[pressed] == True
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
