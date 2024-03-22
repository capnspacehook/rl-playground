import gymnasium as gym
import numpy as np
from PIL import ImageFont
from PIL import ImageDraw
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from pyboy import WindowEvent


actionsToText = {
    WindowEvent.PASS: "NOTHING",
    WindowEvent.PRESS_ARROW_LEFT: "LEFT",
    WindowEvent.PRESS_ARROW_RIGHT: "RIGHT",
    WindowEvent.PRESS_ARROW_UP: "UP",
    WindowEvent.PRESS_ARROW_DOWN: "DOWN",
    WindowEvent.PRESS_BUTTON_B: "B",
    WindowEvent.PRESS_BUTTON_A: "A",
    WindowEvent.PRESS_BUTTON_START: "START",
    WindowEvent.PRESS_BUTTON_SELECT: "START",
}


class Recorder(gym.Wrapper):
    def __init__(self, env, episode_num=1, native_fps=60, rec_steps=2, reward_steps=4):
        super().__init__(env)
        self.env = env

        self.episode_num = episode_num
        self.native_fps = native_fps
        self.rec_steps = rec_steps
        self.reward_steps = reward_steps
        self.font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf", 12)

        self.frames = []
        self.count_episode = 0
        self.cur_step = 0
        self.cur_reward = 0
        self.last_reward = 0

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)

        self.cur_reward += reward
        if self.cur_step % self.reward_steps == self.reward_steps - 1:
            self.last_reward = self.cur_reward
            self.cur_reward = 0
        if self.cur_step % self.rec_steps == 0:
            self.record_step(action)
        self.cur_step += 1

        if term or trunc:
            self.count_episode += 1
            if self.count_episode == self.episode_num:
                self.stop_recording()

        return obs, reward, term, trunc, info

    def record_step(self, action):
        action_list = self.env.unwrapped.actions[action]
        actionText = ""
        for action in action_list:
            actionText += actionsToText[action] + ", "

        frame = self.env.render()
        draw = ImageDraw.Draw(frame)
        draw.text((0, 17), f"ACTION: {actionText[:-2]}", (0, 102, 255), self.font)
        draw.text((0, 30), f"REWARD: {round(self.last_reward, 2)}", (0, 102, 255), self.font)
        self.frames.append(np.array(frame))

    def stop_recording(self):
        print("Writing eval video")

        v = ImageSequenceClip(self.frames, fps=self.native_fps / self.rec_steps)
        v.write_videofile(
            filename="/tmp/eval.mp4",
            codec="libx264",
            bitrate="4000k",
            preset="slow",
            threads=24,
            logger=None,
        )
        v.close()

        self.frames = []
        self.count_episode = 0
        self.cur_step = 0
        self.cur_reward = 0
        self.last_reward = 0


class FrameSkip(gym.Wrapper):
    """Return only every `skip`-th frame"""

    def __init__(self, env, skip):
        self._skip = skip

        super().__init__(env)

    def step(self, action):
        """Repeat action, and sum reward"""

        total_reward = 0.0
        for _ in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info
