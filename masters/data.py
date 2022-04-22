import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from PIL import Image


@dataclass
class Transition:
    prev_observation: np.ndarray
    action: int
    observation: np.ndarray
    reward: float
    done: bool
    render: Optional[np.ndarray]


@dataclass
class Episode:
    transitions: List[Transition]

    @property
    def total_reward(self):
        return sum([t.reward for t in self.transitions])

    @property
    def replay(self):
        frames = []
        for t in self.transitions:
            assert t.render is not None

            frames.append(torch.from_numpy(t.render.copy()))

        video = torch.stack(frames).permute(0, 3, 1, 2).unsqueeze(0)

        return video

    def render_replay(self, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        Image.fromarray(self.transitions[0].render).save(
            output_path,
            save_all=True,
            append_images=[Image.fromarray(t.render) for t in self.transitions[1:]],
        )


@dataclass
class Experience:
    episodes: List[Episode]


@dataclass
class EncoderConfig:
    pass
