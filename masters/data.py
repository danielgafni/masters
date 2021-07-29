from dataclasses import dataclass
from typing import List, Optional

import numpy as np
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

    def render_replay(self, output_path: str):
        Image.fromarray(self.transitions[0].render).save(
            output_path,
            save_all=True,
            append_images=[Image.fromarray(t.render) for t in self.transitions[1:]],
        )


@dataclass
class Experience:
    episodes: List[Episode]


cartpole_info = {
    0: dict(start=-1, end=1, scale=0.05, n=20),
    1: dict(start=-2, end=2, scale=0.1, n=20),
    2: dict(start=-1, end=1, scale=0.05, n=20),
    3: dict(start=-10, end=10, scale=0.5, n=20),
}
