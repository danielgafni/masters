import logging
from dataclasses import dataclass
from typing import List, Optional
from uuid import uuid1

import torch
from torch.utils.tensorboard import SummaryWriter

from masters.a2c.agent import A2CAgent
from masters.constants import LOGS_DIR
from masters.data import Episode, Transition
from masters.env_wrapper import make_wrapped_env
from masters.networks import OUTPUT_LAYER_NAME

logger = logging.getLogger(__name__)


@dataclass
class A2CTrainerConfig:
    """
    Base config for inheritance
    """

    num_episodes: int = 1000
    max_steps: int = 200
    gamma: float = 0.95
    spikes_to_value: float = 1.0
    start_actor_train: int = 100
    num_test_episodes: int = 200
    device: str = "gpu" if torch.cuda.is_available() else "cpu"
    log: bool = True
    heavy_log_freq: int = 10
    experiment_name: str = "a2c"

    _target_: str = "masters.a2c.trainer.A2CTrainer"


class A2CTrainer:
    def __init__(
        self,
        num_episodes: int,
        max_steps: int = 200,
        gamma: float = 0.91,
        spikes_to_value: float = 1.0,
        start_actor_train: int = 100,
        num_test_episodes: int = 200,
        device: torch.device = torch.device("cpu"),
        log: bool = False,
        heavy_log_freq: int = 10,
        experiment_name: str = "a2c",
    ):
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.gamma = gamma
        self.start_actor_train = start_actor_train
        self.num_test_episodes = num_test_episodes
        self.spikes_to_value = spikes_to_value
        self.device = device
        self.log = log
        self.heavy_log_freq = heavy_log_freq
        self.experiment_name = experiment_name

        self.hparams = dict(
            num_episodes=num_episodes,
            max_steps=max_steps,
            gamma=gamma,
            spikes_to_value=spikes_to_value,
            start_actor_train=start_actor_train,
            num_test_episodes=num_test_episodes,
        )

        self.run_id = str(uuid1())
        self.log_dir = str(LOGS_DIR / self.experiment_name / self.run_id)
        self.writer: Optional[SummaryWriter] = None
        if log:
            assert self.experiment_name is not None

            self.writer = SummaryWriter(log_dir=self.log_dir)

            assert self.writer is not None

            logger.info(f"Initialized TensorBoard log_dir:\n{self.writer.log_dir}")

    def fit(self, agent: A2CAgent, env_name: str, num_episodes: Optional[int] = None):
        if num_episodes is None:
            num_episodes = self.num_episodes

        agent.to(self.device)

        assert agent.hparams is not None

        hparams = self.hparams.copy()
        hparams.update(agent.hparams)

        if self.log:
            assert self.writer is not None

            self.writer.add_hparams(hparams, {"hparam/mean_total_reward": -1.0}, run_name=".")

        for i in range(num_episodes):
            render = True if i % self.heavy_log_freq == 0 else False
            episode = self.play_episode(agent=agent, env_name=env_name, render=render)

            logger.info(f"Total reward after {agent.num_episodes} episodes: {episode.total_reward}")

            deltas: List[float] = []

            for transition in episode.transitions:
                observation = torch.from_numpy(transition.observation)
                prev_observation = torch.from_numpy(transition.prev_observation)
                action = transition.action
                reward = transition.reward
                done = transition.done

                value = agent.run_critic(observation=observation, train=False) if not done else 0.0
                prev_value = agent.run_critic(observation=prev_observation, train=False)

                delta = self.compute_delta(value=value, prev_value=prev_value, reward=reward)
                deltas.append(delta)

                self.train_critic(agent=agent, observation=observation, prev_observation=prev_observation, delta=delta)

                if agent.num_episodes > self.start_actor_train:
                    self.train_actor(agent=agent, observation=prev_observation, delta=delta, action=action)

                # if done:
                #     self.update_net(net=agent.actor.network, total_reward=sum(deltas), steps=len(episode.transitions) - 1)
                #     self.update_net(net=agent.critic.network, total_reward=sum(deltas), steps=len(episode.transitions) - 1)

                if self.log:
                    assert self.writer is not None

                    self.writer.add_scalar("Train/delta", delta, agent.num_steps)
                    self.writer.add_scalar("Train/value", value, agent.num_steps)
                    self.writer.add_scalar("Train/prev_value", prev_value, agent.num_steps)

                agent.num_steps += 1

            if self.log:
                assert self.writer is not None

                self.writer.add_scalar("Train/mean_delta", torch.tensor(deltas).mean().item(), agent.num_episodes)
                self.writer.add_scalar("Train/total_reward", episode.total_reward, agent.num_episodes)
                self.writer.add_histogram(
                    "Train/actions",
                    torch.tensor([t.action for t in episode.transitions]),
                    agent.num_episodes,
                    bins=agent.action_space_size,
                )

            if agent.num_episodes % self.heavy_log_freq == 0:
                if self.log:
                    assert self.writer is not None

                    self.writer.add_video("Train/replay", episode.replay, agent.num_episodes)

                    agent.log_weights(self.writer, tag_prefix="Train")
                    agent.log_spikes(self.writer, tag_prefix="Train")
                    # agent.log_voltages(self.writer, tag_prefix="Train")

            agent.num_episodes += 1

        if self.log:
            assert self.writer is not None
            self.writer.flush()

    def test(self, agent: A2CAgent, env_name: str, num_episodes: Optional[int] = None):
        if num_episodes is None:
            num_episodes = self.num_test_episodes

        agent.to(self.device)

        assert agent.hparams is not None

        hparams = self.hparams.copy()
        hparams.update(agent.hparams)

        total_rewards: List[float] = []

        for _ in range(num_episodes):
            episode = self.play_episode(agent=agent, env_name=env_name, render=False)
            total_rewards.append(episode.total_reward)

        total_rewards_tensor = torch.tensor(total_rewards)
        mean_total_reward = total_rewards_tensor.mean().item()

        rendered_episode = self.play_episode(agent=agent, env_name=env_name, render=True)
        rendered_episode.render_replay(output_path=f"{self.log_dir}/{agent.num_episodes}={mean_total_reward}.gif")

        if self.log:
            assert self.writer is not None

            self.writer.add_scalar("Test/mean_total_reward", mean_total_reward, agent.num_episodes)
            self.writer.add_hparams(hparams, {"hparam/mean_total_reward": mean_total_reward}, run_name=".")
            self.writer.add_video("Test/replay", rendered_episode.replay, agent.num_episodes)
            self.writer.flush()

        return total_rewards_tensor

    def play_episode(
        self, agent: A2CAgent, env_name: str, render: bool = False, device: torch.device = torch.device("cpu")
    ) -> Episode:
        agent.actor.network.to(device)

        env = make_wrapped_env(env_name=env_name)
        done = False
        step = 0
        observation = env.reset()

        transitions = []

        while not done and step < self.max_steps:
            observation_tensor = torch.from_numpy(observation).float()
            action = agent.run_actor(observation_tensor)

            prev_observation = observation

            observation, reward, done, info = env.step(action)

            render_ = None
            if render:
                render_ = env.render(mode="rgb_array")

            transition = Transition(
                observation=observation,
                prev_observation=prev_observation,
                done=done,
                reward=reward,
                action=action,
                render=render_,
            )
            transitions.append(transition)

        env.close()

        return Episode(transitions=transitions)

    def train_actor(self, agent: A2CAgent, observation: torch.Tensor, delta: float, action: int):
        # only the neuron corresponding to the action being trained should spike
        out_agent_shape = agent.actor.network.layers[OUTPUT_LAYER_NAME].shape
        clamp = torch.zeros(agent.actor.time, *out_agent_shape).bool()
        clamp[..., action] = 1
        unclamp = ~clamp

        agent.run_actor(
            observation=observation,
            train=True,
            reward=delta,
            # clamp={OUTPUT_LAYER_NAME: clamp},
            unclamp={OUTPUT_LAYER_NAME: unclamp},
        )

    def train_critic(self, agent: A2CAgent, observation: torch.Tensor, prev_observation: torch.Tensor, delta: float):
        agent.run_critic(observation=observation, reward=-delta, train=True)
        agent.run_critic(observation=prev_observation, reward=delta, train=True)

    def compute_delta(self, value: float, prev_value: float, reward: float):
        return self.spikes_to_value * (value * self.gamma - prev_value) + reward

    # def update_net(self, net: Network, total_reward: float, steps: int):
    #     if net.reward_fn is not None:
    #         print("updating")
    #         net.reward_fn.update(
    #             accumulated_reward=total_reward,
    #             steps=steps,
    #         )
