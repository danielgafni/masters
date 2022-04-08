import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from bindsnet.analysis.visualization import summary
from bindsnet.encoding import Encoder
from bindsnet.network import Network
from torch.utils.tensorboard import SummaryWriter

from masters.a2c.action import select_softmax
from masters.networks import INPUT_LAYER_NAME, OUTPUT_LAYER_NAME
from masters.networks.actor import Actor, ActorConfig
from masters.networks.critic import Critic, CriticConfig

logger = logging.getLogger(__name__)


@dataclass
class A2CAgentConfig:
    """
    Base config for inheritance
    """

    actor: ActorConfig
    critic: CriticConfig

    _target_: str = "masters.a2c.agent.A2CAgent"


class A2CAgent:
    def __init__(self, actor: Actor, critic: Critic, encoder: Encoder, encoder_hparams: Optional[Dict] = None):
        """
        actor output must have the shape of [time, ..., population_size, n_actions]
        critic and prev_critic must have the shape of [time, ..., population_size]
        """
        self.actor = actor
        self.critic = critic
        self.encoder = encoder

        if encoder_hparams is None:
            encoder_hparams = {}
        self.encoder_hparams = encoder_hparams

        self.num_steps = 0
        self.num_episodes = 0

        self.action_space_size = self.actor.network.layers[OUTPUT_LAYER_NAME].shape[-1]

        self.hparams = dict(
            # actor
            actor_a_plus=actor.a_plus,
            actor_a_minus=actor.a_minus,
            actor_thresh=actor.thresh,
            actor_time=actor.time,
            actor_input_size=actor.input_size,
            actor_n_hidden=actor.n_hidden,
            actor_action_space_size=actor.action_space_size,
            # critic
            critic_a_plus=critic.a_plus,
            critic_a_minus=critic.a_minus,
            critic_thresh=critic.thresh,
            critic_time=critic.time,
            critic_input_size=critic.input_size,
            critic_n_hidden=critic.n_hidden,
            # encoder
            **encoder_hparams,
        )

        actor_summary = summary(self.actor.network)
        critic_summary = summary(self.critic.network)

        logger.info(f"Actor:\n{actor_summary}")
        logger.info(f"Critic:\n{critic_summary}")

    def run_actor(self, observation: torch.Tensor, train: bool = False, reward: Optional[float] = None, **kwargs):
        spikes = self.run_net(
            observation=observation,
            net=self.actor.network,
            max_time=self.actor.time,
            train=train,
            reward=reward,
            **kwargs,
        )
        return select_softmax(spikes=spikes.float().sum(dim=-2))

    def run_critic(self, observation: torch.Tensor, train: bool = False, reward: Optional[float] = None, **kwargs):
        return (
            self.run_net(
                observation=observation,
                net=self.critic.network,
                max_time=self.critic.time,
                train=train,
                reward=reward,
                **kwargs,
            )
            .float()
            # TODO: take last n timestamps
            .mean(dim=-1)
            .sum()
            .item()
        )

    def run_net(
        self,
        observation: torch.Tensor,
        net: Network,
        max_time: int,
        train: bool = False,
        reward: Optional[float] = None,
        **kwargs,
    ):
        if reward is None:
            reward = 0

        net.reset_state_variables()

        original_training = net.training
        original_device = observation.device
        net.train(train)

        encoded_observation = self.encoder(observation.cpu()).to(original_device)
        inpts = {INPUT_LAYER_NAME: encoded_observation}
        net.run(inpts, time=max_time, reward=reward, **kwargs)
        spikes = net.monitors[OUTPUT_LAYER_NAME].get("s")

        net.train(original_training)

        return spikes

    def get_spikes(self):
        return self.get_state_vars(var="s")

    def get_voltages(self):
        return self.get_state_vars(var="v")

    def get_state_vars(self, var):
        nets = ["actor", "critic", "prev_critic"]
        state_vars = {net: {} for net in nets}

        for net in nets:
            network = getattr(self, net)
            for key in network.monitors:
                state_vars[net][key] = network.monitors[key].get(var)

        return state_vars

    def to(self, device: torch.device):
        self.actor.network.to(device)
        self.critic.network.to(device)

    def log_weights(self, writer: SummaryWriter, tag_prefix: str = ""):
        pass
        # actor_weights = []
        # for conn in self.actor.network.connections:
        #     actor_weights.append(self.actor.network.connections[conn].w.data.unsqueeze(0))
        #
        # writer.add_images(f"{tag_prefix}/actor_weights", torch.stack(actor_weights), self.num_episodes)
        #
        # critic_weights = []
        # for conn in self.critic.network.connections:
        #     critic_weights.append(self.critic.network.connections[conn].w.data.unsqueeze(0))
        #
        # writer.add_images(f"{tag_prefix}/critic_weights", torch.stack(critic_weights), self.num_episodes)

    def log_spikes(self, writer: SummaryWriter, tag_prefix: str = ""):
        pass
        # writer.add_image(
        #     f"{tag_prefix}/actor_spikes",
        #     self.actor.network.monitors[OUTPUT_LAYER_NAME].get("s").view(1, self.actor.time, -1).float(),
        #     self.num_episodes,
        # )
        # writer.add_image(
        #     f"{tag_prefix}/critic_spikes",
        #     self.critic.network.monitors[OUTPUT_LAYER_NAME].get("s").view(1, self.critic.time, -1).float(),
        #     self.num_episodes,
        # )

    # def log_voltages(self, writer: SummaryWriter, tag_prefix: str = ""):
    #     actor_fig = plt.Figure()
    #     writer.add_image(
    #         f"{tag_prefix}/actor_voltages",
    #         self.actor.network.monitors[OUTPUT_LAYER_NAME].get("v").view(1, self.actor.time, -1).float(),
    #         self.num_episodes,
    #     )
    #     writer.add_image(
    #         f"{tag_prefix}/critic_voltages",
    #         self.critic.network.monitors[OUTPUT_LAYER_NAME].get("v").view(1, self.critic.time, -1).float(),
    #         self.num_episodes,
    #     )
