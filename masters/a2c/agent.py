from typing import Optional

import torch
from bindsnet.encoding import Encoder
from bindsnet.encoding.encoders import PositiveEncoder
from bindsnet.network import Network

from masters.a2c.action import select_softmax
from masters.networks import INPUT_LAYER_NAME, OUTPUT_LAYER_NAME


class A2CAgent:
    def __init__(
        self,
        actor: Network,
        critic: Network,
        prev_critic: Network,
        encoder: Encoder,
        output_actor_layer: str = OUTPUT_LAYER_NAME,
        output_critic_layer: str = OUTPUT_LAYER_NAME,
        output_prev_critic_layer: str = OUTPUT_LAYER_NAME,
        actor_max_time: int = 100,
        critic_max_time: int = 100,
        prev_critic_max_time: int = 100,
        spikes_to_value: float = 10.0,
    ):
        """
        actor output must have the shape of [time, ..., population_size, n_actions]
        critic and prev_critic must have the shape of [time, ..., population_size]
        """
        self.actor = actor
        self.output_actor_layer = output_actor_layer
        self.actor_max_time = actor_max_time

        self.critic = critic
        self.output_critic_layer = output_critic_layer
        self.critic_max_time = critic_max_time

        self.prev_critic = prev_critic
        self.output_prev_critic_layer = output_prev_critic_layer
        self.prev_critic_max_time = prev_critic_max_time

        self.spikes_to_value = spikes_to_value

        self.encoder = PositiveEncoder(encoder)

        self.observation = None
        self.prev_observation = None
        self.prev_delta = None

    def run_actor(self, observation: torch.Tensor, train: bool = False, reward: Optional[float] = None, **kwargs):
        spikes = self.run_net(
            observation=observation,
            net=self.actor,
            max_time=self.actor_max_time,
            output_layer=self.output_actor_layer,
            train=train,
            reward=reward,
            **kwargs,
        )
        return select_softmax(spikes=spikes.float().sum(dim=-2))

    def run_critic(self, observation: torch.Tensor, train: bool = False, reward: Optional[float] = None, **kwargs):
        return (
            self.run_net(
                observation=observation,
                net=self.critic,
                max_time=self.critic_max_time,
                output_layer=self.output_critic_layer,
                train=train,
                reward=reward,
                **kwargs,
            )
            .float()
            .mean(dim=-1)
            .sum()
            .item()
        ) * self.spikes_to_value

    def run_prev_critic(self, observation: torch.Tensor, train: bool = False, reward: Optional[float] = None, **kwargs):
        return (
            self.run_net(
                observation=observation,
                net=self.prev_critic,
                max_time=self.prev_critic_max_time,
                output_layer=self.output_prev_critic_layer,
                train=train,
                reward=reward,
                **kwargs,
            )
            .sum()
            .item()
        ) * self.spikes_to_value

    def run_net(
        self,
        observation: torch.Tensor,
        net: Network,
        max_time: int,
        output_layer: str,
        train: bool = False,
        reward: Optional[float] = None,
        **kwargs,
    ):
        if reward is None:
            reward = 0

        net.reset_state_variables()

        original_training = net.training
        net.train(train)

        encoded_observation = self.encoder(observation)
        inpts = {INPUT_LAYER_NAME: encoded_observation}
        net.run(inpts, time=max_time, reward=reward, **kwargs)
        spikes = net.monitors[output_layer].get("s")

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
        self.actor.to(device)
        self.critic.to(device)
        self.critic.to(device)
