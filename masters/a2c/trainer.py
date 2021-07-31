import torch

from masters.a2c.agent import A2CAgent
from masters.data import Episode, Transition
from masters.env_wrapper import make_wrapped_env


class A2CTrainer:
    def __init__(
        self,
        num_episodes: int,
        max_steps: int = 200,
        gamma: float = 0.91,
        lr: float = 0.01,
        spikes_to_value: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ):
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.gamma = gamma
        self.lr = lr
        self.spikes_to_value = spikes_to_value
        self.device = device

    def fit(self, agent: A2CAgent, env_name: str, render: bool = False):
        agent.to(self.device)

        for _ in range(self.num_episodes):
            episode = self.play_episode(agent=agent, env_name=env_name, render=render)
            print(episode.total_reward)

            for transition in episode.transitions:
                observation = torch.from_numpy(transition.observation)
                prev_observation = torch.from_numpy(transition.prev_observation)
                action = transition.action
                reward = transition.reward
                done = transition.done

                delta = self.compute_delta(
                    agent=agent, observation=observation, prev_observation=prev_observation, reward=reward, done=done
                )

                self.train_actor(agent=agent, observation=prev_observation, delta=delta, action=action)
                self.train_critics(agent=agent, observation=observation, prev_observation=prev_observation, delta=delta)

    def play_episode(
        self, agent: A2CAgent, env_name: str, render: bool = False, device: torch.device = torch.device("cpu")
    ) -> Episode:
        agent.actor.to(device)

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
        out_agent_shape = agent.actor.layers[agent.output_actor_layer].shape
        clamp = torch.zeros(agent.actor_max_time, *out_agent_shape).bool()
        clamp[..., action] = 1
        unclamp = ~clamp

        agent.run_actor(
            observation=observation,
            train=True,
            reward=delta * self.lr,
            # clamp={agent.output_actor_layer: clamp},
            unclamp={agent.output_actor_layer: unclamp},
        )

    def train_critics(self, agent: A2CAgent, observation: torch.Tensor, prev_observation: torch.Tensor, delta: float):
        agent.run_critic(observation=observation, reward=-delta * self.lr, train=True)
        agent.run_prev_critic(observation=prev_observation, reward=delta * self.lr, train=True)

    def compute_delta(
        self, agent: A2CAgent, observation: torch.Tensor, prev_observation: torch.Tensor, reward: float, done: bool
    ):
        value = agent.run_critic(observation=observation, train=False) if not done else 0
        prev_value = agent.run_prev_critic(observation=prev_observation, train=False)
        return self.spikes_to_value * (value * self.gamma - prev_value) + reward
