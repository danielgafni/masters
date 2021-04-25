import uuid

import hydra
from bindsnet.analysis.pipeline_analysis import TensorboardAnalyzer
from bindsnet.encoding import bernoulli, convert_to_positive
from bindsnet.environment import GymEnvironment
from bindsnet.learning import MSTDP, PostPre
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.pipeline import EnvironmentPipeline
from bindsnet.pipeline.action import select_softmax
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from masters.configs import CONFIG_DIR, Config
from masters.configs.environment import register_environment_configs
from masters.configs.model import register_model_configs
from masters.constants import LOGS_DIR

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)

register_environment_configs()
register_model_configs()


@hydra.main(config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: Config):
    print(OmegaConf.to_yaml(cfg))

    # Build network.
    network = Network(dt=1.0)

    # Layers of neurons.
    inpt = Input(n=cfg.environment.in_neurons, shape=[1, 1, 2, 4], traces=True)
    middle = LIFNodes(n=100, traces=True)
    out = LIFNodes(n=2, refrac=0, traces=True)

    # Connections between layers.
    inpt_middle = Connection(source=inpt, target=middle, wmin=0, wmax=1e-1)
    middle_out = Connection(
        source=middle,
        target=out,
        wmin=0,
        wmax=1,
        update_rule=MSTDP,
        nu=[1e-3, 1e-4],
        norm=0.5 * middle.n,
    )
    middle_inhibitory = Connection(
        source=middle, target=middle, wmin=0, wmax=1, update_rule=PostPre, nu=[-1e-4, 1e-3], norm=-0.5 * middle.n
    )

    # Add all layers and connections to the network.
    network.add_layer(inpt, name="Input Layer")
    network.add_layer(middle, name="Hidden Layer")
    network.add_layer(out, name="Output Layer")
    network.add_connection(inpt_middle, source="Input Layer", target="Hidden Layer")
    network.add_connection(middle_inhibitory, source="Hidden Layer", target="Hidden Layer")
    network.add_connection(middle_out, source="Hidden Layer", target="Output Layer")

    # Load the Breakout environment.
    environment = GymEnvironment("CartPole-v0")
    environment.reset()

    # Build pipeline from specified components.
    environment_pipeline = EnvironmentPipeline(
        network,
        environment,
        encoding=convert_to_positive(bernoulli),
        action_function=select_softmax,
        output="Output Layer",
        time=100,
        history_length=1,
        delta=1,
        plot_interval=1,
        render_interval=1,
        analyzer=TensorboardAnalyzer,
        plot_config={"summary_directory": str(LOGS_DIR / str(uuid.uuid1())), "reward_eps": 1},
    )

    def run_pipeline(pipeline, episode_count):
        for i in range(episode_count):
            total_reward = 0
            pipeline.reset_state_variables()
            is_done = False
            while not is_done:
                result = pipeline.env_step()
                pipeline.step(result)

                reward = result[1]
                total_reward += reward

                is_done = result[2]
            print(f"Episode {i} total reward:{total_reward}")

    print("Training: ")
    run_pipeline(environment_pipeline, episode_count=1000)

    # stop MSTDP
    environment_pipeline.network.learning = False

    print("Testing: ")
    run_pipeline(environment_pipeline, episode_count=1000)


if __name__ == "__main__":
    main()
