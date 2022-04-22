import pathlib

ROOT_DIR = pathlib.Path(__file__).parents[0].parents[0]
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = DATA_DIR / "logs"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
RENDERS_DIR = DATA_DIR / "renders"
CONFIGS_DIR = ROOT_DIR / "configs"
