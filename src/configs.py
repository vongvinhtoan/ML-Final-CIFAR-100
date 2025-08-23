from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv(".env.local")

class Settings:
    def __init__(self) -> None:
        self.GGDRIVE_TOKEN_PATH = Path(os.getenv("GGDRIVE_TOKEN_PATH", ""))
        self.GGDRIVE_FOLDER_ID = Path(os.getenv("GGDRIVE_FOLDER_ID", ""))
        self.DATASET_PATH = Path(os.getenv("DATASET_PATH", ""))
        self.SWEEP_CONFIG_PATH = Path(os.getenv("SWEEP_CONFIG_PATH", ""))
        self.WEIGHT_PATH = Path(os.getenv("WEIGHT_PATH", ""))
        self.WANDB_PROJECT_NAME = os.getenv("WANDB_PROJECT_NAME")
        self.WANDB_TEAM_NAME = os.getenv("WANDB_TEAM_NAME")

settings = Settings()