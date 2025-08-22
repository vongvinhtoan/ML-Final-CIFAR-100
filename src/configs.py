from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv(".env.local")

class Settings:
    def __init__(self) -> None:
        self.GGDRIVE_TOKEN_PATH = Path(os.getenv("GGDRIVE_TOKEN_PATH", ""))
        self.GGDRIVE_FOLDER_ID = Path(os.getenv("GGDRIVE_FOLDER_ID", ""))
        self.DATASET_PATH = Path(os.getenv("DATASET_PATH", ""))

settings = Settings()