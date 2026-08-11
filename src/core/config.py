from pathlib import Path

# Project Root Directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Project Folders
ASSETS_DIR = BASE_DIR / "assets"
DATASETS_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"
NOTEBOOK_DIR = BASE_DIR / "notebooks"
EXPERIMENTS_DIR = BASE_DIR / "experiments"

# Supported File Types
SUPPORTED_FILE_TYPES = ["csv", "xlsx"]

# Maximum Upload Size (MB)
MAX_FILE_SIZE_MB = 100


def ensure_project_directories() -> None:
    """Create runtime directories required by DataMindAI."""
    for directory in (
        ASSETS_DIR,
        DATASETS_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        LOGS_DIR,
        NOTEBOOK_DIR,
        EXPERIMENTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)