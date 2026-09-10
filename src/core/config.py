import os
from pathlib import Path

# Load .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Environment Variables & Configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
SECRET_KEY = os.getenv("SECRET_KEY", "datamind-ai-dev-secret-key-change-in-prod")
STORAGE_PATH = Path(os.getenv("STORAGE_PATH", str(ARTIFACTS_DIR / "storage")))

# Database Configuration (Development: SQLite, Production: PostgreSQL)
DEFAULT_DB_PATH = ARTIFACTS_DIR / "datamind.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")

# Ollama / LLM Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")

# Supported File Types
SUPPORTED_FILE_TYPES = ["csv", "xlsx", "json", "geojson", "png", "jpg", "jpeg"]

# Maximum Upload Size (MB)
MAX_FILE_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE", "100"))


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
        ARTIFACTS_DIR,
        STORAGE_PATH,
        ARTIFACTS_DIR / "models",
        ARTIFACTS_DIR / "datasets",
        ARTIFACTS_DIR / "images",
    ):
        directory.mkdir(parents=True, exist_ok=True)