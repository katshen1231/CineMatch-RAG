"""
config.py — Centralized configuration management (paths, Azure endpoints, and model parameters)

"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------- Path ----------
PROJECT_ROOT = Path(__file__).parent  
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = PROJECT_ROOT / "indexes"
INDEX_DIR.mkdir(exist_ok=True)  

MOVIES_PATH = DATA_DIR / "ml-1m" / "movies.dat"
RATINGS_PATH = DATA_DIR / "ml-1m" / "ratings.dat"
INTRO_PATH = DATA_DIR / "IMDBPoster" / "info" / "info.csv"


FAISS_INDEX_PATH = INDEX_DIR / "movies.faiss"
METADATA_PATH = INDEX_DIR / "metadata.pkl"

# ---------- Azure OpenAI ----------
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT")

# ---------- Hyperparameters ----------
EMBEDDING_DIM = 1536         
EMBED_BATCH_SIZE = 64         
TOP_K = 10                    

_required = {
    "AZURE_OPENAI_ENDPOINT": AZURE_ENDPOINT,
    "AZURE_OPENAI_API_KEY": AZURE_API_KEY,
    "AZURE_EMBEDDING_DEPLOYMENT": EMBEDDING_DEPLOYMENT,
    "AZURE_CHAT_DEPLOYMENT": CHAT_DEPLOYMENT,
}
_missing = [k for k, v in _required.items() if not v]
if _missing:
    raise RuntimeError(f"Missing enviroment variables: {_missing}, Please check the .env document")