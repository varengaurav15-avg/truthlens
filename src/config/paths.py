from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "data"
RAW_TEXT = DATA / "raw" / "text"
RAW_IMAGES = DATA / "raw" / "images"
RAW_VIDEOS = DATA / "raw" / "videos"
PROCESSED_TEXT = DATA / "processed" / "text"
PROCESSED_IMAGES = DATA / "processed" / "images"
KG_DIR = DATA / "knowledge_graph"
TRIPLES_CSV = KG_DIR / "triples.csv"
GRAPH_DB = KG_DIR / "graph.db"
MODELS_DIR = BASE / "models"
TEXT_MODEL = MODELS_DIR / "text"
DEEPFAKE_MODEL = MODELS_DIR / "deepfake"
GRAPH_MODEL = MODELS_DIR / "graph"
FUSION_MODEL = MODELS_DIR / "fusion"