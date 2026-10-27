import sys
from pathlib import Path
from loguru import logger

# Ensure log directory exists
Path("logs").mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stderr, level="INFO", colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
logger.add("logs/app_{time}.log", rotation="1 day", retention="7 days",
           level="DEBUG", encoding="utf-8")
