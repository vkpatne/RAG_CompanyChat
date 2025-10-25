import logging
from logging.handlers import RotatingFileHandler

def setup_logger():
    logger = logging.getLogger("rag_app")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler("rag_app.log", maxBytes=5*1024*1024, backupCount=2)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger

logger = setup_logger()
