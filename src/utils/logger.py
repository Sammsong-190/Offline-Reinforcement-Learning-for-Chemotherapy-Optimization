"""Logger: WandB / TensorBoard 封装 (可选)"""
import logging


def get_logger(name="offline_chemo", level=logging.INFO):
    """Simple logger. Extend for WandB/TensorBoard."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(h)
        logger.setLevel(level)
    return logger
