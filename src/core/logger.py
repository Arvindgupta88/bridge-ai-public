import logging
import sys


LOGGER_NAME = "BridgeAI"


logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.INFO)

# Prevent messages from being passed to Streamlit's root logger.
logger.propagate = False

# Avoid adding duplicate handlers during Streamlit reruns.
if not logger.handlers:
    console_handler = logging.StreamHandler(sys.stdout)

    console_handler.setLevel(logging.INFO)

    console_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )
    )

    logger.addHandler(console_handler)