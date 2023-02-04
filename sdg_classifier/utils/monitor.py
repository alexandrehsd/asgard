import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s - %(levelname)s: %(message)s",
    datefmt='%d-%m-%Y %H:%M:%S'
)

LOGGER = logging.getLogger()