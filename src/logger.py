from loguru import logger

logger.add(
    ".logs/log_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="INFO",
    rotation="00:00",
    retention="14 days",
    enqueue=True,  # write from a background thread so logging never blocks the event loop
)
