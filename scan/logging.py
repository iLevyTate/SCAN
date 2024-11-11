import logging

from scan.config import settings

logging.basicConfig(
    level=settings.COMPUTED_LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
