"""
Logging configuration for the Childcare RAG System.
"""

import logging
import colorlog
from rich.logging import RichHandler
from .settings import settings


def setup_logging() -> logging.Logger:
    """
    Set up logging configuration with color support and rich formatting.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("childcare_rag")
    logger.setLevel(getattr(logging, settings.log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    if settings.debug_mode:
        # Rich handler for development
        handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=True
        )
        formatter = logging.Formatter(
            fmt="%(message)s",
            datefmt="[%X]"
        )
    else:
        # Color handler for production
        handler = colorlog.StreamHandler()
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


# Global logger instance
logger = setup_logging()
