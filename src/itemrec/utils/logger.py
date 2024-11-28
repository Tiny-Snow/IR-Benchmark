# -------------------------------------------------------------------
# ItemRec / Item Recommendation Benchmark
# Copyright (C) 2024 Tiny Snow / Weiqin Yang @ Zhejiang University
# -------------------------------------------------------------------
# Module: Logger
# Description:
#   This module provides a global logger for the project.
# -------------------------------------------------------------------

# import modules ----------------------------------------------------
from typing import (
    Any, 
    Optional,
    List,
    Tuple,
    Set,
    Dict,
    Callable,
)
import os
import logging

# public functions --------------------------------------------------
__all__ = [
    "logger",
    "setup_logger",
    "set_logfile",
]

# logger ------------------------------------------------------------
def setup_logger(name: str = 'ItemRec', log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    ## Function
    Setup a logger, output to stdout and log file.

    ## Arguments
    name: str
        Name of the logger.
    log_file: str
        Path to the log file. Default: None.
    level: int
        Logging level. Default: logging.INFO.

    ## Returns
    logger: logging.Logger
        A logger object.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if log_file:
        log_file = os.path.abspath(log_file)
        if not os.path.exists(os.path.dirname(log_file)):
            os.makedirs(os.path.dirname(log_file))
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info(f"Succesfully setup logger {name}.")
    return logger

def set_logfile(logger: logging.Logger, log_file: str) -> None:
    """
    ## Function
    Set the log file of a logger.

    ## Arguments
    logger: logging.Logger
        A logger object.
    log_file: str
        Path to the log file.
    """
    log_file = os.path.abspath(log_file)
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    fh = logging.FileHandler(log_file)
    fh.setLevel(logger.level)
    logger.addHandler(fh)
    logger.info(f"Set log file to {log_file}.")

# global logger
logger = setup_logger(name="ItemRec")

