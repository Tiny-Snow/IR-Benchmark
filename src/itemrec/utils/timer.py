# -------------------------------------------------------------------
# ItemRec / Item Recommendation Benchmark
# Copyright (C) 2024 Tiny Snow / Weiqin Yang @ Zhejiang University
# -------------------------------------------------------------------
# Module: Timer
# Description:
#   This module provides a global timer for time cost calculation.
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
import time
from logging import Logger

# public functions --------------------------------------------------
__all__ = [
    "timer",
    "Timer",
]

# time cost calculation ---------------------------------------------
class Timer:
    """
    ## Class
    A simple timer class to calculate time cost.
    This Timer can be used to multiple time records, each recore
    is identified by a string `info`.

    ## Example
    ```python
    >>> timer = Timer()
    >>> timer.start('TIME1')    # info is used as identifier
    >>> timer.start('TIME2')    # if duplicated, the previous record will be overwritten
    >>> timer.end('TIME2')      # end with the same info id
    TIME2 | Time cost: 00:00:04
    4.10322380065918
    >>> timer.end('TIME1')      # multiple records are allowed
    TIME1 | Time cost: 00:00:08
    8.809662580490112
    >>> timer.end('TIME1')      # if not started, an error will be raised
    AssertionError: Timer with info 'TIME1' has not been started.
    ```
    """
    def __init__(self, logger: Optional[Logger] = None):
        """
        ## Function
        Initialize a timer object.

        ## Arguments
        - logger: Logger (optional, default: None)
            A logger object.
        """
        self.logger = logger
        self._starts : Dict[str, float] = {}
    
    def start(self, info: str = ''):
        """
        ## Function
        Start the timer.

        ## Arguments
        - info: str
            Information to be logged.
            The `info` will be used as identifier of this record.
        """
        start_time = time.time()
        if self._starts.get(info) is not None and self.logger is not None:
            self.logger.warning(f"Timer with info '{info}' has already been started."
                                " The previous record will be overwritten.")
        self._starts[info] = start_time
    
    def end(self, info: str = ''):
        """
        ## Function
        End the timer and return the time cost.
        Clear the timer after ending.

        ## Returns
        - cost: float
            Time cost in seconds.
        """
        assert self._starts.get(info) is not None, f"Timer with info '{info}' has not been started."
        end_time = time.time()
        cost = end_time - self._starts[info]
        log_str = f"Time cost: {time.strftime('%H:%M:%S', time.gmtime(cost))}"
        if info:
            log_str = f"{info} | " + log_str
        if self.logger is not None:
            self.logger.info(log_str)
        else:
            print(log_str)
        # clear timer
        self._starts.pop(info)
        return cost
    
    def change_logger(self, logger: Optional[Logger] = None):
        """
        ## Function
        Change the logger of the timer.

        ## Arguments
        - logger: Logger (optional, default: None)
            A logger object.
        """
        self.logger = logger


# global timer
timer = Timer()

