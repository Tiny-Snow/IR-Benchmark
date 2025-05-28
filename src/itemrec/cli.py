# -------------------------------------------------------------------
# ItemRec / Item Recommendation Benchmark
# Copyright (C) 2024 Tiny Snow / Weiqin Yang @ Zhejiang University
# -------------------------------------------------------------------
# Module: Command Line Interface
# Description:
#   This module provides a Command Line Interface (CLI) for ItemRec.
#   The main function is the entry point for the ItemRec.
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
import hashlib
from .args import parse_args
from .utils import logger, timer
from .utils import set_experiments_main, run
from .hyper import get_params

# main function -----------------------------------------------------
def main():
    r"""
    ## Function
    The main Entry point for ItemRec command line interface.
    """
    # parse arguments
    args = parse_args()
    # NNI: update hyper parameters
    args = get_params(args)
    # set up basic configurations
    set_experiments_main(args)
    # start global time record
    timer.start('GLOBAL')
    # run the training and testing process
    run(args)
    # end global time record
    timer.end('GLOBAL')

