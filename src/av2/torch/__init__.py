"""AV2 Pytorch interface."""

import logging

logger = logging.Logger(__file__)

try:
    import torch.multiprocessing as mp

    logging.info("Setting multiprocessing start method to forkserver.")
    mp.set_forkserver_preload(["polars"])
    mp.set_start_method("forkserver", force=True)
except ImportError as _:
    logger.error("Please install Pytorch to use this module.")
