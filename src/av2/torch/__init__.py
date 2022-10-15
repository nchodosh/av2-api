"""AV2 Pytorch interface."""

import logging

logger = logging.getLogger(__name__)

try:
    import torch.multiprocessing as mp

    if mp.get_start_method() != "forkserver":
        logging.warning("Setting multiprocessing start method to forkserver to avoid deadlocking.")
        mp.set_start_method("forkserver", force=True)
    mp.set_forkserver_preload(["polars"])
except ImportError as _:
    logger.error("Please install Pytorch to use this module.")
