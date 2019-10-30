from pathlib import Path
import logging
import datetime as dt
import sys
import os


## logging
def get_logger(name, log_dir="log/"):
    log_path = Path(log_dir)
    path = log_path

    if not os.path.exists(log_dir):
        path.mkdir(parents=True)

    logger = logging.getLogger("Main")
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(
        str(path / (dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "-" + name + ".log")))
    sh = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s")

    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger