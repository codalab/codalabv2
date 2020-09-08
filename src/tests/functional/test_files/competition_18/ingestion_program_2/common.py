# pylint: disable=logging-fstring-interpolation, broad-except
"""common"""
import signal
import math
import logging
import time
import sys
from contextlib import contextmanager


def get_logger(verbosity_level, name, use_error_log=False):
    """Set logging format to something like:
        2019-04-25 12:52:51,924 INFO score.py: <message>
    """
    logger = logging.getLogger(name)
    logging_level = getattr(logging, verbosity_level)
    logger.setLevel(logging_level)
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)s %(filename)s: %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging_level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    if use_error_log:
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.WARNING)
        stderr_handler.setFormatter(formatter)
        logger.addHandler(stderr_handler)
    logger.propagate = False
    return logger


VERBOSITY_LEVEL = 'INFO'
LOGGER = get_logger(VERBOSITY_LEVEL, __file__)


class TimeoutException(Exception):
    """timeoutexception"""


class Timer:
    """timer"""
    def __init__(self):
        self.duration = 0
        self.total = None
        self.remain = None
        self.exec = None

    def set(self, time_budget):
        """set time_budget"""
        self.total = time_budget
        self.remain = time_budget
        self.exec = 0

    @contextmanager
    def time_limit(self, pname):
        """limit time"""
        def signal_handler(signum, frame):
            raise TimeoutException("Timed out!")
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(int(math.ceil(self.remain)))
        start_time = time.time()

        try:
            yield
        finally:
            exec_time = time.time() - start_time
            signal.alarm(0)
            self.exec += exec_time
            self.duration += exec_time
            self.remain = self.total - self.exec

        LOGGER.info(f'{pname} success, time spent so far {self.exec} sec')

        if self.remain <= 0:
            raise TimeoutException("Timed out!")
