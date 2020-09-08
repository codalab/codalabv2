# pylint: disable=logging-fstring-interpolation
"""scoring function for autowsl"""

import argparse
import datetime
import glob
import os
from os.path import join, isfile
import json
import logging
import sys
import time
import yaml
from filelock import FileLock

import psutil
import pandas as pd
from sklearn.metrics import roc_auc_score

# Verbosity level of logging.
# Can be: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
VERBOSITY_LEVEL = 'INFO'
WAIT_TIME = 30
MAX_TIME_DIFF = datetime.timedelta(seconds=30)
DEFAULT_SCORE = -1.0


def get_logger(verbosity_level, use_error_log=False):
    """Set logging format to something like:
        2019-04-25 12:52:51,924 INFO score.py: <message>
    """
    logger = logging.getLogger(__file__)
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


LOGGER = get_logger(VERBOSITY_LEVEL)


def _ls(filename):
    return sorted(glob.glob(filename))


def _here(*args):
    """Helper function for getting the current directory of the script."""
    here_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(join(here_dir, *args))


def _read_pred(pred_file):
    return pd.read_csv(pred_file)


def _get_solution(solution_dir):
    """Get the solution array from solution directory."""
    solution_names = sorted(_ls(join(solution_dir, '*.solution')))
    if len(solution_names) != 1:  # Assert only one file is found
        LOGGER.warning(f"{len(solution_names)} solution files found: "
                       f"{solution_names}! Return `None` as solution.")
        return None
    solution_file = solution_names[0]
    solution = _read_pred(solution_file)
    return solution


def get_prediction_file(prediction_dir):
    """Return prediction files in prediction directory.  """
    return join(prediction_dir, 'prediction')


def _get_score(solution_dir, prediction_dir):
    """Draw learning curve for one task."""
    LOGGER.info('===== get solution')
    solution = _get_solution(solution_dir)
    LOGGER.info('===== read prediction')
    prediction = _read_pred(get_prediction_file(prediction_dir))
    if solution.shape != prediction.shape:
        raise ValueError(f"Bad prediction shape: {prediction.shape}. "
                         f"Expected shape: {solution.shape}")

    LOGGER.info('===== calculate score')
    score = roc_auc_score(solution, prediction)

    return score


def _update_score(args, duration):
    score = _get_score(solution_dir=args.solution_dir,
                       prediction_dir=args.prediction_dir)
    # Update learning curve page (detailed_results.html)
    _write_scores_html(args.score_dir)
    # Write score
    LOGGER.info('===== write score')
    write_score(args.score_dir, score, duration)
    LOGGER.info(f"AUC: {score:.4}")
    return score


def _init_scores_html(detailed_results_filepath):
    html_head = ('<html><head> <meta http-equiv="refresh" content="5"> '
                 '</head><body><pre>')
    html_end = '</pre></body></html>'
    with open(detailed_results_filepath, 'a') as html_file:
        html_file.write(html_head)
        html_file.write("Starting training process... <br> Please be patient. "
                        "Learning curves will be generated when first "
                        "predictions are made.")
        html_file.write(html_end)


def _write_scores_html(score_dir, auto_refresh=True, append=False):
    filename = 'detailed_results.html'
    if auto_refresh:
        html_head = ('<html><head> <meta http-equiv="refresh" content="5"> '
                     '</head><body><pre>')
    else:
        html_head = """<html><body><pre>"""
    html_end = '</pre></body></html>'
    if append:
        mode = 'a'
    else:
        mode = 'w'
    filepath = join(score_dir, filename)
    with open(filepath, mode) as html_file:
        html_file.write(html_head)
        html_file.write(html_end)
    LOGGER.debug(f"Wrote learning curve page to {filepath}")


def write_score(score_dir, score, duration):
    """Write score and duration to score_dir/scores.txt"""
    score_filename = join(score_dir, 'scores.txt')
    
    with open(score_filename, 'w') as ftmp:
        ftmp.write(f'score: {score}\n')
        ftmp.write(f'Duration: {duration}\n')
    
    print('results:')
    scores = {
        'score': score,
        'Duration': duration
    }
    print(scores)
    
    LOGGER.debug(f"Wrote to score_filename={score_filename} with "
                 f"score={score}, duration={duration}")


def get_ingestion_info(prediction_dir):
    """Get info on ingestion program: PID, start time, etc. from 'start.txt'.

    Args:
        prediction_dir: a string, directory containing predictions (output of
        ingestion)
    Returns:
        A dictionary with keys 'ingestion_pid' and 'start_time' if the file
        'start.txt' exists. Otherwise return `None`.
    """
    start_filepath = join(prediction_dir, 'start.txt')
    if os.path.exists(start_filepath):
        with open(start_filepath, 'r') as ftmp:
            ingestion_info = yaml.safe_load(ftmp)
        return ingestion_info
    return None


def _is_process_alive(pid):
    return psutil.pid_exists(pid)


def get_task_name(solution_dir):
    """Get the task name from solution directory."""
    solution_names = sorted(_ls(join(solution_dir, '*.solution')))
    if len(solution_names) != 1:
        LOGGER.warning(
            f"{len(solution_names)} solution files found: {solution_names}! "
            "Return `None` as task name.")
        return None
    solution_file = solution_names[0]
    task_name = solution_file.split(os.sep)[-1].split('.')[0]
    return task_name


class IngestionError(Exception):
    """Ingestion error"""


class ScoringError(Exception):
    """scoring error"""


def _parse_args():
    # Default I/O directories:
    root_dir = _here(os.pardir)
    default_solution_dir = join(root_dir, "sample_data")
    default_prediction_dir = join(root_dir, "sample_result_submission")
    default_score_dir = join(root_dir, "scoring_output")
    parser = argparse.ArgumentParser()
    parser.add_argument('--solution_dir', type=str,
                        default=default_solution_dir,
                        help=("Directory storing the solution with true "
                              "labels, e.g. adult.solution."))
    parser.add_argument('--prediction_dir', type=str,
                        default=default_prediction_dir,
                        help=("Directory storing the predictions. It should"
                              "contain e.g. [start.txt, adult.predict_0, "
                              "adult.predict_1, ..., end.txt]."))
    parser.add_argument('--score_dir', type=str,
                        default=default_score_dir,
                        help=("Directory storing the scoring output e.g. "
                              "`scores.txt` and `detailed_results.html`."))
    args = parser.parse_args()
    LOGGER.debug(f"Parsed args are: {args}")
    LOGGER.debug("-" * 50)
    LOGGER.debug(f"Using solution_dir: {args.solution_dir}")
    LOGGER.debug(f"Using prediction_dir: {args.prediction_dir}")
    LOGGER.debug(f"Using score_dir: {args.score_dir}")
    return args


def _detect_ingestion_alive(args):
    start_filepath = join(args.prediction_dir, 'start.txt')
    lockfile = join(args.prediction_dir, 'start.txt.lock')
    if not os.path.exists(start_filepath):
        return False

    with FileLock(lockfile):
        with open(start_filepath, 'r') as ftmp:
            try:
                last_time = datetime.datetime.fromtimestamp(json.load(ftmp))
            except Exception:
                LOGGER.info(f"the content of start.txt is: {ftmp.read()}")
                raise

    current_time = datetime.datetime.now()
    timediff = current_time - last_time
    if timediff > MAX_TIME_DIFF:
        return False

    return True


def _init(args):
    if not os.path.isdir(args.score_dir):
        os.mkdir(args.score_dir)
    detailed_results_filepath = join(
        args.score_dir, 'detailed_results.html')
    # Initialize detailed_results.html
    _init_scores_html(detailed_results_filepath)

    # Wait 30 seconds for ingestion to start and write 'start.txt',
    # Otherwise, raise an exception.
    # ingestion_info = None
    LOGGER.info('===== wait for ingestion to start')
    for _ in range(WAIT_TIME):
        if _detect_ingestion_alive(args):
            LOGGER.info('===== detect alive ingestion')
            break
        time.sleep(1)
    else:
        raise IngestionError("[-] Failed: scoring didn't detected the start "
                             f"of ingestion after {WAIT_TIME} seconds.")


def _finalize(args, score, scoring_start):
    """finalize the scoring"""
    task_name = get_task_name(args.solution_dir)
    # Use 'end.txt' file to detect if ingestion program ends
    duration = time.time() - scoring_start
    LOGGER.info(
        "[+] Successfully finished scoring! "
        f"Scoring duration: {duration:.2} sec. "
        "The score of your algorithm on the task "
        f"'{task_name}' is: {score:.6}.")

    LOGGER.info("[Scoring terminated]")


def _exist_endfile(args):
    return isfile(join(args.prediction_dir, 'end.txt'))


def _get_ingestion_info(args):
    endfile = join(args.prediction_dir, 'end.txt')
    with open(endfile, 'r') as ftmp:
        ingestion_info = json.load(ftmp)
    return ingestion_info


def main():
    """main entry"""
    scoring_start = time.time()
    LOGGER.info('===== init scoring program')
    args = _parse_args()
    _init(args)
    score = DEFAULT_SCORE
    # Moniter training processes, stop when ingestion stop or detect endfile
    LOGGER.info('===== wait for the exit of ingestion or end.txt file')
    while _detect_ingestion_alive(args) and (not _exist_endfile(args)):
        time.sleep(1)

    if not _exist_endfile(args):
        raise RuntimeError("no end.txt exist, ingestion failed")
    else:
        LOGGER.info('===== end.txt file detected, get ingestion information')
        # Compute/write score
        ingestion_info = _get_ingestion_info(args)
        duration = ingestion_info['ingestion_duration']
        score = _update_score(args, duration)

    _finalize(args, score, scoring_start)


if __name__ == "__main__":
    main()
