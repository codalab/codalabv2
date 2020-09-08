# pylint: disable=logging-fstring-interpolation, broad-except
"""ingestion program for autoWSL"""
import os
from os.path import join
import sys
from sys import path
import argparse
import time
import datetime
import subprocess
import threading
import json
from filelock import FileLock

from common import get_logger, Timer
from dataset import AutoWSLDataset


# pylint: disable=import-error

# Verbosity level of logging:
# Can be: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
VERBOSITY_LEVEL = 'INFO'
LOGGER = get_logger(VERBOSITY_LEVEL, __file__)


def _here(*args):
    """Helper function for getting the current directory of this script."""
    here = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(os.path.join(here, *args))


def write_start_file(output_dir):
    """Create start file 'start.txt' in `output_dir` with updated timestamp
    start time.

    """
    LOGGER.info('===== alive_thd started')
    start_filepath = os.path.join(output_dir, 'start.txt')
    lockfile = os.path.join(output_dir, 'start.txt.lock')
    while True:
        current_time = datetime.datetime.now().timestamp()
        with FileLock(lockfile):
            with open(start_filepath, 'w') as ftmp:
                json.dump(current_time, ftmp)
        time.sleep(10)


class ModelApiError(Exception):
    """Model api error"""


class IngestionError(RuntimeError):
    """Model api error"""


def _parse_args():
    root_dir = _here(os.pardir)
    default_dataset_dir = join(root_dir, "sample_data")
    default_output_dir = join(root_dir, "sample_result_submission")
    default_ingestion_program_dir = join(root_dir, "ingestion_program")
    default_code_dir = join(root_dir, "code_submission")
    default_score_dir = join(root_dir, "scoring_output")
    default_temp_dir = join(root_dir, 'temp_output')
    default_time_budget = 1200
    default_pred_time_budget = 600
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str,
                        default=default_dataset_dir,
                        help="Directory storing the dataset (containing "
                             "e.g. adult.data/)")
    parser.add_argument('--output_dir', type=str,
                        default=default_output_dir,
                        help="Directory storing the predictions. It will "
                             "contain e.g. [start.txt, adult.predict_0, "
                             "adult.predict_1, ..., end.txt] when ingestion "
                             "terminates.")
    parser.add_argument('--ingestion_program_dir', type=str,
                        default=default_ingestion_program_dir,
                        help="Directory storing the ingestion program "
                             "`ingestion.py` and other necessary packages.")
    parser.add_argument('--code_dir', type=str,
                        default=default_code_dir,
                        help="Directory storing the submission code "
                             "`model.py` and other necessary packages.")
    parser.add_argument('--score_dir', type=str,
                        default=default_score_dir,
                        help="Directory storing the scoring output "
                             "e.g. `scores.txt` and `detailed_results.html`.")
    parser.add_argument('--temp_dir', type=str,
                        default=default_temp_dir,
                        help="Directory storing the temporary output."
                             "e.g. save the participants` model after "
                             "trainning.")
    parser.add_argument("--time_budget", type=float,
                        default=default_time_budget,
                        help="Time budget for trainning model if not specified"
                             " in meta.json.")
    parser.add_argument("--pred_time_budget", type=float,
                        default=default_pred_time_budget,
                        help="Time budget for predicting model "
                             "if not specified in meta.json.")
    args = parser.parse_args()
    LOGGER.debug(f'Parsed args are: {args}')
    LOGGER.debug("-" * 50)
    if (args.dataset_dir.endswith('run/input') and
            args.code_dir.endswith('run/program')):
        LOGGER.debug("Since dataset_dir ends with 'run/input' and code_dir "
                     "ends with 'run/program', suppose running on "
                     "CodaLab platform. Modify dataset_dir to 'run/input_data'"
                     " and code_dir to 'run/submission'. "
                     "Directory parsing should be more flexible in the code of"
                     " compute worker: we need explicit directories for "
                     "dataset_dir and code_dir.")

        args.dataset_dir = args.dataset_dir.replace(
            'run/input', 'run/input_data')
        args.code_dir = args.code_dir.replace(
            'run/program', 'run/submission')

        # Show directories for debugging
        LOGGER.debug(f"sys.argv = {sys.argv}")
        LOGGER.debug(f"Using dataset_dir: {args.dataset_dir}")
        LOGGER.debug(f"Using output_dir: {args.output_dir}")
        LOGGER.debug(
            f"Using ingestion_program_dir: {args.ingestion_program_dir}")
        LOGGER.debug(f"Using code_dir: {args.code_dir}")
    return args


def _init_python_path(args):
    path.append(args.ingestion_program_dir)
    path.append(args.code_dir)
    # IG: to allow submitting the starting kit as sample submission
    path.append(args.code_dir + '/sample_code_submission')
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)


def _check_umodel_methed(umodel):
    # Check if the model has methods `train`, `predict`, `save`, `load`.
    for attr in ['train', 'predict', 'save', 'load']:
        if not hasattr(umodel, attr):
            raise ModelApiError("Your model object doesn't have the method "
                                f"`{attr}`. Please implement it in model.py.")


def _train(args, umodel, dataset):
    # Train the model
    timer = Timer()
    timer.set(args.time_budget)
    train_dataset, train_label = dataset.get_train()
    with timer.time_limit('training'):
        umodel.train(train_dataset, train_label)
        umodel.save(args.temp_dir)
    duration = timer.duration
    LOGGER.info(f"Finished training the model. time spent {duration:5.2} sec")

    result = {}
    result['duration'] = duration
    return result


def _predict(args):
    # Make predictions using the trained model
    result_file = join(args.temp_dir, 'predproc_result.json')
    predict_py = join(
        os.path.dirname(os.path.realpath(__file__)), 'predict.py')
    LOGGER.info("===== call subprocess of prediction")
    subprocess.run(
        f"python {predict_py} --dataset_dir {args.dataset_dir} "
        f"--model_dir {args.code_dir} --result_file {result_file} "
        f"--output_dir {args.output_dir} "
        f"--temp_dir {args.temp_dir} "
        f"--pred_time_budget {args.pred_time_budget}", shell=True,
        check=True)
    with open(result_file, 'r') as ftmp:
        result = json.load(ftmp)

    return result


def _finalize(args, train_result, pred_result):
    if pred_result['status'] == 'success':
        # Finishing ingestion program
        end_time = time.time()
        overall_time_spent = train_result['duration'] + pred_result['duration']

        # Write overall_time_spent to a end.txt file
        end_filename = 'end.txt'
        content = {
            'ingestion_duration': overall_time_spent,
            'end_time': end_time}

        with open(join(args.output_dir, end_filename), 'w') as ftmp:
            json.dump(content, ftmp)
            LOGGER.info(
                f'Wrote the file {end_filename} marking the end of ingestion.')

            LOGGER.info("[+] Done. Ingestion program successfully terminated.")
            LOGGER.info(f"[+] Overall time spent {overall_time_spent:5.2} sec")

        # Copy all files in output_dir to score_dir
        os.system(
            f"cp -R {os.path.join(args.output_dir, '*')} {args.score_dir}")
        LOGGER.debug(
            "Copied all ingestion output to scoring output directory.")

        LOGGER.info("[Ingestion terminated]")
    elif pred_result['status'] == 'timeout':
        raise IngestionError('predicting timeout')
    else:
        raise IngestionError('error occurs when predicting')


def main():
    """main entry"""
    LOGGER.info('===== Start ingestion program.')
    # Parse directories from input arguments
    LOGGER.info('===== Initialize args.')
    args = _parse_args()
    _init_python_path(args)
    dataset = AutoWSLDataset(args.dataset_dir)
    LOGGER.info('===== Load metadata.')
    metadata = dataset.get_metadata()
    args.time_budget = metadata.get("time_budget", args.time_budget)
    args.pred_time_budget = metadata.get(
        "pred_time_budget", args.pred_time_budget)
    LOGGER.info(f"Time budget: {args.time_budget}")

    LOGGER.info('===== Set alive_thd')
    alive_thd = threading.Thread(target=write_start_file, name="alive",
                                 args=(args.output_dir,))
    alive_thd.daemon = True
    alive_thd.start()

    LOGGER.info("===== Load user model")
    from model import Model
    umodel = Model(metadata)

    LOGGER.info("===== Check user model methods")
    _check_umodel_methed(umodel)

    LOGGER.info("===== Begin training user model")
    train_result = _train(args, umodel, dataset)

    LOGGER.info("===== Begin preding by user model on test set")
    pred_result = _predict(args)

    _finalize(args, train_result, pred_result)


if __name__ == "__main__":
    main()
