# pylint: disable=logging-fstring-interpolation, broad-except
"""prediction"""
import argparse
import json
import os
from os.path import join
from sys import path

from common import get_logger, Timer, TimeoutException
from dataset import AutoWSLDataset

# pylint: disable=import-error

VERBOSITY_LEVEL = 'WARNING'
LOGGER = get_logger(VERBOSITY_LEVEL, __file__)


def _write_predict(output_dir, prediction):
    os.makedirs(output_dir, exist_ok=True)
    prediction.rename('label', inplace=True)
    LOGGER.debug(f'prediction shape: {prediction.shape}')
    prediction.to_csv(
        join(output_dir, f'prediction'), index=False, header=True)


def _predict(args):
    result = {}
    try:
        dataset = AutoWSLDataset(args.dataset_dir)
        path.append(args.model_dir)
        LOGGER.info('==== Load user model')
        from model import Model
        umodel = Model(dataset.get_metadata())
        timer = Timer()
        timer.set(args.pred_time_budget)
        LOGGER.info('==== start predicting')
        with timer.time_limit('predicting'):
            umodel.load(args.temp_dir)
            y_pred = umodel.predict(dataset.get_test())
        duration = timer.duration
        LOGGER.info(
            f"Finished predicting the model. time spent {duration:5.2} sec")
        # Write predictions to output_dir
        _write_predict(args.output_dir, y_pred)
        result['status'] = 'success'
        result['duration'] = duration
    except TimeoutException as ex:
        LOGGER.error(ex, exc_info=True)
        result['status'] = 'timeout'
    except Exception as ex:
        LOGGER.error(ex, exc_info=True)
        result['status'] = 'failed'

    return result


def _write_result(args, result):
    with open(args.result_file, 'w') as ftmp:
        json.dump(result, ftmp)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str,
                        help="Directory storing the dataset (containing "
                             "e.g. adult.data/)")
    parser.add_argument('--model_dir', type=str,
                        help="Directory storing the model ")

    parser.add_argument('--result_file', type=str,
                        help="a json file save the result")

    parser.add_argument('--temp_dir', type=str,
                        help="Directory storing the temporary output."
                             "e.g. save the participants` model "
                             "after trainning.")

    parser.add_argument('--output_dir', type=str,
                        help="Directory storing the predictions. It will "
                             "contain e.g. [start.txt, adult.predict_0, "
                             "adult.predict_1, ..., end.txt] when ingestion "
                             "terminates.")

    parser.add_argument("--pred_time_budget", type=float,
                        help="Time budget for predicting model "
                             "if not specified in meta.json.")

    args = parser.parse_args()
    return args


def main():
    """main entry"""
    LOGGER.info('==== prediction process started')
    args = _parse_args()
    result = _predict(args)
    LOGGER.info('==== Write result')
    _write_result(args, result)


if __name__ == '__main__':
    main()
