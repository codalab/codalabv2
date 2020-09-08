"""
  AutoWSL datasets.
"""
import os
from datetime import datetime
import json
from glob import glob as ls
import numpy as np
import pandas as pd
from common import get_logger

TYPE_MAP = {
    'time': str,
    'cat': str,
    'multi-cat': str,
    'num': np.float64
}

VERBOSITY_LEVEL = 'WARNING'
LOGGER = get_logger(VERBOSITY_LEVEL, __file__)


class AutoWSLDataset:
    """"AutoWSLDataset"""
    def __init__(self, dataset_dir):
        """
            train_dataset, test_dataset: list of strings
            train_label: np.array
        """
        self.dataset_name_ = dataset_dir
        self.dataset_dir_ = dataset_dir
        self.metadata_ = self._read_metadata(
            os.path.join(dataset_dir, "info.json"))
        self.train_dataset = None
        self.train_label = None
        self.test_dataset = None

    def read_dataset(self):
        """read dataset"""
        self.train_dataset = self._read_dataset(
            os.path.join(self.dataset_dir_, "train.data"))
        self.train_label = self.read_label(
            os.path.join(self.dataset_dir_, "train.solution"))
        self.test_dataset = self._read_dataset(
            os.path.join(self.dataset_dir_, "test.data"))

    def get_train(self):
        """get train"""
        if self.train_dataset is None:
            self.train_dataset = self._read_dataset(
                os.path.join(self.dataset_dir_, "train.data"))
            self.train_label = self.read_label(
                os.path.join(self.dataset_dir_, "train.solution"))
        return self.train_dataset, self.train_label

    def get_test(self):
        """get test"""
        if self.test_dataset is None:
            self.test_dataset = self._read_dataset(
                os.path.join(self.dataset_dir_, "test.data"))
        return self.test_dataset

    def get_metadata(self):
        """get metadata"""
        return self.metadata_

    @staticmethod
    def _read_metadata(metadata_path):
        return json.load(open(metadata_path))

    def _read_dataset(self, dataset_path):
        schema = self.metadata_['schema']
        table_dtype = {key: TYPE_MAP[val] for key, val in schema.items()}
        date_list = [key for key, val in schema.items() if val == 'time']
        date_parser = (
            lambda millisecs: millisecs if np.isnan(float(millisecs))
            else datetime.fromtimestamp(float(millisecs)/1000)
        )

        dataset = pd.read_csv(
            dataset_path, sep='\t', dtype=table_dtype,
            parse_dates=date_list, date_parser=date_parser)

        return dataset

    @staticmethod
    def read_label(label_path):
        """read_label"""
        train_label = pd.read_csv(label_path)['label']
        return train_label

    def get_train_num(self):
        """ return the number of train instance """
        return self.metadata_["train_num"]

    def get_test_num(self):
        """ return the number of test instance """
        return self.metadata_["test_num"]


def inventory_data(input_dir):
    ''' Inventory the datasets in the input directory and
        return them in alphabetical order'''
    # Assume first that there is a hierarchy dataname/dataname_train.data
    training_names = ls(os.path.join(input_dir, '*.data'))
    training_names = [name.split('/')[-1] for name in training_names]

    if not training_names:
        LOGGER.warning('WARNING: Inventory data - No data file found')

    return sorted(training_names)


def get_dataset(args):
    """get dataset"""
    datanames = inventory_data(args.dataset_dir)
    datanames = [x for x in datanames if x.endswith('.data')]

    if len(datanames) != 1:
        raise ValueError(
            f"{len(datanames)} datasets found "
            f"in dataset_dir={args.dataset_dir}!\n"
            "Please put only ONE dataset under dataset_dir.")

    basename = datanames[0]
    dataset_name = basename[:-5]
    dataset = AutoWSLDataset(os.path.join(args.dataset_dir, basename))
    return dataset, dataset_name
