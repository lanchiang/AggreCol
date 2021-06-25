# Created by lan at 2021/2/11
import decimal
import gzip
import json
import math
import os
import re
import time
from decimal import Decimal

import luigi
import numpy as np
from luigi.mock import MockTarget
from tqdm import tqdm

from data import normalize_file, detect_number_format
from elements import File
from helpers import AggregationOperator
from number import str2decimal


class LoadDataset(luigi.Task):
    """
    This task loads a dataset stored in a json.jl.gz compressed file into the memory, selecting only those entries that are useful to aggregation detection.
    """

    dataset_path = luigi.Parameter()
    result_path = luigi.Parameter('/debug/')

    error_level_dict = luigi.DictParameter(default={'Sum': 0, 'Average': 0, 'Division': 0, 'RelativeChange': 0})
    coverage = luigi.FloatParameter(default=0.7)
    use_extend_strategy = luigi.BoolParameter(default=False, parsing=luigi.BoolParameter.EXPLICIT_PARSING)
    timeout = luigi.FloatParameter(default=300)

    debug = luigi.BoolParameter(default=False, parsing=luigi.BoolParameter.EXPLICIT_PARSING)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.result_path, 'load-dataset.jl'))

    def run(self):
        with gzip.open(self.dataset_path, mode='r') as ds_json_file:
            json_file_dicts = np.array([json.loads(line) for line in ds_json_file])
            dataset = [json.dumps({'file_name': jfd['file_name'],
                                   'sheet_name': jfd['table_id'],
                                   'num_rows': jfd['num_rows'],
                                   'num_cols': jfd['num_cols'],
                                   'file_values': jfd['table_array'],
                                   'aggregation_annotations': jfd['aggregation_annotations'],
                                   'parameters': {'error_level_dict': dict(self.error_level_dict),
                                                  'use_extend_strategy': self.use_extend_strategy,
                                                  'coverage': self.coverage,
                                                  'timeout': self.timeout},
                                   'number_format': jfd['number_format'],
                                   'exec_time': {}}) for jfd in json_file_dicts]

        with self.output().open('w') as file_writer:
            for curated_json_file in dataset:
                file_writer.write(curated_json_file + '\n')


class NumberFormatNormalization(luigi.Task):
    """
    This task runs the number format normalization task on the initial input file, and produces a set of valid number formats on each file.
    """

    dataset_path = luigi.Parameter()
    result_path = luigi.Parameter('./debug/')
    error_level_dict = luigi.DictParameter(default={'Sum': 0, 'Average': 0, 'Division': 0, 'RelativeChange': 0})
    use_extend_strategy = luigi.BoolParameter(default=False, parsing=luigi.BoolParameter.EXPLICIT_PARSING)
    coverage = luigi.FloatParameter(default=0.7)
    timeout = luigi.FloatParameter(default=300)
    sample_ratio = luigi.FloatParameter(default=1)

    debug = luigi.BoolParameter(default=False, parsing=luigi.BoolParameter.EXPLICIT_PARSING)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.result_path, 'normalize_number_format.jl'))

    def requires(self):
        return LoadDataset(error_level_dict=self.error_level_dict, use_extend_strategy=self.use_extend_strategy,
                           coverage=self.coverage,
                           timeout=self.timeout,
                           dataset_path=self.dataset_path, debug=self.debug, result_path=self.result_path)

    def run(self):
        with self.input().open('r') as input_file:
            files_dict = [json.loads(line) for line in input_file]

            for file_dict in tqdm(files_dict, desc='Number format detection'):
                start_time = time.time()
                number_format = detect_number_format(np.array(file_dict['file_values']), self.sample_ratio)

                transformed_values_by_number_format = {}
                numeric_line_indices = {}
                file_cell_data_types = {}
                for nf in number_format:
                    transformed_values_by_number_format[nf], numeric_line_indices[nf], file_cell_data_types[nf] = normalize_file(file_dict['file_values'], nf)
                file_dict['valid_number_formats'] = transformed_values_by_number_format
                file_dict['numeric_line_indices'] = numeric_line_indices

                # test to use the best choice
                file_dict['number_format'] = number_format[0]
                file_dict['detected_number_format'] = number_format[0]

                end_time = time.time()
                exec_time = end_time - start_time
                file_dict['exec_time'][self.__class__.__name__] = exec_time

        with self.output().open('w') as file_writer:
            for file_dict in files_dict:
                file_writer.write(json.dumps(file_dict) + '\n')


if __name__ == '__main__':
    luigi.run()
