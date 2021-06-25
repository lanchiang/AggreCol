# Created by lan at 2021/3/14
import decimal
import json
import math
import os
from abc import ABC, abstractmethod
from copy import copy
from decimal import Decimal

import luigi
from concurrent.futures import TimeoutError

from pebble import ProcessPool
from tqdm import tqdm

from elements import File, FileJsonEncoder
from helpers import AggregationOperator, empty_cell_values
from tree import AggregationRelationForest


class Approach(ABC):

    @abstractmethod
    def detect_row_wise_aggregations(self, file_dict):
        pass

    @abstractmethod
    def detect_column_wise_aggregations(self, file_dict):
        pass

    @abstractmethod
    def detect_proximity_aggregation_relations(self, forest: AggregationRelationForest, error_bound: float, error_strategy):
        pass


class AggregationDetection(luigi.Task, ABC):
    dataset_path = luigi.Parameter()
    result_path = luigi.Parameter(default='/debug/')
    error_level_dict = luigi.DictParameter(default={'Sum': 0, 'Average': 0, 'Division': 0, 'RelativeChange': 0})
    target_aggregation_type = luigi.Parameter(default='All')
    coverage = luigi.FloatParameter(default=0.6)
    use_extend_strategy = luigi.BoolParameter(default=False, parsing=luigi.BoolParameter.EXPLICIT_PARSING)
    timeout = luigi.FloatParameter(default=300)
    debug = luigi.BoolParameter(default=False, parsing=luigi.BoolParameter.EXPLICIT_PARSING)

    operator = ''
    task_name = ''

    error_level = 0

    DIGIT_PLACES = 5

    cpu_count = int(os.cpu_count() * 0.5)

    def run(self):
        with self.input().open('r') as file_reader:
            files_dict = [json.loads(line) for line in file_reader]

        files_dict_map = self.setup_file_dicts(files_dict, self.task_name)

        print('Conduct %s ...' % self.task_name)

        with ProcessPool(max_workers=self.cpu_count, max_tasks=10) as pool:
            returned_results = pool.map(self.detect_aggregations, files_dict, timeout=self.timeout).result()

        self.process_results(returned_results, files_dict_map, self.task_name)

        with self.output().open('w') as file_writer:
            for file_output_dict in tqdm(files_dict_map.values(), desc='Serialize %s results' % self.task_name):
                file_writer.write(json.dumps(file_output_dict, cls=FileJsonEncoder) + '\n')

    @abstractmethod
    def detect_aggregations(self, file_dict) -> (File, File):
        pass

    @abstractmethod
    def setup_file_dicts(self, file_dicts, caller_name):
        pass

    def process_results(self, results, files_dict_map, task_name):
        while True:
            try:
                result = next(results)
            except StopIteration:
                break
            except TimeoutError:
                pass
            except ValueError:
                continue
            else:
                row_wise_result = result[0]
                column_wise_result = result[1]
                file_name = row_wise_result.file_name
                sheet_name = row_wise_result.sheet_name
                merged_result = copy(row_wise_result)
                for nf in merged_result.aggregation_detection_result.keys():
                    merged_result.aggregation_detection_result[nf].extend(column_wise_result.aggregation_detection_result[nf])
                merged_result.exec_time[task_name] = merged_result.exec_time['RowWiseDetection'] + column_wise_result.exec_time['ColumnWiseDetection']
                files_dict_map[(file_name, sheet_name)] = merged_result

    def to_number(self, value, operator):
        number = None
        if operator == AggregationOperator.SUM.value:
            if value in empty_cell_values:
                number = Decimal(0)
        elif operator == AggregationOperator.AVERAGE.value:
            pass
        elif operator == AggregationOperator.DIVISION.value:
            pass
        elif operator == AggregationOperator.RELATIVE_CHANGE.value:
            pass
        else:
            raise NotImplementedError

        try:
            number = Decimal(value)
        except decimal.InvalidOperation:
            pass
        return number
