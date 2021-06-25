# Created by lan at 2021/4/28
import math
import os
from abc import ABC

import luigi
from luigi.mock import MockTarget

from aggrecol.individual import SumDetection, SubtractionDetection, AverageDetection, DivisionDetection, RelativeChangeDetection
from aggrecol.approach import AggregationDetection
from dataprep import NumberFormatNormalization
from elements import File
from helpers import AggregationOperator


class IndividualAggregationDetection(AggregationDetection, ABC):

    def requires(self):
        return NumberFormatNormalization(self.dataset_path, self.result_path,
                                         self.error_level_dict, self.use_extend_strategy,
                                         coverage=self.coverage,
                                         timeout=self.timeout,
                                         debug=self.debug)

    def setup_file_dicts(self, file_dicts, caller_name):
        files_dict_map = {}
        for file_dict in file_dicts:
            file_dict['detected_number_format'] = ''
            file_dict['detected_aggregations'] = []
            file_dict['aggregation_detection_result'] = {file_dict['number_format']: []}
            file_dict['exec_time'][caller_name] = math.nan

            files_dict_map[(file_dict['file_name'], file_dict['sheet_name'])] = file_dict
        return files_dict_map


class AdjacentListAggregationDetection(IndividualAggregationDetection, ABC):
    pass


class SlidingAggregationDetection(IndividualAggregationDetection, ABC):
    # Specify how many neighboring cells (in the same rows/columns) must be considered when looking for the aggregatees of an aggregator candidate
    WINDOW_SIZE = 10
    pass


class RelativeChangeDetectionTask(SlidingAggregationDetection):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.operator = AggregationOperator.RELATIVE_CHANGE.value
        self.task_name = self.__class__.__name__
        self.error_level = self.error_level_dict[self.operator] if self.operator in self.error_level_dict else self.error_level

    def output(self):
        return luigi.LocalTarget(os.path.join(self.result_path, 'aggrdet-relative-change-detection.jl'))

    def detect_aggregations(self, file_dict) -> (File, File):
        file = File(file_dict)
        detector = RelativeChangeDetection(file, parameters=file.parameters)
        row_wise_aggregations, column_wise_aggregations = detector.run()
        return row_wise_aggregations, column_wise_aggregations


class DivisionDetectionTask(SlidingAggregationDetection):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.operator = AggregationOperator.DIVISION.value
        self.task_name = self.__class__.__name__
        self.error_level = self.error_level_dict[self.operator] if self.operator in self.error_level_dict else self.error_level

    def output(self):
        return luigi.LocalTarget(os.path.join(self.result_path, 'aggrdet-division-detection.jl'))

    def detect_aggregations(self, file_dict) -> (File, File):
        file = File(file_dict)
        detector = DivisionDetection(file, parameters=file.parameters)
        row_wise_aggregations, column_wise_aggregations = detector.run()
        return row_wise_aggregations, column_wise_aggregations


class AverageDetectionTask(AdjacentListAggregationDetection):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.operator = AggregationOperator.AVERAGE.value
        self.task_name = self.__class__.__name__
        self.error_level = self.error_level_dict[self.operator] if self.operator in self.error_level_dict else self.error_level

    def output(self):
        return luigi.LocalTarget(os.path.join(self.result_path, 'aggrdet-average-detection.jl'))

    def detect_aggregations(self, file_dict) -> (File, File):
        file = File(file_dict)
        detector = AverageDetection(file, parameters=file.parameters)
        row_wise_aggregations, column_wise_aggregations = detector.run()
        return row_wise_aggregations, column_wise_aggregations


class SubtractionDetectionTask(SlidingAggregationDetection):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.operator = AggregationOperator.SUBTRACT.value
        self.task_name = self.__class__.__name__
        self.error_level = self.error_level_dict[self.operator] if self.operator in self.error_level_dict else self.error_level

    def output(self):
        return luigi.LocalTarget(os.path.join(self.result_path, 'aggrdet-subtraction-detection.jl'))

    def is_equal(self, aggregator_value, aggregatees, based_aggregator_value, error_bound):
        pass

    def detect_aggregations(self, file_dict) -> (File, File):
        file = File(file_dict)
        subtraction_detector = SubtractionDetection(file, parameters=file.parameters)
        row_wise_aggregations, column_wise_aggregations = subtraction_detector.run()
        return row_wise_aggregations, column_wise_aggregations


class SumDetectionTask(AdjacentListAggregationDetection):
    """
    The task to execute sum detection.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.operator = AggregationOperator.SUM.value
        self.task_name = self.__class__.__name__
        self.error_level = self.error_level_dict[self.operator] if self.operator in self.error_level_dict else self.error_level

    def output(self):
        return luigi.LocalTarget(os.path.join(self.result_path, 'aggrdet-sum-detection.jl'))

    def detect_aggregations(self, file_dict) -> (File, File):
        file = File(file_dict)
        sum_detector = SumDetection(file, parameters=file.parameters)
        row_wise_aggregations, column_wise_aggregations = sum_detector.run()
        return row_wise_aggregations, column_wise_aggregations
