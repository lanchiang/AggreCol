# Created by lan at 2021/4/30
import decimal
import os
from abc import ABC, abstractmethod
from decimal import Decimal

from elements import File
from helpers import AggregationOperator, empty_cell_values
from tree import AggregationRelationForest


class AggregationDetection(ABC):
    """
    Abstract class for an aggregation detection aggrecol
    """

    # a python dict that stores metadata of the file, such as annotations, number format, file values, etc.
    file: File = None

    # string of operator name being detected, must be a valid value from the AggregationOperator enum
    operator = None

    # task name, used as identifier for individual steps
    task_name = None

    # Time in second that an aggrecol may use at most
    TIMEOUT = 300

    # number of digit places considered for numbers
    DIGIT_PLACES = 5

    # number of used cpu cores
    CPU_COUNT = int(os.cpu_count() * 0.5)

    def run(self):
        row_wise_aggregations = self.detect_row_wise_aggregations()
        column_wise_aggregations = self.detect_column_wise_aggregations()
        return row_wise_aggregations, column_wise_aggregations

    @abstractmethod
    def detect_row_wise_aggregations(self) -> File:
        pass

    @abstractmethod
    def detect_column_wise_aggregations(self) -> File:
        pass

    @abstractmethod
    def detect_proximity_aggregation_relations(self, forest: AggregationRelationForest, error_bound: float):
        pass


class AggrdetApproach(AggregationDetection):

    error_level_setting = {'Sum': 0, 'Average': 0, 'Division': 0, 'RelativeChange': 0}
    target_aggregation_type = 'All'

    debug = False

    # this parameter controls at least how many percentage of the numeric cells in a line must be aggregations,
    # in order to treat this line as an aggregator line.

    def __init__(self, file, parameters) -> None:
        # a python dict that stores metadata of the file, such as annotations, number format, file values, etc.
        self.file = file if isinstance(file, File) else None

        if not isinstance(parameters, dict):
            raise RuntimeError('Illegal parameter type.')

        self.TIMEOUT = parameters.get('timeout', 300)
        self.DIGIT_PLACES = parameters.get('digit_places', 5)
        self.coverage = parameters.get('coverage', 0.7)
        self.error_level_setting = parameters.get('error_level_dict', self.error_level_setting)
        self.error_level_setting[AggregationOperator.SUBTRACT.value] = self.error_level_setting.get(AggregationOperator.SUM.value, None)

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


